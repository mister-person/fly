"""3n L: the output error is set by the HIDDEN neuron's spike count, which nothing measures.

    python3 _plot_3nl_bands.py [seeds]

N1 has one input, so its spike COUNT is a step function of w01 alone.  The suite reports this
case as 100% count-ok because count is checked at the OUTPUT, where targets exist; N1 has no
target, so its count error is invisible and shows up as a few timesteps of output drift.

TOP    N1's count against w01, drawn as the step function it is.  The band the run must reach
       is shaded.  Each seed is placed where it landed, coloured by whether the OUTPUT came
       out exact -- so the correspondence between "right band" and "right answer" is the
       thing the panel shows, not something asserted in a caption.
BOTTOM the two spike trains per seed, N1 above N2, against their true times.  Seeds are
       ordered by N1's count so the two regimes separate visually.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import field_trace as F
from _diag import CASES, steps_for

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
C_STEP, C_BAND = "#2a78d6", "#cde2fb"
C_OK, C_BAD = "#2a78d6", "#eb6834"          # exact / not exact

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 8
name = "3n L"
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
lab = F.wlabels(C)

runs = []
for seed in range(NSEED):
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = np.asarray(F.train(C, N, outs, w0.copy(), T, p, rounds=800, lr=F.LR), float)
    V = F.fsim(C, N, np.asarray(w, np.float32), p)
    s = {n: F.sp(V, n) for n in range(N)}
    dc = len(s[2]) - len(T[2])
    off = [a - b for a, b in zip(s[2], T[2])] if dc == 0 else None
    exact = dc == 0 and max(abs(o) for o in off) == 0
    runs.append(dict(seed=seed, w=w, s1=s[1], s2=s[2], exact=exact,
                     err=(max(abs(o) for o in off) if off else 99)))

xs = np.arange(100, 1401, 2)
cnt = np.array([len(F.sp(F.fsim(C, N, np.asarray(
    [v, Wl[1], Wl[2]], np.float32), p), 1)) for v in xs])
tgt = len(T[1])

fig = plt.figure(figsize=(11.2, 8.2), facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.30)

ax = fig.add_subplot(gs[0])
ax.set_facecolor(SURFACE)
inb = cnt == tgt
if inb.any():
    ax.axvspan(xs[inb].min(), xs[inb].max(), color=C_BAND, zorder=0, lw=0)
    ax.annotate(f"{tgt} spikes — the band that works ({xs[inb].min()}..{xs[inb].max()})",
                xy=((xs[inb].min() + 1400) / 2, tgt + 0.45), ha="center",
                fontsize=8.8, color=INK2)
ax.step(xs, cnt, where="post", color=C_STEP, lw=2.2, zorder=3)
ax.axvline(float(Wl[0]), color=INK, lw=1.6, ls=(0, (5, 3)), zorder=4)
ax.annotate(f"true {lab[0]}={Wl[0]:.0f}", (float(Wl[0]), 0.15), xytext=(6, 0),
            textcoords="offset points", fontsize=8.6, color=INK)
for r in runs:
    c = C_OK if r["exact"] else C_BAD
    ax.plot([r["w"][0]], [len(r["s1"])], "o", ms=9, mfc=c, mec=SURFACE, mew=1.8, zorder=6)
    ax.annotate(f"s{r['seed']}", (r["w"][0], len(r["s1"])), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=7.6, color=INK2)
ax.plot([], [], "o", ms=9, mfc=C_OK, mec=SURFACE, mew=1.8, label="output EXACT")
ax.plot([], [], "o", ms=9, mfc=C_BAD, mec=SURFACE, mew=1.8, label="output wrong")
ax.legend(frameon=False, fontsize=8.4, loc="lower right", labelcolor=INK2, handlelength=1.2)
ax.set_xlabel(f"{lab[0]}   (N0 → N1, its only input)", fontsize=9.2, color=INK2)
ax.set_ylabel("N1 spike count", fontsize=9.2, color=INK2)
ax.set_title("N1's spike count is a step function of one weight — and N1 has no target, "
             "so this error is invisible to the suite",
             loc="left", fontsize=9.6, color=INK, pad=6)

ax = fig.add_subplot(gs[1])
ax.set_facecolor(SURFACE)
order = sorted(runs, key=lambda r: (len(r["s1"]), r["err"]))
y = 0
ylab = []
for r in order:
    c = C_OK if r["exact"] else C_BAD
    ax.plot(r["s1"], [y + 0.22] * len(r["s1"]), "|", ms=11, color=c, mew=2.2, zorder=3)
    ax.plot(r["s2"], [y - 0.22] * len(r["s2"]), "|", ms=11, color=c, mew=2.2, zorder=3)
    ax.plot(T[1], [y + 0.22] * len(T[1]), "o", ms=6.5, mfc="none", mec=MUTED, mew=1.3,
            zorder=2)
    ax.plot(T[2], [y - 0.22] * len(T[2]), "o", ms=6.5, mfc="none", mec=MUTED, mew=1.3,
            zorder=2)
    ylab.append((y, f"seed{r['seed']}   {lab[0]}={r['w'][0]:.0f}   "
                    f"N1×{len(r['s1'])}   |Δt| max {r['err'] if r['err']<99 else '--'}"))
    y -= 1
for yy, txt in ylab:
    ax.text(-16, yy, txt, ha="right", va="center", fontsize=7.8, color=INK2)
ax.set_ylim(y + 0.4, 0.8)
ax.set_xlim(0, p.steps)
ax.set_yticks([])
ax.set_xlabel("timestep", fontsize=9.2, color=INK2)
ax.set_title("upper tick row = N1 (hidden), lower = N2 (output)   ·   "
             "hollow ring = true time   ·   ordered by N1's count",
             loc="left", fontsize=9.6, color=INK, pad=6)
for a in fig.axes:
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        a.spines[s_].set_color("#dcdcd6")
    a.tick_params(colors=INK2, labelsize=8.2, length=3)
fig.suptitle("3n L — the output error is set by a hidden neuron's spike COUNT",
             x=0.008, y=0.985, ha="left", fontsize=12.5, color=INK)
fig.text(0.008, 0.955, f"true {F.wstr(C, Wl)}   ·   w12 is the only inhibitory weight in "
         "the suite", ha="left", fontsize=8.8, color=INK2)
fig.subplots_adjust(top=0.905, left=0.225, right=0.99, bottom=0.065)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3nl_bands.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
for r in order:
    print(f"  seed{r['seed']}: {lab[0]}={r['w'][0]:7.1f}  N1 fires {len(r['s1'])}/"
          f"{len(T[1])}  output |Δt| max "
          f"{r['err'] if r['err'] < 99 else 'count mismatch'}"
          f"   {'EXACT' if r['exact'] else ''}")
