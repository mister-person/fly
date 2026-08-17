"""Every seed's trajectory in one weight plane.

    python3 _plot_wpair_all.py "3n L" [nseeds] [ia] [ib] [rounds]

One seed's phase plot shows a trade-off; all of them show whether that trade-off is where
the failures live.  Trajectories are coloured by OUTCOME (exact / not) rather than by seed:
sixteen categorical hues is not a palette, and the question is not "which seed is this" but
"do the ones that work end up somewhere different".

The shaded region is the exact-solution set in this plane with the THIRD weight held at its
true value -- a fixed reference for all seeds, since each seed ends with its own third weight
and per-seed regions could not be drawn together.  The side panel carries that third weight,
because it turns out to be what separates the two groups.
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
C_OK, C_BAD, C_SOL = "#2a78d6", "#eb6834", "#d8ecd8"

name = sys.argv[1] if len(sys.argv) > 1 else "3n L"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
IA = int(sys.argv[3]) if len(sys.argv) > 3 else 1
IB = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ROUNDS = int(sys.argv[5]) if len(sys.argv) > 5 else 800
IC = [i for i in range(3) if i not in (IA, IB)][0]

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
lab = F.wlabels(C)

runs = []
for seed in range(NS):
    hist = []
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    ret = F.train(C, N, outs, w0.copy(), T, p, rounds=ROUNDS, lr=F.LR,
                  cb=lambda it, w, *a: hist.append(np.asarray(w, float).copy()))
    H = np.array(hist)
    o = F.sp(F.fsim(C, N, np.asarray(ret, np.float32), p), outs[0])
    runs.append(dict(seed=seed, H=H, fin=np.asarray(ret, float),
                     exact=(o == T[outs[0]]), out=o))

allH = np.vstack([r["H"] for r in runs])
xa = np.linspace(min(allH[:, IA].min(), Wl[IA]) * 0.95,
                 max(allH[:, IA].max(), Wl[IA]) * 1.05, 110)
xb = np.linspace(min(allH[:, IB].min(), Wl[IB]) * 1.05,
                 max(allH[:, IB].max(), Wl[IB]) * 0.95, 110)
sol = np.zeros((len(xb), len(xa)), bool)
for j, b in enumerate(xb):
    for i, a in enumerate(xa):
        w = np.array(Wl, float); w[IA] = a; w[IB] = b
        sol[j, i] = F.sp(F.fsim(C, N, np.asarray(w, np.float32), p), outs[0]) == T[outs[0]]

fig = plt.figure(figsize=(11.6, 6.8), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.24)

ax = fig.add_subplot(gs[0])
ax.set_facecolor(SURFACE)
if sol.any():
    ax.contourf(xa, xb, sol.astype(float), levels=[0.5, 1.5], colors=[C_SOL], zorder=0)
    ax.plot([], [], "s", ms=11, mfc=C_SOL, mec="#b9d6b9",
            label=f"exact-solution set (at true {lab[IC]})")
for r in runs:
    c = C_OK if r["exact"] else C_BAD
    ax.plot(r["H"][:, IA], r["H"][:, IB], color=c, lw=1.1, alpha=0.55, zorder=2)
    ax.plot([r["H"][0, IA]], [r["H"][0, IB]], "o", ms=5.5, mfc="none", mec=c, mew=1.3,
            zorder=3)
    ax.plot([r["fin"][IA]], [r["fin"][IB]], "o", ms=8, mfc=c, mec=SURFACE, mew=1.4,
            zorder=4)
ax.plot([Wl[IA]], [Wl[IB]], "*", ms=19, mfc=INK, mec=SURFACE, mew=1.3, zorder=6,
        label="true weights")
ax.plot([], [], "-", color=C_OK, lw=2, label="output EXACT")
ax.plot([], [], "-", color=C_BAD, lw=2, label="output wrong")
ax.set_xlabel(lab[IA], fontsize=9.5, color=INK2)
ax.set_ylabel(lab[IB], fontsize=9.5, color=INK2)
ax.set_title("hollow = start, filled = end", loc="left", fontsize=9.4, color=INK, pad=5)
ax.legend(frameon=False, fontsize=8.2, loc="best", labelcolor=INK2, handlelength=1.5)

ax = fig.add_subplot(gs[1])
ax.set_facecolor(SURFACE)
ax.axvline(float(Wl[IC]), color=INK, lw=1.5, ls=(0, (5, 3)), zorder=3)
for r in sorted(runs, key=lambda r: r["fin"][IC]):
    c = C_OK if r["exact"] else C_BAD
    y = r["seed"]
    ax.plot([r["H"][0, IC], r["fin"][IC]], [y, y], color=c, lw=1.2, alpha=0.5, zorder=2)
    ax.plot([r["H"][0, IC]], [y], "o", ms=5, mfc="none", mec=c, mew=1.2, zorder=3)
    ax.plot([r["fin"][IC]], [y], "o", ms=7.5, mfc=c, mec=SURFACE, mew=1.3, zorder=4)
ax.set_yticks([r["seed"] for r in runs])
ax.set_yticklabels([f"s{r['seed']}" for r in runs], fontsize=7.4)
ax.set_xlabel(lab[IC], fontsize=9.5, color=INK2)
ax.set_title(f"the third weight   ·   true {Wl[IC]:.0f} (dashed)",
             loc="left", fontsize=9.4, color=INK, pad=5)

for a in fig.axes:
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        a.spines[s_].set_color("#dcdcd6")
    a.tick_params(colors=INK2, labelsize=8.2, length=3)
nx = sum(1 for r in runs if r["exact"])
fig.suptitle(f"{name} — all {NS} seeds in the ({lab[IA]}, {lab[IB]}) plane   "
             f"[{nx}/{NS} exact]", x=0.008, y=0.982, ha="left", fontsize=12.5, color=INK)
fig.text(0.008, 0.948, f"true {F.wstr(C, Wl)}", ha="left", fontsize=8.8, color=INK2)
fig.subplots_adjust(top=0.885, left=0.075, right=0.985, bottom=0.09)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"wpairall_{name.replace(' ','_')}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
for r in sorted(runs, key=lambda r: r["fin"][IC]):
    print(f"  seed{r['seed']:>2}: {lab[IC]}={r['fin'][IC]:7.1f}  {lab[IA]}={r['fin'][IA]:7.1f}  "
          f"{lab[IB]}={r['fin'][IB]:8.1f}   {'EXACT' if r['exact'] else r['out']}")
