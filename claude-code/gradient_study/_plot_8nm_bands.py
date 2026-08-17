"""8n M: the accumulator period is a STEP function of one weight, and that is the whole case.

    python3 _plot_8nm_bands.py [seeds]

Each hidden accumulator N1..N4 has exactly ONE input (N0->Nk), so its firing period depends
on that single weight alone -- and only through which quantisation BAND the weight lands in.
The design needs four different periods, so the task is to place four weights in four
specified bands, and w03's band edge sits 3 units from its true value.

TOP  one panel per accumulator: period against its weight, drawn as the step function it is,
     with the true weight and where each seed actually landed.  A band the run must hit is
     shaded; landing anywhere in it is equivalent, landing outside is a different network.
BOTTOM the output raster -- targets against what the run produced -- so the consequence of a
     mis-banded accumulator is visible as the few-timestep output error it turns into.
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
C_STEP, C_TRUE, C_BAND = "#2a78d6", "#0b0b0b", "#cde2fb"
SEEDC = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 4
name = "8n M"
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
lab = F.wlabels(C)
ACC = [(0, 1), (1, 2), (2, 3), (3, 4)]        # (weight index, accumulator neuron)

reached, sp_seed = [], []
for seed in range(NSEED):
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = F.train(C, N, outs, w0.copy(), T, p, rounds=800, lr=F.LR)
    reached.append(np.asarray(w, float))
    V = F.fsim(C, N, np.asarray(w, np.float32), p)
    sp_seed.append({n: F.sp(V, n) for n in range(N)})


def period_of(wi, n, val):
    w = np.array(Wl, float); w[wi] = val
    s = F.sp(F.fsim(C, N, np.asarray(w, np.float32), p), n)
    return int(round(float(np.mean(np.diff(s))))) if len(s) > 2 else 0


fig = plt.figure(figsize=(11.6, 6.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.5], hspace=0.42, wspace=0.28)
xs = np.arange(120, 1001, 3)
for k, (wi, n) in enumerate(ACC):
    ax = fig.add_subplot(gs[0, k])
    ax.set_facecolor(SURFACE)
    per = np.array([period_of(wi, n, v) for v in xs])
    tgt = period_of(wi, n, float(Wl[wi]))
    inband = per == tgt
    if inband.any():
        ax.axvspan(xs[inband].min(), xs[inband].max(), color=C_BAND, zorder=0, lw=0)
    ax.step(xs, per, where="post", color=C_STEP, lw=2.0, zorder=3)
    ax.axvline(float(Wl[wi]), color=C_TRUE, lw=1.6, ls=(0, (5, 3)), zorder=4)
    for si, w in enumerate(reached):
        ax.plot([w[wi]], [period_of(wi, n, w[wi])], "o", ms=8,
                mfc=SEEDC[si % 4], mec=SURFACE, mew=1.6, zorder=6,
                label=f"seed{si}" if k == 0 else None)
    lo = xs[inband].min() if inband.any() else 0
    hi = xs[inband].max() if inband.any() else 0
    ax.set_title(f"{lab[wi]} → N{n}   true {Wl[wi]:.0f}, period {tgt}\n"
                 f"band {lo:.0f}..{hi:.0f}   (edge {abs(Wl[wi]-lo):.0f} away)",
                 loc="left", fontsize=8.6, color=INK, pad=4)
    ax.set_xlabel("weight", fontsize=8.2, color=INK2)
    if k == 0:
        ax.set_ylabel("firing period (steps)", fontsize=8.4, color=INK2)
        ax.legend(frameon=False, fontsize=7.2, loc="lower right", labelcolor=INK2,
                  handlelength=1.0, ncol=2)
ax = fig.add_subplot(gs[1, :])
ax.set_facecolor(SURFACE)
rows = []
for o in outs:
    rows.append((f"N{o} target", T[o], MUTED, "o", 8, "none"))
    for si in range(NSEED):
        rows.append((f"N{o} seed{si}", sp_seed[si][o], SEEDC[si % 4], "|", 11, None))
for i, (nm_, times, col, mk, ms, mfc) in enumerate(rows):
    y = len(rows) - i
    if mk == "o":
        ax.plot(times, [y] * len(times), mk, ms=ms, mfc=mfc, mec=col, mew=1.6, zorder=3)
    else:
        ax.plot(times, [y] * len(times), mk, ms=ms, color=col, mew=2.0, zorder=3)
    ax.text(-14, y, nm_, ha="right", va="center", fontsize=7.8, color=INK2)
ax.set_ylim(0.3, len(rows) + 0.7)
ax.set_xlim(0, p.steps)
ax.set_yticks([])
ax.set_xlabel("timestep", fontsize=9, color=INK2)
ax.set_title("output raster — hollow ring = target, tick = what the run produced",
             loc="left", fontsize=9, color=INK, pad=5)
for a in fig.axes:
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        a.spines[s_].set_color("#dcdcd6")
    a.tick_params(colors=INK2, labelsize=7.8, length=3)
fig.suptitle("8n M — each accumulator's period is a step function of one weight",
             x=0.008, y=0.985, ha="left", fontsize=12.5, color=INK)
fig.text(0.008, 0.955, "shaded = the band the weight must land in; anywhere inside is "
         "equivalent, outside is a different network", ha="left", fontsize=8.6, color=INK2)
fig.subplots_adjust(top=0.90, left=0.075, right=0.99, bottom=0.065)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "8nm_bands.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
for k, (wi, n) in enumerate(ACC):
    tgt = period_of(wi, n, float(Wl[wi]))
    got = [period_of(wi, n, w[wi]) for w in reached]
    print(f"  {lab[wi]} -> N{n}: true {Wl[wi]:.0f} (period {tgt});  seeds reached "
          f"{[int(round(w[wi])) for w in reached]} -> periods {got}  "
          f"{'ALL OK' if all(g == tgt for g in got) else 'MIS-BANDED'}")
