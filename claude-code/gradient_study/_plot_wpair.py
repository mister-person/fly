"""Two weights over training, separately and against each other.

    python3 _plot_wpair.py "3n L" [seed] [ia] [ib] [rounds]

TOP TWO   each weight against iteration -- the ordinary view, which shows WHEN things happen.
BOTTOM    the same trajectory in the (wa, wb) plane, coloured by iteration.  A time series
          cannot show whether two weights move together, trade off, or circle; a phase plot
          can, and that is the question here.  The SOLUTION SET is shaded underneath (every
          pair that reproduces the target exactly, with the other weights held at their final
          values), so "did it approach the answer and slide along it" is directly readable.

Sequential blue for time, because iteration is a magnitude.  The solution set is a neutral
fill rather than a hue: it is a region, not a series.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import field_trace as F
from _diag import CASES, steps_for

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("t", SEQ)
C_SOL, C_TRUE = "#d8ecd8", "#0b0b0b"

name = sys.argv[1] if len(sys.argv) > 1 else "3n L"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
IA = int(sys.argv[3]) if len(sys.argv) > 3 else 1
IB = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ROUNDS = int(sys.argv[5]) if len(sys.argv) > 5 else 800

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
lab = F.wlabels(C)

hist = []
w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
F.train(C, N, outs, w0.copy(), T, p, rounds=ROUNDS, lr=F.LR,
        cb=lambda it, w, *a: hist.append(np.asarray(w, float).copy()))
H = np.array(hist)
fin = H[-1]

# solution set in the (IA, IB) plane, other weights at their final values
xa = np.linspace(min(H[:, IA].min(), Wl[IA]) * 0.9, max(H[:, IA].max(), Wl[IA]) * 1.1, 90)
xb = np.linspace(min(H[:, IB].min(), Wl[IB]) * 1.1, max(H[:, IB].max(), Wl[IB]) * 0.9, 90)
sol = np.zeros((len(xb), len(xa)), bool)
for j, b in enumerate(xb):
    for i, a in enumerate(xa):
        w = fin.copy(); w[IA] = a; w[IB] = b
        sol[j, i] = F.sp(F.fsim(C, N, np.asarray(w, np.float32), p), outs[0]) == T[outs[0]]

fig = plt.figure(figsize=(10.6, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(3, 1, height_ratios=[0.72, 0.72, 1.7], hspace=0.45)
it = np.arange(1, len(H) + 1)
for k, wi in enumerate((IA, IB)):
    ax = fig.add_subplot(gs[k])
    ax.set_facecolor(SURFACE)
    ax.axhline(float(Wl[wi]), color=C_TRUE, lw=1.4, ls=(0, (5, 3)), zorder=3)
    ax.plot(it, H[:, wi], color="#2a78d6", lw=1.8, zorder=4)
    ax.set_ylabel(lab[wi], fontsize=9, color=INK2)
    ax.set_title(f"{lab[wi]}   {H[0, wi]:.0f} → {H[-1, wi]:.0f}   ·   true {Wl[wi]:.0f} (dashed)",
                 loc="left", fontsize=9, color=INK, pad=4)
    if k == 1:
        ax.set_xlabel("iteration", fontsize=9, color=INK2)

ax = fig.add_subplot(gs[2])
ax.set_facecolor(SURFACE)
if sol.any():
    ax.contourf(xa, xb, sol.astype(float), levels=[0.5, 1.5], colors=[C_SOL], zorder=0)
    ax.plot([], [], "s", ms=10, mfc=C_SOL, mec="#b9d6b9", label="exact-solution set")
pts = np.column_stack([H[:, IA], H[:, IB]]).reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap=CMAP, norm=Normalize(1, len(H)), lw=2.0, zorder=3)
lc.set_array(it[:-1])
ax.add_collection(lc)
ax.plot([H[0, IA]], [H[0, IB]], "o", ms=9, mfc=SEQ[0], mec=INK, mew=1.4, zorder=5,
        label="start")
ax.plot([H[-1, IA]], [H[-1, IB]], "o", ms=9, mfc=SEQ[-1], mec=SURFACE, mew=1.6, zorder=5,
        label="end")
ax.plot([Wl[IA]], [Wl[IB]], "*", ms=17, mfc=C_TRUE, mec=SURFACE, mew=1.2, zorder=6,
        label="true weights")
ax.set_xlabel(lab[IA], fontsize=9.5, color=INK2)
ax.set_ylabel(lab[IB], fontsize=9.5, color=INK2)
ax.set_title("the same trajectory in the weight plane, coloured by iteration",
             loc="left", fontsize=9.5, color=INK, pad=5)
ax.legend(frameon=False, fontsize=8.4, loc="best", labelcolor=INK2, handlelength=1.4)
cb = fig.colorbar(lc, ax=ax, pad=0.012, fraction=0.04)
cb.set_label("iteration", fontsize=8.6, color=INK2)
cb.ax.tick_params(colors=INK2, labelsize=7.8, length=3)
cb.outline.set_visible(False)
for a in fig.axes:
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        a.spines[s_].set_color("#dcdcd6")
    a.tick_params(colors=INK2, labelsize=8.2, length=3)
V = F.fsim(C, N, np.asarray(fin, np.float32), p)
o = F.sp(V, outs[0])
fig.suptitle(f"{name} seed {seed} — {lab[IA]} and {lab[IB]} over training",
             x=0.008, y=0.986, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.957, f"final {F.wstr(C, fin)}   ·   output {o} vs target {T[outs[0]]}",
         ha="left", fontsize=8.6, color=INK2)
fig.subplots_adjust(top=0.905, left=0.10, right=0.95, bottom=0.065)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"wpair_{name.replace(' ','_')}_s{seed}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
print(f"  {lab[IA]}: {H[0,IA]:.0f} -> {H[-1,IA]:.0f}  (true {Wl[IA]:.0f})")
print(f"  {lab[IB]}: {H[0,IB]:.0f} -> {H[-1,IB]:.0f}  (true {Wl[IB]:.0f})")
print(f"  solution cells in the scanned window: {int(sol.sum())} of {sol.size}")
print(f"  output {o}  target {T[outs[0]]}")
