"""Topology as an arc diagram, under the minimum-feedback-arc-set ordering.

    python3 _plot_topo.py "14n Q"

An arc diagram is the right form here because the question is about an ORDERING: nodes sit on
one line in the chosen order, forward edges arc below it and backward edges arc above, so the
count of arcs above IS the objective being minimised and can be read off directly.  A
force-directed blob would hide exactly that.

Geometry carries direction (below = forward, above = backward); colour carries SIGN, which is
the physical property -- excitatory vs inhibitory.  Nothing is encoded twice.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
from _diag import CASES
from _order import min_fas_order, classify

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
C_EXC, C_INH = "#2a78d6", "#eb6834"

name = sys.argv[1] if len(sys.argv) > 1 else "14n Q"
E, N, outs, Wl = CASES[name]
W = np.asarray(Wl, float)

order, nb = min_fas_order(E, N, fixed_first=0)
pos = {n: i for i, n in enumerate(order)}
fwd, bwd = classify(E, order)
print(f"{name}: N={N}, {len(E)} edges, exact minimum backward edges = {nb}")
print("order:", " -> ".join(f"N{n}" for n in order))
print("backward:", [f"N{E[i][0]}->N{E[i][1]}" for i in bwd])

fig, ax = plt.subplots(figsize=(max(9.0, 0.85 * N + 2.0), 4.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(0, color="#e0e0da", lw=1.2, zorder=1)

for i, (a, b) in enumerate(E):
    xa, xb = pos[int(a)], pos[int(b)]
    c = C_INH if W[i] < 0 else C_EXC
    up = i in bwd
    # `rad` bends relative to the DIRECTION OF TRAVEL, so one constant value sends every
    # forward edge below the line and every backward edge above it, with no case analysis.
    # Long spans get a shallower arc or they swamp the panel.
    span = abs(xb - xa)
    rad = 0.55 if span <= 2 else 0.55 * (2.0 / span) ** 0.55
    ax.add_patch(FancyArrowPatch(
        (xa, 0), (xb, 0), connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=9, lw=1.9 if up else 1.1,
        color=c, alpha=1.0 if up else 0.5, zorder=4 if up else 2,
        shrinkA=9, shrinkB=9))

for n in order:
    x = pos[n]
    if n == 0:
        mfc, mec, lw = INK, INK, 1.6
    elif n in outs:
        mfc, mec, lw = SURFACE, INK, 2.2
    else:
        mfc, mec, lw = "#e9e9e3", MUTED, 1.4
    ax.plot([x], [0], "o", ms=17, mfc=mfc, mec=mec, mew=lw, zorder=6)
    ax.text(x, 0, f"{n}", ha="center", va="center", fontsize=8.2, zorder=7,
            color=SURFACE if n == 0 else INK)
    role = "in" if n == 0 else ("out" if n in outs else "")
    if role:
        ax.text(x, 0.42, role, ha="center", va="bottom", fontsize=7.6, color=INK2)

ax.set_xlim(-0.8, N - 0.2)
ax.set_ylim(-2.45, 1.75)
ax.axis("off")
for c, lb in ((C_EXC, "excitatory"), (C_INH, "inhibitory")):
    ax.plot([], [], "-", color=c, lw=2.0, label=lb)
ax.legend(frameon=False, fontsize=8.6, loc="upper right", labelcolor=INK2, ncol=2)
fig.suptitle(f"{name} — topology under the minimum-feedback-arc-set order", x=0.008, y=0.99,
             ha="left", fontsize=12.5, color=INK)
fig.text(0.008, 0.912,
         f"{len(E)} edges, {sum(1 for w in W if w < 0)} inhibitory · "
         f"arcs BELOW the line run forward, arcs ABOVE run backward · "
         f"{nb} backward edge" + ("" if nb == 1 else "s") +
         f" (exact minimum over all {N}! orderings, by subset DP)",
         ha="left", fontsize=8.8, color=INK2)
fig.subplots_adjust(top=0.845, left=0.02, right=0.98, bottom=0.03)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"topo_{name.replace(' ', '_')}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
print("F_ORDER=" + ",".join(str(n) for n in order))
