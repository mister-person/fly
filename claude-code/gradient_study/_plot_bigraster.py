"""Output raster for a large net: which target spikes were hit, and what was produced.

    python3 _plot_bigraster.py "50n A" [seed] [rounds]

One row per output neuron, split into two tick rows: TARGET above, PRODUCED below.  With ten
outputs and ~90 target spikes a table is unreadable and a summary statistic hides where the
errors are, so the raster is the right form -- position carries the answer directly.

Colour marks the pairing, not identity: a produced spike is scored by its distance to the
nearest unclaimed target (exact / within 5 / adrift), and a target with nothing near it is
drawn hollow-red.  Two hues plus a neutral, which is all the categorical load there is.
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
C_HIT, C_NEAR, C_BAD = "#2a78d6", "#eda100", "#eb6834"

name = sys.argv[1] if len(sys.argv) > 1 else "50n A"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
NEAR = 5

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
w = np.asarray(F.train(C, N, outs, w0.copy(), T, p, rounds=ROUNDS, lr=F.LR), float)
V = F.fsim(C, N, np.asarray(w, np.float32), p)
got = {o: F.sp(V, o) for o in outs}

rows, tally = [], dict(exact=0, near=0, adrift=0, missed=0, extra=0)
for o in outs:
    tg, gt = list(T[o]), list(got[o])
    free = list(range(len(tg)))
    pairs, used = {}, set()
    for _d, i, j in sorted((abs(tg[i] - gt[j]), i, j)
                           for i in range(len(tg)) for j in range(len(gt))):
        if i in used or j in pairs:
            continue
        used.add(i); pairs[j] = i
    rows.append((o, tg, gt, pairs, used))
    for j, t in enumerate(gt):
        if j not in pairs:
            tally["extra"] += 1
        else:
            d = abs(t - tg[pairs[j]])
            tally["exact" if d == 0 else ("near" if d <= NEAR else "adrift")] += 1
    tally["missed"] += len(tg) - len(used)

fig, ax = plt.subplots(figsize=(12.4, 0.62 * len(outs) + 2.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for r, (o, tg, gt, pairs, used) in enumerate(rows):
    y = len(rows) - r
    for i, t in enumerate(tg):
        c = MUTED if i in used else C_BAD
        ax.plot([t], [y + 0.17], "o", ms=5.5, mfc="none", mec=c, mew=1.5, zorder=3)
    for j, t in enumerate(gt):
        if j not in pairs:
            c = C_BAD
        else:
            d = abs(t - tg[pairs[j]])
            c = C_HIT if d == 0 else (C_NEAR if d <= NEAR else C_BAD)
        ax.plot([t], [y - 0.17], "|", ms=9, color=c, mew=2.0, zorder=3)
        if j in pairs:
            ax.plot([tg[pairs[j]], t], [y + 0.17, y - 0.17], "-", color=c, lw=0.7,
                    alpha=0.5, zorder=2)
    ax.text(-14, y, f"N{o}", ha="right", va="center", fontsize=8, color=INK2)
ax.set_ylim(0.35, len(rows) + 0.85)
ax.set_xlim(0, p.steps)
ax.set_yticks([])
ax.set_xlabel("timestep", fontsize=9.5, color=INK2)
for c, lb in ((C_HIT, "exact"), (C_NEAR, f"within {NEAR}"), (C_BAD, "adrift / extra / missed"),
              (MUTED, "target (hit)")):
    ax.plot([], [], "|", ms=10, color=c, mew=2.2, label=lb)
ax.legend(frameon=False, fontsize=8.4, loc="upper center", ncol=4, labelcolor=INK2,
          bbox_to_anchor=(0.5, 1.14), handlelength=1.0)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color("#dcdcd6")
ax.tick_params(colors=INK2, labelsize=8.4, length=3)
ntg = sum(len(t) for t in T.values() if t is not None and False) or sum(len(T[o]) for o in outs)
fig.suptitle(f"{name} seed {seed} — output spikes after {ROUNDS} rounds "
             f"(upper ring = target, lower tick = produced)",
             x=0.008, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.945,
         f"{tally['exact']} exact · {tally['near']} within {NEAR} · {tally['adrift']} adrift · "
         f"{tally['missed']} targets missed · {tally['extra']} spurious   "
         f"(of {ntg} target spikes across {len(outs)} outputs)",
         ha="left", fontsize=8.8, color=INK2)
fig.subplots_adjust(top=0.80, left=0.055, right=0.99, bottom=0.13)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"bigraster_{name.replace(' ','_')}_s{seed}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
print(f"  {tally}")
for o, tg, gt, pairs, used in rows:
    off = [gt[j] - tg[pairs[j]] for j in sorted(pairs)]
    print(f"   N{o}: {len(gt)} produced / {len(tg)} target   offsets {off[:10]}")
