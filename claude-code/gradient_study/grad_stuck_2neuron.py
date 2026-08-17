"""Minimal reproduction of the create/suppress cancellation — TWO neurons, no
hidden layer.

One input neuron (a single pulse) -> one output neuron (single weight w).  The
output's crossing time is a monotndecreasing function of w (more weight = earlier
crossing).  Put the TARGET a bit more than WINDOW steps AFTER the reachable
crossing.  Then:

  * the current spike is > WINDOW from the target  -> SUPPRESS it (push V down =
    lower w = crossing moves LATER, toward the target)   [correct direction!]
  * the target has no spike within WINDOW           -> CREATE one (push V up =
    raise w = crossing moves EARLIER, away from the target) [wrong direction]

Both act on the same single weight and oppose.  If they roughly cancel, the net
gradient ~ 0 and the spike is STUCK at the wrong time — even though nothing is
hidden and the target is perfectly reachable.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from grad_method import lif_tangent, TH, DIR
from grad_multi_neuron import voltage_grad, WINDOW

T = 220
T_IN = 15
ia = np.zeros((1, T), bool); ia[0, T_IN] = True     # one input pulse


def crossing(w):
    sp = lif_tangent(np.array([float(w)]), ia, T)[1]
    return sp[0] if sp else None


def train(w0, target, iters=300, step=3.0):
    w = np.array([float(w0)])
    traj = []
    for _ in range(iters):
        sp = lif_tangent(w, ia, T)[1]
        traj.append(sp[0] if sp else np.nan)
        g = voltage_grad(w, ia, [target], T, suppress=True)[0]
        gn = np.linalg.norm(g)
        if gn > 1e-30:
            w = np.clip(w - step * g / gn, 20, 3000)
    return float(w[0]), traj


def decompose(w, target):
    """create-only, suppress-only, and net gradient on the single weight."""
    g_create = voltage_grad(np.array([float(w)]), ia, [target], T, suppress=False)[0][0]
    g_full   = voltage_grad(np.array([float(w)]), ia, [target], T, suppress=True)[0][0]
    return g_create, g_full - g_create, g_full     # create, suppress, net


def main():
    # pick an init weight whose crossing is ~25 (> WINDOW) BEFORE the target
    w0 = 700.0
    c0 = crossing(w0)
    target = c0 + (WINDOW + 5)          # just past the match window, LATER than crossing
    print(f"WINDOW={WINDOW}")
    print(f"init w={w0}: crossing={c0}   target={target}  (offset {target-c0} > WINDOW)")

    # is the target reachable at all?  find a weight whose crossing == target
    ws = np.arange(200, 900, 2.0)
    cs = np.array([crossing(w) or 1e9 for w in ws])
    w_reach = ws[np.argmin(np.abs(cs - target))]
    print(f"target IS reachable: w={w_reach:.0f} gives crossing={crossing(w_reach)}")

    # gradient decomposition at the init point
    gc, gs, net = decompose(w0, target)
    print(f"\ngradient at init w={w0}:")
    print(f"  create push  = {gc:+.3e}  (would move w this way)")
    print(f"  suppress push= {gs:+.3e}")
    print(f"  NET          = {net:+.3e}   -> |net| is {abs(net)/max(abs(gc),1e-30)*100:.0f}% "
          f"of the create push (they cancel)")

    # train and show it's stuck
    wf, traj = train(w0, target)
    print(f"\ntrained: w {w0:.0f}->{wf:.0f}   spike {traj[0]}->{traj[-1]}   target {target}")
    print(f"  stuck? final spike still {abs(traj[-1]-target):.0f} steps from target "
          f"(reachable weight was {w_reach:.0f})")

    # sweep target time -> final spike (the dead band)
    targets = list(range(c0 - 40, c0 + 45, 3))
    finals = []
    for tg in targets:
        _, tr = train(w0, tg, iters=250)
        finals.append(tr[-1])

    # ── figure ──
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    wgrid = np.arange(400, 900, 5.0)
    gcs, gss, nets = [], [], []
    for w in wgrid:
        c, s, n = decompose(w, target)
        gcs.append(c); gss.append(s); nets.append(n)
    a.plot(wgrid, gcs, color="green", label="create push")
    a.plot(wgrid, gss, color="red", label="suppress push")
    a.plot(wgrid, nets, color="k", lw=2, label="NET")
    a.axhline(0, color="gray", lw=.5); a.axvline(w0, color="C0", ls=":", label="init w")
    a.set_title("create & suppress oppose on the one weight\n(NET~0 over a band = stuck)")
    a.set_xlabel("weight w"); a.set_ylabel("gradient"); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1]
    a.plot(traj, color="C0")
    a.axhline(target, color="green", ls="--", label=f"target {target}")
    a.axhline(crossing(w_reach), color="gray", ls=":", label="reachable")
    a.set_title("training trajectory of the spike time (stuck)")
    a.set_xlabel("iteration"); a.set_ylabel("spike time"); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[2]
    a.plot(targets, finals, "o-", ms=4, label="final spike")
    a.plot(targets, targets, color="gray", ls="--", label="ideal (final=target)")
    a.axvspan(c0 - WINDOW, c0 + WINDOW, color="orange", alpha=.15, label="±WINDOW of natural crossing")
    a.set_title("final spike vs requested target\n(flat = can't move / stuck)")
    a.set_xlabel("target time"); a.set_ylabel("achieved spike time"); a.legend(fontsize=8); a.grid(alpha=.3)

    fig.suptitle("Two neurons, no hidden layer: a BAND of target times is unreachable — "
                 "the mistimed spike freezes\n(create↑ and suppress↓ oppose near the "
                 "creation transition; the hard match-WINDOW adds a tolerance floor)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{DIR}/grad_stuck_2neuron.png", dpi=120)
    print(f"\nwrote {DIR}/grad_stuck_2neuron.png")

    # dead-band width: targets whose achieved spike stayed within 3 of the natural crossing
    stuck = [tg for tg, fn in zip(targets, finals) if abs(fn - c0) <= 3]
    if stuck:
        print(f"UNREACHABLE band: targets {min(stuck)}..{max(stuck)} all froze at ~{c0} "
              f"(width {max(stuck)-min(stuck)} steps); natural crossing {c0}, WINDOW {WINDOW}")


if __name__ == "__main__":
    main()
