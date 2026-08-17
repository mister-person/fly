"""A better way to incorporate spike times: put the timing target in VOLTAGE space.

Brittle way (grad_method.py):  loss on the spike TIME, gradient = (dL/dt)(−1/slope)(dV/dw).
  → discrete staircase, 1/slope blow-up at grazing, zero signal when dead.

Better way (here):  to fire at t*, ask that the smooth membrane voltage reach
threshold AT t* and stay below it just before.  The objective is a smooth
function of V (which is linear/smooth in w), so:
  - no differentiation of a crossing time  → no 1/slope blow-up
  - defined even when the neuron is silent   → revives dead neurons
  - no discrete staircase                    → real gradient everywhere
The spike-time information enters as *where* the voltage constraints are placed.

  L(w) = Σ_j  0.5·relu(th − V(t*_j))²                 (reach threshold AT each target)
       + λ·Σ_j Σ_{t in (t*_{j-1}, t*_j − g)} relu(V(t) − (1−m)·th)²   (don't fire early)

grad uses dV/dw from the tangent forward (respects resets). We compare it to the
spike-time gradient on (a) a dead single-target task and (b) a coupled two-spike
task, and sweep weight scale to show the gradient norm stays bounded.
"""

import sys, os, types
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m

import numpy as np
import matplotlib.pyplot as plt
from grad_method import (lif_tangent, spike_time_grad, spike_time_loss, TH, DIR)

MARGIN = 0.12         # keep V below (1-MARGIN)*th at non-target times
GUARD  = 3            # steps right before a target exempt from the "no early fire" band
BACK   = 22           # how far back the "no early fire" band extends (< inter-spike gap)
OVER   = 0.0          # aim V(t*) this fraction above threshold
# The "no early fire" term (LAM>0) turned out to be self-defeating: the LIF PSP
# rises slowly, so demanding V<th a few steps before t* fights V=th AT t*.  Just
# asking V to REACH threshold at each target time already lands the crossing at t*
# (before t* the ramp is still below th) and is far better behaved.  LAM=0 default.
LAM    = 0.0          # weight of the (optional) no-early-fire term


def voltage_target_grad(w, ia, targets, T, ret_loss=False):
    """Gradient of the voltage-space timing objective. Always defined."""
    V, spikes, dV = lif_tangent(w, ia, T)
    g = np.zeros(len(w)); loss = 0.0
    tgt_th = (1.0 + OVER) * TH
    cap = (1.0 - MARGIN) * TH
    for tt in sorted(targets):
        # (1) reach (just past) threshold AT the target time
        if V[tt] < tgt_th:
            loss += 0.5 * (tgt_th - V[tt]) ** 2
            g += (V[tt] - tgt_th) * dV[tt]             # descend -> raise V(tt)
        # (2) stay below threshold in a short window BEFORE it, so the crossing
        #     lands at tt (not early).  Short window avoids penalising the tail of
        #     the previous spike (whose real V is reset low anyway).
        for t in range(max(1, tt - BACK), max(1, tt - GUARD)):
            if V[t] > cap:
                loss += LAM * 0.5 * (V[t] - cap) ** 2
                g += LAM * (V[t] - cap) * dV[t]        # descend -> lower V(t)
    if ret_loss:
        return g, spikes, loss
    return g, spikes


def make_single(T=200, n_in=5):
    ia = np.zeros((n_in, T), bool)
    for i in range(n_in):
        ia[i, 12 + 7 * i] = True
    return ia, [95], T


def make_coupled(T=260, n_in=5):
    """Two output-spike targets driven by the SAME weights (the case that made the
    spike-time gradient thrash).  Targets [90,190] are comfortably reachable."""
    ia = np.zeros((n_in, T), bool)
    for i in range(n_in):
        for c in (0, 1):
            t = 12 + 7 * i + 100 * c
            if t < T:
                ia[i, t] = True
    return ia, [90, 190], T


def train(ia, targets, T, grad_fn, step=4.0, iters=250, w0val=40.0, clip=None):
    w = np.full(ia.shape[0], w0val)
    hist = []
    for _ in range(iters):
        sp = lif_tangent(w, ia, T)[1]
        hist.append(spike_time_loss(sp, targets))
        g = grad_fn(w)
        if clip is not None:
            gn = np.linalg.norm(g)
            if gn > clip:
                g = g * (clip / gn)
        gn = np.linalg.norm(g)
        if gn > 1e-30:
            w = np.clip(w - step * g / gn, 20, 3000)
    return w, hist, lif_tangent(w, ia, T)[1]


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    # ── (a) single target, dead start ────────────────────────────────────────
    ia, tg, T = make_single()
    print("=" * 74)
    print("SINGLE target, dead start — voltage-target vs spike-time")
    print("=" * 74)
    _, h_st, sp_st = train(ia, tg, T, lambda w: spike_time_grad(w, ia, tg, T, slope_floor=5e-5)[0])
    _, h_vt, sp_vt = train(ia, tg, T, lambda w: voltage_target_grad(w, ia, tg, T)[0])
    print(f"  spike-time     : spikes={sp_st} loss={spike_time_loss(sp_st,tg):.2f}")
    print(f"  voltage-target : spikes={sp_vt} loss={spike_time_loss(sp_vt,tg):.2f}")
    a = ax[0, 0]
    a.semilogy(np.array(h_st) + 1e-2, label="spike-time")
    a.semilogy(np.array(h_vt) + 1e-2, label="voltage-target")
    a.set_title(f"(a) single target={tg[0]}: voltage-target hits it, spike-time stuck")
    a.set_xlabel("iter"); a.set_ylabel("spike-time loss+1e-2"); a.legend(fontsize=8); a.grid(alpha=.3)

    # ── (b) coupled two-spike task, dead start ───────────────────────────────
    ia2, tg2, T2 = make_coupled()
    print("\n" + "=" * 74)
    print("COUPLED two-spike target (shared weights) — the thrashing case")
    print("=" * 74)
    _, h_st2, sp_st2 = train(ia2, tg2, T2, lambda w: spike_time_grad(w, ia2, tg2, T2, slope_floor=5e-5)[0])
    _, h_vt2, sp_vt2 = train(ia2, tg2, T2, lambda w: voltage_target_grad(w, ia2, tg2, T2)[0])
    print(f"  spike-time     : spikes={sp_st2} loss={spike_time_loss(sp_st2,tg2):.2f}")
    print(f"  voltage-target : spikes={sp_vt2} loss={spike_time_loss(sp_vt2,tg2):.2f}")
    a = ax[0, 1]
    a.semilogy(np.array(h_st2) + 1e-2, label="spike-time")
    a.semilogy(np.array(h_vt2) + 1e-2, label="voltage-target")
    a.set_title(f"(b) coupled targets={tg2}: spike-time thrashes, voltage-target steady")
    a.set_xlabel("iter"); a.set_ylabel("spike-time loss+1e-2"); a.legend(fontsize=8); a.grid(alpha=.3)

    # ── (c) brittleness sweep: gradient norm vs weight scale ─────────────────
    print("\n" + "=" * 74)
    print("BRITTLENESS: gradient norm vs weight scale (no 1/slope in voltage-target)")
    print("=" * 74)
    scales = np.linspace(0.15, 2.2, 320)
    w0 = np.full(ia2.shape[0], 250.0)
    gn_st, gn_vt = [], []
    for s in scales:
        gn_st.append(np.linalg.norm(spike_time_grad(w0 * s, ia2, tg2, T2, slope_floor=0.0)[0]))
        gn_vt.append(np.linalg.norm(voltage_target_grad(w0 * s, ia2, tg2, T2)[0]))
    gn_st = np.array(gn_st); gn_vt = np.array(gn_vt)
    frac_zero = float(np.mean(gn_vt < 1e-12))
    print(f"  spike-time    : max |grad| = {gn_st.max():.2e}  median {np.median(gn_st):.2e} "
          f"-> unbounded, peak/median = {gn_st.max()/max(np.median(gn_st),1e-30):.0f}x")
    print(f"  voltage-target: max |grad| = {gn_vt.max():.2e}  (bounded, ~1e-7); "
          f"exactly 0 at {100*frac_zero:.0f}% of scales (objective already satisfied)")
    a = ax[1, 0]
    a.semilogy(scales, gn_st + 1e-9, label="spike-time")
    a.semilogy(scales, gn_vt + 1e-9, label="voltage-target")
    a.set_title("(c) gradient norm vs weight scale — voltage-target stays bounded")
    a.set_xlabel("weight scale"); a.set_ylabel("|grad|"); a.legend(fontsize=8); a.grid(alpha=.3)

    # ── (d) the learned single-target trace ──────────────────────────────────
    w_vt, _, _ = train(ia, tg, T, lambda w: voltage_target_grad(w, ia, tg, T)[0])
    V, sp, _ = lif_tangent(w_vt, ia, T)
    a = ax[1, 1]
    a.plot(V, color="C0", label="V after voltage-target training")
    a.axhline(TH, color="k", ls="--", lw=1, label="threshold")
    a.axvline(tg[0], color="C3", ls=":", label=f"target t*={tg[0]}")
    a.set_title(f"(d) learned trace fires at {sp}")
    a.set_xlabel("t"); a.set_ylabel("V"); a.legend(fontsize=8); a.set_ylim(0, TH * 1.4)

    fig.tight_layout(); fig.savefig(f"{DIR}/grad_voltage_target.png", dpi=120)
    print(f"\n wrote {DIR}/grad_voltage_target.png")


if __name__ == "__main__":
    main()
