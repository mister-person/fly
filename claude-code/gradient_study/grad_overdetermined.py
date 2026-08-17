"""Why a gradient method: on an OVER-DETERMINED hidden neuron, a gradient finds a
least-squares compromise where the direct linear solve can only conflict.

Net N0(input) -> N1(hidden) -> N2(output), two weights [w01, w12].
N1 fires ~periodically (period set by w01); N2 fires at N1's spikes + a latency
(set by w12).  So N2's spike times are strongly coupled: two weights cannot place
3 output spikes at arbitrary irregular times -> over-determined.

  DIRECT solve (target prop): invert each output target to an N1 target + latency,
    solve per constraint -> the constraints disagree, so it can satisfy only some.
  GRADIENT (van-Rossum output loss, which is built on the linear PSP): descends to
    the least-squares compromise -> lower TOTAL timing error.

The gradient's dV/dw is the exact linear PSP (the same h the direct solve uses);
the difference is the gradient minimises the residual across ALL constraints
jointly instead of solving one hop exactly.
"""

import sys, os, dataclasses, types
sys.path.insert(0, "/workspace/project/gradient_study")
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax_spiking_model as sim
from grad_method import lif_tangent, TH, DIR

params = dataclasses.replace(sim.default_params, steps=460)
T = params.steps
GSW = params.global_synapse_weight
C = np.array([[0, 1], [1, 2]], np.int32)
N = 3


def full_sim(w):
    V, _, _ = sim.run_sim(params, jnp.array(C), N,
                          jnp.array(np.asarray(w, np.float32)), jnp.array([0]))
    return np.array(V)


def spikes(V, n):
    return np.where(V[:, n] >= TH)[0].tolist()


def expconv(x, decay):
    S = np.zeros_like(x, float); a = 0.0
    for i in range(len(x)):
        a = a * decay + x[i]; S[i] = a
    return S


def vr_loss(w, targets, tau=12.0):
    """van-Rossum loss on N2's spike train vs targets."""
    V = full_sim(w)
    decay = np.exp(-1.0 / tau)
    found = (V[:, 2] >= TH).astype(float)
    tgt = np.zeros(T)
    for t in targets:
        tgt[t] = 1.0
    return float(np.sum((expconv(found, decay) - expconv(tgt, decay)) ** 2))


def fd_grad(w, targets, eps=4.0):
    g = np.zeros(len(w))
    for i in range(len(w)):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        g[i] = (vr_loss(wp, targets) - vr_loss(wm, targets)) / (2 * eps)
    return g


def timing_err(w, targets):
    sp = spikes(full_sim(w), 2)
    if len(sp) != len(targets):
        return None, sp
    return float(np.sum([(a - b) ** 2 for a, b in zip(sorted(sp), sorted(targets))])), sp


def psp(t_in):
    ia = np.zeros((1, T), bool); ia[0, t_in] = True
    return lif_tangent(np.array([50.0]), ia, T)[0] / 50.0


def solve_w(t_in, t_star):
    h = psp(t_in)
    return TH / h[t_star] if (0 <= t_star < T and h[t_star] > 0) else None


def main():
    w0 = np.array([500., 500.])
    nat = spikes(full_sim(w0), 2)
    print(f"natural N2 spikes at w={w0.tolist()}: {nat}")

    # over-determined target: 4 spikes with IRREGULAR spacing that no single
    # (period, latency) pair can fit (natural is periodic ~100).
    targets = [123, 263, 343, 443]
    sp_gaps = [targets[i+1]-targets[i] for i in range(len(targets)-1)]
    print(f"OVER-DETERMINED output target: {targets}  (spacings {sp_gaps} — not equal)")

    # ── DIRECT solve (target-prop inversion) ─────────────────────────────────
    # pick a latency, invert each output target to an N1 target, solve w01 per
    # constraint; the three implied w01 disagree -> use their median (best hard).
    tN0 = spikes(full_sim(w0), 0)[0]
    N1nat = spikes(full_sim(w0), 1)
    lat = nat[0] - N1nat[0]                        # N1->N2 latency at w=500
    w12_direct = solve_w(N1nat[0], nat[0])          # latency-matching w12
    n1_targets = [t - lat for t in targets]
    w01_candidates = [solve_w(tN0, max(20, t)) for t in n1_targets]
    w01_candidates = [x for x in w01_candidates if x]
    w01_direct = float(np.median(w01_candidates))
    w_direct = np.array([w01_direct, w12_direct])
    e_direct, sp_direct = timing_err(w_direct, targets)
    print(f"\nDIRECT solve: w={np.round(w_direct,0).tolist()}  N2 spikes={sp_direct}  "
          f"target={targets}")

    # ── GRADIENT descent (van-Rossum) with Adam ──────────────────────────────
    w = w0.copy().astype(float); m = np.zeros(2); v = np.zeros(2)
    hist = []
    for t in range(1, 401):
        g = np.nan_to_num(fd_grad(w, targets))
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
        w = np.clip(w - 6.0 * mh / (np.sqrt(vh) + 1e-12), 40, 3000)
        if t % 40 == 0:
            hist.append(vr_loss(w, targets))
    e_grad, sp_grad = timing_err(w, targets)
    print(f"GRADIENT:     w={np.round(w,0).tolist()}  N2 spikes={sp_grad}  target={targets}")

    def rms(e):
        return f"{np.sqrt(e/len(targets)):.1f}" if e is not None else "count mismatch"
    print(f"\nRMS timing error:  DIRECT solve = {rms(e_direct)}   "
          f"GRADIENT = {rms(e_grad)}  steps")

    # ── figure: N2 voltage & spikes, direct vs gradient, vs targets ──────────
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a, w_, lab, c in [(ax[0], w_direct, "DIRECT solve", "C3"),
                          (ax[1], w, "GRADIENT (least-squares)", "C0")]:
        V = full_sim(w_)
        a.plot(V[:, 2], color=c)
        a.axhline(TH, color="k", ls="--", lw=1)
        for tt in targets:
            a.axvline(tt, color="green", ls=":", lw=1)
        sp = spikes(V, 2)
        a.plot(sp, [TH] * len(sp), "v", color=c, ms=9)
        e, _ = timing_err(w_, targets)
        a.set_title(f"{lab}: N2 spikes {sp}  (RMS err {rms(e)})", fontsize=10)
        a.set_ylabel("V(N2)"); a.set_ylim(0, TH * 1.5)
    ax[-1].set_xlabel("t")
    fig.suptitle("Over-determined hidden neuron: gradient finds the least-squares "
                 "compromise the direct solve can't\n(green ⋮ = output targets)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{DIR}/grad_overdetermined.png", dpi=120)
    print(f"wrote {DIR}/grad_overdetermined.png")


if __name__ == "__main__":
    main()
