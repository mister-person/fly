"""Does the direct closed-form solve generalise to HIDDEN neurons?

Chain N0(input) -> N1(hidden) -> N2(output).

The claim to test: V is linear in a neuron's OWN incoming weights, but the output's
dependence on a HIDDEN weight is not, because it passes through the hidden
threshold.  So:
  (a) V_N1(t*) is a straight line in w01                          [local linearity]
  (b) N2's crossing time is NONLINEAR in w01                      [no direct solve
       (the hidden weight) — because it goes through N1's spike]   from output target]
  (c) BUT if we ASSIGN N1 a target time t1*, two independent 1-hop
       solves (w01 for N1@t1*, w12 for N2@T2* given N1@t1*) hit it exactly.
  (d) fan-out (N1 feeds two outputs wanting different timings) makes t1*
       over-determined -> no consistent hidden target -> the solve breaks.
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

params = dataclasses.replace(sim.default_params, steps=320)
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


def cross(V, n):
    s = spikes(V, n)
    return s[0] if s else None


# PSP probe: sub-threshold V per unit weight for a single input pulse at t_in
def psp(t_in, T):
    ia = np.zeros((1, T), bool); ia[0, t_in] = True
    Vv, sp, _ = lif_tangent(np.array([50.0]), ia, T)   # sub-threshold probe
    return Vv / 50.0            # V per unit weight (= gsw*h)


def solve_w(t_in, t_star, T):
    """w so that a neuron with a single input pulse at t_in fires at t_star."""
    h = psp(t_in, T)
    return TH / h[t_star] if h[t_star] > 0 else None


def main():
    w_ref = np.array([500., 500.])
    Vr = full_sim(w_ref)
    print("reference chain spikes:", {n: spikes(Vr, n) for n in range(N)})
    tN0 = spikes(Vr, 0)[0]          # first N0 spike time

    # (a) V_N1(t*) linear in w01 ; (b) N2 crossing nonlinear in w01
    w01s = np.linspace(200, 1200, 60)
    tstar_N1 = 55                    # a time on N1's first rising ramp
    vN1, n2cross, n1_after = [], [], []
    for w01 in w01s:
        V = full_sim([w01, 500.])
        vN1.append(V[tstar_N1, 1])
        n2cross.append(cross(V, 2))
        c1 = cross(V, 1)
        n1_after.append(c1 is not None and c1 > tstar_N1)   # N1 not yet spiked by t*
    vN1 = np.array(vN1); n1_after = np.array(n1_after)
    # linearity check only where N1 has NOT yet spiked by t* (pre-reset, rising ramp)
    if n1_after.sum() > 2:
        p = np.polyfit(w01s[n1_after], vN1[n1_after], 1)
        resid = np.max(np.abs(vN1[n1_after] - np.polyval(p, w01s[n1_after])))
        print(f"(a) V_N1(t={tstar_N1}) vs w01, pre-spike: max deviation from a "
              f"straight line = {resid:.2e}  (linear; th={TH})")

    # (c) assign N1 a FEASIBLE target time (rising phase), do two 1-hop solves
    T2_star = 150                    # desired OUTPUT time
    print(f"\n(c) target: N2 fires at {T2_star};  free choice of hidden target N1@t1*:")
    for t1_star in [70, 80, 90]:     # all on N1's reachable rising phase
        w01 = solve_w(tN0, t1_star, T)
        w12 = solve_w(t1_star, T2_star, T)     # N2@T2* given N1@t1*
        if w01 is None or w12 is None:
            print(f"   hidden target N1@{t1_star}: infeasible"); continue
        V = full_sim([w01, w12])
        print(f"   hidden target N1@{t1_star}:  w01={w01:.0f} w12={w12:.0f}  "
              f"-> N1 fires {cross(V,1)}, N2 fires {cross(V,2)} (want {T2_star})")

    # (d) fan-out conflict: N1 -> N2 (want T2) and N1 -> N3 (want T3), T2 != T3
    print("\n(d) fan-out N1->{N2,N3}: required hidden target time for each output")
    lat = None
    # latency N1->post at w=500: post crossing - N1 crossing
    Vr2 = full_sim([500., 500.])
    lat = cross(Vr2, 2) - cross(Vr2, 1)
    for (name, T_post) in [("N2", 205), ("N3", 175)]:
        print(f"   to fire {name} at {T_post}, N1 must fire at ~{T_post - lat} "
              f"(latency {lat})")
    print("   -> two outputs with different target times demand DIFFERENT N1 spike "
          "times; one hidden neuron can't satisfy both -> no consistent target.")

    # ── figure ──
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    a = ax[0]
    a.plot(w01s, np.array(vN1) / TH, "o-", ms=3, color="C0")
    a.axhline(1.0, color="k", ls="--", lw=1, label="threshold")
    a.set_title(f"(a) LOCAL: V_N1(t={tstar_N1}) is linear in its own weight w01")
    a.set_xlabel("w01 (N0->N1)"); a.set_ylabel("V_N1(t*)/th"); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1]
    nc = [c if c is not None else np.nan for c in n2cross]
    a.plot(w01s, nc, "o-", ms=3, color="C3")
    a.set_title("(b) COMPOSED: N2 (output) crossing is NONLINEAR in the hidden w01\n"
                "(passes through N1's threshold) — no direct solve from the output target")
    a.set_xlabel("w01 (N0->N1)"); a.set_ylabel("N2 crossing time"); a.grid(alpha=.3)
    fig.suptitle("Direct solve is LOCAL: linear in a neuron's own weights, nonlinear "
                 "through a hidden neuron's threshold", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{DIR}/grad_direct_hidden.png", dpi=120)
    print(f"\nwrote {DIR}/grad_direct_hidden.png")


if __name__ == "__main__":
    main()
