"""Extract a MINIMAL example that actually replicates the 50-neuron failure.

Take one real neuron n from a 50-neuron case, feed it its presynaptic neurons'
TRUE spike times as fixed inputs, reconstruct its incoming weights from those true
inputs + its true target, then simulate THAT NEURON ALONE.  If it fires extra
spikes, we have the failure reproduced in isolation (no coupling, no input drift)
— the minimal faithful example to study and fix.
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
from homotopy_core import hard_sim as _hard_sim
from grad_method import lif_tangent, TH
import grad_robust_recurrent as RR

Tw = 1000


def isolated(pre_times, w):
    """Simulate one LIF neuron driven by fixed presynaptic spike trains."""
    K = len(pre_times)
    ia = np.zeros((K, Tw), bool)
    for k, ts in enumerate(pre_times):
        for t in ts:
            if 0 <= t < Tw:
                ia[k, t] = True
    return lif_tangent(np.asarray(w, float), ia, Tw)[1]


def main():
    for ci in [1]:                      # case 304 had the worst over-firing
        tc, params, C, w_true, N, outs = RR.build(ci)
        Cj = jnp.array(C)
        tv = np.array(_hard_sim(jnp.array(w_true), params, Cj, N, jnp.array([0])))
        T = {n: np.where(tv[:, n] >= TH)[0].tolist() for n in range(N)}
        lo, hi = w_true * 0.1, w_true * 5.0
        inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}

        # find the smallest-fan-in neuron that over-fires in isolation with true inputs
        found = None
        for n in range(1, N):
            if not T[n]:
                continue
            syn, pres = inc[n]
            if len(syn) == 0:
                continue
            pre_times = [T[int(p)] for p in pres]
            # sanity: true weights reproduce in isolation?
            sp_true_iso = isolated(pre_times, w_true[syn])
            w_rec = RR.solve(pre_times, T[n], lo[syn], hi[syn], robust=True)
            sp_rec = isolated(pre_times, w_rec) if w_rec is not None else None
            over = sp_rec is not None and len(sp_rec) > len(T[n])
            if over and (found is None or len(syn) < found["k"]):
                found = dict(n=n, k=len(syn), pres=pres.tolist(), pre_times=pre_times,
                             target=T[n], w_true=w_true[syn], w_rec=w_rec,
                             sp_true_iso=sp_true_iso, sp_rec=sp_rec)
                if len(syn) <= 2:
                    break

        f = found
        print(f"case{ci}: minimal over-firing neuron = N{f['n']}  ({f['k']} inputs)")
        print(f"  presynaptic neurons: {f['pres']}")
        for p, ts in zip(f['pres'], f['pre_times']):
            print(f"    N{p} true spikes: {ts}")
        print(f"  target (true N{f['n']} spikes): {f['target']}")
        print(f"  true weights {np.round(f['w_true'],0).tolist()} -> isolated spikes "
              f"{f['sp_true_iso']}  (== target? {f['sp_true_iso']==f['target']})")
        print(f"  RECON weights {np.round(f['w_rec'],0).tolist()} -> isolated spikes "
              f"{f['sp_rec']}  ({len(f['sp_rec'])} vs target {len(f['target'])})  <-- EXTRA SPIKES")
        # which spikes are the extras?
        extras = [s for s in f['sp_rec'] if all(abs(s - t) > 8 for t in f['target'])]
        print(f"  extra spikes at: {extras}")


if __name__ == "__main__":
    main()
