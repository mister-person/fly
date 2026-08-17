"""Credit-weighted target inference (partial credit + edge-level silence).

Fixes the over-attribution failure (grad_overfire_minimal.py) with two changes to the
relaxation inference:

1. PER-SPIKE COMPETITIVE CREDIT.  Credit is scored for EACH downstream target spike
   separately, and each spike is owned by the presynaptic most responsible for it, so a
   downstream's spikes are SPLIT among its drivers.  Scoring per downstream NEURON
   instead (one scalar over its whole train, thresholded at r >= 1/fanin) can only
   include or exclude the entire train: a hidden neuron owing only SOME of a
   downstream's spikes is then demanded all of them (over-fire) or none (silenced) --
   see grad_overdemand_minimal.py.
   Per spike, "who CAUSED it" (kernel-weighted drive) is used where d already fires;
   where d is MISSING the spike nobody caused it, so we ask "who should GROW it"
   (structural weight share).  That fallback also keeps silence from being ABSORBING --
   a silenced neuron has zero dynamic share and could otherwise never earn credit back.

2. SILENCE CONSTRAINS THE EDGE, NOT THE SOURCE.  A downstream with an empty target
   places no demand, and is enforced by scaling ITS OWN incoming weights sub-threshold
   in train() (previously empty targets were skipped, so nothing ever suppressed a
   neuron).  An earlier version instead let silence VETO the presynaptic neuron's
   firing; that is too absolute and wrongly silences a neuron another downstream needs
   -- see grad_veto_minimal.py (veto 0/4, edge-level 4/4).

Both minimal cases pass: over-fire 4/4 (N1 correctly silent), veto 4/4 (N1 correctly
fires).  Feed-forward all 4/4, 3-cycle 4/4.
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
import numpy as np, jax.numpy as jnp
import jax_spiking_model as sim
from homotopy_core import hard_sim as _hard_sim
from grad_fork_test import solve_vsub
from grad_infer_relax import anchor_targets, MAX_LAT
import grad_unified as U

TH = U.TH
hk = U.hk
KWIN = 400          # kernel support used when attributing drive
REFR = sim.default_params.refractory_iters
SPLIT_FRAC = float(os.environ.get("SPLIT_FRAC", "0.75"))  # owner if share >= frac * top


def mkparams(steps): return dataclasses.replace(sim.default_params, steps=steps)


def fsim(C, N, w, params):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, jnp.array(C), N, jnp.array([0])))


def sp(V, n): return np.where(V[:, n] >= TH)[0].tolist()


def spike_share(C, w, n, d, spall, t, syn_d):
    """PER-SPIKE credit: fraction of d's drive at the single time t due to edge n->d.

    Per-downstream-neuron credit (one scalar over all of d's spikes) can only include or
    exclude d's WHOLE train, so a neuron that owes only SOME of d's spikes is either
    demanded all of them (over-fire) or none (silenced).  Scoring each target spike
    separately lets n be credited for exactly the spikes it actually drives.

    If nothing currently drives d at t (a spike d must grow), fall back to structural
    weight share so silent/weak presynaptics can still be recruited.
    """
    if any(abs(s - t) <= REFR for s in spall[d]):      # d already fires here: who CAUSED it
        tot = 0.0; mine = 0.0
        for si in syn_d:
            p = int(C[si, 0])
            v = float(w[si]) * sum(hk(t - s) for s in spall[p] if 0 < t - s < KWIN)
            tot += v
            if p == n:
                mine += v
        if tot > 1e-12:
            return mine / tot
    # d is MISSING this spike: nobody caused it, so ask who should GROW it -> weight share
    wsum = float(sum(w[si] for si in syn_d))
    mw = float(sum(w[si] for si in syn_d if int(C[si, 0]) == n))
    return mw / wsum if wsum > 0 else 0.0


def infer_credit(C, N, out_targets, spall, tgt, w, sweeps=4):
    down = {n: [int(d) for d in C[C[:, 0] == n][:, 1]] for n in range(N)}
    innn = {d: np.where(C[:, 1] == d)[0] for d in range(N)}
    tgt = {n: list(tgt.get(n, spall[n])) for n in range(N)}
    for o, t in out_targets.items():
        tgt[o] = list(t)
    for _ in range(sweeps):
        # own each downstream target spike -> the presynaptic(s) most responsible for it
        owners = {}
        for d in range(N):
            syn_d = innn[d]
            pres_d = sorted({int(C[si, 0]) for si in syn_d})
            owners[d] = {}
            for t in tgt.get(d, []):
                # OWNERSHIP BY TIMING FEASIBILITY: every presynaptic with a spike in the
                # causal window [t-MAX_LAT, t) is a co-owner, so one spike's credit SPLITS
                # across all sources that could have contributed to it.  Magnitude is NOT
                # used to pick owners -- a co-driver firing at the wrong times has a tiny
                # share, and once its weight shrinks the share shrinks further, an
                # absorbing state that permanently starves it (grad_coincidence_minimal:
                # N2's share collapses 0.23->0.09).  The actual DRIVE split is left to
                # solve_vsub fitting d's incoming weights.
                # ...but a negligible edge (share ~0.09 in grad_overfire_minimal) must not
                # be credited just for being timing-feasible, so also require a
                # non-negligible STRUCTURAL weight share.  Structural share is used rather
                # than the realised drive because the latter collapses once the co-driver's
                # timing drifts, which is the absorbing state that starved it.
                st = {p: sum(w[si] for si in syn_d if int(C[si, 0]) == p) for p in pres_d}
                stop = max(st.values()) if st else 0.0
                feas = tuple(p for p in pres_d
                             if any(t - MAX_LAT <= s < t for s in spall[p])
                             and st[p] >= SPLIT_FRAC * stop)
                if feas:
                    owners[d][t] = feas
                    continue
                # nobody could have caused it -> recruit by structural weight share
                sh = {p: spike_share(C, w, p, d, spall, t, syn_d) for p in pres_d}
                if not sh:
                    continue
                top = max(sh.values())
                if top <= 0:
                    continue
                # SPLIT credit: a spike may legitimately have SEVERAL sources that all
                # deserve credit, so every materially-contributing presynaptic is an owner,
                # not just the winner.  Winner-take-all (>= 0.9*top) starves a genuine
                # co-driver on a different schedule and silences it
                # (grad_coincidence_minimal.py).  The floor still excludes negligible
                # contributors (~0.09 share in grad_overfire_minimal).  The DRIVE split
                # itself is handled downstream by solve_vsub fitting d's incoming weights.
                owners[d][t] = tuple(p for p, v in sh.items() if v >= SPLIT_FRAC * top)
        new = dict(tgt)
        for n in range(1, N):
            if n in out_targets or not down[n]:
                continue
            # PER-SPIKE COMPETITIVE credit: each downstream target spike is owned by the
            # presynaptic most responsible for THAT spike, so a downstream's spikes are
            # SPLIT among its drivers.  A hidden neuron owing only some of a downstream's
            # spikes gets exactly those -- not all of them (over-fire) and not none
            # (a hard share threshold can zero out a neuron entirely).
            dmd = {}
            for d in down[n]:
                keep = [t for t in tgt.get(d, []) if n in owners[d].get(t, ())]
                if keep:
                    dmd[d] = keep
            # A SILENT downstream places no demand -- but it does NOT veto n's firing.
            # Its silence constrains the EDGE w[n->d], not the SOURCE: that is handled
            # by scaling d's own incoming weights sub-threshold in train().  Vetoing
            # the source wrongly silences a neuron that another downstream needs
            # (grad_veto_minimal.py: veto 0/4 vs no-veto 4/4).
            new[n] = anchor_targets(list(dmd), dmd) if dmd else []
        tgt = new
    return tgt


def train(C, N, outs, w, T_true, params, rounds, alpha=0.5, sweeps=4, lo=None, hi=None):
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
    out_t = {o: T_true[o] for o in outs}
    tgt = {}
    for _ in range(rounds):
        V = fsim(C, N, w, params); spall = {p: sp(V, p) for p in range(N)}
        tgt = infer_credit(C, N, out_t, spall, tgt, w, sweeps=sweeps)
        for n in range(1, N):
            syn, pres = inc[n]
            if len(syn) == 0 or n not in tgt:
                continue
            if not tgt[n]:                        # must be SILENT -> scale drive below th
                vmax = float(V[:, n].max())
                if vmax >= TH:
                    w[syn] = w[syn] * (0.9 * TH / vmax)
                    if lo is not None:
                        w[syn] = np.clip(w[syn], lo[syn], hi[syn])
                continue
            sol = solve_vsub([spall[int(p)] for p in pres], tgt[n], robust=(len(pres) > 1))
            w[syn] = (1 - alpha) * w[syn] + alpha * sol
            if lo is not None:
                w[syn] = np.clip(w[syn], lo[syn], hi[syn])
    return w, tgt


def run(name, C, N, outs, w_true, seeds=4, rounds=30, steps=520, verbose=False):
    params = mkparams(steps)
    C = np.array(C, np.int32); w_true = np.array(w_true, np.float32)
    tv = fsim(C, N, w_true, params); T_true = {n: sp(tv, n) for n in range(N)}
    succ = 0; last = None
    for seed in range(seeds):
        w = (w_true * np.random.default_rng(seed).uniform(0.5, 1.5, len(w_true))).astype(float)
        w, tgt = train(C, N, outs, w, T_true, params, rounds)
        last = tgt
        V = fsim(C, N, w, params)
        succ += int(all(sp(V, o) == T_true[o] for o in outs))
    tag = "" if not verbose else "".join(
        f"\n   N{n}: inferred {last.get(n, '-')}   true {T_true[n]}" for n in range(N))
    print(f"{name}: recovered {succ}/{seeds}{tag}")
    return succ


def main():
    print("Credit-weighted inference: partial credit + edge-level silence\n")
    print("the failing minimal case:")
    run("over-fire minimal (N1 must stay SILENT)",
        [[0, 1], [0, 2], [1, 2], [1, 3]], 4, [2, 3], [150., 500., 50., 500.], verbose=True)

    print("\nfeed-forward regressions (relaxation loop got 4/4 on all):")
    run("BREAK divergent", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 900., 470.])
    run("chain", [[0, 1], [1, 2], [2, 3]], 4, [3], [500., 500., 500.])
    run("fanout equal", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 500.])
    run("fanout hard", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 200.])

    print("\nsmall recurrent (relaxation loop: 3-cycle 4/4, others 0/4):")
    run("2-cycle", [[0, 1], [1, 2], [2, 1], [2, 3]], 4, [3], [500., 500., 60., 500.])
    run("3-cycle", [[0, 1], [1, 2], [2, 3], [3, 1]], 4, [3], [500., 500., 500., 60.])
    run("cycle+fanout", [[0, 1], [1, 2], [2, 1], [2, 3], [2, 4]], 5, [3, 4],
        [500., 500., 60., 700., 400.])


if __name__ == "__main__":
    main()
