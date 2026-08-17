"""Partial-observation target propagation.

Question: can target prop recover the OUTPUT behaviour of a 50-neuron recurrent
network when we only observe the spike times of a *fraction* of the neurons?

Setup (per recurrent case):
  - We always observe the 3 output neurons [47,48,49] (that's the thing we want
    to match).  We additionally observe a random fraction f of the 46 hidden
    neurons (neuron 0 is the fixed external driver, never trained).
  - TP trains only the incoming weights of OBSERVED neurons, using each observed
    neuron's TRUE spike times as its target (RHS: "fire at these times").
  - Weights of UNOBSERVED neurons are left at their random init and never learn.
  - Presynaptic INPUT spike times (the A-matrix) come from a forward sim with the
    *current* weights — you can always simulate, so inputs are "free"; only
    TARGETS are withheld.  A second variant feeds ORACLE inputs (true spikes) to
    isolate how much damage comes from wrong hidden dynamics vs missing targets.

We sweep f = 0 (outputs only) .. 1 (full oracle TP) and report the output MSE
loss and output spike-count match, averaged over random init/subset seeds.

Baselines printed alongside:
  - random init (no training)     -> floor
  - soft homotopy saved weights   -> reference (best_weights_caseX.npy)
  - f=1.0 full-observation TP      -> ceiling for TP

Env vars:
  CASES   comma list of case indices (default 0,1,2)
  FRACS   comma list of observed hidden fractions (default 0,0.1,0.25,0.5,0.75,1.0)
  SEEDS   number of random init/subset seeds (default 3)
  N_ITER  TP refinement iterations (default 8)
  ALPHA   damping (default 0.4)
  MARGIN  QP no-spurious-fire margin (default 0.10)
"""

import sys, os, types, dataclasses, time, json
os.environ.setdefault("LOSS", "st")
for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); [setattr(_m, k, v) for k, v in _attrs.items()]; sys.modules[_n] = _m
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import jax.numpy as jnp
from scipy.optimize import nnls, minimize

from homotopy_core import hard_sim as _hard_sim
import jax_spiking_model as sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights

CASES  = [int(x) for x in os.environ.get("CASES", "0,1,2").split(",")]
FRACS  = [float(x) for x in os.environ.get("FRACS", "0,0.1,0.25,0.5,0.75,1.0").split(",")]
SEEDS  = int(os.environ.get("SEEDS", "3"))
N_ITER = int(os.environ.get("N_ITER", "8"))
ALPHA  = float(os.environ.get("ALPHA", "0.4"))
MARGIN = float(os.environ.get("MARGIN", "0.10"))
# INIT: starting weights for the UNOBSERVED neurons (which never get trained).
#   random -> cold start, w = w_true * U(0.5,1.5)   (partial obs from scratch)
#   found  -> warm start, w = soft-homotopy best_weights_caseX.npy
#             (TP refines observed neurons on top of another method's solution)
INIT   = os.environ.get("INIT", "random")

MAX_H = 600


def build_case(case_idx):
    tc = RECURRENT_CASES[case_idx]
    conns, tw = _make_recurrent_weights(
        tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
        tc["num_neurons"], tc["output_neurons"])
    params = dataclasses.replace(sim.default_params, steps=1000)
    C = jnp.array(conns)
    N = tc["num_neurons"]
    outs = tc["output_neurons"]
    A = jnp.array([0])
    w_true = np.array(tw, np.float32)
    ctx = dict(
        tc=tc, params=params, C=C, N=N, outs=outs, A=A,
        conns=np.array(conns, np.int32), w_true=w_true,
        lo=w_true * 0.1, hi=w_true * 5.0,
        th=params.threshold, gsw=params.global_synapse_weight,
        delay=params.delay_iters, refr=params.refractory_iters,
        nd=float(params.neuron_decay), rd=float(params.rise_decay),
    )
    # impulse response
    h = np.zeros(MAX_H); R = V = 0.0
    for t in range(MAX_H):
        upd = 1.0 if t == ctx["delay"] else 0.0
        R = (R + upd) * ctx["rd"]; V = (V - R) * ctx["nd"] + R
        h[t] = V
    ctx["h"] = h
    # presynaptic map
    edges = ctx["conns"]
    ctx["pre_map"] = {n: (np.where(edges[:, 1] == n)[0], edges[np.where(edges[:, 1] == n)[0], 0])
                      for n in range(N)}
    ctx["target_v"] = np.array(_hard_sim(jnp.array(w_true), params, C, N, A))
    ctx["T_true"] = {n: np.where(ctx["target_v"][:, n] >= ctx["th"])[0].tolist() for n in range(N)}
    return ctx


def spikes_of(V, n, th):
    return np.where(V[:, n] >= th).tolist() if False else np.where(V[:, n] >= th)[0].tolist()


def build_contrib(ctx, pres, spikes_by_neuron, eval_times, epoch_starts):
    h = ctx["h"]
    rows = []
    for j, Tj in enumerate(eval_times):
        T_prev = epoch_starts[j]
        row = np.zeros(len(pres))
        for ci, p in enumerate(pres):
            for t_k in spikes_by_neuron.get(int(p), ()):
                if T_prev < t_k < Tj:
                    dt = Tj - t_k
                    if 0 < dt < MAX_H:
                        row[ci] += h[dt]
        rows.append(row)
    return np.array(rows)


def tp_neuron(ctx, n, input_spikes, tgt_times, margin):
    th, gsw = ctx["th"], ctx["gsw"]
    lo, hi = ctx["lo"], ctx["hi"]
    syn_idxs, pres = ctx["pre_map"][n]
    if len(pres) == 0 or len(tgt_times) == 0:
        return None
    tgt = list(tgt_times)
    epoch_lo = [0] + tgt[:-1]
    A_lo = build_contrib(ctx, pres, input_spikes, tgt, epoch_lo)
    b = np.full(len(tgt), th / gsw)
    if A_lo.size == 0:
        return None
    valid = A_lo.max(axis=1) > 1e-12
    if not valid.any():
        return None
    A_lo, b = A_lo[valid], b[valid]
    w_nnls, _ = nnls(A_lo, b)
    if margin <= 0:
        return syn_idxs, np.clip(w_nnls, lo[syn_idxs], hi[syn_idxs])
    # upper-bound rows (no spurious firing between target spikes)
    nt_times, nt_epochs = [], []
    for j in range(len(tgt) - 1):
        T_ref_end = tgt[j] + ctx["refr"] + 2
        T_next = tgt[j + 1]
        if T_ref_end >= T_next:
            continue
        for frac in (0.25, 0.45, 0.65, 0.85):
            nt_times.append(int(T_ref_end + frac * (T_next - T_ref_end)))
            nt_epochs.append(tgt[j])
    cons = [{'type': 'ineq', 'fun': lambda w, A=A_lo, bb=b: A @ w - bb}]
    if nt_times:
        A_up = build_contrib(ctx, pres, input_spikes, nt_times, nt_epochs)
        b_up = np.full(len(nt_times), (1.0 - margin) * th / gsw)
        vu = A_up.max(axis=1) > 1e-12
        A_up, b_up = A_up[vu], b_up[vu]
        if len(A_up):
            cons.append({'type': 'ineq', 'fun': lambda w, A=A_up, bb=b_up: bb - A @ w})
    bounds = [(float(lo[si]), float(hi[si])) for si in syn_idxs]
    res = minimize(lambda w: 0.5 * float(np.dot(w - w_nnls, w - w_nnls)),
                   w_nnls.copy(), jac=lambda w: (w - w_nnls),
                   method='SLSQP', constraints=cons, bounds=bounds,
                   options={'ftol': 1e-10, 'maxiter': 2000, 'disp': False})
    w_sol = res.x if (res.success or res.status in (0, 4)) else w_nnls
    return syn_idxs, np.clip(w_sol, lo[syn_idxs], hi[syn_idxs])


def tp_solve(ctx, observed, input_spikes, w_base, margin):
    """Solve incoming weights for the OBSERVED neurons only, using T_true targets."""
    w_new = w_base.copy()
    for n in observed:
        if n == 0:
            continue
        tgt = ctx["T_true"][n]
        if not tgt:
            continue
        r = tp_neuron(ctx, n, input_spikes, tgt, margin)
        if r is None:
            continue
        syn_idxs, w_sol = r
        w_new[syn_idxs] = w_sol
    return w_new


def out_loss(ctx, V):
    return float(sum(np.sum((ctx["target_v"][:, n] - V[:, n]) ** 2) for n in ctx["outs"]))


def out_spike_match(ctx, V):
    """Number of OUTPUT neurons (of 3) whose spike count matches target."""
    return sum(1 for n in ctx["outs"]
               if int(np.sum(V[:, n] >= ctx["th"])) == len(ctx["T_true"][n]))


def net_spike_match(ctx, V):
    """Number of ALL neurons whose spike count matches target (recovery breadth)."""
    return sum(1 for n in range(ctx["N"])
               if int(np.sum(V[:, n] >= ctx["th"])) == len(ctx["T_true"][n]))


def run_partial(ctx, observed, w_init, oracle_inputs, margin):
    """TP refinement loop under partial observation.

    Returns (best_output_loss, weights_at_best, out_match_at_best, net_match_at_best).
    """
    params, C, N, A, th = ctx["params"], ctx["C"], ctx["N"], ctx["A"], ctx["th"]
    w = w_init.copy()
    best = float("inf"); best_w = w.copy(); best_om = 0; best_nm = 0
    for _ in range(N_ITER):
        if oracle_inputs:
            input_spikes = ctx["T_true"]                        # true presyn times
        else:
            V = np.array(_hard_sim(jnp.array(w.astype(np.float32)), params, C, N, A))
            input_spikes = {n: np.where(V[:, n] >= th)[0].tolist() for n in range(N)}
        w_tp = tp_solve(ctx, observed, input_spikes, w, margin)
        w = np.clip((1 - ALPHA) * w + ALPHA * w_tp, ctx["lo"], ctx["hi"]).astype(np.float32)
        Vn = np.array(_hard_sim(jnp.array(w), params, C, N, A))
        l = out_loss(ctx, Vn)
        if l < best:
            best, best_w = l, w.copy()
            best_om, best_nm = out_spike_match(ctx, Vn), net_spike_match(ctx, Vn)
    return best, best_w, best_om, best_nm


def main():
    print(f"Partial-observation target prop  |  INIT={INIT}  cases={CASES}  fracs={FRACS}  "
          f"seeds={SEEDS}  N_ITER={N_ITER}  ALPHA={ALPHA}  MARGIN={MARGIN}", flush=True)

    all_results = {}
    for ci in CASES:
        ctx = build_case(ci)
        N, outs = ctx["N"], ctx["outs"]
        hidden = [n for n in range(1, N) if n not in outs]

        # references
        w_true = ctx["w_true"]
        rand_losses = []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            w_r = (w_true * rng.uniform(0.5, 1.5, len(w_true))).astype(np.float32)
            V_r = np.array(_hard_sim(jnp.array(w_r), ctx["params"], ctx["C"], N, ctx["A"]))
            rand_losses.append(out_loss(ctx, V_r))
        rand_loss = float(np.mean(rand_losses))

        # whole-network match at random init (floor) and true weights (ceiling)
        rand_netmatch = []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            w_r = (w_true * rng.uniform(0.5, 1.5, len(w_true))).astype(np.float32)
            V_r = np.array(_hard_sim(jnp.array(w_r), ctx["params"], ctx["C"], N, ctx["A"]))
            rand_netmatch.append(net_spike_match(ctx, V_r))
        rand_nm = float(np.mean(rand_netmatch))
        n_true_active = sum(1 for n in range(N) if ctx["T_true"][n])

        soft_loss = None
        w_found = None
        sp = f"best_weights_case{ci}.npy"
        if os.path.exists(sp):
            w_s = np.load(sp).astype(np.float32)
            if len(w_s) == len(w_true):
                w_found = w_s
                V_s = np.array(_hard_sim(jnp.array(w_s), ctx["params"], ctx["C"], N, ctx["A"]))
                soft_loss = out_loss(ctx, V_s)
        if INIT == "found" and w_found is None:
            print("  [INIT=found but no saved weights; falling back to random]", flush=True)

        print(f"\n{'='*74}\n{ctx['tc']['name']}  ({len(w_true)} syn, {N} neurons, "
              f"{len(hidden)} hidden)  targets N{outs}", flush=True)
        tgt_counts = "  ".join(f"N{n}={len(ctx['T_true'][n])}sp" for n in outs)
        print(f"  target spikes: {tgt_counts}", flush=True)
        print(f"  reference: random-init loss={rand_loss:.3e} (net_match={rand_nm:.1f}/{n_true_active})"
              + (f"   soft-homotopy loss={soft_loss:.3e}" if soft_loss is not None else ""),
              flush=True)
        print(f"  {'obs f':>6} {'n_obs':>5}  {'self loss':>11}  {'oracle loss':>11}  "
              f"{'out/3':>6}  {'net_match/'+str(n_true_active):>12}", flush=True)

        case_rows = []
        for f in FRACS:
            n_obs_hidden = int(round(f * len(hidden)))
            self_losses, oracle_losses = [], []
            self_om, self_nm = [], []
            for s in range(SEEDS):
                rng = np.random.default_rng(100 * ci + s)
                obs_h = list(rng.choice(hidden, n_obs_hidden, replace=False)) if n_obs_hidden else []
                observed = set(outs) | set(int(x) for x in obs_h)
                if INIT == "found" and w_found is not None:
                    # warm start: unobserved neurons keep the soft-homotopy solution;
                    # jitter only so repeated seeds differ in the observed subset.
                    w_init = w_found.copy()
                else:
                    w_init = (w_true * rng.uniform(0.5, 1.5, len(w_true))).astype(np.float32)

                l_self, _, om, nm = run_partial(ctx, observed, w_init, oracle_inputs=False, margin=MARGIN)
                l_orac, _, _, _    = run_partial(ctx, observed, w_init, oracle_inputs=True,  margin=MARGIN)
                self_losses.append(l_self); oracle_losses.append(l_orac)
                self_om.append(om); self_nm.append(nm)

            row = dict(f=f, n_obs=len(set(outs)) + n_obs_hidden,
                       self_mean=float(np.mean(self_losses)), self_std=float(np.std(self_losses)),
                       oracle_mean=float(np.mean(oracle_losses)), oracle_std=float(np.std(oracle_losses)),
                       out_match=float(np.mean(self_om)), net_match=float(np.mean(self_nm)))
            case_rows.append(row)
            print(f"  {f:>6.2f} {row['n_obs']:>5}  "
                  f"{row['self_mean']:>9.3e}±{row['self_std']:.0e}  "
                  f"{row['oracle_mean']:>9.3e}±{row['oracle_std']:.0e}  "
                  f"{row['out_match']:>5.1f}  {row['net_match']:>12.1f}", flush=True)

        all_results[ctx['tc']['name']] = dict(
            rand_loss=rand_loss, soft_loss=soft_loss, rand_netmatch=rand_nm,
            n_true_active=n_true_active, rows=case_rows,
            target_counts={n: len(ctx['T_true'][n]) for n in outs})

    out = f"/workspace/project/performance/partial_obs_{INIT}.json"
    with open(out, "w") as fh:
        json.dump(dict(init=INIT, cases=CASES, fracs=FRACS, seeds=SEEDS, n_iter=N_ITER,
                       alpha=ALPHA, margin=MARGIN, results=all_results), fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
