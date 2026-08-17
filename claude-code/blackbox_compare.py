"""Black-box optimization on the 50-neuron recurrent cases.

Methods:
  cmaes  – CMA-ES (cma library).  Population evaluated in a single batched JAX
            call via vmap, so one generation ≈ one forward pass.
  de     – Differential Evolution (scipy).  Vectorized=True sends the whole
            population to a batched objective at once.

Same test cases, seeds, NR restarts, and output format as recurrent_compare.py.

Env vars:
  NR      – restarts per case/method  (default 8)
  FEVALS  – function-eval budget per restart  (default 20000)
  NW      – parallel workers  (default 6)
  N_TRAIN – neurons to train on from the end; 0 = all  (default 0)
"""

import sys, os, types, dataclasses, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _attrs.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m

sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import jax, jax.numpy as jnp

from homotopy_core import hard_sim as _hard_sim
import jax_spiking_model as sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights

N_RESTARTS   = int(os.environ.get("NR",       "8"))
MAX_FEVALS   = int(os.environ.get("FEVALS", "500"))
N_WORKERS    = int(os.environ.get("NW",       "6"))
_N_TRAIN_ENV = int(os.environ.get("N_TRAIN",   "0"))
TOL          = 1e-6


def spike_counts(V_np, outs, th):
    return {n: int(np.sum(V_np[:, n] >= th)) for n in outs}


def run_case_method(case_idx, method):
    import cma
    from scipy.optimize import differential_evolution

    tc = RECURRENT_CASES[case_idx]
    conns, tw = _make_recurrent_weights(
        tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
        tc["num_neurons"], tc["output_neurons"])

    params   = dataclasses.replace(sim.default_params, steps=1000)
    C        = jnp.array(conns)
    N        = tc["num_neurons"]
    A        = jnp.array([0])
    outs     = tc["output_neurons"]
    th       = params.threshold
    n_train  = _N_TRAIN_ENV if _N_TRAIN_ENV > 0 else N
    n_train  = min(n_train, N)
    train_ns = list(range(N - n_train, N))

    true_strs = jnp.array(tw, jnp.float32)
    lo        = true_strs * 0.1
    hi        = true_strs * 5.0
    target_v  = jnp.array(_hard_sim(true_strs, params, C, N, A))
    seeds     = list(range(42, 42 + N_RESTARTS))

    target_np = np.array(target_v)
    lo_np     = np.array(lo)
    hi_np     = np.array(hi)
    true_np   = np.array(true_strs)
    n_weights = len(tw)

    # ── batched objective via vmap ────────────────────────────────────────────
    target_jnp = target_v

    def _single(w):
        V = _hard_sim(w, params, C, N, A)
        return sum(jnp.sum((target_jnp[:, n] - V[:, n]) ** 2) for n in train_ns)

    _batch_jit = jax.jit(jax.vmap(_single))

    # warm-up JIT
    _batch_jit(jnp.ones((4, n_weights), jnp.float32))

    def obj_batch(W_np):
        """W_np: (pop, n_weights) → losses: (pop,)"""
        return np.array(_batch_jit(jnp.array(W_np, jnp.float32)))

    def obj_scalar(w_np):
        return float(obj_batch(w_np[None])[0])

    # ── eval loss on output neurons ───────────────────────────────────────────
    def eval_loss(w_np):
        v = np.array(_hard_sim(jnp.array(w_np, jnp.float32), params, C, N, A))
        return float(sum(np.sum((target_np[:, n] - v[:, n]) ** 2) for n in outs))

    t0        = time.perf_counter()
    best_loss = float("inf")
    best_w    = true_np.copy()
    total_evals = 0

    if method == "cmaes":
        for seed in seeds:
            rng = np.random.default_rng(seed)
            x0 = np.clip(true_np * rng.uniform(0.5, 1.5, n_weights), lo_np, hi_np)
            # sigma = ~1/6 of average weight range so ±3σ covers the box
            sigma0 = float(np.mean((hi_np - lo_np) / 6.0))

            es = cma.CMAEvolutionStrategy(
                x0, sigma0,
                {
                    'seed':      int(seed),
                    'maxfevals': MAX_FEVALS,
                    'bounds':    [lo_np.tolist(), hi_np.tolist()],
                    'verbose':   -9,
                    'tolx':      1e-9,
                    'tolfun':    1e-12,
                },
            )

            while not es.stop():
                X = np.array(es.ask())          # (lambda, n_weights)
                F = obj_batch(X)                # one batched JAX call
                es.tell(X.tolist(), F.tolist())
                total_evals += len(X)

            w_best = np.clip(es.result.xbest, lo_np, hi_np)
            l = eval_loss(w_best)
            if l < best_loss:
                best_loss = l
                best_w    = w_best
            v_r = np.array(_hard_sim(jnp.array(w_best, jnp.float32), params, C, N, A))
            sp_r = spike_counts(v_r, outs, th)
            sp_str = " ".join(f"N{n}:{sp_r[n]}/{spike_counts(target_np, outs, th)[n]}sp" for n in outs)
            print(f"    cmaes case{case_idx} restart={seed} evals={es.result.evaluations} "
                  f"loss={l:.3e} best={best_loss:.3e}  {sp_str}", flush=True)
            if best_loss < TOL:
                break

    elif method == "de":
        # popsize in scipy = multiplier × n_weights; use 5 so pop ≈ 5×n
        # maxiter computed so total evals ≈ MAX_FEVALS
        pop_count = max(10, 5 * n_weights)
        maxiter   = max(10, MAX_FEVALS // pop_count)

        for seed in seeds:
            def _obj_vec(W_np):   # scipy sends (n_weights, pop) when vectorized
                return obj_batch(W_np.T)

            result = differential_evolution(
                _obj_vec,
                list(zip(lo_np, hi_np)),
                seed=int(seed),
                maxiter=maxiter,
                popsize=5,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.7,
                vectorized=True,
                polish=False,
            )
            total_evals += result.nfev
            w = np.clip(result.x, lo_np, hi_np)
            l = eval_loss(w)
            if l < best_loss:
                best_loss = l
                best_w    = w
            v_r = np.array(_hard_sim(jnp.array(w, jnp.float32), params, C, N, A))
            sp_r = spike_counts(v_r, outs, th)
            sp_str = " ".join(f"N{n}:{sp_r[n]}/{spike_counts(target_np, outs, th)[n]}sp" for n in outs)
            print(f"    de case{case_idx} restart={seed} evals={result.nfev} "
                  f"loss={l:.3e} best={best_loss:.3e}  {sp_str}", flush=True)
            if best_loss < TOL:
                break

    wall    = time.perf_counter() - t0
    v_found = np.array(_hard_sim(jnp.array(best_w, jnp.float32), params, C, N, A))
    sp      = spike_counts(v_found, outs, th)
    sp_true = spike_counts(target_np, outs, th)

    return {
        "case_idx":    case_idx,
        "method":      method,
        "loss":        best_loss,
        "conv":        best_loss < TOL,
        "wall":        wall,
        "total_evals": total_evals,
        "sp":          sp,
        "sp_true":     sp_true,
        "name":        tc["name"],
        "n_syn":       n_weights,
        "n_train":     len(train_ns),
        "n":           N,
        "outs":        outs,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    n_cases      = len(RECURRENT_CASES)
    methods      = ["cmaes", "de"]
    work         = [(ci, m) for ci in range(n_cases) for m in methods]
    n_procs      = min(N_WORKERS, len(work))
    n_train_disp = _N_TRAIN_ENV if _N_TRAIN_ENV > 0 else "N"

    print(f"FEVALS={MAX_FEVALS}/restart  NR={N_RESTARTS}  N_TRAIN={n_train_disp}  workers={n_procs}/{len(work)}")
    print()

    ctx     = multiprocessing.get_context("spawn")
    results = {}

    with ProcessPoolExecutor(max_workers=n_procs, mp_context=ctx) as ex:
        futs = {ex.submit(run_case_method, ci, m): (ci, m) for ci, m in work}
        for f in as_completed(futs):
            ci, m = futs[f]
            try:
                results[(ci, m)] = f.result()
            except Exception as e:
                results[(ci, m)] = {"error": str(e), "case_idx": ci, "method": m}
            print(f"  [case {ci} {m} done]", flush=True)

    print()
    for ci in range(n_cases):
        rc = results.get((ci, "cmaes"), {})
        rd = results.get((ci, "de"),    {})
        if "error" in rc:
            print(f"  cmaes error case {ci}: {rc['error']}")
        if "error" in rd:
            print(f"  de error case {ci}: {rd['error']}")
        if "error" in rc or "error" in rd:
            continue

        outs    = rc["outs"]
        sp_true = rc["sp_true"]
        print(f"{'─'*70}")
        print(f"{rc['name']}  ({rc['n_syn']} syn, {rc['n']} neurons, train on {rc['n_train']} neurons)")
        print(f"  Target spikes: " + "  ".join(f"N{n}={sp_true[n]}sp" for n in outs))
        for r, label in [(rc, "CMA-ES      "), (rd, "Diff.Evol   ")]:
            sp   = r["sp"]
            conv = "YES" if r["conv"] else "NO "
            print(f"  {label}: loss={r['loss']:.3e}  conv={conv}  {r['wall']:.1f}s"
                  f"  evals={r['total_evals']:,}  "
                  + "  ".join(f"N{n}:{sp[n]}/{sp_true[n]}sp" for n in outs))
