"""Headless parallel test suite for the 3-neuron chain homotopy training.

Each test case runs in its own worker process so all 24 cases proceed in parallel.
Results are printed as they finish; a summary table in original order follows.

Covers:
  - starting weights way too low / high on each synapse independently, both together
  - crossed starts (one synapse way high, the other way low)
  - asymmetric and extreme true (target) weights
  - targets where N1 or N2 ends up silent (synapse too weak to drive neuron)
  - starts where N1 or N2 begins silent (init weight below firing threshold)

Run:  python3 test_cases.py
      NOPT=100 NR=2 python3 test_cases.py    # quick smoke-test
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault("MPLBACKEND", "Agg")

BETAS         = [0.5, 1, 2, 3, 5, 8, 13, 21, 34]
NOPT          = int(os.environ.get("NOPT",    "300"))
N_RESTARTS    = int(os.environ.get("NR",      "4"))
PATIENCE      = int(os.environ.get("PATIENCE","50"))   # early-stop window (steps)
RTOL          = float(os.environ.get("RTOL",  "1e-3")) # min relative improvement
TOL           = 1e-6
RUNTIME_STEPS = 1000   # params.steps
OBSERVE_LAST  = os.environ.get("OBSERVE_LAST", "0") == "1"

# ── test case definitions ─────────────────────────────────────────────────────
#
# Fields:
#   name        short identifier printed in output
#   true_strs   [s0, s1]  target synapse weights (what training should recover)
#   init_mods   [m0, m1]  starting multipliers for restart 0; None = random
#   lo_scale    lower bound = true_strs * lo_scale  (default 0.3)
#   hi_scale    upper bound = true_strs * hi_scale  (default 3.0)
#   desc        human-readable label printed in summary

TEST_CASES = [

    # ── starting weights way too low ──────────────────────────────────────────
    # lo_scale must be below init_mods so the start isn't clipped on step 1

    dict(name="s0_way_low",
         true_strs=[420, 420], init_mods=[0.05, 1.00],
         lo_scale=0.04, hi_scale=3.0,
         desc="synapse 0 starts 20× below target, synapse 1 on target"),

    dict(name="s1_way_low",
         true_strs=[420, 420], init_mods=[1.00, 0.05],
         lo_scale=0.04, hi_scale=3.0,
         desc="synapse 0 on target, synapse 1 starts 20× below target"),

    dict(name="both_way_low",
         true_strs=[420, 420], init_mods=[0.05, 0.05],
         lo_scale=0.04, hi_scale=3.0,
         desc="both synapses start 20× below target"),

    # ── starting weights way too high ─────────────────────────────────────────
    # hi_scale must be above init_mods

    dict(name="s0_way_high",
         true_strs=[420, 420], init_mods=[10.0,  1.0],
         lo_scale=0.3, hi_scale=11.0,
         desc="synapse 0 starts 10× above target, synapse 1 on target"),

    dict(name="s1_way_high",
         true_strs=[420, 420], init_mods=[ 1.0, 10.0],
         lo_scale=0.3, hi_scale=11.0,
         desc="synapse 0 on target, synapse 1 starts 10× above target"),

    dict(name="both_way_high",
         true_strs=[420, 420], init_mods=[10.0, 10.0],
         lo_scale=0.3, hi_scale=11.0,
         desc="both synapses start 10× above target"),

    # ── crossed starts ────────────────────────────────────────────────────────

    dict(name="s0_hi_s1_lo",
         true_strs=[420, 420], init_mods=[10.0, 0.05],
         lo_scale=0.04, hi_scale=11.0,
         desc="synapse 0 way high, synapse 1 way low"),

    dict(name="s0_lo_s1_hi",
         true_strs=[420, 420], init_mods=[0.05, 10.0],
         lo_scale=0.04, hi_scale=11.0,
         desc="synapse 0 way low, synapse 1 way high"),

    # ── different (asymmetric) true weights ───────────────────────────────────

    dict(name="true_300_600",
         true_strs=[300, 600],
         desc="synapse 1 twice as strong as synapse 0"),

    dict(name="true_600_300",
         true_strs=[600, 300],
         desc="synapse 0 twice as strong as synapse 1"),

    dict(name="true_140_800",
         true_strs=[140, 800],
         desc="synapse 0 just above firing threshold (1 spike), synapse 1 very strong"),

    dict(name="true_800_100",
         true_strs=[800, 100],
         desc="synapse 0 very strong, synapse 1 weak"),

    dict(name="true_200_200",
         true_strs=[200, 200],
         desc="symmetric, weaker than default 420"),

    dict(name="true_1000_1000",
         true_strs=[1000, 1000],
         desc="both very strong — neurons fire many times per window"),

    dict(name="true_140_600",
         true_strs=[140, 600],
         desc="synapse 0 just above firing threshold (1 spike), synapse 1 strong"),

    dict(name="true_600_50",
         true_strs=[600, 50],
         desc="synapse 0 strong, synapse 1 near firing threshold"),

    # ── target has N2 silent (synapse 1 too weak to drive N2 above threshold) ─

    dict(name="target_n2_silent",
         true_strs=[420, 15],
         desc="true s1 very small — N2 should never fire in target"),

    dict(name="target_n2_silent_asym",
         true_strs=[300, 20],
         desc="moderate s0, tiny s1 — N2 should never fire in target"),

    # ── target has N1 and N2 both silent (synapse 0 too weak) ────────────────

    dict(name="target_n1n2_silent",
         true_strs=[15, 420],
         desc="true s0 very small — N1 silent, so N2 also silent"),

    dict(name="target_both_silent",
         true_strs=[15, 15],
         desc="both synapses tiny — N1 and N2 both silent in target"),

    # ── init weights below firing threshold (neuron starts silent) ────────────
    # true_strs cause firing; init_strs = true_strs * init_mods are too small.
    # lo_scale < init_mods so the starting point stays in the search region.

    dict(name="start_n1_silent",
         true_strs=[420, 420], init_mods=[0.04, 1.0],
         lo_scale=0.03, hi_scale=3.0,
         desc="N1 starts silent (s0 init too small); target has N1 firing"),

    dict(name="start_n1n2_silent",
         true_strs=[420, 420], init_mods=[0.04, 0.04],
         lo_scale=0.03, hi_scale=3.0,
         desc="N1 and N2 both start silent; both should fire at true weights"),

    dict(name="start_n1_too_high_n2_silent",
         true_strs=[420, 420], init_mods=[10.0, 0.04],
         lo_scale=0.03, hi_scale=11.0,
         desc="N1 fires way too much (s0 high), N2 starts silent (s1 too low)"),

    dict(name="start_n1_silent_s1_too_high",
         true_strs=[420, 420], init_mods=[0.04, 10.0],
         lo_scale=0.03, hi_scale=11.0,
         desc="N1 starts silent, s1 way too high (can't help until N1 fires)"),
]


# ── 50-neuron random recurrent test cases ─────────────────────────────────────
#
# Topology is a random sparse directed graph (no self-loops) with connection
# probability p_connect.  True weights are drawn from a random uniform range.
# Both are fully determined by (topo_seed, p_connect, trial_idx).
#
# Observed neurons: OUTPUT_NEURONS_50 (indices 47, 48, 49).
# Input neuron:     0  (activated every 100 steps, same as 3-neuron chain).
#
# Run just these:  python3 test_cases.py recurrent
# Run chain only:  python3 test_cases.py  (default) OR  python3 test_cases.py chain
# Run everything:  python3 test_cases.py all

OUTPUT_NEURONS_50 = [47, 48, 49]
RUNTIME_STEPS_50  = 1000


def _make_recurrent_topology(topo_seed, p_connect, num_neurons=50,
                              output_neurons=None):
    """Reconstruct random recurrent connection list from generative params."""
    import numpy as _np
    if output_neurons is None:
        output_neurons = OUTPUT_NEURONS_50
    rng = _np.random.default_rng(topo_seed)
    pairs = [(i, j) for i in range(num_neurons)
                    for j in range(num_neurons) if i != j]
    mask  = rng.random(len(pairs)) < p_connect
    conns = [p for p, m in zip(pairs, mask) if m]
    # guarantee each output has at least 2 incoming connections
    for out in output_neurons:
        inc = [p for p in conns if p[1] == out]
        while len(inc) < 2:
            pre = int(rng.integers(0, num_neurons))
            if pre != out and (pre, out) not in conns:
                conns.append((pre, out))
                inc.append((pre, out))
    return conns


def _make_recurrent_weights(topo_seed, p_connect, trial_idx, num_neurons=50,
                             output_neurons=None):
    """Reconstruct topology and true weight vector from generative params."""
    import numpy as _np
    conns = _make_recurrent_topology(topo_seed, p_connect, num_neurons,
                                     output_neurons)
    n_syn = len(conns)
    rng   = _np.random.default_rng(topo_seed * 100 + int(p_connect * 100))
    for _ in range(trial_idx + 1):
        lo = rng.uniform(50, 300)
        hi = rng.uniform(lo * 2, lo * 6)
        w  = rng.uniform(lo, hi, n_syn)
    return conns, w


RECURRENT_CASES = [
    # Spike counts at true weights (observed during exploration):
    dict(name="recurrent_132_7_2_1",
         group="recurrent",
         topo_seed=6, p_connect=0.06, trial_idx=13,
         num_neurons=50, output_neurons=OUTPUT_NEURONS_50,
         observed_spikes={47: 7, 48: 2, 49: 1},
         desc="132 syn p=0.06; N47=7sp N48=2sp N49=1sp; timing spread ~800 steps"),

    dict(name="recurrent_304_4_2_7",
         group="recurrent",
         topo_seed=0, p_connect=0.12, trial_idx=3,
         num_neurons=50, output_neurons=OUTPUT_NEURONS_50,
         observed_spikes={47: 4, 48: 2, 49: 7},
         desc="304 syn p=0.12; N47=4sp N48=2sp N49=7sp"),

    dict(name="recurrent_240_3_3_7",
         group="recurrent",
         topo_seed=0, p_connect=0.09, trial_idx=8,
         num_neurons=50, output_neurons=OUTPUT_NEURONS_50,
         observed_spikes={47: 3, 48: 3, 49: 7},
         desc="240 syn p=0.09; N47=3sp N48=3sp N49=7sp"),
]

ALL_CASES = TEST_CASES + RECURRENT_CASES


# ── worker (runs in a child process) ─────────────────────────────────────────

def _worker(args):
    tc, seed = args
    os.environ.setdefault("MPLBACKEND", "Agg")

    import types, dataclasses
    import numpy as np
    import jax, jax.numpy as jnp
    from functools import partial

    for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
        if _n not in sys.modules:
            _m = types.ModuleType(_n)
            for _k, _v in _attrs.items(): setattr(_m, _k, _v)
            sys.modules[_n] = _m

    from homotopy_core import (
        hard_sim as _hard_sim,
        homotopy_stage as _homotopy_stage,
        soft_sim as _soft_sim,
    )
    import jax_spiking_model as sim

    # ── recurrent 50-neuron case ──────────────────────────────────────────────
    if tc.get("group") == "recurrent":
        params = dataclasses.replace(sim.default_params,
                                     steps=tc.get("steps", RUNTIME_STEPS_50))
        th   = params.threshold
        conns, true_w_np = _make_recurrent_weights(
            tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
            tc["num_neurons"], tc["output_neurons"])
        C    = jnp.array(conns)
        N_R  = tc["num_neurons"]
        A    = jnp.array([0])
        outs = tc["output_neurons"]

        true_strs = jnp.array(true_w_np, dtype=jnp.float32)
        lo = true_strs * 0.1
        hi = true_strs * 5.0

        target_v    = _hard_sim(true_strs, params, C, N_R, A)
        target_fire = [bool(jnp.any(target_v[:, n] >= th)) for n in outs]

        @jax.jit
        def stage(w0, base, lo, hi, beta, lr):
            tgt = _soft_sim(base, beta, params, C, N_R, A)
            def loss_fn(w):
                v = _soft_sim(w, beta, params, C, N_R, A)
                return sum(jnp.sum((tgt[:, n] - v[:, n]) ** 2) for n in outs)
            vg = jax.value_and_grad(loss_fn)
            def cond(c):
                _, _, _, _, _, t, _, done = c
                return (t < NOPT) & ~done
            def body(c):
                w, m, v, bw, bl, t, l_check, done = c
                l, g = vg(w)
                g = jnp.nan_to_num(g)
                bw, bl = jax.lax.cond(l < bl, lambda: (w, l), lambda: (bw, bl))
                m  = 0.9   * m + 0.1   * g
                v  = 0.999 * v + 0.001 * g * g
                t1 = (t + 1).astype(jnp.float32)
                step = (m / (1 - 0.9 ** t1)) / (
                    jnp.sqrt(v / (1 - 0.999 ** t1)) + 1e-12)
                w_new  = jnp.clip(w - lr * step, lo, hi)
                new_t  = t + 1
                at_end = (new_t % PATIENCE == 0)
                rel_imp = (l_check - bl) / (jnp.abs(l_check) + 1e-10)
                done_now    = done | (at_end & (rel_imp < RTOL))
                l_check_new = jax.lax.cond(at_end, lambda: bl, lambda: l_check)
                return (w_new, m, v, bw, bl, new_t, l_check_new, done_now)
            l0 = vg(w0)[0]
            z  = jnp.zeros_like(w0)
            init = (w0, z, z, w0, l0, jnp.int32(0), l0, jnp.bool_(False))
            _, _, _, bw, _, _, _, _ = jax.lax.while_loop(cond, body, init)
            return bw

        rng_r     = np.random.default_rng(seed)
        best_loss = float("inf")
        best_w    = true_strs

        for restart in range(N_RESTARTS):
            w = true_strs * jnp.array(
                rng_r.uniform(0.5, 1.5, len(true_strs)), dtype=jnp.float32)
            for beta in BETAS:
                lr = 1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)
                w  = stage(w, true_strs, lo, hi,
                           jnp.float32(beta), jnp.float32(lr))
            v_found = np.array(_hard_sim(w, params, C, N_R, A))
            hl = float(sum(
                np.sum((np.array(target_v)[:, n] - v_found[:, n]) ** 2)
                for n in outs))
            if hl < best_loss:
                best_loss = hl
                best_w    = w
            if best_loss < TOL:
                break

        v_best = np.array(_hard_sim(best_w, params, C, N_R, A))
        sp_found = {n: int(np.sum(v_best[:, n] >= th))         for n in outs}
        sp_true  = {n: int(np.sum(np.array(target_v)[:, n] >= th)) for n in outs}
        mods = "  ".join(f"N{n}:{sp_found[n]}/{sp_true[n]}sp" for n in outs)

        return dict(
            name        = tc["name"],
            true_strs   = [],
            init_mods   = None,
            converged   = best_loss < TOL,
            loss        = best_loss,
            mods        = mods,
            target_fire = target_fire,
        )

    # ── 3-neuron chain case ───────────────────────────────────────────────────
    CONNECTIONS      = jnp.array([[0, 1], [1, 2]])
    NEURONS_ACTIVATE = jnp.array([0])
    NUM_NEURONS      = 10

    def hard_sim(w, params):
        return _hard_sim(w, params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE)

    @partial(jax.jit, static_argnames=["params", "nopt", "observe_last", "patience"])
    def homotopy_stage(w0, base, lo, hi, beta, lr, params,
                       nopt=300, observe_last=False, patience=30):
        return _homotopy_stage(w0, base, lo, hi, beta, lr,
                               params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE,
                               nopt=nopt, observe_last=observe_last,
                               patience=patience)

    params = dataclasses.replace(sim.default_params, steps=RUNTIME_STEPS)
    rng    = np.random.default_rng(seed)
    th     = params.threshold

    true_strs   = jnp.array(tc["true_strs"], dtype=jnp.float32)
    lo          = true_strs * tc.get("lo_scale", 0.3)
    hi          = true_strs * tc.get("hi_scale", 3.0)
    target_v    = hard_sim(true_strs, params)
    target_fire = [bool(jnp.any(target_v[:, i] >= th)) for i in range(3)]

    best_loss = float("inf")
    best_w    = true_strs

    for restart in range(N_RESTARTS):
        if restart == 0 and tc.get("init_mods") is not None:
            w = true_strs * jnp.array(tc["init_mods"], dtype=jnp.float32)
        else:
            w = true_strs * jnp.array(rng.uniform(0.5, 1.5, size=2), dtype=jnp.float32)

        for beta in BETAS:
            lr = 1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)
            w  = homotopy_stage(w, true_strs, lo, hi,
                                jnp.float32(beta), jnp.float32(lr),
                                params, nopt=NOPT, observe_last=OBSERVE_LAST,
                                patience=PATIENCE)

        hl = float(jnp.sum((target_v - hard_sim(w, params)) ** 2))
        if hl < best_loss:
            best_loss = hl
            best_w    = w
        if best_loss < TOL:
            break

    return dict(
        name        = tc["name"],
        true_strs   = tc["true_strs"],
        init_mods   = tc.get("init_mods"),
        converged   = best_loss < TOL,
        loss        = best_loss,
        mods        = np.array(best_w / true_strs).round(3).tolist(),
        target_fire = target_fire,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import multiprocessing

    # Filtering:
    #   python3 test_cases.py              → chain cases only (default)
    #   python3 test_cases.py chain        → chain cases only
    #   python3 test_cases.py recurrent    → recurrent 50-neuron cases
    #   python3 test_cases.py all          → every case
    #   python3 test_cases.py name1 name2  → specific cases by name
    names = sys.argv[1:]
    _grp  = lambda tc: tc.get("group", "chain")
    if not names or names == ["chain"]:
        cases = [tc for tc in ALL_CASES if _grp(tc) == "chain"]
    elif names == ["recurrent"]:
        cases = [tc for tc in ALL_CASES if _grp(tc) == "recurrent"]
    elif names == ["all"]:
        cases = ALL_CASES
    else:
        cases = [tc for tc in ALL_CASES if tc["name"] in names]
    if not cases:
        print(f"No test cases matched: {names}")
        print(f"Groups: chain, recurrent, all")
        print(f"Names:  {[tc['name'] for tc in ALL_CASES]}")
        return

    n_workers = min(len(cases), os.cpu_count() or 4)

    obs_str = "N2 only" if OBSERVE_LAST else "all neurons"
    print(f"cases={len(cases)}  workers={n_workers}  observe={obs_str}  "
          f"NOPT={NOPT}  NR={N_RESTARTS}  TOL={TOL:.0e}  steps={RUNTIME_STEPS}\n")

    work = [(tc, 42 + i) for i, tc in enumerate(cases)]
    results = {}
    t0 = time.time()

    tc_by_name = {tc["name"]: tc for tc in cases}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_worker, w): w[0]["name"] for w in work}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                r  = results[name] = fut.result()
                ok = "YES" if r["converged"] else "NO "
                if tc_by_name[name].get("group") == "recurrent":
                    print(f"  [{ok}] {name:<34}  {r['mods']}  "
                          f"loss={r['loss']:.3e}", flush=True)
                else:
                    n1 = "Y" if r["target_fire"][1] else "N"
                    n2 = "Y" if r["target_fire"][2] else "N"
                    print(f"  [{ok}] {name:<34}  N1={n1} N2={n2}  "
                          f"loss={r['loss']:.3e}  mods={r['mods']}", flush=True)
            except Exception as exc:
                results[name] = dict(name=name, converged=False, loss=float("inf"),
                                     target_fire=[False]*3, mods="ERR")
                print(f"  [ERR] {name:<34}  {exc}", flush=True)

    elapsed = time.time() - t0

    # ── summary table in original case order ──────────────────────────────────
    W = 34
    print(f"\n{'name':{W}} {'true_strs':>12} {'init_mods':>14}  "
          f"{'N1':>2} {'N2':>2}  {'conv':>5}  {'loss':>11}  mods")
    print("─" * 105)
    n_pass = 0
    for tc in cases:
        r = results.get(tc["name"])
        if r is None:
            continue
        ok = "YES" if r["converged"] else "NO"
        if r["converged"]:
            n_pass += 1
        if tc.get("group") == "recurrent":
            outs = tc["output_neurons"]
            fire = " ".join(f"N{n}={'Y' if f else 'N'}"
                            for n, f in zip(outs, r["target_fire"]))
            print(f"{tc['name']:{W}} {fire}  {ok:>5}  {r['loss']:>11.3e}  {r['mods']}")
        else:
            n1 = "Y" if r["target_fire"][1] else "N"
            n2 = "Y" if r["target_fire"][2] else "N"
            ts = str(tc["true_strs"])
            im = str(tc.get("init_mods", "rand"))
            print(f"{tc['name']:{W}} {ts:>12} {im:>14}  "
                  f"{n1:>2} {n2:>2}  {ok:>5}  {r['loss']:>11.3e}  {r['mods']}")

    print(f"\n{n_pass}/{len(cases)} converged  ({elapsed:.1f}s wall-clock)")

    failures = [tc["name"] for tc in cases
                if results.get(tc["name"]) and not results[tc["name"]]["converged"]]
    if failures:
        print(f"Failed: {failures}")


if __name__ == "__main__":
    main()
