"""Diagnostic analysis for case 2 (240_3_3_7) — the stubborn unsolved case.

Runs the soft+ST optimization, saves best weights, then dissects:
  - Full spike raster (target vs found) for all 50 neurons
  - Voltage traces of output neurons (N47, N48, N49)
  - How far below threshold each output neuron sits at the times it should spike
  - Pre-synaptic neurons of N47/N48 and whether *they* are spiking correctly
  - Weight magnitudes and signs of incoming connections to N47/N48
"""

import sys, os, types, dataclasses, time
os.environ.setdefault("LOSS", "st")

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
from test_cases import RECURRENT_CASES, _make_recurrent_weights, BETAS
from recurrent_compare import make_soft_stage, TAU_SCHEDULE

CASE_IDX = int(os.environ.get("CASE", "2"))
NR       = int(os.environ.get("NR",    "8"))
NOPT     = int(os.environ.get("NOPT", "600"))
SKIP_OPT = os.environ.get("SKIP_OPT", "0") == "1"   # use saved weights if available

tc = RECURRENT_CASES[CASE_IDX]
conns, tw = _make_recurrent_weights(
    tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
    tc["num_neurons"], tc["output_neurons"])

params    = dataclasses.replace(sim.default_params, steps=1000)
C         = jnp.array(conns)
N         = tc["num_neurons"]
A         = jnp.array([0])
outs      = tc["output_neurons"]
th        = params.threshold
train_ns  = list(range(N))

true_strs = jnp.array(tw, jnp.float32)
lo        = true_strs * 0.1
hi        = true_strs * 5.0
target_v  = np.array(_hard_sim(true_strs, params, C, N, A))

save_path = f"best_weights_case{CASE_IDX}.npy"

if SKIP_OPT and os.path.exists(save_path):
    print(f"Loading saved weights from {save_path}")
    best_w = jnp.array(np.load(save_path), jnp.float32)
    best_loss = float("inf")
else:
    print(f"Optimizing case {CASE_IDX}: {tc['name']} ({len(tw)} syn, {N} neurons)")
    stage = make_soft_stage(params, C, N, A, train_ns)
    _d0 = jnp.float32(np.exp(-1.0 / TAU_SCHEDULE[0]))
    stage(true_strs, true_strs, lo, hi, jnp.float32(1.0), jnp.float32(1.0), _d0)

    best_loss = float("inf")
    best_w    = true_strs
    t0 = time.perf_counter()
    for seed in range(42, 42 + NR):
        rng = np.random.default_rng(seed)
        w = true_strs * jnp.array(rng.uniform(0.5, 1.5, len(true_strs)), jnp.float32)
        for beta, tau_i in zip(BETAS, TAU_SCHEDULE):
            lr    = 1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)
            decay = jnp.float32(np.exp(-1.0 / tau_i))
            w     = stage(w, true_strs, lo, hi, jnp.float32(beta), jnp.float32(lr), decay)
        v_f = np.array(_hard_sim(w, params, C, N, A))
        hl  = float(sum(np.sum((target_v[:, n] - v_f[:, n]) ** 2) for n in outs))
        if hl < best_loss:
            best_loss = hl
            best_w    = w
        print(f"  restart {seed}  loss={hl:.3e}  best={best_loss:.3e}", flush=True)
    print(f"Wall: {time.perf_counter()-t0:.1f}s")
    np.save(save_path, np.array(best_w))

found_v   = np.array(_hard_sim(best_w, params, C, N, A))
found_w   = np.array(best_w)
true_w    = np.array(true_strs)
edges     = np.array(C)          # edges[k] = [pre, post];  tw[k] = weight of synapse k
T         = target_v.shape[0]

def pre_of(n):
    """Synapse indices and pre-neuron IDs for all inputs to neuron n."""
    syn_idxs = np.where(edges[:, 1] == n)[0]
    return syn_idxs, edges[syn_idxs, 0]

def post_of(n):
    """Synapse indices and post-neuron IDs for all outputs from neuron n."""
    syn_idxs = np.where(edges[:, 0] == n)[0]
    return syn_idxs, edges[syn_idxs, 1]

def spikes(V, n):
    return np.where(V[:, n] >= th)[0].tolist()

# ── 1. Full spike raster comparison ──────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"FULL SPIKE RASTER  (target vs found)  case {CASE_IDX}: {tc['name']}")
print(f"{'═'*70}")
print(f"{'n':>3}  {'tgt_sp':>6}  {'fnd_sp':>6}  {'match':>5}  target_times → found_times")

for n in range(N):
    t_sp = spikes(target_v, n)
    f_sp = spikes(found_v,  n)
    nt, nf = len(t_sp), len(f_sp)
    flag = "★" if nt == nf and nt > 0 else ("0" if nt == 0 else ("✗" if nt != nf else "~"))
    marker = " ←OUT" if n in outs else ""
    if nt > 0 or nf > 0:
        print(f"{n:>3}  {nt:>6}  {nf:>6}  {flag:>5}  {t_sp} → {f_sp}{marker}")

# ── 2. Output neuron voltage traces at spike times ────────────────────────────
print(f"\n{'═'*70}")
print(f"OUTPUT NEURON VOLTAGES AT TARGET SPIKE TIMES")
print(f"Threshold = {th:.3f}")
print(f"{'═'*70}")

for n in outs:
    t_sp = spikes(target_v, n)
    f_sp = spikes(found_v,  n)
    print(f"\n  Neuron {n}  (target={len(t_sp)}sp  found={len(f_sp)}sp)")
    print(f"  {'t':>5}  {'V_target':>10}  {'V_found':>10}  {'below_th':>10}  {'status'}")
    for tt in t_sp:
        vt = target_v[tt, n]
        vf = found_v[tt, n]
        gap = th - vf
        status = "FIRES" if vf >= th else f"SILENT  gap={gap:.3f} ({gap/th*100:.0f}%th)"
        print(f"  {tt:>5}  {vt:>10.4f}  {vf:>10.4f}  {gap:>10.4f}  {status}")
    # Also show max voltage of found simulation for this neuron
    print(f"  max V_found[{n}] = {found_v[:, n].max():.4f}  "
          f"(vs th={th:.3f}, {found_v[:, n].max()/th*100:.0f}%)")

# ── 3. Pre-synaptic analysis for each output neuron ──────────────────────────
print(f"\n{'═'*70}")
print(f"PRE-SYNAPTIC INPUTS TO OUTPUT NEURONS")
print(f"{'═'*70}")

WINDOW = 50
for n in outs:
    syn_idxs, pre = pre_of(n)
    t_sp_n = spikes(target_v, n)
    print(f"\n  Neuron {n} receives input from {len(pre)} neurons: {sorted(pre.tolist())}")
    print(f"  True weight range:  [{true_w[syn_idxs].min():.4f}, {true_w[syn_idxs].max():.4f}]"
          f"  sum={true_w[syn_idxs].sum():.4f}")
    print(f"  Found weight range: [{found_w[syn_idxs].min():.4f}, {found_w[syn_idxs].max():.4f}]"
          f"  sum={found_w[syn_idxs].sum():.4f}")

    for tt in t_sp_n:
        t_start = max(0, tt - WINDOW)
        print(f"\n    At target spike t={tt} (window [{t_start},{tt}]):")
        for si, p in zip(syn_idxs, pre):
            t_pre = [s for s in spikes(target_v, p) if t_start <= s <= tt]
            f_pre = [s for s in spikes(found_v,  p) if t_start <= s <= tt]
            if t_pre or f_pre:
                print(f"      pre N{p:>2}  tw={true_w[si]:+.4f}  fw={found_w[si]:+.4f}"
                      f"  tgt={t_pre}  fnd={f_pre}")

# ── 4. Weight comparison for output neuron inputs ────────────────────────────
print(f"\n{'═'*70}")
print(f"WEIGHT COMPARISON: inputs to output neurons")
print(f"{'═'*70}")

for n in outs:
    syn_idxs, pre = pre_of(n)
    print(f"\n  Neuron {n} ({len(pre)} inputs):")
    print(f"  {'pre':>4}  {'true_w':>10}  {'found_w':>10}  {'ratio':>7}")
    for si, p in zip(syn_idxs, pre):
        tw_v, fw_v = true_w[si], found_w[si]
        ratio = fw_v / tw_v if abs(tw_v) > 1e-9 else float("nan")
        print(f"  {p:>4}  {tw_v:>10.4f}  {fw_v:>10.4f}  {ratio:>7.3f}")

# ── 5. Summary voltage statistics ────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"VOLTAGE STATISTICS (all neurons, found simulation)")
print(f"{'═'*70}")
print(f"{'n':>3}  {'max_V':>8}  {'%th':>6}  {'tgt_sp':>7}  {'fnd_sp':>7}  {'note'}")
for n in range(N):
    max_v  = found_v[:, n].max()
    fnd_sp = int(np.sum(found_v[:, n] >= th))
    tgt_sp = int(np.sum(target_v[:, n] >= th))
    note   = " ←OUTPUT" if n in outs else ""
    if max_v > 0.5 * th or fnd_sp > 0 or tgt_sp > 0 or n in outs:
        print(f"{n:>3}  {max_v:>8.4f}  {max_v/th*100:>5.0f}%  {tgt_sp:>7}  {fnd_sp:>7}  {note}")
