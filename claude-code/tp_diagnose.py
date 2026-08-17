"""Diagnose the residual 50-neuron TP gap: per-neuron recovery vs error compounding.

Two hypotheses for why oracle TP gives loss>0 even with TRUE pre-spike times:
  (A) recovery error  — recovered w mistimes a neuron's OWN spikes even when
                        that neuron is driven by the TRUE input spike trains.
  (B) compounding     — every neuron is fine in isolation (true inputs), but in
                        the FULL simultaneous sim small timing errors propagate
                        and amplify through the recurrent graph.

Discriminator per neuron n:
  single_hop_err[n] = timing error of n's spikes when an isolated LIF neuron is
                      driven by n's TRUE input spikes using recovered weights.
  full_sim_err[n]   = timing error of n in the full sim with all recovered w.

If single_hop_err ≈ 0 everywhere but full_sim_err > 0  → (B) compounding.
If some single_hop_err > 0                            → (A), and it names them.
"""

import sys, os, types, dataclasses, time
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

CASE_IDX = int(os.environ.get("CASE", "2"))
RIDGE    = float(os.environ.get("RIDGE", "1e-3"))
MARGIN   = float(os.environ.get("MARGIN", "0.05"))

tc = RECURRENT_CASES[CASE_IDX]
conns, tw = _make_recurrent_weights(
    tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
    tc["num_neurons"], tc["output_neurons"])

params = dataclasses.replace(sim.default_params, steps=1000)
th, gsw = params.threshold, params.global_synapse_weight
delay   = params.delay_iters
refr    = params.refractory_iters
nd      = float(params.neuron_decay)
rd      = float(params.rise_decay)
A_ext   = jnp.array([0])

C_np = np.array(conns, dtype=np.int32)
C    = jnp.array(C_np)
N    = tc["num_neurons"]
outs = tc["output_neurons"]
edges = C_np
w_true = np.array(tw, np.float32)
lo, hi = w_true * 0.1, w_true * 5.0

MAX_H = 600
h = np.zeros(MAX_H)
_R = _V = 0.0
for t in range(MAX_H):
    _R = (_R + (1.0 if t == delay else 0.0)) * rd
    _V = (_V - _R) * nd + _R
    h[t] = _V

def pre_of(n):
    idx = np.where(edges[:, 1] == n)[0]
    return idx, edges[idx, 0]

def spikes_of(V_np, n):
    return np.where(V_np[:, n] >= th)[0].tolist()

target_v = np.array(_hard_sim(jnp.array(w_true), params, C, N, A_ext))
T_true   = {n: spikes_of(target_v, n) for n in range(N)}

# ── single-neuron LIF driven by given input spike trains ──────────────────
def lif_single(input_spikes_by_neuron, syn_weights, steps=1000):
    """Isolated LIF: events = (t_pre + delay, w*gsw). Returns spike times.
    input_spikes_by_neuron: {pre_id: [spike times]}; syn_weights: {pre_id: w}."""
    events = []
    for pid, sp in input_spikes_by_neuron.items():
        w = syn_weights[pid] * gsw
        for tk in sp:
            ta = tk + delay
            if ta < steps:
                events.append((ta, w))
    events.sort()
    ev = iter(events); nxt = next(ev, None)
    V = R = 0.0; ref = 0; out = []
    for t in range(steps):
        upd = 0.0
        while nxt and nxt[0] == t:
            upd += nxt[1]; nxt = next(ev, None)
        R = (R + upd) * rd * (ref != 1)
        V = (V - R) * nd + R
        V = V * (ref == 0)
        if V >= th and ref == 0:
            out.append(t); ref = refr + 1
        elif ref > 0:
            ref -= 1
    return out

# ── ridge NNLS + QP recovery (same as tp_50neuron) ─────────────────────────
def nnls_ridge(A_mat, b_vec, ridge_frac):
    ncol = A_mat.shape[1]
    if ridge_frac <= 0:
        return nnls(A_mat, b_vec)
    lam = ridge_frac * np.trace(A_mat.T @ A_mat) / max(ncol, 1)
    A_aug = np.vstack([A_mat, np.sqrt(lam) * np.eye(ncol)])
    b_aug = np.concatenate([b_vec, np.zeros(ncol)])
    return nnls(A_aug, b_aug)

def contrib(n, pre_spikes, eval_times, epoch_starts):
    syn_idxs, pres = pre_of(n)
    rows = []
    for j, Tj in enumerate(eval_times):
        Tp = epoch_starts[j]
        row = np.zeros(len(pres))
        for ci, p in enumerate(pres):
            for tk in pre_spikes.get(int(p), []):
                if Tp < tk < Tj:
                    dt = Tj - tk
                    if 0 < dt < MAX_H:
                        row[ci] += h[dt]
        rows.append(row)
    return syn_idxs, pres, np.array(rows)

def recover_neuron(n, pre_spikes, tgt, margin, ridge):
    syn_idxs, pres = pre_of(n)
    if len(pres) == 0 or len(tgt) == 0:
        return None
    tgt = list(tgt)
    starts = [0] + tgt[:-1]
    _, _, A_lo = contrib(n, pre_spikes, tgt, starts)
    b = np.full(len(tgt), th / gsw)
    valid = A_lo.max(axis=1) > 1e-12
    A_lo, b = A_lo[valid], b[valid]
    if len(A_lo) == 0:
        return None
    w_nnls, _ = nnls_ridge(A_lo, b, ridge)
    if margin <= 0:
        return syn_idxs, np.clip(w_nnls, lo[syn_idxs], hi[syn_idxs])
    nt, nte = [], []
    for j in range(len(tgt) - 1):
        e = tgt[j] + refr + 2; nx = tgt[j + 1]
        if e >= nx: continue
        for f in [0.25, 0.45, 0.65, 0.85]:
            nt.append(int(e + f * (nx - e))); nte.append(tgt[j])
    A_up = np.zeros((0, len(pres))); b_up = np.zeros(0)
    if nt:
        _, _, A_up = contrib(n, pre_spikes, nt, nte)
        b_up = np.full(len(nt), (1 - margin) * th / gsw)
        v = A_up.max(axis=1) > 1e-12
        A_up, b_up = A_up[v], b_up[v]
    cons = [{'type': 'ineq', 'fun': lambda w, A=A_lo, bb=b: A @ w - bb}]
    if len(A_up) > 0:
        cons.append({'type': 'ineq', 'fun': lambda w, A=A_up, bb=b_up: bb - A @ w})
    bnds = [(float(lo[si]), float(hi[si])) for si in syn_idxs]
    r = minimize(lambda w: 0.5 * float(np.dot(w - w_nnls, w - w_nnls)), w_nnls.copy(),
                 jac=lambda w: w - w_nnls, method='SLSQP', constraints=cons,
                 bounds=bnds, options={'ftol': 1e-10, 'maxiter': 2000})
    w_sol = r.x if (r.success or r.status in [0, 4]) else w_nnls
    return syn_idxs, np.clip(w_sol, lo[syn_idxs], hi[syn_idxs])

# ── recover all weights (oracle: true pre-spikes + true targets) ───────────
w_tp = w_true.copy()
for n in range(N):
    if n == 0 or not T_true[n]:
        continue
    res = recover_neuron(n, T_true, T_true[n], MARGIN, RIDGE)
    if res is not None:
        idxs, ws = res
        w_tp[idxs] = ws

# ── full sim with recovered weights ────────────────────────────────────────
v_full = np.array(_hard_sim(jnp.array(w_tp.astype(np.float32)), params, C, N, A_ext))
T_full = {n: spikes_of(v_full, n) for n in range(N)}
loss = float(sum(np.sum((target_v[:, n] - v_full[:, n])**2) for n in outs))

def terr(found, tgt):
    if len(found) != len(tgt):
        return None
    return max((abs(f - t) for f, t in zip(found, tgt)), default=0)

# ── per-neuron: single-hop (true inputs) vs full-sim ───────────────────────
print(f"Case {CASE_IDX}  ridge={RIDGE}  margin={MARGIN}")
print(f"Full-sim output loss = {loss:.4e}\n")
print("Per-neuron: single-hop err (true inputs+recovered w) vs full-sim err")
print(f"{'n':>4} {'depth':>5} {'#in':>4} {'wErr%':>6} {'sHop':>6} {'full':>6}  note")

# compute a rough topological depth (longest path from external N0)
depth = {0: 0}
for _ in range(N):
    for k in range(len(edges)):
        pre, post = int(edges[k, 0]), int(edges[k, 1])
        if pre in depth:
            depth[post] = max(depth.get(post, 0), depth[pre] + 1)

def neuron_system(n):
    """Return the lower-bound TP linear system for neuron n on true inputs:
    A_lo (rows=target spikes, cols=inputs), plus cond and NNLS residual."""
    idxs, pres = pre_of(n)
    tgt = list(T_true[n]); starts = [0] + tgt[:-1]
    _, _, A_lo = contrib(n, T_true, tgt, starts)
    b = np.full(len(tgt), th / gsw)
    valid = A_lo.max(axis=1) > 1e-12
    A_lo, b = A_lo[valid], b[valid]
    if len(A_lo) == 0:
        return None
    w_nnls, _ = nnls_ridge(A_lo, b, RIDGE)
    resid = float(np.linalg.norm(A_lo @ w_nnls - b) / (np.linalg.norm(b) + 1e-12))
    cond  = float(np.linalg.cond(A_lo)) if A_lo.shape[1] > 1 else 1.0
    return dict(n_in=A_lo.shape[1], n_eq=A_lo.shape[0], cond=cond, resid=resid)

rows_summary = []
print("  (cond=cond(A), resid=relative NNLS residual, over/under = n_eq vs n_in)")
for n in range(1, N):
    if not T_true[n]:
        continue
    idxs, pres = pre_of(n)
    inp = {int(p): T_true[int(p)] for p in pres}
    wmap = {int(p): float(w_tp[si]) for si, p in zip(idxs, pres)}
    sh = lif_single(inp, wmap)
    she = terr(sh, T_true[n])
    fe  = terr(T_full[n], T_true[n])
    werr = 100 * np.mean(np.abs(w_tp[idxs] - w_true[idxs]) / (w_true[idxs] + 1e-6))
    sys = neuron_system(n) or dict(n_in=len(pres), n_eq=0, cond=np.inf, resid=np.inf)
    bad = (she is None or she > 2)
    rows_summary.append(dict(n=n, she=she, fe=fe, werr=werr, bad=bad, **sys))

# ── per-neuron table sorted by conditioning ────────────────────────────────
print(f"\n{'n':>4} {'n_in':>4} {'n_eq':>4} {'eq/in':>5} {'cond':>9} {'resid':>7} "
      f"{'wErr%':>6} {'sHop':>5}  group")
for r in sorted(rows_summary, key=lambda x: -x['cond']):
    grp = "RECOVERY-BAD" if r['bad'] else "compounds"
    she_s = f"{r['she']}" if r['she'] is not None else "cnt✗"
    ratio = r['n_eq'] / max(r['n_in'], 1)
    print(f"{r['n']:>4} {r['n_in']:>4} {r['n_eq']:>4} {ratio:>5.1f} {r['cond']:>9.1e} "
          f"{r['resid']:>7.3f} {r['werr']:>5.0f}% {she_s:>5}  {grp}")

# ── grouped comparison: what distinguishes bad from good recovery? ─────────
def stats(key, group):
    vals = [r[key] for r in rows_summary if r['bad'] == group and np.isfinite(r[key])]
    if not vals:
        return "—"
    return f"med={np.median(vals):.2g} mean={np.mean(vals):.2g}"

print(f"\n{'metric':>10}  {'RECOVERY-BAD (sHop>2)':>28}  {'GOOD (sHop<=2)':>28}")
for key in ['n_in', 'n_eq', 'cond', 'resid', 'werr']:
    print(f"{key:>10}  {stats(key, True):>28}  {stats(key, False):>28}")

n_bad  = sum(1 for r in rows_summary if r['bad'])
n_good = sum(1 for r in rows_summary if not r['bad'])
print(f"\n{len(rows_summary)} firing neurons: {n_bad} recovery-bad, {n_good} good single-hop")
# under-determined counts
und_bad  = sum(1 for r in rows_summary if r['bad'] and r['n_eq'] < r['n_in'])
und_good = sum(1 for r in rows_summary if not r['bad'] and r['n_eq'] < r['n_in'])
print(f"under-determined (n_eq<n_in): {und_bad}/{n_bad} of bad, {und_good}/{n_good} of good")

# ── correlation of structural features with weight error ───────────────────
def corr(xk, yk):
    xs = np.array([r[xk] for r in rows_summary], float)
    ys = np.array([r[yk] for r in rows_summary], float)
    m  = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3: return float('nan')
    return float(np.corrcoef(xs[m], ys[m])[0, 1])
for r in rows_summary:
    r['eqin'] = r['n_eq'] / max(r['n_in'], 1)
print(f"\nPearson corr with weight-error (wErr%):")
for k in ['n_in', 'n_eq', 'eqin', 'cond', 'resid']:
    print(f"  {k:>6}: {corr(k, 'werr'):+.2f}")
print("\nDone.")
