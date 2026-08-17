"""Look at the real dynamics of tiny spiking models and how they respond to a
weight change.  Pure observation — no proposed method.

We want to see, per neuron, the data that (a) says how a weight should change and
(b) how it couples to upstream neurons.  So for each model we extract:

  - the voltage trace V[t, n] and the spike (threshold-crossing) times
  - the finite-difference sensitivity dV[t,n]/dw of every neuron's voltage to a
    single synapse weight  (the "true" smooth signal the hard step throws away)
  - at each threshold crossing t*: the margin (th - V just before), the slope
    dV/dt across the crossing, dV/dw at the crossing, and the empirical spike-time
    shift dt*/dw  vs the event-based prediction  -(dV/dw)/(dV/dt).

Models:
  A  single synapse   N0(driven) -> N1
  B  three-neuron chain  N0 -> N1 -> N2
"""

import sys, os, types, dataclasses
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax_spiking_model as sim

STEPS = 260
params = dataclasses.replace(sim.default_params, steps=STEPS)
TH  = params.threshold          # 0.007
GSW = params.global_synapse_weight
DIR = "/workspace/project/gradient_study"


def run(conns, N, w, act=(0,)):
    """Return voltage array (STEPS, N) for weights w (raw, pre-gsw)."""
    C = jnp.array(np.array(conns, np.int32))
    A = jnp.array(np.array(act, np.int32))
    V, _, _ = sim.run_sim(params, C, N, jnp.array(np.array(w, np.float32)), A)
    return np.array(V)


def spike_times(v, n):
    return np.where(v[:, n] >= TH)[0].tolist()


def crossings(v, n):
    """Timesteps where neuron n crosses threshold from below (spike onsets)."""
    x = v[:, n]
    return [t for t in range(1, len(x)) if x[t] >= TH and x[t-1] < TH]


def fd_dVdw(conns, N, w, syn, eps_frac=1e-3, act=(0,)):
    """Central finite-difference dV[t,n]/dw for synapse index `syn`."""
    w = np.array(w, np.float64)
    e = max(abs(w[syn]) * eps_frac, 1e-3)
    wp = w.copy(); wp[syn] += e
    wm = w.copy(); wm[syn] -= e
    return (run(conns, N, wp, act) - run(conns, N, wm, act)) / (2 * e)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL A: single synapse  N0 -> N1
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 74)
print("MODEL A: single synapse  N0(driven) -> N1")
print("=" * 74)

connsA = [[0, 1]]
NA = 2

# sweep the weight and watch N1
ws = [100, 200, 300, 350, 400, 500, 700]
print(f"\n threshold th = {TH}")
print(f"{'w':>6} {'max V(N1)':>12} {'V/th':>7} {'#spikes':>8}  first spike t")
for w in ws:
    v = run(connsA, NA, [w])
    st = spike_times(v, 1)
    cr = crossings(v, 1)
    print(f"{w:>6} {v[:,1].max():>12.6f} {v[:,1].max()/TH:>7.3f} {len(cr):>8}  "
          f"{cr[0] if cr else '-'}")

# find where the first spike appears (count discontinuity) by fine sweep
fine = np.arange(150, 780, 4)
counts = [len(crossings(run(connsA, NA, [w]), 1)) for w in fine]
firstw = next((w for w, c in zip(fine, counts) if c >= 1), None)
print(f"\n first spike appears between w={firstw-2} and w={firstw} "
      f"(count jumps 0 -> {counts[list(fine).index(firstw)]})")

# spike-time vs w once firing (continuous shift)
print(f"\n once firing, first-spike time vs w (continuous shift):")
for w in [max(firstw,340), 360, 400, 460, 540, 640]:
    cr = crossings(run(connsA, NA, [w]), 1)
    print(f"   w={w:>4}  first spike t = {cr[0] if cr else '-'}   "
          f"n_spikes={len(cr)}")

# sensitivity of N1 voltage to w  (should be a smooth PSP-shaped trace)
w0 = 460.0
vA = run(connsA, NA, [w0])
dV = fd_dVdw(connsA, NA, [w0], 0)          # (STEPS, 2)
crA = crossings(vA, 1)
print(f"\n at w={w0}: N1 spikes at {crA}")
print(f" dV(N1)/dw is smooth and nonzero everywhere N1 has input;")
print(f"   peak dV/dw = {dV[:,1].max():.3e} at t={dV[:,1].argmax()}  "
      f"(hard dspike/dV = 0 here except exactly at crossing)")

# per-crossing quantities + event-based spike-time gradient check
print(f"\n per-crossing data (N1), and empirical vs event-based dt*/dw:")
print(f"  {'t*':>4} {'margin(th-V[t*-1])':>18} {'slope dV/dt':>12} "
      f"{'dV/dw@t*':>12} {'dt*/dw emp':>11} {'-dVdw/slope':>12}")
eps = 5.0
crp = crossings(run(connsA, NA, [w0 + eps]), 1)
crm = crossings(run(connsA, NA, [w0 - eps]), 1)
for i, t in enumerate(crA):
    margin = TH - vA[t-1, 1]
    slope = vA[t, 1] - vA[t-1, 1]
    dvdw = dV[t, 1]
    dt_emp = ((crp[i] if i < len(crp) else np.nan) -
              (crm[i] if i < len(crm) else np.nan)) / (2 * eps)
    pred = -dvdw / slope if slope != 0 else np.nan
    print(f"  {t:>4} {margin:>18.3e} {slope:>12.3e} {dvdw:>12.3e} "
          f"{dt_emp:>11.4f} {pred:>12.4f}")

# ── figure A ──
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
axw = ax[0, 0]
for w in [200, 300, firstw, 460, 640]:
    v = run(connsA, NA, [w])
    axw.plot(v[:, 1], label=f"w={w}")
axw.axhline(TH, color="k", ls="--", lw=1, label="threshold")
axw.set_title("A1: N1 voltage vs time, sweeping the single weight")
axw.set_xlabel("timestep"); axw.set_ylabel("V(N1)"); axw.legend(fontsize=8)
axw.set_ylim(0, TH * 1.6)

axc = ax[0, 1]
axc.plot(fine, counts, "o-", ms=3)
axc.set_title("A2: spike COUNT vs w  (discontinuous — spikes are created)")
axc.set_xlabel("w"); axc.set_ylabel("# spikes in N1"); axc.grid(alpha=.3)

axt = ax[1, 0]
sweep2 = np.arange(firstw, 700, 4)
ft = [(crossings(run(connsA, NA, [w]), 1) or [np.nan])[0] for w in sweep2]
axt.plot(sweep2, ft, ".-", ms=4)
axt.set_title("A3: first-spike TIME vs w  (continuous — spikes slide earlier)")
axt.set_xlabel("w"); axt.set_ylabel("first spike timestep"); axt.grid(alpha=.3)

axs = ax[1, 1]
axs.plot(vA[:, 1], label="V(N1)", color="C0")
axs.axhline(TH, color="k", ls="--", lw=1)
axs.set_ylim(0, TH * 1.3); axs.set_ylabel("V(N1)", color="C0")
axs.set_xlabel("timestep")
axs2 = axs.twinx()
axs2.plot(dV[:, 1], color="C3", lw=1.2, label="dV(N1)/dw")
axs2.set_ylim(-0.3e-5, 2.0e-5)      # clip: subthreshold PSP ramp visible; resets go off-scale
axs2.set_ylabel("dV(N1)/dw  (clipped)", color="C3")
for t in crA:
    axs.axvline(t, color="gray", ls=":", lw=1)
axs.set_title("A4: subthreshold dV/dw = smooth PSP ramp; at spikes (dotted) it spikes off-scale")
axs.legend(loc="upper left", fontsize=8); axs2.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{DIR}/modelA_single_synapse.png", dpi=120)
print(f"\n wrote {DIR}/modelA_single_synapse.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL B: chain  N0 -> N1 -> N2
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("MODEL B: chain  N0(driven) -> N1 -> N2   (syn0=w01, syn1=w12)")
print("=" * 74)

connsB = [[0, 1], [1, 2]]
NB = 3
wB = [460.0, 460.0]
vB = run(connsB, NB, wB)
for n in range(NB):
    print(f"  N{n} spikes: {crossings(vB, n)}")

# perturb w01 (upstream) vs w12 (downstream); watch N2
print("\n effect on N2 of perturbing each weight (+20):")
for syn, name in [(0, "w01 (upstream)"), (1, "w12 (downstream)")]:
    wp = list(wB); wp[syn] += 20
    v2 = run(connsB, NB, wp)
    print(f"   {name:18s}: N1 {crossings(v2,1)}   N2 {crossings(v2,2)} "
          f"(base N2 {crossings(vB,2)})")

# sensitivity of N2 voltage to the UPSTREAM weight w01 — routed through N1 spikes
dV2_dw01 = fd_dVdw(connsB, NB, wB, 0)
dV2_dw12 = fd_dVdw(connsB, NB, wB, 1)
print(f"\n peak |dV(N2)/dw01| = {np.abs(dV2_dw01[:,2]).max():.3e}  "
      f"(upstream weight, coupled through N1's spike times)")
print(f" peak |dV(N2)/dw12| = {np.abs(dV2_dw12[:,2]).max():.3e}  (direct)")

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
a = ax[0, 0]
for n, c in zip(range(3), ["C0", "C1", "C2"]):
    a.plot(vB[:, n], color=c, label=f"N{n}")
a.axhline(TH, color="k", ls="--", lw=1)
a.set_title("B1: chain voltages (base)"); a.set_xlabel("t"); a.legend(fontsize=8)
a.set_ylim(0, TH * 1.6)

a = ax[0, 1]
a.plot(vB[:, 2], label="N2 base", color="C2")
wp = list(wB); wp[0] += 20
a.plot(run(connsB, NB, wp)[:, 2], label="N2 (w01+20)", color="C3", ls="--")
a.axhline(TH, color="k", ls="--", lw=1)
a.set_title("B2: upstream weight change shifts N2 via N1 timing")
a.set_xlabel("t"); a.legend(fontsize=8); a.set_ylim(0, TH * 1.6)

a = ax[1, 0]
a.plot(dV2_dw01[:, 1], label="dV(N1)/dw01", color="C1")
a.plot(dV2_dw01[:, 2], label="dV(N2)/dw01", color="C2")
for t in crossings(vB, 1):
    a.axvline(t, color="gray", ls=":", lw=1)
a.set_title("B3: sensitivity to UPSTREAM w01 (note spikes of N1 as gray lines)")
a.set_xlabel("t"); a.legend(fontsize=8)

a = ax[1, 1]
a.plot(dV2_dw12[:, 2], label="dV(N2)/dw12", color="C2")
a.set_title("B4: sensitivity to DIRECT w12 (smooth PSP train)")
a.set_xlabel("t"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{DIR}/modelB_chain.png", dpi=120)
print(f"\n wrote {DIR}/modelB_chain.png")
print("\nDone.")
