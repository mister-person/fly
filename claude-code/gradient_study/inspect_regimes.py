"""Two regimes the single-synapse study left open (observation only):

  C  a neuron with TWO summing inputs — how credit splits at a crossing, and
     the degenerate coincident-input case.
  D  the DEAD-neuron regime — what (if anything) is smooth when a neuron never
     crosses threshold, and which losses keep a gradient there.

We use the exact subthreshold voltage  V_post(t) = gsw * Σ_i w_i * Σ_k h(t - t_ik),
where h is the LIF post-synaptic-potential kernel (same one target_prop uses).
This is exact up to the first reset, which is all we need to look at crossings.
"""

import sys, os, types, dataclasses
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m

import numpy as np
import matplotlib.pyplot as plt
import jax_spiking_model as sim

p     = sim.default_params
TH    = p.threshold
GSW   = p.global_synapse_weight
DELAY = p.delay_iters
ND    = float(p.neuron_decay)
RD    = float(p.rise_decay)
DIR   = "/workspace/project/gradient_study"
MAXH  = 400

# LIF PSP kernel: V per unit synaptic update, single pre-spike at t=0
h = np.zeros(MAXH)
_R = _V = 0.0
for t in range(MAXH):
    _R = (_R + (1.0 if t == DELAY else 0.0)) * RD
    _V = (_V - _R) * ND + _R
    h[t] = _V
H_MAX = h.max()
print(f"th={TH}  gsw={GSW}  PSP peak h={H_MAX:.4f} at t={h.argmax()}")
print(f"single-input firing threshold w* = th/(gsw*h_max) = {TH/(GSW*H_MAX):.1f}")


def hh(dt):
    dt = np.asarray(dt)
    out = np.zeros_like(dt, dtype=float)
    m = (dt >= 0) & (dt < MAXH)
    out[m] = h[dt[m].astype(int)]
    return out


def voltage(T, inputs):
    """inputs = list of (w, [spike_times]).  Returns V over 0..T-1 (subthreshold)."""
    ts = np.arange(T)
    V = np.zeros(T)
    for w, sp in inputs:
        for tk in sp:
            V += w * GSW * hh(ts - tk)
    return V


def first_cross(V):
    idx = np.where(V >= TH)[0]
    return int(idx[0]) if len(idx) else None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL C: two summing inputs A, B -> one neuron
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("MODEL C: two inputs A,B -> N. Credit at the crossing = gsw*h(t* - t_input)")
print("=" * 74)

T = 200
tA = 20
# each input alone is SUBTHRESHOLD; only together do they cross (coincidence)
wA = wB = 300.0
print(f"\n each input alone: w={wA}, single-input w*={TH/(GSW*H_MAX):.0f} -> "
      f"alone {'FIRES' if wA>TH/(GSW*H_MAX) else 'stays dead'}")

print(f"\n vary the inter-input gap Δ = t_B - t_A  (wA=wB={wA}):")
print(f"  {'Δ':>4} {'cross t*':>9} {'creditA':>10} {'creditB':>10} {'A/(A+B)':>9}")
deltas = [0, 5, 10, 20, 30, 45, 60]
rows = []
for d in deltas:
    V = voltage(T, [(wA, [tA]), (wB, [tA + d])])
    tc = first_cross(V)
    if tc is None:
        print(f"  {d:>4} {'(no spike)':>9}    PSPs too far apart to sum to threshold")
        rows.append((d, None, 0, 0))
        continue
    cA = GSW * hh(np.array([tc - tA]))[0]        # dV/dwA at crossing
    cB = GSW * hh(np.array([tc - (tA + d)]))[0]  # dV/dwB at crossing
    frac = cA / (cA + cB + 1e-30)
    print(f"  {d:>4} {tc:>9} {cA:>10.3e} {cB:>10.3e} {frac:>9.3f}")
    rows.append((d, tc, cA, cB))

print("\n Δ=0 (coincident): creditA == creditB exactly -> the two columns are")
print(" proportional; only wA+wB is identifiable from this crossing (the ridge case).")

# ── figure C ──
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
a = ax[0, 0]
for d, c in zip([0, 10, 30, 60], ["C0", "C1", "C2", "C3"]):
    V = voltage(T, [(wA, [tA]), (wB, [tA + d])])
    a.plot(V, color=c, label=f"Δ={d}")
a.axhline(TH, color="k", ls="--", lw=1)
a.set_title("C1: summed voltage at N for different input gaps Δ")
a.set_xlabel("t"); a.set_ylabel("V(N)"); a.legend(fontsize=8)

a = ax[0, 1]
ds = np.arange(0, 90)
fr = []
for d in ds:
    V = voltage(T, [(wA, [tA]), (wB, [tA + d])])
    tc = first_cross(V)
    if tc is None:
        fr.append(np.nan); continue
    cA = GSW * hh(np.array([tc - tA]))[0]
    cB = GSW * hh(np.array([tc - (tA + d)]))[0]
    fr.append(cA / (cA + cB))
a.plot(ds, fr, ".-", ms=4)
a.axhline(0.5, color="gray", ls=":")
a.set_title("C2: credit fraction to input A vs gap Δ\n(0.5=shared; →1 as B lands after the crossing)")
a.set_xlabel("Δ = t_B - t_A"); a.set_ylabel("creditA / (creditA+creditB)")
a.grid(alpha=.3)

a = ax[1, 0]
# coincidence detection: fires only if BOTH present
for label, inp, c in [("A only", [(wA, [tA])], "C0"),
                      ("B only", [(wB, [tA + 10])], "C1"),
                      ("A+B (Δ=10)", [(wA, [tA]), (wB, [tA + 10])], "C2")]:
    a.plot(voltage(T, inp), color=c, label=label)
a.axhline(TH, color="k", ls="--", lw=1)
a.set_title("C3: coincidence — neither input alone fires, together they do")
a.set_xlabel("t"); a.set_ylabel("V(N)"); a.legend(fontsize=8)

a = ax[1, 1]
# credit vs the crossing point, Δ=10 fixed, sweep wA=wB scale
V = voltage(T, [(wA, [tA]), (wB, [tA + 10])])
tc = first_cross(V)
a.plot(GSW * hh(np.arange(T) - tA), color="C0", label="dV/dwA(t) = gsw·h(t-tA)")
a.plot(GSW * hh(np.arange(T) - (tA + 10)), color="C1", label="dV/dwB(t) = gsw·h(t-tB)")
if tc:
    a.axvline(tc, color="k", ls="--", lw=1, label=f"crossing t*={tc}")
a.set_title("C4: per-input sensitivity traces; credit = their value at t*")
a.set_xlabel("t"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{DIR}/modelC_summing.png", dpi=120)
print(f"\n wrote {DIR}/modelC_summing.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL D: dead neuron — one input, weight below firing threshold
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("MODEL D: dead neuron. Single input; sweep w through the firing threshold w*")
print("=" * 74)

wstar = TH / (GSW * H_MAX)
T = 200
tA = 20
ws = np.linspace(20, 2.0 * wstar, 200)
maxV = np.array([voltage(T, [(w, [tA])]).max() for w in ws])
margin = TH - maxV
nsp = np.array([1 if first_cross(voltage(T, [(w, [tA])])) is not None else 0 for w in ws])

# a target trace: neuron firing at w_target (above w*)
w_target = 1.4 * wstar
V_target = voltage(T, [(w_target, [tA])])
sp_target = (V_target >= TH).astype(float)

def exp_conv(x, tau=20.0):
    dec = np.exp(-1.0 / tau); S = np.zeros_like(x); acc = 0.0
    for i, xi in enumerate(x):
        acc = acc * dec + xi; S[i] = acc
    return S
S_target = exp_conv(sp_target)

def losses_and_grads(w, eps=1.0):
    def L_volt(ww):
        return float(np.sum((voltage(T, [(ww, [tA])]) - V_target) ** 2))
    def L_spike(ww):
        V = voltage(T, [(ww, [tA])])
        S = exp_conv((V >= TH).astype(float))
        return float(np.sum((S - S_target) ** 2))
    gv = (L_volt(w + eps) - L_volt(w - eps)) / (2 * eps)
    gs = (L_spike(w + eps) - L_spike(w - eps)) / (2 * eps)
    return gv, gs

gv = np.array([losses_and_grads(w)[0] for w in ws])
gs = np.array([losses_and_grads(w)[1] for w in ws])

print(f"\n w* (first fires) = {wstar:.1f}   target weight = {w_target:.1f}")
print(f"\n {'w':>6} {'maxV/th':>8} {'margin':>10} {'#sp':>4} "
      f"{'|grad voltage-loss|':>19} {'|grad spike-loss|':>18}")
for w in [30, 60, wstar*0.7, wstar*0.95, wstar*1.05, wstar*1.3, w_target]:
    V = voltage(T, [(w, [tA])])
    tc = first_cross(V)
    gvw, gsw_ = losses_and_grads(w)
    print(f" {w:>6.0f} {V.max()/TH:>8.3f} {TH-V.max():>10.3e} "
          f"{0 if tc is None else 1:>4} {abs(gvw):>19.3e} {abs(gsw_):>18.3e}")

print("\n In the DEAD region (w<w*): margin is smooth & linear (crosses 0 at w*),")
print(" voltage-loss gradient is NONZERO and points toward firing, but the")
print(" spike-loss gradient is exactly 0 (no spikes -> flat loss). A dead neuron")
print(" is only recoverable via a signal that reads its subthreshold voltage.")

# ── figure D ──
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
a = ax[0, 0]
a.plot(ws, maxV / TH, color="C0")
a.axhline(1.0, color="k", ls="--", lw=1, label="threshold")
a.axvline(wstar, color="C3", ls=":", label=f"w*={wstar:.0f}")
a.set_title("D1: max V vs w (smooth, linear)"); a.set_xlabel("w")
a.set_ylabel("max V / th"); a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[0, 1]
a.plot(ws, nsp, color="C2")
a.axvline(wstar, color="C3", ls=":")
a.set_title("D2: #spikes vs w (step at w*)"); a.set_xlabel("w")
a.set_ylabel("# spikes"); a.grid(alpha=.3)

a = ax[1, 0]
a.plot(ws, margin, color="C0")
a.axhline(0, color="k", ls="--", lw=1)
a.axvline(wstar, color="C3", ls=":", label=f"w*={wstar:.0f}")
a.set_title("D3: margin (th - maxV) — smooth per-neuron 'distance to firing'")
a.set_xlabel("w"); a.set_ylabel("th - max V"); a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[1, 1]
a.plot(ws, np.abs(gv) / (np.abs(gv).max() + 1e-30), color="C0", label="|grad| voltage-MSE loss")
a.plot(ws, np.abs(gs) / (np.abs(gs).max() + 1e-30), color="C3", label="|grad| spike-timing loss")
a.axvline(wstar, color="C3", ls=":")
a.set_title("D4: gradient vs w (normalized).\nspike-loss grad = 0 while dead; voltage-loss grad is not")
a.set_xlabel("w"); a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{DIR}/modelD_dead.png", dpi=120)
print(f"\n wrote {DIR}/modelD_dead.png")
print("\nDone.")
