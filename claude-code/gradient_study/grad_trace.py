"""Backward TRACE method: replace discrete target inference with per-neuron signals.

The timing approach forces combinatorial decisions -- which presynaptic OWNS a downstream
spike, HOW MANY spikes a hidden neuron owes, fire-or-silent -- and every threshold chosen
for them breaks some case (see grad_overfire / grad_overdemand / grad_coincidence
_minimal).  Here credit is continuous and splits automatically.

Each neuron stores two time-varying traces; messages travel backwards only along edges:

  FORWARD, per presynaptic:  eligibility  eps_k(t) = sum_s h(t - s_k)
      the PSP that neuron contributes -- exactly dV_n(t)/dw_kn, computed locally.

  BACKWARD, per neuron:      learning signal  L_n(t)
      a demand on that neuron's sub-threshold voltage over time.
      outputs:  L_o(t*) = th - Vsub_o(t*)   at each target time (push up to threshold)
                L_o(t)  = 0.9*th - Vsub_o(t) at each spurious spike (push down)
      hidden :  L_n(t) = [ sum_d w_nd * sum_t' L_d(t') h(t'-t) ] * g_n(t)
                g_n(t) = exp(-((Vsub_n(t)-th)/(SIG*th))^2)  -- the neuron's own local
                sensitivity: it can only affect downstream where it is near threshold.

  UPDATE:  dw_kn = sum_t L_n(t) * eps_k(t)

Credit to each source is automatically proportional to w_nd*h, so several sources SPLIT
one spike's credit with no ownership rule, nothing can be thresholded to zero (no
absorbing state), and "how many spikes" is never decided -- it emerges from the drive.
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
import grad_unified as U
from grad_infer_relax import infer_relax as _infer_relax

TH = U.TH
KWIN = 400
DELAY_ITERS = int(sim.default_params.delay_iters)
REFRAC_ITERS = int(sim.default_params.refractory_iters)
SHARP_LAT = int(os.environ.get("SHARP_LAT", "1"))            # place the sharp request at the weight-implied latency
SHARP_GAIN = float(os.environ.get("SHARP_GAIN", "1.0"))      # collapse the request to one time (29 -> 34 on the suite)
SHARP_WIN = int(os.environ.get("SHARP_WIN", "30"))           # how near the requested time counts as "at" it
LAST_SHARP = {}   # debug hook: neuron -> (taus chosen, #downstream pools, all candidates)
SHARP_DEBUG = int(os.environ.get("SHARP_DEBUG", "0"))         # populate LAST_SHARP
SHARP_MULTI = int(os.environ.get("SHARP_MULTI", "0"))         # request a SET of times, not one
SHARP_FLOOR = float(os.environ.get("SHARP_FLOOR", "0.3"))     # keep extra taus at this share of the strongest
SUPP_FIX = int(os.environ.get("SUPP_FIX", "1"))                # suppression cannot invert into creation
MOVE_COHERE = int(os.environ.get("MOVE_COHERE", "0"))          # gate the timing term on demand agreement
MOVE_ACT = int(os.environ.get("MOVE_ACT", "0"))                # score the timing term against ACTUAL-reset eps
MOVE_GAIN = float(os.environ.get("MOVE_GAIN", "0.25"))         # signed timing demand at matched output spikes
# 0 -> 58/72, 0.25 -> 60/72, 0.5 -> 60/72, 1.0 -> 51/72.  Too strong and the
# timing term overwhelms the creation hinge; 0.25 keeps 3n A/3n E/4n F at 8/8.
EARLY_STOP = int(os.environ.get("EARLY_STOP", "1"))            # halt on a zero-gradient exact match
OCCL_DEBUG = int(os.environ.get("OCCL_DEBUG", "0"))
OCCL_MASK = int(os.environ.get("OCCL_MASK", "1"))             # request may only ask at times that CAN serve
OCCL_GAIN = float(os.environ.get("OCCL_GAIN", "1.0"))         # route demand out of refractory/epoch occlusion
SHARP_PROTECT = int(os.environ.get("SHARP_PROTECT", "0"))     # never flip a spike driving an ON-TARGET output spike
PROTECT_SHARE = float(os.environ.get("PROTECT_SHARE", "0.2")) # share of the downstream drive that counts as "driving"
PROTECT_TOL = int(os.environ.get("PROTECT_TOL", "3"))         # how close an output spike must be to count as ON TARGET
SHARP_FLIP = int(os.environ.get("SHARP_FLIP", "0"))           # force the SIGN at spikes far from tau: assumes the
                                                              # neuron fires ONCE at tau -- true only of 3n D, and
                                                              # costs chain -3, 3-cycle -1, over-demand -1.  Off.
BLOCK_GAIN = float(os.environ.get("BLOCK_GAIN", "0.0"))      # suppress a spike that blocks a requested one
WREQ_GAIN = float(os.environ.get("WREQ_GAIN", "0.0"))        # direct weight demand from unsatisfiable requests
SHARED_MATCH = int(os.environ.get("SHARED_MATCH", "1"))      # suppression uses the same closest-pair matching
PAIR_MATCH = int(os.environ.get("PAIR_MATCH", "1"))          # requests use closest-pair rather than target-order
EXTEND_FLOOR = float(os.environ.get("EXTEND_FLOOR", "0.5"))   # epoch drive below this share of th is "cannot explain the target"
EPOCH_EXTEND = int(os.environ.get("EPOCH_EXTEND", "1"))       # extend only EMPTY epochs back to the nearest driving spike
EPOCH_ACTUAL = int(os.environ.get("EPOCH_ACTUAL", "0"))       # extend epochs back to the real reset
REFRAC_ACTUAL = int(os.environ.get("REFRAC_ACTUAL", "0"))    # mask on ACTUAL spikes, not target times
EDGE_SIGN = int(os.environ.get("EDGE_SIGN", "1"))            # swap create/suppress along an
# INHIBITORY edge.  R and S are non-negative demand magnitudes combined as
# L = R - SUPP_GAIN*S, and along a negative edge the roles invert: "n must fire MORE"
# means its inhibitory presynaptic must fire LESS, and "n fires spuriously" means that
# presynaptic must fire MORE.  Without this the backward message is exactly inverted on
# every inhibitory edge -- the same defect field_trace had (its EDGE_SIGN fix cut 3n L's
# timing error 6x).
REFRAC_MASK = int(os.environ.get("REFRAC_MASK", "1"))        # drop input landing in a refractory shadow.
# The shadow is (r, r+REFRAC_ITERS) EXCLUSIVE at both ends, i.e. arrivals r+1..r+RF-1.
# jax_spiking_model.py:61 sets the timer to refractory_iters+1 on firing and decrements
# immediately, so it is 22,21,..,1 on steps r..r+21 and reaches 0 at r+RF; and `out` at
# step r itself is gated by the OLD timer, which is still 0.  The old inclusive window
# [r, r+RF] wrongly discarded both ends -- on 3n E that dropped N0@201 (arriving at
# 197+22) so vsub read 0.0000e+00 at a PERFECTLY placed target and fabricated a
# full-deficit hinge, breaking stationarity at the true weights.
SIG = float(os.environ.get("SIG", "0.5"))       # width of the near-threshold sensitivity
LR = float(os.environ.get("LR", "10.0"))         # step size (relative, per synapse)
SWEEPS = int(os.environ.get("SWEEPS", "6"))     # backward relaxation sweeps (handles cycles)
CREATE_FLOOR = float(os.environ.get("CREATE_FLOOR", "0.2"))  # below this share of th there is nothing to amplify
PROX_BAND = float(os.environ.get("PROX_BAND", "1.0"))        # free displacement (rel) before any penalty
PROX = float(os.environ.get("PROX", "0.0"))                # pull-back toward the initial weights
USE_GN = int(os.environ.get("USE_GN", "0"))                  # Gauss-Newton step per neuron
GN_LAM = float(os.environ.get("GN_LAM", "1e-3"))            # Levenberg damping (relative to trace)
GN_ALPHA = float(os.environ.get("GN_ALPHA", "0.5"))         # damping on the GN step
DEEP_SUP = int(os.environ.get("DEEP_SUP", "0"))              # per-neuron targets, as the discrete method has
RESTART_EVERY = int(os.environ.get("RESTART_EVERY", "100"))   # periodic Adam restart (0=off)
KEEP_BEST = int(os.environ.get("KEEP_BEST", "1"))            # retain the best-measured iterate
KEEP_EVERY = int(os.environ.get("KEEP_EVERY", "3"))          # how often to measure it
FREEZE_RUNAWAY = int(os.environ.get("FREEZE_RUNAWAY", "0"))  # pin weights detected as runaways
RUN_WIN = int(os.environ.get("RUN_WIN", "30"))               # window for the runaway test
RUN_THRESH = float(os.environ.get("RUN_THRESH", "0.95"))     # |net|/TV above this = runaway
TRUST = float(os.environ.get("TRUST", "2.0"))                # max predicted spike-time shift per step (0=off)
GLOBAL_NORM = int(os.environ.get("GLOBAL_NORM", "0"))        # 1 = keep relative gradient scale across weights
SLOPE_FLOOR = float(os.environ.get("SLOPE_FLOOR", "0.002"))   # floor on dV/dt when converting timing->voltage
USE_REQ_LAT = int(os.environ.get("USE_REQ_LAT", "0"))        # 1 = per-request latency from traces
EARLY_TOL = float(os.environ.get("EARLY_TOL", "2"))          # earliness tolerated before suppressing
EARLY_GAIN = float(os.environ.get("EARLY_GAIN", "0.0"))      # strength of early-spike suppression
REQ_SELFNORM = int(os.environ.get("REQ_SELFNORM", "1"))      # rescale each hop by its ACTUAL gain
REQ_PEAKNORM = int(os.environ.get("REQ_PEAKNORM", "1"))      # peak-normalise (compensates per-hop decay)
R_CAP = int(os.environ.get("R_CAP", "0"))                    # bound the request magnitude to the seed
REQ_GAIN = float(os.environ.get("REQ_GAIN", "3.0"))          # strength of the creation request per hop
LOOP_CAP = int(os.environ.get("LOOP_CAP", "0"))              # bound backward loop gain to <= 1
PIVOT_GAIN = float(os.environ.get("PIVOT_GAIN", "0.0"))      # suppress spikes pivotal for EARLY downstream spikes
ABLATE_GAIN = float(os.environ.get("ABLATE_GAIN", "0.0"))    # suppress spikes whose removal improves timing
SUPP_TOL = float(os.environ.get("SUPP_TOL", "30"))           # distance beyond which a spike is unwanted
SUPP_GAIN = float(os.environ.get("SUPP_GAIN", "0.0"))        # strength of upstream suppression
DEAD_ZONE = float(os.environ.get("DEAD_ZONE", "5"))          # error tolerated before any response
GRADE_SUPP = int(os.environ.get("GRADE_SUPP", "0"))          # graded vs binary late-spike suppression
GRADE_REQ = int(os.environ.get("GRADE_REQ", "0"))            # graded vs binary creation-request seeding
OCCL_FROMDEMAND = int(os.environ.get("OCCL_FROMDEMAND", "0"))  # window from downstream L when R is empty
FIELD_XING = float(os.environ.get("FIELD_XING", "0.0"))        # phase-referenced demand at implied_w crossings
FIELD_SMOOTH = float(os.environ.get("FIELD_SMOOTH", "0.0"))    # Gaussian sigma smoothing the urgency density
FIELD_POW = float(os.environ.get("FIELD_POW", "1.0"))          # concentration of the positive field
FIELD_REACH = float(os.environ.get("FIELD_REACH", "0.0"))      # weight by how close n is to firing
FIELD_SWEEPS = int(os.environ.get("FIELD_SWEEPS", "3"))        # iterations so demand can travel a loop
OUT_FIELD = int(os.environ.get("OUT_FIELD", "1"))              # an output that feeds other neurons gets a field too
TIM_REFRAC = float(os.environ.get("TIM_REFRAC", "0.0"))        # discarded spikes get a push, not a timing nudge
FIELD_REFRAC = int(os.environ.get("FIELD_REFRAC", "1"))        # a discarded arrival earns no credit
FIELD_SUPP = float(os.environ.get("FIELD_SUPP", "0.25"))       # suppression depth, share of peak
FIELD_WIN = int(os.environ.get("FIELD_WIN", "20"))             # half-width of the suppressed region
FIELD_TOL = float(os.environ.get("FIELD_TOL", "1.0"))          # weight mismatch, in units of w, that kills urgency
XING_DEMAND = float(os.environ.get("XING_DEMAND", "1.0"))      # share of FIELD_XING that puts demand AT crossings
XING_MOVE = float(os.environ.get("XING_MOVE", "1.0"))          # share that pushes existing spikes toward a crossing
FIELD_SUPP_LOCAL = float(os.environ.get("FIELD_SUPP_LOCAL", "1.0"))  # depth of the local suppression
FIELD_LOCAL = float(os.environ.get("FIELD_LOCAL", "0.0"))      # build L[n] from n's OWN field alone
FIELD_MIST = int(os.environ.get("FIELD_MIST", "1"))            # grade jobs by the timing error of a MATCHED spike
FIELD_BASE = float(os.environ.get("FIELD_BASE", "0.0"))        # constant suppression on every existing spike
FIELD_PROP = int(os.environ.get("FIELD_PROP", "1"))            # 1 = propagate via urgency peaks, 0 = via crossings
FIELD_GAP = int(os.environ.get("FIELD_GAP", "1"))              # gap that separates two urgency bumps
FIELD_OCCL = int(os.environ.get("FIELD_OCCL", "1"))            # candidate whose arrival is eaten earns no field
FIELD_PHYS = int(os.environ.get("FIELD_PHYS", "1"))            # field reads PHYSICAL resets, not target epochs
FIELD_HARD = int(os.environ.get("FIELD_HARD", "0"))            # clip urgency to 0 past the tolerance (old)
XING_NOVEL = int(os.environ.get("XING_NOVEL", "0"))            # drop crossings this close to an existing spike
FIELD_NEG = int(os.environ.get("FIELD_NEG", "1"))              # unwanted downstream spikes get the mirrored field
FIELD_FLAT = int(os.environ.get("FIELD_FLAT", "1"))            # keep the flat window for spikes with no field at all
BETA1 = float(os.environ.get("BETA1", "0.9"))                  # Adam momentum; higher filters zigzag
KICK_DEAD = float(os.environ.get("KICK_DEAD", "0.0"))          # superseded by CREATE; kept for comparison          # per-neuron kick for a SILENT hidden neuron
KICK_STALL = float(os.environ.get("KICK_STALL", "2.0"))        # kick when the weights stop moving
KICK_WIN = int(os.environ.get("KICK_WIN", "300"))              # window for the stall test
KICK_FEW = float(os.environ.get("KICK_FEW", "0.0"))            # also kick when the output has too few spikes
KICK_GAIN = float(os.environ.get("KICK_GAIN", "10.0"))       # escape a frozen-and-wrong state
ORACLE_T = {}   # *** CHEATING *** diagnostic only: true hidden spike times
FIELD = int(os.environ.get("FIELD", "0"))                      # signed should-fire-here field
HID_TARGETS = int(os.environ.get("HID_TARGETS", "0"))          # infer hidden spike times explicitly
NEW_DEMAND = int(os.environ.get("NEW_DEMAND", "0"))            # direct feasible+reachable demand density
NEW_GAIN = float(os.environ.get("NEW_GAIN", "1.0"))
FIELD_ADD = float(os.environ.get("FIELD_ADD", "0.0"))          # add the URGENCY DENSITY to the working demand
LN_RELOC = int(os.environ.get("LN_RELOC", "0"))                # retime the rejected propagated demand
OCCL_RELOC = int(os.environ.get("OCCL_RELOC", "0"))            # move an infeasible request to the nearest feasible time
BARRIER_CLAMP = int(os.environ.get("BARRIER_CLAMP", "0"))     # never step a weight down through w_crit
GRADE_PROP = int(os.environ.get("GRADE_PROP", "0"))          # graded vs hard upstream propagation gate
GRADE_SCALE = float(os.environ.get("GRADE_SCALE", "30"))     # steps of lateness for full suppression/request
HIT_TOL = int(os.environ.get("HIT_TOL", "60"))                # a target counts as achieved only within this
DECAY = float(os.environ.get("DECAY", "0.05"))               # step decay per iteration
NOM_LAT = int(os.environ.get("NOM_LAT", "71"))               # nominal edge latency for propagating a request
FIELD_SLOPE = float(os.environ.get("FIELD_SLOPE", "0.0"))      # timing from the creation field's own slope
TIM_GAIN = float(os.environ.get("TIM_GAIN", "1.0"))            # weight of the backward TIMING (pull-to-nearest) term
CREATE = float(os.environ.get("CREATE", "0.3"))  # creation demand weight in the backward
# relaxation.  This is the ONLY path by which a demand reaches a neuron that has not spiked:
#     vol = sum_d w[n->d] * back_corr(L[d], HK);  Ln = vol * near_threshold_gate * CREATE
# None of it needs n's own spikes, unlike the timing term, which is attached AT them.  With
# CREATE=0 (the old default) a silent neuron got exactly zero demand and zero gradient on
# its inputs, so nothing in a chain could ever start firing -- measured on 8n M, vol*gate
# was a healthy 2.06e-06 and then multiplied by zero.  At 0.3 the silent N3/N4 get
# correctly-signed gradients (+1.4e-07, +2.0e-07) and the suite goes 76 -> 80/104, which
# also makes KICK_DEAD redundant (80 either way).  Larger is much worse: 1.0 -> 73,
# 10.0 -> 57, the per-hop amplification this project has hit repeatedly.
# The creation term correlates against the all-positive kernel, so it is a PERSISTENT
# upward push on every hidden weight whenever downstream demand has positive mass -- it
# never balances, and hidden weights diverge (chain: 568->1345 vs true 500).  The signed
# timing term alone is self-limiting because h' changes sign either side of the peak.
MATCH_WIN = int(os.environ.get("MATCH_WIN", "60"))  # a spike within this of a target is that target's, just mistimed
HK = np.array([U.hk(dt) for dt in range(KWIN)])
W_CRIT = TH / float(HK.max())   # single-spike firing weight (444.5): below it a neuron
#                                 must ACCUMULATE, so crossing changes the spike COUNT
# step-nudged kernel: aim the crossing at the CENTRE of the step containing t*, not its
# edge, so the discrete spike lands on t* instead of a step early/late (same fix as
# solve_vsub's hk_nud -- without it the fit converges to ~495-505 and misses by 1-2 steps)
HKN = np.array([0.5 * (U.hk(dt - 1) + U.hk(dt)) for dt in range(KWIN)])
# kernel DERIVATIVE: gives the SIGNED timing demand.  Moving n's spike from s to s-delta
# changes V_d(t) by w_nd*h'(t-s)*delta, so sum_t L_d(t) h'(t-s) is positive exactly when
# the spike should fire EARLIER.  Correlating against h itself (as a plain voltage demand
# does) carries magnitude only and cannot say WHICH WAY in time the spike must move.
HKP = np.array([U.hk(dt) - U.hk(dt - 1) for dt in range(KWIN)])


def mkparams(steps): return dataclasses.replace(sim.default_params, steps=steps)


def fsim(C, N, w, params):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, jnp.array(C), N, jnp.array([0])))


def sp(V, n): return np.where(V[:, n] >= TH)[0].tolist()


def eligibility(spikes, T, resets=(), refrac_at=None):
    """eps(t) = sum over presynaptic spikes s SINCE THE LAST RESET of h(t-s).

    Resets are the postsynaptic neuron's TARGET times (its own spikes if it has no
    target).  Without them the PSP sum accumulates over the whole run and the demand
    stops being monotonic in w: a neuron driven too hard fires EARLY, has already reset
    by the target time, and a raw voltage reading there says "push up" when the spike
    actually needs delaying.  Resetting the accumulation at each target makes the
    quantity monotonic in w, so the demand is signed correctly in BOTH directions
    (this is the V_sub construction from solve_vsub)."""
    e = np.zeros(T)
    R = sorted(resets)
    for s in spikes:
        if REFRAC_MASK and R:
            # INPUT ARRIVING DURING REFRACTORY IS DISCARDED BY THE SIMULATOR:
            #   out = out * (refractory_timers == 0)        -- voltage held at zero
            #   rise_values *= (refractory_timers != 1)     -- accumulator wiped at the end
            # so a presynaptic spike whose ARRIVAL (s + delay) lands inside [r, r+refr]
            # after a reset r contributes nothing at all.  Ignoring this made the
            # reconstruction 42% too high on 3n A -- vsub(253)=9.92e-03 against a real
            # V(253)=6.97e-03 -- which fabricated a -2.9e-03 "firing too early" demand at
            # weights where the output was already EXACTLY right, so truth was not a
            # stationary point and the fit was driven away from it.
            # Mask against the times the neuron ACTUALLY fired, not the epoch resets.
            # The epochs are the TARGET times (that is what makes the objective monotone),
            # but a neuron is only refractory after a spike it really produced.  At the
            # solution the two coincide; during training they do not, and masking on
            # targets drops input the real neuron does integrate.  Measured: masking on
            # targets FIXES 3n A (0/8 -> 7/8) but BREAKS 3n D (4/8 -> 1/8).
            RF = R if refrac_at is None else refrac_at
            arr = s + DELAY_ITERS
            if any(r < arr < r + REFRAC_ITERS for r in RF):   # (r, r+RF) -- see note
                continue
        # epoch is (previous reset, reset] -- INCLUSIVE of the reset time, since the
        # demand is evaluated exactly AT the target and must see the spikes preceding it
        nxt = next((r for r in R if r > s), T)
        hi = min(s + KWIN, nxt + 1, T)
        if hi > s:
            # Use the TRUE kernel, not the nudged one.  HKN = (h(dt-1)+h(dt))/2 averages a
            # RISING kernel and so biases vsub ~0.4% LOW -- at the true weights that put
            # vsub(t*) = 6.9837e-03 under th = 7.0e-03 while the simulation was actually at
            # 7.0100e-03, so the hinge fired forever and the gradient was never zero at the
            # solution.  Discretisation is now handled by the hinge interval instead, which
            # is what it is for; the nudge was both redundant and biasing.
            e[s:hi] += HK[:hi - s]
    return e


_PEAK_IDX = int(np.argmax(HK))
_HK_RISE = HK[:_PEAK_IDX + 1]          # monotonically increasing, so searchsortable


def request_lat_vec(w_kn, deficits):
    """Vectorised request_lat: smallest dt with w*h(dt) >= deficit, for MANY deficits.

    The scalar version scans up to KWIN=400 steps per call, and the sharpening needs one
    call per nonzero request entry (~275) per edge per sweep per round -- which made the
    accurate version intractably slow, forcing a peak-only approximation that picked worse
    times and lost the benefit.  HK's rising phase is monotonic, so the same query is a
    binary search, and it vectorises over all deficits at once.
    Returns -1 where even the kernel peak cannot cover the deficit.
    """
    if w_kn <= 0:
        return np.full(len(deficits), -1, dtype=int)
    v = np.asarray(deficits, dtype=float) / w_kn
    dt = np.searchsorted(_HK_RISE, v, side="left")
    return np.where(dt > _PEAK_IDX, -1, dt)


def request_lat(w_kn, deficit):
    """Latency the REQUEST itself carries, from the traces -- no fixed constant.

    If n is short by `deficit` at the requested time tau, presynaptic k must supply
    w_kn*h(dt) = deficit.  Taking the RISING branch (smallest such dt) means k's
    contribution is BELOW the deficit at every earlier time, so n first reaches threshold
    exactly at tau instead of before it.  The answer depends on the edge's own weight --
    a strong edge needs a short dt, a weak one a long dt -- which is why a single constant
    (NOM_LAT) dragged spikes to the wrong time: true N1->N2 latency here is 47, not 71.
    Returns None when even the kernel peak cannot cover the deficit (edge too weak).
    """
    if deficit <= 0:
        return 1
    for dt in range(1, KWIN):
        if w_kn * HK[dt] >= deficit:
            return dt
    return None


def back_corr(Ld, K=HK):
    """(L_d correlated with K)(s) = sum_dt L_d(s+dt) K(dt) -- the backward message.
    K=HK gives a voltage/creation demand; K=HKP gives the SIGNED timing demand."""
    # vectorised: m[s] = sum_dt Ld[s+dt] K[dt] is a correlation, so it equals
    # convolve(Ld, reversed K) offset by KWIN-1.  The explicit 400-step loop was the
    # bottleneck at scale (called per edge per sweep: 0.86 s/round on the 50-neuron nets).
    T = len(Ld)
    return np.convolve(Ld, K[::-1], mode="full")[KWIN - 1:KWIN - 1 + T]


def _cross(v, th):
    """times where a trace crosses th upward (a cheap stand-in for re-simulating)"""
    a = v >= th
    return np.nonzero(a[1:] & ~a[:-1])[0] + 1


def _errdrop(base_t, abl_t, tgt):
    """how much closer the crossings get to the targets when a spike is ablated"""
    if len(tgt) == 0:
        return 0.0
    def err(ts):
        if len(ts) == 0:
            return float(len(tgt))
        return float(np.mean([min(abs(int(x) - q) for q in ts) for x in tgt]))
    return err(base_t) - err(abl_t)



def demand_direct(C, N, w, spall, T, out_targets, vsub, L, inc):
    """DIRECT DEMAND: one construction replacing seed + propagate + mask + sharpen + retime.

    Those five stages exist because each patches the previous one's failure, which is why
    they are conjunctive (REQ+SHARP+OCCL all three -> 8/8 on 4n G, every pair <= 2/8) and
    why the whole thing needs ~10 interacting gains.  But the measurements say a demand is
    determined by exactly two computable quantities:

      FEASIBLE   a spike at q can serve a demand at t on d only if its arrival lands in the
                 SAME EPOCH as t and clear of d's refractory shadows:
                     q + DELAY in ( last_reset_of_d_before(t), t ]  minus shadows
      REACHABLE  n must have drive at q for a weight increase to amplify: vsub_n(q) > 0.
                 (Feasibility alone sends the demand where eps is zero -- measured: it
                 landed at 437, past N2's reset at 376, and did nothing.)

    So place the demand as a DENSITY over the feasible-and-reachable times, weighted by how
    much a spike at q would actually deliver at t (w * h(t-q)) times how close n already is
    to firing there.  No seeding constant, no per-hop normalisation, no rejection step and
    nothing to retime -- an infeasible time simply gets zero mass instead of being placed
    and then removed.
    """
    Dm = {n: np.zeros(T) for n in range(N)}
    for n in range(N - 1, -1, -1):
        if n in out_targets or len(inc[n]) == 0:
            continue
        for si in np.where(C[:, 0] == n)[0]:
            d = int(C[si, 1])
            dem = L[d] if d in out_targets else Dm[d]
            pos = np.nonzero(dem > 0)[0]
            if len(pos) == 0:
                continue
            rd = sorted(out_targets[d]) if d in out_targets else sorted(spall[d])
            for t in pos:
                t = int(t)
                lo = max([r for r in rd if r < t], default=-1)
                if t < lo + 1:
                    continue
                arr = np.arange(lo + 1, t + 1)
                for r in rd:                       # refractory shadow is (r, r+RF)
                    arr = arr[~((arr > r) & (arr < r + REFRAC_ITERS))]
                if len(arr) == 0:
                    continue
                q = arr - DELAY_ITERS
                q = q[(q >= 0) & (q < T) & (t - (arr - DELAY_ITERS) < KWIN)]
                if len(q) == 0:
                    continue
                deliver = float(w[si]) * HK[t - q]          # what a spike at q gives at t
                reach = np.clip(vsub[n][q] / TH, 0.0, 1.0)  # can n be recruited at q
                dens = deliver * reach
                mx = float(dens.max())
                if mx <= 0:
                    continue
                Dm[n][q] += float(dem[t]) * (dens / mx)
        # MAGNITUDE-PRESERVING HOP.  Each downstream demand time contributes a full-height
        # density, so a neuron serving many of them accumulates their sum and the signal
        # grows with depth: measured g(0->1) = 1.05e-03 against g(1->2) = 8.47e-06, a 124x
        # imbalance two hops from the output, which drives the upstream weight wildly.
        # Rescale each neuron's demand to the largest downstream demand that produced it,
        # so a hop carries direction and shape without gain.  This is the one thing the old
        # path's REQ_SELFNORM was for, and it is the only normalisation this needs.
        mxn = float(Dm[n].max())
        if mxn > 0:
            src = max((float((L[int(C[si, 1])] if int(C[si, 1]) in out_targets
                              else Dm[int(C[si, 1])]).max())
                       for si in np.where(C[:, 0] == n)[0]), default=0.0)
            if src > 0:
                Dm[n] = Dm[n] * (src / mxn)
    return Dm






def field_crossings(C, N, w, urgency, implied, T):
    """Where should each hidden neuron fire, GIVEN the weights it currently has?

    implied_w(q) is the outgoing weight that would make a spike at q useful.  It is monotone
    within each candidate window (every window sits on HK's rising flank, dt <= 96 < peak
    110), so implied_w crosses the neuron's ACTUAL outgoing weight at most once per window.
    That crossing is a PHASE REFERENCE: fire there and the spike is consistent with the
    weight you already have.  Every other timing signal in this method is a scalar
    earlier/later with no anchor; this one names a time.

    Returns, per hidden neuron, the crossing times and the urgency at each.
    """
    out = {}
    for n in range(N):
        iw, ug = implied[n], urgency[n]
        if not np.isfinite(iw).any():
            continue
        outs_n = np.where(C[:, 0] == n)[0]
        if len(outs_n) == 0:
            continue
        w_now = float(np.mean([abs(w[si]) for si in outs_n]))
        d = iw - w_now
        fin = np.isfinite(d)
        xs = []
        idx = np.nonzero(fin)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:          # window boundary: not a crossing, just a gap
                continue
            if d[a] == 0 or (d[a] < 0) != (d[b] < 0):
                q = a if abs(d[a]) <= abs(d[b]) else b
                # do NOT require urgency>0: near the solution the deficit vanishes but the
                # crossing is still the correct phase reference
                xs.append((int(q), float(max(ug[q], 0.0))))
        if xs:
            out[n] = xs
    return out


def local_demand(F, spikes, T):
    """n's learning signal from n's OWN field and n's OWN spikes -- nothing downstream.

    demand_field already used the downstream to BUILD F[n]; once it exists, "where should
    this neuron's weights move" is a local question.  F's positive bumps are the spikes it
    wants and their peaks are when it wants them, so the whole decision is a comparison of
    two spike trains on the same neuron:

        bump with no spike   ->  CREATE    positive at the bump
        spike with no bump   ->  SUPPRESS  negative at the spike
        bump paired to spike ->  MOVE      positive at the bump, negative at the spike,
                                           both scaled by the error, so a correct pairing
                                           contributes nothing

    Pairing is greedy-nearest and ONE-TO-ONE, with no distance cutoff: a spike is only
    surplus once every bump has been claimed.  That is the part MATCH_WIN got wrong --
    4n F's N1 had ONE spike against FIVE demands and the spike was 72 steps from the
    nearest, so a 60-step cutoff called it "unwanted outright" and asked to delete the
    neuron's only spike.  With nothing to delete, a far-off spike is mistimed, not surplus.
    """
    L = np.zeros(T)
    pos = np.nonzero(F > 0)[0]
    bumps = []
    if len(pos):
        for r in np.split(pos, np.nonzero(np.diff(pos) > FIELD_GAP)[0] + 1):
            if len(r):
                q = int(r[np.argmax(F[r])])
                bumps.append((q, float(F[q])))
    sp = [int(q) for q in spikes if 0 <= q < T]
    free_b = list(range(len(bumps)))
    pairs, used = [], set()
    # nearest pairs first, so a confident match is not stolen by a distant one
    cand = sorted(((abs(bumps[i][0] - f), i, f) for i in free_b for f in sp),
                  key=lambda x: x[0])
    claimed_b = set()
    for _d, i, f in cand:
        if i in claimed_b or f in used:
            continue
        claimed_b.add(i); used.add(f); pairs.append((i, f))
    for i, (q, u) in enumerate(bumps):
        if i not in claimed_b:
            L[q] += u                                   # wanted, absent -> create
    for f in sp:
        if f not in used:
            L[f] -= FIELD_SUPP_LOCAL * TH               # present, unwanted -> suppress
    for i, f in pairs:
        q, u = bumps[i]
        err = abs(q - f)
        if err <= DEAD_ZONE:
            continue                                    # close enough: contribute nothing
        amt = min(1.0, (err - DEAD_ZONE) / max(GRADE_SCALE, 1e-9))
        L[q] += amt * u                                 # pull the spike toward the bump
        if f < q:
            # ONLY AN EARLY SPIKE GETS THE NEGATIVE.  Suppression removes drive, which moves
            # a spike LATER: right for an early one, actively wrong for a late one.  A late
            # spike is corrected by the positive at the bump alone, which is EARLIER than it.
            # Applying it to both directions inverted the gradient outright: eligibility
            # grows through an epoch, so on 4n F's N1 (one spike at 442, bumps at
            # 73/173/220/273/370) the lone negative at 442 multiplied a much larger eps than
            # all five positives combined, giving g(w0) = -1.7e-05 when the weight needed to
            # RISE from 156 toward 240.
            L[f] -= amt * FIELD_SUPP_LOCAL * TH
    return L


def demand_field(C, N, w, spall, T, out_targets, vsub, eps):
    """A signed "should you fire here" FIELD per hidden neuron, instead of a target train.

    Discrete targets force the count decision, and the count is what the inference gets
    wrong ~4 times in 5 -- a train with the wrong number of spikes gives bad directions
    however well each time is placed (measured: exact latency placement left sign accuracy
    at 71.6/68.4/80.0, unchanged).  A field never decides a count.

    POSITIVE region, per unmet downstream target t: a spike at q works iff the outgoing
    weight lies in an interval, and the WIDTH of that interval is exactly "how many weights
    would make this work" --
        w_min = (TH - other(t)) / h(t-q)                     enough to cross AT t
        w_max = min over tau<t of (TH - other(tau)) / h(tau-q)   not so much it fires EARLY
    Empty interval => q cannot serve t at any weight => no mass there.  Wide interval => q
    is a robust place to fire => more mass.  `other` is the co-drivers' profile with n's own
    contribution removed (using vsub directly is self-referential).

    NEGATIVE region, per spurious spike: a spike one step later is still spurious, so
    suppression spreads over a region rather than a point.
    """
    # PHYSICAL RESETS FOR THE FIELD.  vsub/eps reset an OUTPUT at its TARGET times -- a
    # counterfactual that is load-bearing for the creation hinge but wrong here.  The field
    # asks a forward question: "what weight would make a spike at q push d over threshold at
    # t?"  What decides that is the drive accumulated since d ACTUALLY last reset, not since
    # a target it may not have hit.  Worse, EPOCH_EXTEND moves a shared boundary: on 4n F it
    # pulled the boundary at target 233 back to 204 in order to widen the starved epoch for
    # 290, so 233 -- a target N3 HITS, whose own epoch carried 1.018*TH of drive -- fell out
    # of the boundary list entirely and read a full deficit.  That phantom deficit laid a
    # whole spurious urgency bump over q=116..215.  Hidden neurons already reset physically
    # (rst[n] = spall[n]), so only outputs need this.
    _phys = {}
    if FIELD_PHYS:
        for _d in out_targets:
            for _k in range(N):
                _phys[(_k, _d)] = eligibility(spall[_k], T, spall[_d], refrac_at=spall[_d])
    _demand_t = {}                                  # times each neuron should fire at
    _sweeps = max(1, FIELD_SWEEPS)
    F = {n: np.zeros(T) for n in range(N)}          # URGENCY
    WSUM = {n: np.zeros(T) for n in range(N)}       # urgency * implied_w
    WCNT = {n: np.zeros(T) for n in range(N)}       # urgency, for the mean
    for _sw in range(_sweeps):
        # a fresh field each sweep, but _demand_t persists, so on sweep 2 an
        # output can see the demand of the neuron it feeds -- the loop closes
        for _k in range(N):
            F[_k][:] = 0.0; WSUM[_k][:] = 0.0; WCNT[_k][:] = 0.0
        for n in range(N - 1, 0, -1):
            if n in out_targets:
                if not OUT_FIELD:
                    continue
                # seed straight from the KNOWN targets, in the same volt units as a hidden
                # neuron's urgency: how much drive is missing at each target it is not hitting,
                # and a full negative at any spike far from every target.
                for tq in sorted(out_targets[n]):
                    if not (0 <= tq < T):
                        continue
                    if any(abs(tq - q) <= 2 for q in spall[n]):
                        continue                      # already hit
                    F[n][tq] += max(TH - float(vsub[n][tq]), 0.0)
                for q in spall[n]:
                    if 0 <= q < T and all(abs(tq - q) > MATCH_WIN for tq in out_targets[n]):
                        F[n][q] -= TH
                # fall through: an output that FEEDS other neurons must also collect their
                # demand, which is what closes a loop like N1->N2->N3->N1
            for si in np.where(C[:, 0] == n)[0]:
                d = int(C[si, 1])
                # BACKWARD PROPAGATION.  An output supplies its target times directly.  A HIDDEN
                # downstream has none -- what it has is its own field, already built because the
                # loop runs downstream-first.  Use ITS CROSSINGS as the demand times: those are
                # where d itself should fire, so they are exactly what n must serve.  This is
                # the recursion that lets the field reach neurons more than one hop from an
                # output (chain's N1 and 4n G's N1 previously got NO field at all, while the
                # neurons feeding an output directly landed on their true times exactly).
                if d in out_targets:
                    tg = sorted(out_targets[d])
                    rd = tg
                else:
                    tg = sorted(_demand_t.get(d, []))
                    rd = sorted(spall[d])           # a hidden neuron resets at its OWN spikes
                    if not tg:
                        continue
                if FIELD_PHYS and d in out_targets:
                    _vd = sum(float(w[_si]) * _phys[(int(C[_si, 0]), d)]
                              for _si in np.where(C[:, 1] == d)[0])
                    other = _vd - float(w[si]) * _phys[(n, d)]
                else:
                    other = vsub[d] - float(w[si]) * eps[(n, d)]
                wnow = abs(float(w[si]))
                # TWO KINDS OF JOB, ONE CONSTRUCTION.  A demand time of d is a place n should
                # help it cross (+); a spike of d that no demand time claims is a place n
                # should stop helping it cross (-).  Both ask the same question of a candidate
                # q -- "what outgoing weight would make a spike at q put d over threshold at
                # t?" -- so they run through identical code and differ only in the sign of the
                # mass laid down.  Suppression built this way is no longer a flat stamp on
                # every spike with no field nearby: it is strongest exactly where the CURRENT
                # weight is what produces the unwanted spike, which is the case where changing
                # this weight actually removes it.
                # A MATCHED-BUT-MISTIMED SPIKE IS NEITHER FULLY SERVED NOR FULLY MISSING.
                # With physical resets a spike 10 steps EARLY resets the accumulation at the
                # early time, so by the target only 10 steps of drive have built up and
                # other[t] reads 0.000*TH -- a FULL deficit, identical to never having fired.
                # Measured on 4n F with N3 at [33,123,223,323,423] against targets
                # [33,133,233,333,433]: N2's field had FIVE full-strength positive bumps
                # (requests at 100/200/257/300/400) and ZERO negative points, so the four
                # spurious requests from 10-step errors were indistinguishable from the one
                # real request for the genuinely missing target at 290.
                # Grade both sides by the error instead, so perfect timing extinguishes both:
                #   positive at t   scaled by |f - t| / GRADE_SCALE   (served => no request)
                #   negative at f   scaled by (t - f) / GRADE_SCALE   for an EARLY spike only
                # Only EARLY spikes get the negative: suppressing drive moves a spike LATER,
                # which fixes an early one and worsens a late one.  A late spike is already
                # handled by the positive job at its target.
                _free = list(tg); _claim = {}
                for _f in sorted(spall[d]):
                    if not _free:
                        _claim[int(_f)] = None; continue
                    _tt = min(_free, key=lambda x: abs(x - _f))
                    if abs(_tt - _f) > MATCH_WIN:
                        _claim[int(_f)] = None; continue
                    _free.remove(_tt); _claim[int(_f)] = int(_tt)
                _served = {tt: f for f, tt in _claim.items() if tt is not None}
                jobs = []
                for t in tg:
                    _jw = 1.0
                    if FIELD_MIST and int(t) in _served:
                        _jw = min(1.0, abs(_served[int(t)] - int(t))
                                  / max(GRADE_SCALE, 1e-9))
                    if _jw > 0:
                        jobs.append((int(t), 1.0, _jw))
                if FIELD_NEG:
                    for _f in spall[d]:
                        if not (0 <= _f < T):
                            continue
                        _tt = _claim.get(int(_f))
                        if _tt is None:
                            jobs.append((int(_f), -1.0, 1.0))      # unwanted outright
                        elif FIELD_MIST and _f < _tt:              # matched but EARLY
                            _g = min(1.0, (_tt - int(_f)) / max(GRADE_SCALE, 1e-9))
                            if _g > 0:
                                jobs.append((int(_f), -1.0, _g))
                for t, sgn, jw in jobs:
                    if not (0 <= t < T):
                        continue
                    # NOTE: do NOT skip when other[t] >= TH.  implied_w is "what weight would be
                    # CONSISTENT with this spike serving this target", which is defined whether
                    # or not there is a shortfall -- and `other` already excludes this edge's own
                    # contribution, so at the TRUE weights need = w*h(dt) and implied_w = w
                    # exactly.  Gating it on a deficit made it evaporate at convergence (3n D,
                    # chain and 4n G all had NO crossings at the true weights), which is
                    # precisely where a phase reference is needed.  URGENCY still requires a
                    # deficit; only implied_w is computed unconditionally.
                    # ... and for a NEGATIVE job, other[t] >= TH means d crosses at t whether
                    # or not n helps, so n is not to blame and must not be pushed for it.
                    _has_deficit = other[t] < TH
                    _base = tg if sgn > 0 else rd
                    lo = max([r for r in _base if r < t], default=-1)
                    qs = np.arange(max(0, lo + 1 - DELAY_ITERS), t - DELAY_ITERS + 1)
                    if len(qs) == 0:
                        continue
                    tau = np.arange(lo + 1, t)        # times that must NOT cross
                    need = TH - other[t]
                    # ALL CANDIDATE q AT ONCE.  Every quantity below is a pure function of q
                    # over a contiguous range, and the wmax test is a min over tau -- i.e. a
                    # (q, tau) reduction.  Done as a Python loop with a numpy call per q this
                    # was 56x the cost of traces() (3-cycle 336 ms, 8n M 1914 ms per call),
                    # which made the field-on suite unaffordable.  Same arithmetic, one pass.
                    dts = t - qs
                    okq = (dts >= 0) & (dts < KWIN)
                    # take(mode="clip") gathers and bounds in one C pass; clip-then-index
                    # was 10% of the whole function on 8n M, all of it in temporaries.
                    hk = np.where(okq, np.take(HK, dts, mode="clip"), 0.0)
                    okq &= hk > 0
                    if not okq.any():
                        continue
                    wmin_a = np.where(okq, need / np.where(hk > 0, hk, 1.0), np.inf)
                    if len(tau):
                        # (Q, TAU): the weight at which q would ALREADY have crossed at tau.
                        # wmax is the smallest such -- fire any harder and the spike comes
                        # early.  Rows are q, columns the times that must not cross.
                        d2 = tau[None, :] - qs[:, None]
                        oth_t = other[tau]
                        m = (d2 >= 0) & (d2 < KWIN) & (oth_t < TH)[None, :]
                        hh = np.where(m, np.take(HK, d2, mode="clip"), 0.0)
                        m &= hh > 0
                        wmax_a = np.where(m, (TH - oth_t)[None, :] / np.where(hh > 0, hh, 1.0),
                                          np.inf).min(axis=1)
                    else:
                        wmax_a = np.full(qs.shape, np.inf)
                    feas = okq & (wmax_a > wmin_a)     # some weight can place a spike at q
                    # AN ARRIVAL THAT GETS EATEN EARNS NO FIELD.  If q's arrival lands in the
                    # refractory shadow of a spike d ACTUALLY produced, the simulator throws
                    # the contribution away, so firing at q cannot help d cross at t at ANY
                    # weight.  Without this the field's own maximum sat on such a time: on
                    # 4n F, target 290 admitted q=216..272 and the peak was at q=216, whose
                    # arrival at 234 falls inside N3@233's shadow (233,255) -- 21 of the 57
                    # candidates were dead, all at the front of the window, which is exactly
                    # where the score concentrates.  Masking them leaves q=237..272, and the
                    # true N2 spike (256) is inside that.
                    if FIELD_OCCL and spall[d]:
                        arr = qs + DELAY_ITERS
                        occ = np.zeros(qs.shape, bool)
                        for r in spall[d]:
                            occ |= (arr > r) & (arr < r + REFRAC_ITERS)
                        feas &= ~occ
                    if not feas.any():
                        continue
                    # SCORE q BY THE WEIGHT IT WOULD TAKE, AGAINST THE WEIGHT WE HAVE.
                    # wmin is "the outgoing weight at which a spike at q makes d cross at t".
                    # The best q is the one where wmin is the weight ALREADY on the edge: at
                    # the current weight, firing there lands the downstream spike on time, so
                    # no weight change is asked for as the price of the timing.
                    #
                    # THE FALLOFF MUST NEVER REACH ZERO.  A hard max(0, 1 - mis) clip made
                    # urgency vanish outright once the weight was more than 2x off, which is
                    # a category error: it collapses the two field variables back into one.
                    # Urgency answers "do we want a spike here", implied_w answers "and what
                    # would the weight have to be" -- the second is not licensed to silence
                    # the first.  Measured on 4n F seed 2 at its stuck weights, w(N2->N3) had
                    # collapsed to 25 while the cheapest workable weight was 445 (mis = 16.8),
                    # so BOTH hidden neurons had an identically zero field: no urgency, no
                    # implied_w, no crossings, at exactly the moment the signal was needed.
                    # And it is self-reinforcing -- once an edge drifts out of band the field
                    # goes silent for it, so nothing can pull it back.
                    #
                    # 1/(1 + mis/TOL) decays but stays positive, and it also unifies the two
                    # earlier scores instead of choosing between them.  When w is far BELOW
                    # any workable weight the ranking is dominated by wmin being smallest,
                    # i.e. by max drive -- the kernel peak, which is the right answer when
                    # the weight is too weak.  As w approaches a workable value the maximum
                    # slides onto the q where wmin == w, which is the phase-correct answer.
                    # FIELD_HARD=1 restores the clip for comparison.
                    mis = np.abs(wmin_a - wnow) / max(wnow, 1e-9)
                    if FIELD_HARD:
                        sc = np.maximum(0.0, 1.0 - mis / max(FIELD_TOL, 1e-9))
                    else:
                        sc = 1.0 / (1.0 + mis / max(FIELD_TOL, 1e-9))
                    reach = np.clip(vsub[n][qs] / TH, 0.0, 1.0)
                    score = np.where(feas, sc * reach ** FIELD_REACH, 0.0)
                    # CONCENTRATE.  Spreading `need` over every feasible q reproduces the old
                    # request-plateau failure ("fire somewhere in here" is satisfied by firing
                    # EVERYWHERE, grad_trace.py:564) in a new form -- a neuron owing ONE spike
                    # got mass across ~100 steps, and 3n D degraded worst.  Sharpen the profile
                    # by raising it to a power before laying it down: FIELD_POW=1 is the flat
                    # relative-width weighting, larger values put high mass on a narrow range.
                    mx = float(score.max())
                    if mx <= 0:
                        continue
                    sel = np.nonzero(score > 0)[0]
                    qsel = qs[sel]                     # each q appears once, so += is safe
                    u = (sgn * jw * (need if _has_deficit else 0.0)
                         * (score[sel] / mx) ** FIELD_POW)
                    F[n][qsel] += u                    # URGENCY: additive
                    # SECOND FIELD VARIABLE.  wmin is the OUTGOING weight that a spike at q
                    # would need for it to actually serve the demand at t.  Unlike urgency it
                    # is a REQUIRED VALUE, not a quantity, so it must not accumulate -- two
                    # requests both wanting w ~= 500 mean 500, not 1000.  Carry it as an
                    # urgency-weighted mean via a running (sum of u*wmin, sum of u), with a
                    # floor on the weight so implied_w is not dropped merely because urgency
                    # happens to be 0 there.  Only a WANTED spike implies a weight.
                    if sgn > 0:
                        _wt = np.maximum(u, 1e-12)
                        WSUM[n][qsel] += _wt * wmin_a[sel]
                        WCNT[n][qsel] += _wt
            # DEFAULT SUPPRESSION ON EVERY EXISTING SPIKE.  A spike that costs nothing at
            # the CURRENT weight can still be the thing blocking progress: on 4n F, N2's
            # four surplus spikes drive nothing while w(N2->N3) = 20, so no negative job is
            # generated for them -- but raising that weight toward the ~445 the field is
            # ASKING for drags every matched N3 spike earlier (133->123, 233->223) without
            # ever producing the missing one at 290.  The damage is timing drift on MATCHED
            # spikes, and the mirrored jobs only fire on UNMATCHED ones, so the field is
            # silent about them.  A small constant tax on existing spikes makes "fewer
            # spikes" the default and forces each one to be earned by positive demand.
            if FIELD_BASE > 0:
                for q in spall[n]:
                    if 0 <= q < T:
                        a, b = max(0, q - FIELD_WIN), min(T, q + FIELD_WIN + 1)
                        F[n][a:b] -= FIELD_BASE * TH
            # RESIDUAL FLAT SUPPRESSION.  The mirrored negative jobs above cover the spikes of
            # n that DO drive an unwanted downstream spike.  They say nothing about a spike
            # that drives nothing at all -- one whose arrival every downstream discards -- so
            # there is no t to mirror and no wmin to score.  That case still needs a push, and
            # a flat window is all that is available for it.  It now fires only when there is
            # no field of EITHER sign nearby, so it no longer overrides the graded answer.
            if FIELD_FLAT and F[n].max() > 0:
                # THRESHOLD OFF A TYPICAL VALUE, NOT THE GLOBAL MAX.  The field is spiky,
                # so max(F) is set by narrow window-edge peaks; scaling the suppression
                # threshold to it puts thr above almost every sample, and then EVERY spike
                # fails the 'is there field near me' test.  Measured at the TRUE weights of
                # 3-cycle: all four of N2's perfectly-placed spikes took a full -1.70e-03,
                # and the nearest positive sample to any correct spike was 21 steps away.
                # The median of the positive field tracks the bulk instead of the extremes.
                _pos = F[n][F[n] > 0]
                _ref = float(np.median(_pos)) if _pos.size else float(F[n].max())
                thr = FIELD_SUPP * _ref
                for q in spall[n]:
                    if not (0 <= q < T):
                        continue
                    # A SPIKE WHOSE ARRIVAL IS DISCARDED GETS NO CREDIT.  The window test below
                    # asks only whether there is field mass NEARBY, so a spike survives on the
                    # strength of demand it cannot actually serve.  3-cycle: N2@312 arrives at
                    # 330, inside N3@316's refractory shadow (to 338), so the simulator throws
                    # it away and the output target at 404 goes unserved -- yet the spike keeps
                    # ~0 demand (the other three N2 spikes get a full -1.0) because the field at
                    # 295-305 is close enough to shelter it.  If every downstream discards this
                    # arrival, the spike is doing nothing and must be pushed like any other.
                    blocked = FIELD_REFRAC and len(np.where(C[:, 0] == n)[0]) > 0
                    if blocked:
                        for si_ in np.where(C[:, 0] == n)[0]:
                            d_ = int(C[si_, 1])
                            # REFRACTORINESS IS PHYSICAL.  The simulator discards an
                            # arrival that lands behind a spike the neuron REALLY produced;
                            # a target it is missing casts no shadow.  Testing against
                            # out_targets made 4n F's N2@285 (arrival 303) "blocked" by the
                            # TARGET at 290 that N3 does not fire, stamping a flat -thr over
                            # 265..305 -- penalising the spike for colliding with the very
                            # target the field was asking it to serve.
                            rd_ = (sorted(spall[d_]) if FIELD_PHYS
                                   else (sorted(out_targets[d_]) if d_ in out_targets
                                         else sorted(spall[d_])))
                            arr_ = q + DELAY_ITERS
                            if not any(r < arr_ < r + REFRAC_ITERS for r in rd_):
                                blocked = False      # at least one downstream accepts it
                                break
                    a, b = max(0, q - FIELD_WIN), min(T, q + FIELD_WIN + 1)
                    # FIELD_FLAT=1: only the drives-nothing case, which the mirrored jobs
                    # cannot see.  FIELD_FLAT=2 restores the old blanket "no field nearby"
                    # test, which now double-counts -- it stamped -0.22 on N1's perfectly
                    # placed spike at 461 with 3-cycle sitting at its TRUE weights.
                    if blocked or (FIELD_FLAT > 1 and float(np.abs(F[n][a:b]).max()) <= thr):
                        F[n][a:b] -= thr
            # having finished n's field, record WHERE IT SHOULD FIRE so upstream neurons can
            # use it as their demand times on the next iteration of the (downstream-first) loop
            if F[n].any() and FIELD_PROP:
                # PROPAGATE VIA THE URGENCY BUMPS, NOT THE implied_w CROSSINGS.  Crossings
                # were load-bearing in TWO places -- the final consumer AND this recursion --
                # and replacing only the consumer left the second hop starved: on 4n F under
                # the density consumer, N2's crossings were [] so N1 received no demand times
                # and had an identically empty field, while N2 (one hop from an output, so it
                # reads out_targets directly) had a clean correct one.
                # Each contiguous run of positive urgency is ONE requested spike, and its
                # argmax is where that spike is wanted.  This also restores a COUNT: the
                # number of bumps is the number of spikes being asked for, which the raw
                # density cannot express.
                _p = np.nonzero(F[n] > 0)[0]
                if len(_p):
                    _runs = np.split(_p, np.nonzero(np.diff(_p) > FIELD_GAP)[0] + 1)
                    _xs = [int(r[np.argmax(F[n][r])]) for r in _runs if len(r)]
                    if _xs:
                        _demand_t[n] = sorted(set(_xs))
            elif F[n].any():
                _iw = np.divide(WSUM[n], WCNT[n], out=np.full(T, np.nan), where=WCNT[n] > 0)
                _o = np.where(C[:, 0] == n)[0]
                if len(_o):
                    _wn = float(np.mean([abs(w[k]) for k in _o]))
                    _dd = _iw - _wn
                    _fin = np.nonzero(np.isfinite(_dd))[0]
                    _xs = []
                    for a, b in zip(_fin[:-1], _fin[1:]):
                        if b == a + 1 and (_dd[a] == 0 or (_dd[a] < 0) != (_dd[b] < 0)):
                            _xs.append(int(a if abs(_dd[a]) <= abs(_dd[b]) else b))
                    if _xs:
                        _demand_t[n] = sorted(set(_xs))
    if FIELD_SMOOTH > 0:
        # URGENCY IS A DENSITY, SO SMOOTH IT.  The raw field is spiky: `room` = 1 - wmin/wmax
        # jumps discontinuously wherever wmax changes, which happens at epoch boundaries, so
        # the profile shows narrow spikes that are an artefact of the boundary rather than
        # real structure (3n D: peaks at ~215 and ~272 with the TRUE spike at 246 sitting in
        # the trough BETWEEN them).  Convolve with a Gaussian so the field expresses "a
        # spike is wanted around here" rather than "at exactly this sample".
        _r = int(max(1, round(3 * FIELD_SMOOTH)))
        _x = np.arange(-_r, _r + 1)
        _k = np.exp(-0.5 * (_x / FIELD_SMOOTH) ** 2)
        _k /= _k.sum()
        for n in range(N):
            if F[n].any():
                F[n] = np.convolve(F[n], _k, mode="same")
    # implied_w = urgency-weighted mean; NaN where no request touched that time
    implied = {n: np.divide(WSUM[n], WCNT[n], out=np.full(T, np.nan),
                            where=WCNT[n] > 0) for n in range(N)}
    return F, implied


def solve_latency(t, lo, other, wsi, T):
    """When must n fire so that d crosses threshold AT t and NOT BEFORE?

    Every heuristic tried here approximated this: request_lat_vec returns the FIRST dt at
    which w*h(dt) alone covers the remaining deficit -- the earliest marginal crossing --
    whereas the truth sits well up the rising phase where the PSP COMBINES with the
    co-drivers.  On 3n D that put N1 at 170 (and 197/257) against a true 246, and correcting
    the deficit to exclude n's own share did not move it.

    It is a well-posed 1-D problem, so solve it exactly.  With `other` the co-drivers'
    voltage profile, a spike at q gives d

        V(tau) = other(tau) + wsi * h(tau - q)

    and we want the FIRST tau in the epoch with V(tau) >= TH to be exactly t.  Scan the
    admissible q (arrival inside the epoch) and keep those that satisfy it; prefer the
    LATEST, which is the least drive that still works and so the least disruptive elsewhere.
    Returns None when no q can do it -- which is itself the right answer, meaning this edge
    cannot serve t at its current weight and the demand belongs upstream.
    """
    tau = np.arange(max(0, lo + 1), t + 1)
    if len(tau) == 0:
        return None
    oth = other[tau]
    if oth[-1] >= TH:                 # co-drivers already fire d at t without n
        return None
    good = []
    q_lo = max(0, lo + 1 - DELAY_ITERS)
    for q in range(q_lo, t - DELAY_ITERS + 1):
        d_ = tau - q
        contrib = np.where((d_ >= 0) & (d_ < KWIN), wsi * HK[np.clip(d_, 0, KWIN - 1)], 0.0)
        V = oth + contrib
        if V[-1] >= TH and (len(V) == 1 or V[:-1].max() < TH):
            good.append(q)
    return good[-1] if good else None


def infer_hidden_targets(C, N, w, spall, T, out_targets, vsub, eps=None):
    """Decide WHERE each hidden neuron should fire, then let the ordinary hinge do the rest.

    The oracle test says this is the missing piece: handed their TRUE spike times, hidden
    neurons get 90-100% correct weight-direction from the existing hinge (3n D 100%, 4n G
    90%), versus 71-82% from the inferred request path.  So the machinery that turns a
    demand into a direction is fine; what is broken is choosing the times.  Do that
    explicitly instead of implicitly through five request stages.

    For each downstream target t that d cannot reach on its own drive, n must fire at the
    latency its CURRENT weight implies for the shortfall, and that firing must land in d's
    epoch for t and clear of its refractory shadows.  Collect those times, enforce n's own
    refractory separation, and hand the result to the same hinge the outputs use.
    """
    tgts = {k: list(v) for k, v in out_targets.items()}
    for n in range(N - 1, 0, -1):                 # downstream-first
        if n in out_targets:
            continue
        outs_n = np.where(C[:, 0] == n)[0]
        if len(outs_n) == 0:
            continue
        want = []
        for si in outs_n:
            d = int(C[si, 1])
            if d not in tgts:
                continue
            rd = sorted(tgts[d])
            for t in rd:
                if not (0 <= t < T):
                    continue
                # DEFICIT FROM THE OTHER EDGES ONLY.  Using TH - vsub[d][t] is
                # self-referential: vsub already contains n's own current contribution, so
                # the neuron is asked to cover a shortfall it is itself partly creating,
                # and the implied latency comes out wrong -- measured on 3n D with the
                # direct edge at TRUTH, N1 was placed at 170 (and at 197/257) against a
                # true 246.  What the truth actually satisfies is that n's PSP COMBINES
                # with the co-drivers to cross threshold at t, so subtract n's own share
                # and solve for the latency that closes what remains.
                lo = max([r for r in rd if r < t], default=-1)
                own = (np.zeros(T) if eps is None
                       else float(w[si]) * eps[(n, d)])
                other = vsub[d] - own             # the co-drivers' profile, n removed
                q = solve_latency(int(t), int(lo), other, float(w[si]), T)
                if q is None:
                    continue
                a = q + DELAY_ITERS
                if any(r < a < r + REFRAC_ITERS for r in rd):
                    continue
                if 0 <= q < T:
                    want.append(int(q))
        if not want:
            continue
        keep = []
        for q in sorted(set(want)):
            if not keep or q - keep[-1] > REFRAC_ITERS:
                keep.append(q)
        tgts[n] = keep
    return tgts


def traces(C, N, w, spall, T, out_targets, Vsim, hidden_targets=None, full=False):
    """Vsim: the simulated voltage (WITH resets).  A no-reset sum of PSPs accumulates
    over the whole run and sits far above threshold later on, which flips the sign of the
    output demand even where the neuron is under-driven -- so use the real trace."""
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    # reset points: a neuron's targets if it has them, else its own current spikes
    # Epoch resets.  Using the TARGET times for outputs makes the demand monotone in w,
    # but it has a failure mode: the epoch for t* starts at the PREVIOUS TARGET, so once
    # the network is mistimed the presynaptic spike that actually drives t* can fall just
    # BEFORE that boundary and be excluded entirely.  Measured on the chain: with the output
    # 6 steps early, N2 fires at 213 against an epoch (214, 314] -- excluded -- leaving only
    # its 313 spike, whose PSP is ~0 one step later.  vsub(314) reads 0.0 and the hinge
    # demands the FULL deficit (+7e-03), asking for MORE drive, which moves the spikes
    # earlier still: a runaway (-6 -> -24 -> -37 -> spikes lost).
    # EPOCH_ACTUAL extends the epoch back to the neuron's real reset when that is earlier,
    # so a driving spike can never be cut off by the boundary.
    rst = {}
    for n in range(N):
        if n not in out_targets:
            rst[n] = spall[n]
            continue
        tg = sorted(out_targets[n])
        if not EPOCH_EXTEND:
            rst[n] = tg
            continue
        # NARROW FIX for the epoch-boundary bug.  An epoch (prev_target, t*] that contains
        # NO presynaptic spike cannot possibly explain a spike at t*, so vsub reads 0 and
        # the hinge demands the full deficit -- which asks for more drive, moves the spikes
        # earlier, and excludes them further (chain: -6 -> -24 -> -37 -> spikes lost).
        # Only in that case, pull the boundary back just far enough to admit the most recent
        # presynaptic spike before t*.  Epochs that already contain a spike are untouched,
        # so the target-based monotonicity is preserved everywhere it was working.
        # The test is on achievable DRIVE, not on spike presence: the broken epoch is not
        # empty, it holds a spike 1 step before the target whose PSP is ~0.  If the whole
        # epoch cannot get near threshold, pull the boundary back to admit the previous
        # presynaptic spike, which is the one that actually drives that target.
        def _drive(lo, t):
            tot = 0.0
            for si in inc[n]:
                for q in spall[int(C[si, 0])]:
                    if lo < q <= t and 0 <= t - q < KWIN:
                        tot += float(w[si]) * HK[t - q]
            return tot
        pre = sorted({q for si in inc[n] for q in spall[int(C[si, 0])]})
        bounds = list(tg)
        for i, t in enumerate(tg):
            lo = bounds[i - 1] if i > 0 else -1
            if _drive(lo, t) >= EXTEND_FLOOR * TH or i == 0:
                continue
            earlier = [q for q in pre if q <= lo]
            if earlier:
                bounds[i - 1] = min(bounds[i - 1], earlier[-1] - 1)
        rst[n] = sorted(set(b for b in bounds if b >= 0))
    # eps[(k,n)]: k's PSP as seen by n, accumulated within n's epochs
    eps = {(k, n): eligibility(spall[k], T, rst[n],
                               refrac_at=(spall[n] if REFRAC_ACTUAL else None))
           for n in range(N) for k in range(N)}
    # SECOND ELIGIBILITY, ON THE ACTUAL RESETS, for the TIMING term only.
    # eps above is deliberately counterfactual: epochs reset at the TARGET times, i.e. it
    # evaluates the world in which the neuron already fires on schedule.  That is correct
    # and load-bearing for the creation hinge (it is what makes the demand monotone in w
    # and truth an exact fixed point), but it is WRONG for a demand placed at an ACTUAL
    # spike.  On the chain trap the two disagree totally: N3's spike at 264 is caused
    # entirely by N2@212 (w2*HK[52] = 0.00702862 vs simulated V(264) = 0.00702861), yet
    # under target-anchored resets N2@212 arrives at 230, inside the refractory shadow of
    # the COUNTERFACTUAL spike at 214, so it is discarded and eps[(2,3)] is empty
    # EVERYWHERE -- the model zeroes out the sole cause of the spike it is asked to move.
    # The two terms genuinely want different epoch structures, so build both.
    eps_act = {(k, n): eligibility(spall[k], T, spall[n], refrac_at=spall[n])
               for n in out_targets for k in range(N)}
    vsub = {n: (sum(w[si] * eps[(int(C[si, 0]), n)] for si in inc[n])
                if len(inc[n]) else np.zeros(T)) for n in range(N)}
    L = {n: np.zeros(T) for n in range(N)}
    Lmove = {n: np.zeros(T) for n in range(N)}   # the timing part, scored against eps_act
    # DEEP SUPERVISION.  The discrete relaxation solves the 3-cycle 4/4 while its inferred
    # hidden targets are WRONG (N1 [50,150,240,340] vs a true [72,172,262,362,461]) -- what
    # matters is that every neuron gets its OWN target, so the problem decomposes into
    # independent local ones and the global barrier is never traversed.  Give each hidden
    # neuron the same hinge its outputs get.
    seeded = dict(out_targets)
    if ORACLE_T:
        # *** CHEATING ***  true hidden spike times injected as targets.  Not a method --
        # an UPPER BOUND on what perfect hidden-target inference could buy, used to decide
        # whether the inference is worth chasing at all.  Never enable outside diagnostics.
        seeded.update({k: list(v) for k, v in ORACLE_T.items() if k not in out_targets})
    if HID_TARGETS:
        # explicit hidden targets, seeded through the SAME hinge/matching/suppression path
        seeded = infer_hidden_targets(C, N, w, spall, T, out_targets, vsub, eps)
    if hidden_targets:
        for n, tg in hidden_targets.items():
            if n not in out_targets and n != 0 and tg:
                seeded[n] = tg
    for o, tgt in seeded.items():
        # Match current spikes to targets (nearest, one-to-one) BEFORE deciding what is
        # spurious.  A merely LATE spike is not an extra spike: suppressing it cancels the
        # push-up toward its own target and the fit settles short of the true weight.
        # GRADED matching: record HOW FAR each found spike is from the target it claims,
        # instead of a yes/no verdict at MATCH_WIN.  A spike one step late is nearly right
        # and should barely be suppressed; one 30 steps late is mostly wrong and should be.
        # ONE matching, greedy by CLOSEST PAIR, shared with the request seeding below.
        # Previously this path matched greedily over spikes in TIME order while the request
        # path used closest-pair, and the two disagreed: on 3n D seed2 the misplaced spike
        # at 396 was paired with 433 here (so never suppressed) while the correct spike at
        # 436 was paired with 293 and pushed UP by +6.3e-03.
        lateness = {t: None for t in spall[o]}; claimed = {t: None for t in spall[o]}
        if SHARED_MATCH:
            _pairs = sorted(((abs(t - q), t, q) for t in spall[o] for q in tgt), key=lambda x: x[0])
            _ut = set(); _uq = set()
            for _d, t, q in _pairs:
                if t in _ut or q in _uq:
                    continue
                claimed[t] = q; lateness[t] = _d; _ut.add(t); _uq.add(q)
        else:                                   # original: greedy over spikes in TIME order
            _free = list(tgt)
            for t in sorted(spall[o]):
                if not _free:
                    continue
                q = min(_free, key=lambda x: abs(x - t))
                _free.remove(q); lateness[t] = abs(q - t); claimed[t] = q
        for t in tgt:
            # HINGE, not an equality.  A spike lands exactly on t* for a whole INTERVAL of
            # weights -- those with V(t*) >= th AND V(t*-1) < th -- so demanding
            # vsub(t*) == th keeps pushing even when the timing is already perfect.  That
            # is why the gradient was NOT zero at the true weights and why the optimiser
            # drifted off correct solutions.  Two one-sided terms, both zero on the
            # interval, make truth a genuine stationary point:
            if 0 <= t < T:
                L[o][t] += max(0.0, TH - vsub[o][t])          # under-driven -> push up
            if 0 <= t - 1 < T:
                L[o][t - 1] += min(0.0, TH - vsub[o][t - 1])  # already firing early -> push down
        # EARLY-SPIKE SUPPRESSION.  A spike that lands EARLIER than its own target gets no
        # demand at all today: the hinge at t* sees vsub(t*) already above threshold and
        # calls the target satisfied, while nothing attaches to the spike itself.  Measured
        # in case D: target 233, spike at 226, V_sim(233)=0 (refractory), L(233)=0, and all
        # five found spikes carried L=0.  This is the dominant miss (26 of 59 across the
        # failing cases, vs 3 caused by a genuinely different spike).
        # Fix: put a NEGATIVE demand at the spike's OWN time, scaled by how early it is --
        # less drive there makes it fire later.
        for q, tq in claimed.items():
            if tq is None or not (0 <= q < T):
                continue
            early = tq - q
            if early > EARLY_TOL:
                frac = min(1.0, (early - EARLY_TOL) / GRADE_SCALE)
                L[o][q] += EARLY_GAIN * frac * (0.9 * TH - vsub[o][q])
        # COHERENCE OF THE TIMING DEMAND.  A weight change shifts ALL of a neuron's spikes
        # the same way, so matched spikes that want OPPOSITE shifts cannot both be
        # satisfied -- and that disagreement is itself the signal that the deficit is a
        # COUNT problem, not a timing one, where creation should be left to act alone.
        # over-demand seed0: 2 output spikes against 3 targets, matched 174<-140 (34 LATE,
        # wants earlier) and 374<-399 (25 EARLY, wants later).  The two demands cancel and
        # the run FREEZES near its start ([284,539,162] -> [281,518,183], vs reaching
        # [250,685,300] with the term off).  The chain trap is the opposite: 264<-214 and
        # 464<-414, both +50, both wanting earlier -- coherent, and it escapes.
        # Scale by |sum of signs| / count: 1.0 when all agree, 0 when evenly split.
        _offs = [t - claimed[t] for t in spall[o]
                 if claimed.get(t) is not None and abs(t - claimed[t]) > DEAD_ZONE]
        _coh = 1.0
        if MOVE_COHERE and _offs:
            _coh = abs(sum(1 if o_ > 0 else -1 for o_ in _offs)) / len(_offs)
        for t in spall[o]:
            if not (0 <= t < T):
                continue
            d = lateness.get(t)
            # suppression strength grows with how far the spike is from its target:
            # 0 when exactly on target, full once it is GRADE_SCALE steps away
            # DEAD ZONE then GRADE.  Pure grading responds to every tiny error, so
            # something is always perturbing and near-correct configurations never settle
            # (suite 14/16 -> 7/16).  Pure thresholding is blind to a sparse perturbation
            # (a target missed by 27 steps counted as "hit").  Dead zone gives the
            # stability, the ramp beyond it gives the sensitivity.
            frac = 1.0 if d is None else (
                min(1.0, max(0.0, d - DEAD_ZONE) / GRADE_SCALE) if GRADE_SUPP
                else (0.0 if d <= MATCH_WIN else 1.0))
            if frac > 0:
                # SUPPRESSION MUST NOT INVERT.  0.9*TH - vsub is meant to go NEGATIVE at a
                # spurious spike (vsub is above threshold there, so removing drive lowers
                # it).  But the epoch-reconstructed vsub can read ZERO at a spike whose
                # driving arrival was masked out -- on 8n M the three extra N6 spikes at
                # 248/548/848 each got 0.9*TH - 0 = +6.300e-03, a CREATION demand at
                # exactly the spikes that should be removed.
                # The neuron demonstrably FIRED at t, so its true drive was >= TH whatever
                # the reconstruction says.  Clamp vsub up to TH before differencing, which
                # bounds the term at <= -0.1*TH and makes an inversion impossible.
                _vs = max(float(vsub[o][t]), TH) if SUPP_FIX else float(vsub[o][t])
                L[o][t] += frac * (0.9 * TH - _vs)
            # SIGNED TIMING DEMAND AT A MATCHED-BUT-MISTIMED OUTPUT SPIKE.
            # Without this a spike matched within MATCH_WIN carries NO demand at its own
            # time at all: `frac` above is 0 inside the window, so lateness is measured and
            # then dropped.  All that travels backward is "CREATE a spike at the target",
            # never "MOVE the spike you already have".  Measured on the chain trap
            # w=[456,415,626]: N3 fires at 264/464 against targets 214/414, the matching
            # correctly says "fire 50 earlier", and L[3] is nonzero ONLY at the four target
            # times -- nothing at 264 or 464.
            # More drive => crosses threshold sooner, so a LATE spike wants a POSITIVE
            # demand and an EARLY one a NEGATIVE demand (same convention as SHARP_FLIP).
            # Graded from DEAD_ZONE so it vanishes at perfect timing and truth stays a
            # stationary point.
            if MOVE_GAIN > 0 and claimed.get(t) is not None:
                off = t - claimed[t]                    # >0 late, <0 early
                amt = min(1.0, max(0.0, abs(off) - DEAD_ZONE) / GRADE_SCALE)
                if amt > 0:
                    _mv = MOVE_GAIN * _coh * (1.0 if off > 0 else -1.0) * amt * TH
                    L[o][t] += _mv          # so it still propagates to hidden neurons
                    Lmove[o][t] += _mv      # and is re-scored against eps_act below
    # ---- SPIKE-CREATION REQUESTS -------------------------------------------------
    # A missing spike is not "push the weights up": it is a request for a spike AT A
    # SPECIFIC TIME.  Its demand is the DEFICIT th - vsub_n(tau), which vanishes once the
    # spike exists, so it is self-limiting (unlike a propagated magnitude, which never
    # turns off and drove hidden weights to divergence).  If the neuron has no drive to
    # amplify at tau, no weight can create the spike there, so the request PROPAGATES
    # backward to its presynaptics at tau - latency, and keeps propagating until it
    # reaches a neuron that can actually satisfy it.
    # Seed a request at EVERY target the neuron does not already hit (not just the
    # unmatched ones).  A spike that is merely LATE also wants to move earlier, and often
    # no weight increase can do it: firing earlier needs presynaptic drive EARLIER, and if
    # that presynaptic spike does not exist, amplifying what is there cannot help.  Such a
    # "move earlier" demand must therefore raise a CREATION request in the inputs -- which
    # the propagation below does whenever there is nothing at tau to amplify.
    # A request does not commit to ONE latency.  Picking a single dt needs the current
    # weight, which is wrong during training, so the request lands at the wrong time and
    # chases a moving target (that cost 20/40 -> 14/40).  Instead the request spans the
    # WHOLE range of weight/latency combinations: it is a TRACE over time, spread across
    # the causal window weighted by kernel influence h(tau - s).  The location then comes
    # only from the kernel, which is FIXED, while the weight merely scales magnitude.
    # Seeded from DEFICITS so it still vanishes once satisfied (a propagated magnitude
    # with no deficit is what diverged before).
    R = {n: np.zeros(T) for n in range(N)}
    # S is allocated HERE, not at its own sweep, because the create sweep now needs to
    # write into it: along an inhibitory edge a creation demand becomes a SUPPRESSION
    # demand on the presynaptic neuron (EDGE_SIGN).
    S = {n: np.zeros(T) for n in range(N)}
    for o, tgt in out_targets.items():
        # ONE-TO-ONE matching, as the suppression path already does.  The old test was
        # `any(|t-q| <= HIT_TOL)`, which is NOT exclusive: a single spike could mark
        # arbitrarily many targets as "hit".  Measured with 6 spikes against 8 targets,
        # ALL EIGHT counted as hit -- targets 448 and 508 both claimed the same spike at
        # 464 -- so no request was raised even though two spikes were missing.  Combined
        # with suppression, which acts freely, that made the process a RATCHET: spikes
        # could be removed but never restored.
        # GREEDY BY CLOSEST PAIR, not in target order.  Matching targets in time order
        # lets an early target steal a later one's spike and cascade: measured on 3n D with
        # found [33,133,233,333,433] against targets [33,133,233,293,333,433], target 293
        # claimed the spike at 333 (40 away), 333 then claimed 433 (100 away) and 433 was
        # left unmatched -- so the "missing spike" demand landed on the wrong target.
        # Taking the closest available pair each time gives 293 -> None, which is right.
        avail = sorted(spall[o]); claim_t = {t: None for t in tgt}
        if PAIR_MATCH:
            pairs = sorted(((abs(t - q), t, q) for t in tgt for q in avail), key=lambda x: x[0])
            used_t = set(); used_q = set()
            for _, t, q in pairs:
                if t in used_t or q in used_q:
                    continue
                claim_t[t] = q; used_t.add(t); used_q.add(q)
        else:                                   # original: greedy over targets in TIME order
            for t in sorted(tgt):
                if not avail:
                    continue
                q = min(avail, key=lambda x: abs(x - t)); avail.remove(q); claim_t[t] = q
        for t in tgt:
            if not (0 <= t < T):
                continue
            q = claim_t.get(t)
            miss = None if q is None else abs(t - q)
            frac = 1.0 if miss is None else (
                min(1.0, max(0.0, miss - DEAD_ZONE) / GRADE_SCALE) if GRADE_REQ
                else (0.0 if miss <= HIT_TOL else 1.0))
            if frac > 0:
                R[o][t] += frac * max(TH - vsub[o][t], 0.0)
    # NORMALISE BY KERNEL MASS, NOT PEAK.  back_corr accumulates the kernel over its whole
    # ~400-step support, so its output scales with sum(HK)=4.00e-03, while peak=1.58e-05.
    # Dividing by the peak therefore AMPLIFIED every hop by mass/peak = 254x: measured on
    # the 3-cycle the request grew 7.0e-03 -> 1.09e+02 in one sweep and 9.5e+11 by sweep 5,
    # producing a gradient of 1.4e+07 that drove the weights into the 3000 clip.  Dividing
    # by the mass makes a hop magnitude-preserving.
    peak = float(HK.max()) if REQ_PEAKNORM else float(HK.sum())
    _rref = max((float(np.abs(R[o]).max()) for o in out_targets), default=0.0)
    for _ in range(SWEEPS):
        for n in range(N - 1, -1, -1):
            if R[n].max() <= 0:
                continue
            # only the part n cannot satisfy locally is passed on
            # graded propagation: pass on the FRACTION of threshold still missing rather
            # than gating on a hard drive floor.  With a hard floor, partial drive (4.3e-3
            # against a 1.4e-3 floor) blocked the request entirely and the upstream neuron
            # was never asked to fire.
            # A SILENT neuron always forwards the graded remainder.  The hard floor asks
            # "does n have almost no drive here" (vsub < 0.2*th) and, if not, keeps the
            # request for itself.  For 4n G's dead relay that is exactly wrong: N2's vsub
            # peaks at 0.99*th, so the floor concludes N2 can serve the request locally and
            # forwards NOTHING -- while N2 in fact fires not at all and the 1% it is short
            # of is precisely what its own input weight has to supply.  A neuron with no
            # spikes has demonstrably NOT satisfied the request, whatever its vsub says.
            # Applying the graded form everywhere instead costs 62 -> 52 (3-cycle 7->3,
            # 4n F 8->2), so restrict it to the case that needs it.
            unmet = (R[n] * np.clip(1.0 - vsub[n] / TH, 0.0, 1.0) if GRADE_PROP
                     else R[n] * (vsub[n] < CREATE_FLOOR * TH))
            if unmet.max() <= 0:
                continue
            # SHAPE-INDEPENDENT NORMALISATION.  Dividing by peak(HK) is right only while
            # the signal is SPARSE: a lone target time gives a profile of height U*peak.
            # After one hop the request is smeared over the kernel's ~400-step support, so
            # the correlation then accumulates the whole mass and peak-normalising
            # amplifies by mass/peak = 254 EVERY subsequent hop (7e-3 -> 1.1e2 -> 9.5e11).
            # Dividing by mass has the mirror flaw: correct for a broad signal, it crushes
            # a sparse one by the same 254x.  Instead rescale by the operation's ACTUAL
            # gain, so a hop preserves magnitude whatever the shape of its input.
            msg = back_corr(unmet, HK)
            if REQ_SELFNORM:
                mmax = float(np.abs(msg).max()); umax = float(np.abs(unmet).max())
                msg = msg * (umax / mmax) if mmax > 0 else msg
            else:
                msg = msg / peak
            msg = REQ_GAIN * msg
            for si in inc[n]:
                k = int(C[si, 0])
                if k != n:
                    if EDGE_SIGN and float(w[si]) < 0:
                        S[k] = S[k] + msg      # it holds n down; to help n, fire LESS
                    else:
                        R[k] = R[k] + msg
        if R_CAP:
            # Peak normalisation compensates the per-hop signal DECAY (it multiplies by
            # mass/peak = 254 per hop, and the backward signal loses roughly that much with
            # depth), which is why removing it costs 14/16 -> 11-12/16 even with the gain
            # retuned.  But uncapped it explodes on a cycle (7e-3 -> 9.5e11 by sweep 5).
            # Keep the compensation, bound the magnitude: clip each request to the seeded
            # output demand, which is fixed and cannot drift with the loop.
            for n in range(N):
                mx = float(np.abs(R[n]).max())
                if mx > _rref > 0:
                    R[n] = R[n] * (_rref / mx)
    # ---- SUPPRESSION REQUESTS: the mirror image of creation --------------------------
    # Creation requests propagate a DEFICIT upstream so a missing spike can be grown.
    # Nothing propagated the opposite signal, so hidden neurons had NO suppression at all:
    # requests could add hidden spikes but nothing could remove them.  That asymmetry is
    # why opening creation up (CREATE_FLOOR=1, or any graded variant) always over-fired.
    # Here an EXCESS -- a spike present where none is wanted -- travels upstream by the
    # same route, so the two forces balance instead of creation running away.
    for o, tgt in out_targets.items():
        for t in spall[o]:
            if not (0 <= t < T):
                continue
            if not any(abs(t - q) <= SUPP_TOL for q in tgt):     # spike nobody asked for
                S[o][t] += max(vsub[o][t] - 0.9 * TH, 0.0)
    for _ in range(SWEEPS):
        for n in range(N - 1, -1, -1):
            if S[n].max() <= 0:
                continue
            msg = back_corr(S[n], HK) / peak
            for si in inc[n]:
                k = int(C[si, 0])
                if k != n:
                    if EDGE_SIGN and float(w[si]) < 0:
                        R[k] = R[k] + msg      # firing MORE is how it suppresses n
                    else:
                        S[k] = S[k] + msg
    # ---- DIRECT WEIGHT DEMAND FROM AN UNSATISFIABLE REQUEST --------------------------
    # A request reaches a weight only through the eligibility, g = L . eps -- so when the
    # presynaptic does not fire inside the epoch, eps is 0 and the weight gets NO signal at
    # all, however badly it is needed.  Measured on 3n D: the output needs a spike at 293,
    # N1's spikes (183,383) lie outside the epoch (233,293], so eps_1(293)=0 and w(1->2)'s
    # gradient is exactly 0 while it sits at the lower clip (20 against a true 700).  The
    # request then lands entirely on w(0->1) as "more drive to N1", which makes N1 fire
    # EARLIER rather than at the time actually wanted.
    # Fix: an unsatisfied request also states what the EDGE would have to be.  If k fired at
    # the best possible moment it would contribute w_kn * peak(h), so covering a deficit D
    # needs w_kn >= D / peak(h).  That is a demand on the weight itself and does not
    # multiply through the presynaptic's current firing.
    wreq = np.zeros(len(w))
    if WREQ_GAIN > 0:
        pk = float(HK.max())
        for n in range(N):
            idx = np.nonzero(R[n])[0]
            for tau in idx:
                D = float(R[n][tau])
                if D <= 0:
                    continue
                for si in inc[n]:
                    need = D / pk
                    if w[si] < need:
                        wreq[si] += (need - w[si]) / max(len(idx), 1)
    # ---- BLOCKING SPIKES: suppress a spike that PREVENTS a requested one ---------------
    # A creation request can only ask for MORE drive, which makes an accumulating neuron
    # fire EARLIER -- it can never ask it to fire LATER.  But a request at tau is often
    # unsatisfiable precisely because the neuron already fired at some q < tau: the RESET
    # sends it back to zero and it cannot re-accumulate to threshold by tau.  Measured on
    # 3n D: the output needs a spike at 293, requiring N1 to fire in (233,293]; N1 fires at
    # 224 and 324, straddling the window, and the 224 spike is what stops it -- after
    # resetting there it cannot reach threshold again until 324.  No weight change fixes
    # that; the 224 spike itself has to go.
    if BLOCK_GAIN > 0:
        for n in range(N):
            if n in out_targets:
                continue
            for tau in np.nonzero(R[n])[0]:
                prev = [q for q in spall[n] if q < tau]
                if not prev:
                    continue
                q = max(prev)
                # can it re-accumulate to threshold over (q, tau] after resetting at q?
                reach = sum(float(w[si]) * sum(HK[tau - t] for t in spall[int(C[si, 0])]
                                               if q < t <= tau and 0 <= tau - t < KWIN)
                            for si in inc[n])
                if reach < TH and 0 <= q < T:      # blocked by its own earlier spike
                    L[n][q] -= BLOCK_GAIN * max(vsub[n][q] - 0.9 * TH, 0.0)
    # ---- SHARPEN THE REQUEST: fire AT this time, not merely somewhere -----------------
    # The request is deliberately SPREAD over all feasible latencies, so on 3n D it is
    # nonzero across [0,274].  A broad "fire somewhere in here" demand is satisfied by
    # firing EVERYWHERE -- which is what N1 does (5 spikes at 24,124,224,324,424 against a
    # true single spike at 246).  If a request names one time the neuron should converge to
    # firing AT that time, which needs both signs: pull towards the requested moment and
    # push down on the spikes that are not it.
    # ---- FEASIBILITY MASK ON THE REQUEST ----------------------------------------------
    # The request says "fire at tau to serve a demand downstream", but nothing checks that
    # a spike at tau COULD serve it.  On 3n D seed0 the request lands at 183: that arrives
    # at 201, BEFORE N2's reset at 238, so its PSP is wiped out at the epoch boundary and
    # the spike cannot contribute to the target at 293 for ANY weight.  The demand is
    # unsatisfiable as posed, yet it supplies the dominant term in g(0->1) (+3.57e-06 at
    # t=183 vs +4.79e-07 at N1's real spike) and pushes the weight the wrong way.
    # A presynaptic spike q can serve a demand at t on d only if its arrival lands in the
    # SAME EPOCH as t and clear of d's refractory shadows:
    #       q + DELAY_ITERS  in  ( last_reset_of_d_before(t),  t ]   minus shadows
    # Zero the request everywhere else.  This is exact and local -- no matching, no
    # resimulation -- and it constrains WHERE the request may ask for a spike rather than
    # trying to correct the wrong answer afterwards.
    _occl_ok = {}
    if OCCL_MASK:
        for n in range(N):
            if n in out_targets:
                continue
            # The window is a property of the DOWNSTREAM DEMAND and the reset geometry, not
            # of the request.  Deriving it from R[n] coupled the retiming to the whole
            # creation-request machinery -- with REQ_GAIN=0 the mask skipped every neuron,
            # _occl_ok was never populated and LN_RELOC silently never fired.  Fall back to
            # the downstream L when there is no request, so the two are independent.
            if R[n].max() <= 0 and not (OCCL_FROMDEMAND and any(
                    int(C[si, 1]) in out_targets for si in np.where(C[:, 0] == n)[0])):
                continue
            ok_q = np.zeros(T, bool)
            for si in np.where(C[:, 0] == n)[0]:
                d = int(C[si, 1])
                # PHYSICAL reset structure, NOT rst[]: EPOCH_EXTEND deliberately widens an
                # epoch precisely when nothing in it can reach threshold -- which is the
                # occlusion case -- and that widening would re-admit the very times this
                # mask exists to reject (measured: window 227..274 instead of 238..274,
                # letting the infeasible 227 through).  Feasibility must be judged against
                # where the neuron actually resets.
                rd = sorted(out_targets[d]) if d in out_targets else sorted(spall[d])
                tds = np.nonzero(R[d])[0]
                if OCCL_FROMDEMAND and len(tds) == 0 and d in out_targets:
                    tds = np.nonzero(L[d] > 0)[0]     # demand, when no request exists
                if len(tds) == 0:
                    continue
                live = np.zeros(T, bool)
                for t in tds:
                    lo = max([r for r in rd if r < t], default=-1)
                    live[lo + 1:int(t) + 1] = True
                for r in rd:                       # arrivals inside a shadow are discarded
                    live[max(0, r + 1):min(T, r + REFRAC_ITERS)] = False
                a = np.nonzero(live)[0] - DELAY_ITERS
                a = a[(a >= 0) & (a < T)]
                ok_q[a] = True
                if OCCL_DEBUG:
                    print(f'   [occl] n={n} d={d} demands={tds.tolist()[:8]}(n={len(tds)}) q-window {a.min() if len(a) else None}..{a.max() if len(a) else None}')
            # RELOCATE, DO NOT DELETE.  Zeroing an infeasible request throws away the
            # DEMAND along with the bad TIME, and the neuron is then told nothing at all.
            # Measured on 4n G seed4's resting point w=[289,420,1262,538]: the request sits
            # at 228 while the feasible window for the missing mark at 291 is q in [237,273]
            # -- the mask is right that 228 cannot serve it, but deleting leaves L[2] EMPTY
            # and g = [0,0,0,0], whereas the demand moved to 244 acts through
            # eps[(1,2)][244] = 1.57e-05 and gives g(1->2) = +2.26e-06, the direction that
            # re-crosses the barrier (w 420 -> needs 447 to fire N2@244).
            # So carry each rejected entry to the NEAREST feasible time instead.
            if OCCL_RELOC and ok_q.any():
                feas = np.nonzero(ok_q)[0]
                bad = np.nonzero((~ok_q) & (R[n] > 0))[0]
                if len(bad):
                    moved = np.zeros(T)
                    tgt_i = feas[np.abs(feas[None, :] - bad[:, None]).argmin(axis=1)]
                    np.add.at(moved, tgt_i, R[n][bad])
                    R[n] = R[n] * ok_q + moved
                else:
                    R[n] = R[n] * ok_q
            else:
                R[n] = R[n] * ok_q
            _occl_ok[n] = ok_q
    _sharp_tau = {}
    if SHARP_GAIN > 0:
        for n in range(N):
            if n in out_targets or R[n].max() <= 0:
                continue
            # WHICH time to ask for.  argmax of the spread is the latency of maximum
            # kernel influence, argmax(HK)=110 -- not the latency that actually causes a
            # crossing.  On 3n D the output needs 293 and N1 must fire at 246 (latency 47),
            # but the argmax sits at 293-110 = 183, so the neuron converges to the WRONG
            # moment (measured: N1 -> 356, the extra output spike landing at 394 not 293).
            # Use the latency implied by the CURRENT weight instead: the first dt at which
            # w*h(dt) covers the deficit.
            tau = int(np.argmax(R[n]))
            if SHARP_LAT:
                # Scan EVERY nonzero downstream request entry, not just its peak -- the
                # peak-only approximation picks a worse time and loses the benefit.  This is
                # affordable now that the latency query is a vectorised binary search on
                # HK's monotonic rising phase (355x faster than the 400-step scalar scan,
                # verified identical on 1200 cases).
                best = None; pool = []
                for si in np.where(C[:, 0] == n)[0]:
                    d = int(C[si, 1])
                    tds = np.nonzero(R[d])[0]
                    if len(tds) == 0:
                        continue
                    dts = request_lat_vec(float(w[si]), R[d][tds])
                    cands = tds - dts
                    ok = (dts >= 0) & (cands >= 0) & (cands < T)
                    if not np.any(ok):
                        continue
                    cc = cands[ok]
                    pool.append(cc)
                    c = int(cc[int(np.argmax(R[n][cc]))])
                    if best is None or R[n][c] > R[n][best]:
                        best = c
                if best is not None:
                    tau = best
            taus = [tau]
            if SHARP_MULTI and pool:
                # A SET of requested times, not one.  Collapsing to a single tau forces the
                # "fire once, at tau" reading on every neuron, and the sign rule downstream
                # then drags a neuron's OTHER legitimate spikes toward that one time --
                # over-demand's N1 must fire at 173 AND 373, and the correct 373 is pushed
                # up until N1's input weight goes supercritical and it fires every cycle.
                # Keep every candidate that is a separate request: greedily take the
                # strongest, then any further one more than SHARP_WIN from all taken.  On a
                # genuinely fire-once neuron (3n D) this yields a single tau and reduces to
                # the old behaviour exactly.
                cand = np.unique(np.concatenate(pool))
                cand = cand[R[n][cand] >= SHARP_FLOOR * float(R[n][tau])]
                for c in cand[np.argsort(-R[n][cand])]:
                    if all(abs(int(c) - t) > SHARP_WIN for t in taus):
                        taus.append(int(c))
            amps = [float(R[n][t]) for t in taus]
            R[n] = np.zeros(T)                          # collapse to those times only
            for t, a in zip(taus, amps):
                R[n][t] = a
            _sharp_tau[n] = taus                        # applied AFTER the relaxation
            if SHARP_DEBUG:   # off by default: the unique/concat is not free
                LAST_SHARP[n] = (list(taus), len(pool),
                                 [int(x) for x in (np.unique(np.concatenate(pool)) if pool else [])])
    # ---- OCCLUSION: route the demand to whatever makes the edge LIVE again -----------
    # An edge is EXACTLY inert -- derivative identically zero for every weight -- when its
    # presynaptic arrival cannot land in the same epoch as the demand.  Measured on 3n D
    # seed0: w(1->2) scanned over its WHOLE range 20..3000 never changes the output, because
    # N1@223 arrives at 241 inside N2@238's refractory shadow [238,260].  No local rule of
    # any sign can see past that, since there is no slope to read.
    #
    # What IS computable, exactly and locally, is the window the arrival must land in to
    # serve a demand at t.  It is NOT merely "outside the refractory shadow": clearing the
    # shadow by moving EARLIER is useless because the PSP would then be reset away at the
    # epoch boundary.  The arrival must be in the SAME EPOCH as t:
    #
    #     arrival in ( last_reset_of_n_before(t),  t ]   minus refractory shadows
    #
    # On seed0/target 293 that is (260, 293], i.e. the presynaptic must fire in (242, 275].
    # The true N1 fires at 246 -- inside the window -- and the current 223 is BELOW it, so
    # the spike must move LATER, i.e. WEAKEN w(0->1).  That is the true direction (243->200),
    # and it is the OPPOSITE of the cheapest way out of the shadow.  Direction comes from
    # the window geometry, not from a create-vs-suppress verdict, which is why it yields
    # either sign as needed.
    if OCCL_GAIN > 0:
        for n in range(N):
            resets = sorted(rst.get(n, spall[n]))
            if not resets:
                continue
            want = np.where(L[n] > 0)[0]        # unmet demand: this neuron needs drive here
            for t in want:
                t = int(t)
                lo = max([r for r in resets if r < t], default=-1)   # epoch start
                a_lo, a_hi = lo + 1, t                              # arrival window
                if a_hi < a_lo:
                    continue

                def _live(a):                    # arrival lands in-epoch and unshadowed
                    return (a_lo <= a <= a_hi and
                            not any(r < a < r + REFRAC_ITERS for r in resets))

                # only act when NOTHING can currently serve t -- otherwise the ordinary
                # gradient is live and this would just add noise on top of it
                served = any(_live(int(q) + DELAY_ITERS)
                             for si in inc[n] for q in spall[int(C[si, 0])])
                if OCCL_DEBUG:
                    print(f'   [occl-gain] n={n} t={t} epoch=({lo},{t}] served={served}')
                if served:
                    continue
                feas = [a for a in range(a_lo, a_hi + 1) if _live(a)]
                if OCCL_DEBUG:
                    print(f'      feasible arrivals {feas[:1]}..{feas[-1:]} '
                          f'-> q window {(feas[0]-DELAY_ITERS) if feas else None}'
                          f'..{(feas[-1]-DELAY_ITERS) if feas else None}')
                if not feas:
                    continue                     # unreachable this epoch; nothing to ask
                qlo, qhi = feas[0] - DELAY_ITERS, feas[-1] - DELAY_ITERS
                mag = OCCL_GAIN * float(L[n][t])
                for si in inc[n]:
                    k = int(C[si, 0])
                    for q in spall[k]:
                        if not (0 <= q < T) or qlo <= q <= qhi:
                            continue
                        # move q INTO the window: later => less drive, earlier => more
                        L[k][q] += -mag if q < qlo else +mag
                        if OCCL_DEBUG:
                            print(f'      WRITE L[{k}][{q}] += '
                                  f'{-mag if q < qlo else mag:+.4e}')
    if FIELD or FIELD_XING > 0 or FIELD_ADD > 0 or FIELD_LOCAL > 0:
        _Dm, _implied = demand_field(C, N, w, spall, T, out_targets, vsub, eps)
    if FIELD_XING > 0:
        # PHASE-REFERENCED DEMAND.  At each crossing q* the neuron's current weight is the
        # right weight for firing there, so q* is where it SHOULD fire.  Put a positive
        # demand at q* (fire here), and for each existing spike push it TOWARD the nearest
        # crossing with the sign given by the direction -- later => less drive => negative.
        _x = field_crossings(C, N, w, _Dm, _implied, T)
        for n, xs in _x.items():
            if n in out_targets:
                continue
            # A CROSSING ON AN EXISTING SPIKE CARRIES NO INFORMATION.  implied_w meets w
            # wherever the current weight already explains the current timing, so at a spike
            # the crossing test is satisfied BY CONSTRUCTION.  Measured over 147 field peaks
            # at stuck points, 79 land within 2 steps of a current spike and only 14 within
            # 2 of a true one.  Feeding those back as "fire here" is a spike-count ratchet:
            # it reinforces every spike that exists and can never remove one.  On 4n F, whose
            # N1 must fire ONCE, it drove N1 to the 4-spike input-locked rhythm by it400 and
            # the addition disagreed with the working demand's sign on 219 of 245 points.
            if XING_NOVEL > 0 and spall[n]:
                xs = [(q, u) for q, u in xs
                      if all(abs(int(q) - int(s_)) > XING_NOVEL for s_ in spall[n])]
                if not xs:
                    continue
            for q, u in xs:
                L[n][q] += FIELD_XING * XING_DEMAND * u
            if spall[n] and XING_MOVE != 0:
                qs = np.array([q for q, _ in xs])
                for s_ in spall[n]:
                    if not (0 <= s_ < T):
                        continue
                    q = int(qs[np.argmin(np.abs(qs - s_))])
                    if q == s_:
                        continue
                    amt = min(1.0, abs(q - s_) / GRADE_SCALE)
                    L[n][s_] += (FIELD_XING * XING_MOVE
                                 * (-1.0 if q > s_ else 1.0) * amt * TH)
    elif NEW_DEMAND:
        _Dm = demand_direct(C, N, w, spall, T, out_targets, vsub, L, inc)
    for n in range(N):
        if n in out_targets:
            continue          # outputs already carry these from the seeding above
        if FIELD or NEW_DEMAND:
            L[n] = L[n] + NEW_GAIN * _Dm[n]
        else:
            L[n] = L[n] + R[n] - SUPP_GAIN * S[n]   # creation vs suppression, balanced
            if FIELD_ADD > 0:
                # USE THE URGENCY DENSITY DIRECTLY, NOT THE CROSSINGS.  FIELD_XING consumes
                # only field_crossings(), i.e. the LOCATIONS where implied_w meets w, and
                # never reads urgency at all -- which is why four successive improvements to
                # urgency left recovery untouched.  The density is the thing that carries
                # "a spike here would help": broad, signed, and defined even where implied_w
                # is a step function with no crossing to find.
                L[n] = L[n] + FIELD_ADD * _Dm[n]

    # ---- PIVOTAL-CONTRIBUTOR SUPPRESSION (cause A, corrected) ------------------------
    # A hidden spike is harmful if it is PIVOTAL for a downstream spike that fires EARLY:
    # pivotal meaning removing its contribution w*h(f-q) would drop the drive at f below
    # threshold, so without it that spike would not have happened when it did.  Pure
    # arithmetic on the spike trains and the kernel -- NO re-simulation.
    # This is what the plain backward message cannot do: measured on the 5-neuron net it
    # pushed DOWN on all three spikes of a neuron that was UNDER-firing, whereas this test
    # flags none of them and fingers only the genuine offenders.
    if PIVOT_GAIN > 0:
        for d, tgt in out_targets.items():
            free = list(tgt); claim = {}
            for f in sorted(spall[d]):
                if not free:
                    claim[f] = None; continue
                tt = min(free, key=lambda x: abs(x - f)); free.remove(tt); claim[f] = tt
            for i, f in enumerate(spall[d]):
                tt = claim.get(f)
                if tt is None or f >= tt:        # only EARLY spikes are harmful
                    continue
                prev = spall[d][i - 1] if i > 0 else -10 ** 9
                parts = []; drive = 0.0
                for si in inc[d]:
                    k = int(C[si, 0])
                    for q in spall[k]:
                        if prev < q <= f and 0 <= f - q < KWIN:
                            c = float(w[si]) * HK[f - q]
                            drive += c; parts.append((k, q, c))
                for k, q, c in parts:
                    if drive - c < TH and k not in out_targets and 0 <= q < T:
                        L[k][q] -= PIVOT_GAIN * (TH - (drive - c))

    # fixed reference for the loop-gain cap: the magnitude of the seeded output demand,
    # captured BEFORE any backward sweep so it cannot drift upward with the loop
    _ref = max((float(np.abs(L[o]).max()) for o in out_targets), default=0.0)

    # backward relaxation: hidden signals from downstream, gated by own sensitivity
    down = {n: [(int(si), int(C[si, 1])) for si in np.where(C[:, 0] == n)[0]] for n in range(N)}
    for _ in range(SWEEPS):
        for n in range(N):
            if n in out_targets or not down[n]:
                continue
            # SIGNED TIMING DEMAND at the spikes this neuron already has: positive means
            # "fire earlier" -> needs more drive -> positive voltage demand there.
            tim = np.zeros(T); vol = np.zeros(T)
            for si, d in down[n]:
                tim += w[si] * back_corr(L[d], HKP)
                vol += w[si] * back_corr(L[d], HK)
            # Both demands act TOGETHER, not either/or: a neuron firing too FEW times has
            # no spike to move, so a timing-only signal is silent about the spikes it is
            # missing (and a creation-only signal cannot say which way an existing spike
            # should shift).  Timing acts at the spikes it has; creation acts wherever it
            # is near threshold and could be recruited.
            # Convert the TIMING demand into a VOLTAGE demand by the local slope.
            # sum_t L_d(t) h'(t-s) has units of benefit-per-unit-time-shift, not volts;
            # the conversion factor is ds/dV = 1/(dV/dt) at the spike.  Without it each
            # hop carries a stray factor ~w*h' ~ 5e-5 and the signal vanishes about 1e5
            # per layer (measured L_1 ~1e-13 vs ~1e-3 at the output, gradients ~1e-18).
            # Rescaling to a fixed reference instead also destroys convergence, since the
            # demand can then never decay to zero at the solution.
            _field = vol * np.exp(-((vsub[n] - TH) / (SIG * TH)) ** 2)
            Ln = _field * CREATE
            if FIELD_SLOPE > 0:
                # TIMING FROM THE FIELD'S OWN SLOPE.  The creation density already contains
                # timing information that is currently thrown away: if the density is RISING
                # at a spike, more demand sits later, so the spike should move later (less
                # drive => negative voltage demand); if FALLING, earlier.  That is a signed
                # timing signal derived from the SAME object that carries the rate, rather
                # than from a separate correlation against the kernel derivative -- so it
                # inherits whatever target structure the field has, instead of being a bare
                # earlier/later scalar that pulls toward the nearest demand.
                _d = np.gradient(_field)
                for s_ in spall[n]:
                    if 0 <= s_ < T:
                        Ln[s_] -= FIELD_SLOPE * float(_d[s_])
            # LOOP-GAIN BOUND.  On a CYCLIC graph the backward message feeds back into
            # itself every sweep, so it amplifies geometrically: measured on the 3-cycle,
            # max|L| ran 1.3 -> 4.2e5 -> 1.8e10 -> 1.0e12 over 1,2,4,6 sweeps, producing a
            # gradient of 1.4e+07 and driving the weights straight into the 3000 clip.  The
            # acyclic chain grows only linearly (1e-6 -> 2e-4), so this is purely a loop
            # effect.  A hop must not amplify: cap the message at the magnitude of the
            # downstream demand that produced it, which bounds the loop gain to <= 1 while
            # leaving direction and within-layer structure untouched.
            slope = np.diff(vsub[n], prepend=vsub[n][0])
            for s_ in spall[n]:
                if TIM_REFRAC and 0 <= s_ < T:
                    # A SPIKE WHOSE ARRIVAL IS DISCARDED IS SERVING NOTHING, so it should
                    # not receive a timing correction as if it were.  3-cycle: N2@312
                    # arrives at 330 inside N3@316's refractory shadow and is thrown away by
                    # the simulator, yet it collects a normal earlier/later demand -- which
                    # holds it in place while the output target at 404 goes unserved.  Give
                    # it a plain push instead of a timing nudge.
                    _blk = True
                    for _si in np.where(C[:, 0] == n)[0]:
                        _d = int(C[_si, 1])
                        _rd = (sorted(out_targets[_d]) if _d in out_targets
                               else sorted(spall[_d]))
                        if not any(r < s_ + DELAY_ITERS < r + REFRAC_ITERS for r in _rd):
                            _blk = False
                            break
                    if _blk:
                        Ln[s_] -= TIM_REFRAC * TH
                        continue
                if 0 <= s_ < T:
                    sl = abs(float(slope[s_]))
                    # TIM_GAIN scales the backward TIMING term -- the part that drags a
                    # hidden spike toward whichever downstream demand is nearest.  That is
                    # the same "pull to the closest target" logic as the output-side
                    # matching, one layer up, and it is a plausible source of phase error:
                    # it can only say earlier/later, never "at 173", so it pushes an
                    # accumulator's crossing around without a phase reference.
                    Ln[s_] += TIM_GAIN * tim[s_] / max(sl, SLOPE_FLOOR * TH)
            if LOOP_CAP and _ref > 0:
                # cap against the SEEDED OUTPUT demand, which is fixed, not against the
                # downstream L -- that is itself growing inside the loop, so capping to it
                # bounds nothing (measured: 1.0e12 -> 9.1e11, no real change).
                mx = float(np.abs(Ln).max())
                if mx > _ref:
                    Ln = Ln * (_ref / mx)
            # The FORWARD pass masks occluded spikes (REFRAC_MASK); the backward message
            # must respect the same structure.  A POSITIVE demand at a time whose arrival
            # cannot reach the downstream demand is asking for a spike that would do
            # nothing -- on 3n D seed0 that is the +0.0165 sitting on N1@223, which is the
            # term that keeps pushing w(0->1) the wrong way.  Drop the positive part there.
            # The NEGATIVE part is kept: a spike at a time that can serve nothing SHOULD be
            # discouraged, and that is exactly the signal needed to move it out.
            #
            # CLAMP THE INCREMENT, NOT THE RUNNING TOTAL.  Clamping L[n] after the add is
            # wrong: the occlusion term is written ONCE before the sweeps, while this
            # message re-injects its positive at the SAME infeasible time on EVERY sweep.
            # Measured on 4n G seed7, L[2][201] went -2.80e-02 (correct, "move this spike
            # out") -> +6.84e-03 added six times -> net positive -> clamped to 0, losing the
            # signal entirely and leaving g(1->2) = 0.  Clamping Ln keeps the accumulated
            # negative intact.
            if OCCL_MASK and n in _occl_ok:
                ok_n = _occl_ok[n]; bad = ~ok_n
                if LN_RELOC and ok_n.any():
                    # RETIME, don't discard.  Zeroing the positive part throws away the
                    # DEMAND along with the unusable TIME, and the neuron is left with
                    # nothing: at 4n G seed4's resting point that empties L[2] entirely and
                    # g goes to [0,0,0,0], while the same demand at a feasible time acts
                    # through eps[(1,2)][244] = 1.57e-05 and gives g(1->2) = +2.26e-06 --
                    # the direction that re-crosses the barrier.  Carry each rejected
                    # POSITIVE entry to the nearest feasible time; negatives stay put
                    # (a spike that can serve nothing should still be discouraged there).
                    pos = np.nonzero(bad & (Ln > 0))[0]
                    # The destination must be feasible AND REACHABLE: ok_n only says a
                    # spike there could serve the downstream demand, not that this neuron's
                    # inputs can produce any drive there.  Without the second condition the
                    # demand lands where the eligibility is zero and still does nothing --
                    # measured, it moved to 437, past N2's reset at 376 where N1's PSPs are
                    # truncated, so g(1->2) stayed 0.  Require vsub[n] > 0, i.e. there is
                    # something at that time for a weight increase to amplify.
                    reach = ok_n & (vsub[n] > 0)
                    if len(pos) and reach.any():
                        feas = np.nonzero(reach)[0]
                        dest = feas[np.abs(feas[None, :] - pos[:, None]).argmin(axis=1)]
                        carried = np.zeros(T)
                        np.add.at(carried, dest, Ln[pos])
                        Ln = Ln.copy(); Ln[pos] = 0.0; Ln = Ln + carried
                    Ln[bad] = np.minimum(Ln[bad], 0.0)
                else:
                    Ln[bad] = np.minimum(Ln[bad], 0.0)
            L[n] = L[n] + Ln
    # SET THE SIGN of the demand at spikes far from the requested time -- AFTER the
    # backward relaxation, which is what adds the large timing term.  Doing it before was
    # useless: the correction is worth |vsub-0.9th| ~ 7e-4 while the timing term at the
    # same spike is ~0.46 (it carries a 1/slope factor) and is added afterwards, so it was
    # swamped 650:1 and then overwritten.  A spike LATER than tau needs MORE drive to fire
    # earlier; one EARLIER than tau needs less.
    for n, taus in (_sharp_tau.items() if SHARP_FLIP else ()):
        tl = taus if isinstance(taus, (list, tuple)) else [taus]
        # DO NOT DRAG A SPIKE THAT IS ALREADY DOING USEFUL WORK.  The request R lists only
        # UNMET demands, so when one of a neuron's several jobs is already satisfied the
        # request names only the OTHER time -- and the flip then treats the correct spike as
        # a mistimed copy of it.  That is exactly over-demand: N1 must fire at 173 and 373,
        # only one is ever unmet, and the correct one is pushed until N1 goes supercritical.
        # A spike that materially drives an ON-TARGET output spike is protected.
        prot = set()
        if SHARP_PROTECT:
            for si in np.where(C[:, 0] == n)[0]:
                d = int(C[si, 1])
                if d not in out_targets:
                    continue
                for sd in spall[d]:
                    if not any(abs(sd - t) <= PROTECT_TOL for t in out_targets[d]):
                        continue                      # that downstream spike is not on target
                    tot = 0.0
                    for sj in inc[d]:
                        p = int(C[sj, 0])
                        tot += sum(float(w[sj]) * HK[sd - qq] for qq in spall[p]
                                   if 0 <= sd - qq < KWIN)
                    if tot <= 0:
                        continue
                    for qq in spall[n]:
                        if 0 <= sd - qq < KWIN and \
                                float(w[si]) * HK[sd - qq] >= PROTECT_SHARE * tot:
                            prot.add(qq)
        for q in spall[n]:
            if not (0 <= q < T) or q in prot:
                continue
            # measure against the NEAREST requested time: a spike that is serving some
            # other request must not be dragged toward this one
            tau = min(tl, key=lambda t: abs(q - t))
            if abs(q - tau) <= SHARP_WIN:
                continue
            cur = float(L[n][q])
            want_up = q > tau
            if ((cur < 0) if want_up else (cur > 0)):
                L[n][q] = -cur * SHARP_GAIN

    if FIELD_LOCAL > 0:
        # REPLACE the hidden demand entirely, and do it LAST.  Placing this earlier in the
        # function did nothing: the BACKWARD RELAXATION runs afterwards and does
        # `L[n] = L[n] + Ln` (its own comment calls Ln "the large timing term", ~0.46
        # because it carries a 1/slope factor).  Measured on 4n F seed 3 at the stuck point,
        # local_demand contributed neg -2.33e-04 while the final L[2] carried neg -4.74e-01
        # -- 2000x larger -- so the field was buried, not consulted.  n's own field against
        # n's own spikes is meant to BE the decision, so nothing may be added after it.
        for n in range(N):
            if n in out_targets:
                continue
            L[n] = FIELD_LOCAL * local_demand(_Dm[n], spall[n], T)
            Lmove[n][:] = 0.0        # the timing term is part of what is being replaced

    if full:
        return eps, L, vsub, wreq, eps_act, Lmove
    return eps, L, vsub, wreq          # 4-tuple kept for the diagnostic callers


def train(C, N, outs, w, T_true, params, rounds, lr=LR, cb=None):
    """Adam on the trace gradient.  Normalising by max|g| (as a plain relative step does)
    throws the magnitude away -- with a single synapse every step is then exactly +-lr, so
    it creeps and cannot fine-tune.  Adam rescales by the running RMS, giving a step in
    weight units that adapts as the gradient shrinks near the solution."""
    T = params.steps
    out_t = {o: T_true[o] for o in outs}
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    m = np.zeros(len(w)); v = np.zeros(len(w))
    # runaway freezing: a weight whose motion never reverses over a window, while the
    # output error fails to improve, is absorbing error rather than being constrained by
    # it (see the 3-cycle feedback edge, 91 -> 480 with error stuck at 99).  Detected from
    # observable quantities only, then pinned in place.
    _wsign = np.where(np.asarray(w, float) < 0, -1.0, 1.0)   # fixed for the whole run
    w = _wsign * np.clip(np.abs(np.asarray(w, float)), 20, 3000)
    w_init = w.copy()           # anchor for the proximal penalty
    frozen = np.zeros(len(w), bool); whist = [w.copy()]; prev_err = None
    best_w = w.copy(); best_e = np.inf
    def out_err():
        Vh = fsim(C, N, w, params)
        tot = 0.0
        for o in outs:
            f = sp(Vh, o); t = T_true[o]
            tot += 99.0 if len(f) != len(t) else float(np.mean([abs(a - b) for a, b in zip(f, t)]))
        return tot / max(len(outs), 1)
    ait = 0                     # Adam's own step counter, reset at each restart
    _wh = []                    # recent weight history, for the stall test
    for it in range(1, rounds + 1):
        if RESTART_EVERY > 0 and it % RESTART_EVERY == 1 and it > 1:
            # PERIODIC RESTART of the Adam state.  The gradient does not vanish at a
            # correct solution (discretisation residual), so a single continuous run drifts
            # off it and the stale moment estimates carry it further.  Clearing m, v and the
            # step counter re-energises the search and it revisits good regions; combined
            # with KEEP_BEST this converts near-misses into exact hits (3-cycle 0/4 -> 2/4).
            m = np.zeros(len(w)); v = np.zeros(len(w)); ait = 0
        ait += 1
        V = fsim(C, N, w, params); spall = {p: sp(V, p) for p in range(N)}
        ht = None
        if DEEP_SUP:
            ht = _infer_relax(C, N, out_t, spall, {}, sweeps=4)
        eps, L, vsub, wreq, eps_act, Lmove = traces(
            C, N, w, spall, T, out_t, V, hidden_targets=ht, full=True)
        g = np.zeros(len(w))
        for n in range(N):
            for si in inc[n]:
                k = int(C[si, 0])
                if MOVE_ACT and n in out_t and MOVE_GAIN > 0:
                    # score the TIMING part against the ACTUAL-reset eligibility and the
                    # rest against the counterfactual one: they need different epochs
                    g[si] = float(np.dot(L[n] - Lmove[n], eps[(k, n)])
                                  + np.dot(Lmove[n], eps_act[(k, n)]))
                else:
                    g[si] = float(np.dot(L[n], eps[(k, n)]))
        _kick = (float(np.abs(g).max()) == 0.0
                 and not all(spall[o] == T_true[o] for o in outs))
        if KICK_STALL > 0 and not _kick and it > KICK_WIN:
            # STALLED, not frozen.  g == 0 is the extreme case; a run can equally sit in a
            # tiny neighbourhood with a live-but-useless gradient -- 4n G seed3 travels 0.10
            # weight units over its last 500 iterations while never once hitting g == 0.
            # Gate on the weights not MOVING rather than on the gradient vanishing, which
            # covers both.  Gating on "output has too few spikes" instead was tried and is
            # catastrophic (4n G 5/8 -> 0/8, suite 66 -> 24): that is true for most of
            # training, so the kick fires constantly and swamps the real gradient.  The
            # frozen/stalled test works BECAUSE it is rare.
            _wh.append(w.copy())
            if len(_wh) > KICK_WIN:
                _wh.pop(0)
                if float(np.abs(_wh[-1] - _wh[0]).max()) < KICK_STALL:
                    _kick = True
        if KICK_FEW > 0 and not _kick:
            # UNDER-FIRING, not just frozen.  With the freeze fixed, 4n G's remaining seeds
            # move freely and still converge to N1=1,N2=1,N3=5 against a true 2,2,7 --
            # w(0->1) is driven DOWN to ~190-205 (true 250) until the accumulator is too
            # slow to fire twice.  seed5 visits (2,2,5) and (2,2,6) and drifts away, so the
            # count is reachable and actively abandoned.  g is nonzero throughout, so the
            # frozen test never fires.  But "the output has FEWER spikes than targets" is
            # just as observable, and false at truth.
            if any(len(spall[o]) < len(T_true[o]) for o in outs):
                _kick = True
        if KICK_GAIN > 0 and _kick:
            # FROZEN AND WRONG.  g == 0 on EVERY edge while the output is still wrong is not
            # a solution, it is an absorbing state: on 4n G the relay N2 falls below the
            # single-spike weight, stops firing, and with no N2 spike there is no
            # eligibility on the 2->3 edge for any demand to attach to, so the run sits
            # there for 2900 iterations.  Perfect hidden targets do NOT fix it (oracle 4/8),
            # so this is not an inference problem -- there is simply no gradient to follow.
            # The state is exactly detectable, and the only neurons that can be responsible
            # are the hidden ones that are firing least.  Push their inputs up: a silent or
            # near-silent hidden neuron that the output still needs can only be revived by
            # more drive.  Magnitude-free -- it just has to move.
            hid = [n for n in range(N) if n not in outs and n != 0 and len(inc[n])]
            if hid:
                fewest = min(len(spall[n]) for n in hid)
                for n in hid:
                    if len(spall[n]) == fewest:
                        for si in inc[n]:
                            g[si] = g[si] + KICK_GAIN * TH
        if KICK_DEAD > 0 and not all(spall[o] == T_true[o] for o in outs):
            # PER-NEURON kick.  The global test above needs max|g| == 0 on EVERY edge, so a
            # run with two dead hidden neurons and live gradients elsewhere never triggers
            # it -- 8n M sits exactly there: N3 and N4 silent with g == 0 on their inputs,
            # while N1/N2's edges are still moving, so the run looks healthy globally.
            # A hidden neuron that is SILENT and has zero gradient on every incoming edge is
            # unrecoverable by descent no matter what the rest of the network is doing: it
            # contributes nothing, so the derivative really is zero and the improvement is a
            # JUMP (measured on 8n M, w(0->4) 135 -> 190 takes the error 99.0 -> 45.8, but
            # 135 -> 136 changes nothing at all).  Kick that neuron on its own.
            for n in range(N):
                if n in out_t or n == 0 or len(inc[n]) == 0 or spall[n]:
                    continue
                if all(g[si] == 0.0 for si in inc[n]):
                    for si in inc[n]:
                        g[si] = g[si] + KICK_DEAD * TH
        if EARLY_STOP and float(np.abs(g).max()) == 0.0 and \
                all(spall[o] == T_true[o] for o in outs):
            # GENUINE FIXED POINT: the output already matches every target AND the gradient
            # is identically zero, so no further iterate can differ.  Continuing only risks
            # the drift KEEP_BEST exists to undo (the optimiser reaching err=0 and walking
            # away), and on a converged case it burns the whole remaining budget.  Tested
            # against the spall/V already computed this iteration, so it costs no extra
            # simulation.
            best_w = w.copy(); best_e = 0.0
            break
        if WREQ_GAIN > 0:
            # Scale by a FIXED reference, not by max|g|.  Scaling to the working gradient
            # makes the term vanish exactly when the ordinary gradient is frozen -- which is
            # precisely when it is needed (3n D locks at w=[3000,1205,200] with the gradient
            # at zero and one output spike permanently missing).
            wr = float(np.abs(wreq).max())
            if wr > 0:
                g = g + WREQ_GAIN * 1e-7 * wreq / wr
        # BETA1 as a knob.  The hidden edges oscillate rather than converge: on 4n G
        # seed2, w(0->1) covers 3567 units of path for 64 units of NET progress (1.8%
        # efficiency) and w(1->2) 2194 for 96 (4.4%), while the direct edge w(0->3) runs at
        # 55.7%.  That is the 71% sign accuracy showing up as zigzag, and it is why the case
        # needs 10k iterations for ~600 iterations' worth of travel.  More momentum averages
        # the reversals out.
        m = BETA1 * m + (1 - BETA1) * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - BETA1 ** ait); vh = v / (1 - 0.999 ** ait)
        if GLOBAL_NORM:  # (kept for comparison; see notes)
            # Per-parameter normalisation rescales EVERY direction to a full-size step, so
            # a weakly-determined (flat) weight moves as fast as a sharply-determined one
            # and drifts far while the outputs stay put -- the 3-cycle feedback weight ran
            # to 92-256 against a true 60 that way.  Normalising by the LARGEST second
            # moment instead keeps relative gradient scale, so flat directions move little.
            vh = np.full_like(vh, float(vh.max()))
        # decay the step: with a fixed step the fit oscillates around the solution
        # (output seen bouncing 209/219/215 about a target of 214) instead of settling
        step = lr / (1.0 + DECAY * ait)
        prop = step * mh / (np.sqrt(vh) + 1e-18)
        if TRUST > 0:
            # TRUST REGION IN SPIKE-TIME UNITS.  Bounding the step in WEIGHT units is the
            # wrong currency when directions are ill-conditioned: the same weight change
            # moves one spike by 30 steps and another not at all.  Predict each spike's
            # shift, ds = -(sum_k dw_k eps_k(s)) / (dV/dt at s), and rescale the whole step
            # so the worst predicted shift stays within TRUST steps.  Sharp directions are
            # restrained (this is what stopped the runaways) while flat ones, which barely
            # move any spike, are left free.
            worst = 0.0
            for n in range(N):
                if not spall[n] or len(inc[n]) == 0:
                    continue
                sl_tr = np.diff(vsub[n], prepend=vsub[n][0])
                for s_ in spall[n]:
                    if not (0 <= s_ < T):
                        continue
                    dv = sum(prop[si] * eps[(int(C[si, 0]), n)][s_] for si in inc[n])
                    sl = max(abs(float(sl_tr[s_])), SLOPE_FLOOR * TH)
                    worst = max(worst, abs(dv) / sl)
            if worst > TRUST:
                prop = prop * (TRUST / worst)
        upd = prop
        if USE_GN:
            # GAUSS-NEWTON where a genuine voltage RESIDUAL exists.  vsub is LINEAR in a
            # neuron's incoming weights, so the demands form a least-squares system
            # A dw ~= r with A[j,k] = eps_k(t_j) (already computed) and r_j = L_n(t_j).
            # First-order uses A^T r; Gauss-Newton preconditions by (A^T A)^-1 and lands ON
            # the minimiser instead of stepping past it, solving a neuron's incoming weights
            # JOINTLY -- the jump the discrete solver makes.
            # Restricted to OUTPUT neurons: there L is a real hinge residual in volts, so
            # driving it to zero is meaningful.  For a hidden neuron L is a timing-derived
            # descent DIRECTION of arbitrary scale; zeroing it is meaningless and blew the
            # weights to the 3000 clip from a 10% start.  Hidden neurons keep first-order.
            gn = np.zeros(len(w)); gmask = np.zeros(len(w), bool)
            for n in list(out_t):
                syn = inc[n]
                if len(syn) == 0:
                    continue
                times = np.nonzero(L[n])[0]
                if len(times) == 0:
                    continue
                A = np.stack([eps[(int(C[si, 0]), n)][times] for si in syn], axis=1)
                r = L[n][times]
                tr = max(float(np.trace(A.T @ A)), 1e-30) / len(syn)
                try:
                    gn[syn] = np.linalg.solve(A.T @ A + GN_LAM * tr * np.eye(len(syn)), A.T @ r)
                    gmask[syn] = True
                except np.linalg.LinAlgError:
                    pass
            upd = np.where(gmask, GN_ALPHA * gn, prop)
        if PROX > 0:
            # DISTANCE PENALTY, applied as a MULTIPLIER ON THE STEP rather than as an
            # opposing force.  An additive pull-back is fatal: correct solutions are exact
            # fixed points (g = 0), so the pull would be the only force there and would drag
            # the weights off the solution -- measured, even a 7% cumulative pull took the
            # suite 4/12 -> 0/12.  Multiplying the step instead means zero gradient still
            # gives zero motion, and the step direction can never be reversed, only damped.
            # The initial weights are a legitimate anchor: the true weights are known to lie
            # within 0.5-1.5x of them, so displacement beyond that is a priori suspect.
            # Penalise only displacement BEYOND the a-priori plausible band.  init is
            # true * U(0.5,1.5), so truth can legitimately sit anywhere up to 2x the init
            # (rel = 1.0) -- seed0 genuinely needs 1.85x and 1.94x moves.  A penalty that
            # starts at rel = 0 therefore damps exactly the travel that is required, and
            # measured monotonically worse: 4/12 (off), 3/12, 1/8, 0/8 as it strengthened.
            # Flat inside the band, growing outside it.
            rel = np.abs(w - w_init) / np.maximum(np.abs(w_init), 1e-9)
            excess = np.maximum(0.0, rel - PROX_BAND)
            upd = upd / (1.0 + PROX * excess ** 2)
        if BARRIER_CLAMP:
            # DO NOT CROSS THE SINGLE-SPIKE THRESHOLD IN ONE STEP.
            # w_crit = th/max(HK) is the weight below which ONE presynaptic spike can no
            # longer fire the neuron.  Above it the neuron fires once per input spike;
            # below it it must accumulate, so crossing does not DELAY a spike, it DELETES
            # one -- and the resulting state has no gradient at all, so it is absorbing.
            # Measured on 4n G seeds 4 and 5: both START with N2 firing the correct 2
            # spikes (just ~46 steps early), the demand correctly says "fire later" =>
            # lower w(1->2), and that walks 506->420 / 654->411 straight through 444.5.
            # N2 collapses to one spike and the run freezes at it301 with g = [0,0,0,0].
            # Clamp the step at the barrier instead; the weight rests ON w_crit and the
            # other weights are free to move, which is what the timing demand actually
            # needed.  Only blocks a DOWNWARD crossing from above -- a weight already below
            # w_crit (a legitimate accumulator, e.g. every true w(0->1) in the suite) is
            # untouched.
            # Land just ABOVE the barrier, not exactly on it: clamping to W_CRIT itself
            # makes the next iteration's `w > W_CRIT` test false, so the following step
            # walks straight through and the guard never fires twice (measured: w(1->2)
            # still ended at 422 with the clamp "on").
            _cross = (w >= W_CRIT) & (w + upd < W_CRIT)
            if np.any(_cross):
                upd = np.where(_cross, W_CRIT * 1.002 - w, upd)
        upd[frozen] = 0.0
        if cb is not None: cb(it, w, upd, g, spall, vsub, L)
        # SIGN-PRESERVING CLAMP.  A synapse is excitatory or inhibitory and does not change
        # type, so the sign is part of the problem statement.  np.clip(w, 20, 3000) made an
        # inhibitory weight unrepresentable: on 4n V, 3n R and 3n L every seed initialises
        # correctly negative and the FIRST clip snapped it to +20, so all three scored 0/8
        # with the inhibitory weight pinned at the floor.  Clamp the MAGNITUDE instead.
        w = _wsign * np.clip(_wsign * (w + upd), 20, 3000)
        if KEEP_BEST and it % KEEP_EVERY == 0:
            # The gradient does NOT vanish at a correct solution -- the discretisation
            # residual keeps pushing -- so the optimiser reaches err=0 and then WALKS AWAY
            # (3-cycle seed0: 99 -> 5.5 -> 0.5 -> 0.0 -> 1.0 -> 4.0 ...).  The output error
            # is observable from the given targets, so simply retain the best iterate.
            e_cur = out_err()
            if e_cur < best_e:
                best_e = e_cur; best_w = w.copy()
        if FREEZE_RUNAWAY:  # (heuristic; see notes)
            whist.append(w.copy())
            if it % RUN_WIN == 0 and len(whist) > RUN_WIN:
                H = np.array(whist[-RUN_WIN:]); st = np.diff(H, axis=0)
                net = np.abs(H[-1] - H[0]); tvv = np.abs(st).sum(axis=0)
                ratio = net / np.maximum(tvv, 1e-12)
                e_now = out_err()
                if prev_err is not None and e_now >= prev_err - 1e-9:
                    frozen |= (ratio > RUN_THRESH) & (tvv > 1.0)
                prev_err = e_now
    return best_w if (KEEP_BEST and best_e < np.inf) else w


def run(name, C, N, outs, w_true, seeds=4, rounds=150, steps=520, verbose=False, lr=None):
    params = mkparams(steps)
    C = np.array(C, np.int32); w_true = np.array(w_true, np.float32)
    tv = fsim(C, N, w_true, params); T_true = {n: sp(tv, n) for n in range(N)}
    ok = 0; last = None
    for seed in range(seeds):
        w = (w_true * np.random.default_rng(seed).uniform(0.5, 1.5, len(w_true))).astype(float)
        w = train(C, N, outs, w, T_true, params, rounds, lr=LR if lr is None else lr)
        V = fsim(C, N, w, params)
        ok += all(sp(V, o) == T_true[o] for o in outs)
        last = {n: sp(V, n) for n in range(N)}
    print(f"{name}: recovered {ok}/{seeds}")
    if verbose:
        for n in range(N):
            print(f"    N{n}: found {last[n]}   true {T_true[n]}")
    return ok


def main():
    print(f"BACKWARD TRACE method (SIG={SIG}, LR={LR}, SWEEPS={SWEEPS})\n")
    print("the four minimal cases that broke the timing method:")
    run("  over-fire  (N1 must stay silent)", [[0, 1], [0, 2], [1, 2], [1, 3]], 4, [2, 3],
        [150., 500., 50., 500.])
    run("  veto       (N1 must fire)", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 50.])
    run("  over-demand(hidden fires less)", [[0, 1], [1, 2], [0, 2]], 3, [2], [250., 700., 300.])
    run("  coincidence(two co-drivers)", [[0, 1], [0, 2], [1, 3], [2, 3]], 4, [3],
        [500., 300., 350., 400.], verbose=True)
    print("\nfeed-forward + recurrent regressions:")
    run("  BREAK divergent", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 900., 470.])
    run("  chain", [[0, 1], [1, 2], [2, 3]], 4, [3], [500., 500., 500.])
    run("  fanout equal", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 500.])
    run("  fanout hard", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 200.])
    run("  3-cycle", [[0, 1], [1, 2], [2, 3], [3, 1]], 4, [3], [500., 500., 500., 60.])
    run("  2-cycle", [[0, 1], [1, 2], [2, 1], [2, 3]], 4, [3], [500., 500., 60., 500.])


if __name__ == "__main__":
    main()
