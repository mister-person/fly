"""Weight recovery by a per-neuron DEMAND FIELD -- a separate pathway from grad_trace.

WHY A REWRITE.  grad_trace's backward relaxation and this field grew in the same file, and
the relaxation dominates: measured on 4n F, the field contributed -2.33e-04 to L[2] while
the final L[2] carried -4.74e-01, ~2000x larger.  Every field number ever recorded there
(FIELD_XING, FIELD_ADD, FIELD_BASE, FIELD_MIST) was therefore an ADDITION to a signal that
swamped it, and measured how much perturbation the relaxation tolerates rather than what the
field can do.  The two cannot be compared while they share a gradient.

WHAT IS SHARED.  Only the physics -- the simulator, the PSP kernel, the eligibility
accumulator, the constants.  None of the method: no relaxation, no occlusion rewrite, no
TIM/PIVOT/SHARP/MOVE terms, no hidden-target inference, no kick.

THE METHOD, in three lines that recurse:

    bumps(o)  = the known target times, for an output
    L[n]      = local_demand(bumps(n), spikes(n))          -- signed, and PURELY LOCAL
    F[n][q]   = sum_d sum_t L[d][t] * K(q -> t | d)        -- backward through plausibility
    bumps(n)  = the positive peaks of F[n]

`local_demand` compares two spike trains on the SAME neuron: what the field asks for against
what the neuron does.  Nothing downstream is matched -- that was the source of the worst bug
in the old path, where a spike 72 steps from its demand fell outside MATCH_WIN=60 and was
reclassified from "late" to "unwanted", asking to delete a neuron's only spike.

K is the PLAUSIBILITY that a spike at q produces a crossing at t given the CURRENT weight:
feasible timing, arrival not swallowed by a refractory shadow, and the weight that would be
needed close to the weight already present.  Everything is measured against PHYSICAL resets
(the neuron's actual spikes); the counterfactual target-epoch structure that grad_trace needs
for its hinge is wrong here, and importing it carried a real bug (EPOCH_EXTEND shares a
boundary between adjacent epochs, so widening a starved one deletes its predecessor).
"""
import os
import numpy as np

# ---- physics only ------------------------------------------------------------------
from grad_trace import (fsim, sp, mkparams, eligibility,
                        TH, HK, KWIN, DELAY_ITERS, REFRAC_ITERS, W_CRIT)

# ---- configuration -----------------------------------------------------------------
LR = float(os.environ.get("F_LR", "10.0"))
BETA1 = float(os.environ.get("F_BETA1", "0.9"))
DECAY = float(os.environ.get("F_DECAY", "0.05"))
RESTART_EVERY = int(os.environ.get("F_RESTART", "100"))
TRUST = float(os.environ.get("F_TRUST", "2.0"))
WMAG_MIN = float(os.environ.get("F_WMAG_MIN", "20"))    # clamp on |w|; the SIGN is fixed
WMAG_MAX = float(os.environ.get("F_WMAG_MAX", "3000"))  # per weight at its initial value.
# A synapse is excitatory or inhibitory and does not change type, so the sign is given
# rather than learned.  Two bugs at once from not doing this: with a hard [20, 3000] clamp
# an INHIBITORY edge was unrepresentable (3n L's true w12 is -700, every seed initialises
# correctly negative at -378..-920 and the first clip snapped it to +20; grad_trace has the
# same clamp at grad_trace.py:2106 and also scores 3n L 0/8, so that case has never been
# solvable by either pathway).  Opening the clamp to [-3000, 3000] instead let a POSITIVE
# weight collapse through zero and sit there -- over-demand seed 0 ended with w12 = -2
# against a true +700.  Fixing the sign and clamping the magnitude rules out both.
WFLOOR = float(os.environ.get("F_WFLOOR", "0.05")) # |w| floor for the plausibility ratio,
# as a share of W_CRIT: the score divides by |w|, so a weight passing through zero would
# otherwise send the mismatch to infinity and silence the field exactly at the crossing.

SWEEPS = int(os.environ.get("F_SWEEPS", "3"))      # so demand can travel a cycle
TOL = float(os.environ.get("F_TOL", "1.0"))        # weight mismatch that halves plausibility
GAP = int(os.environ.get("F_GAP", "1"))            # gap that separates two urgency bumps
DEAD_ZONE = float(os.environ.get("F_DEAD", "0"))   # timing error tolerated before responding.
# MUST DEFAULT TO ZERO.  Recovery is judged on EXACT spike times, so any dead zone is a
# basin of states the field calls finished and the scorer calls wrong.  At 5 (carried over
# from grad_trace, where it suits a hinge) 3n D halted at N2 = [38,138,238,289,338,438]
# against targets [33,133,233,293,333,433] -- every spike present, each ~5 steps off, field
# empty, gradient identically zero.  Suite: 0/52 at 5, 1/52 at 2, 3/52 at 1, 16/52 at 0.
GRADE = float(os.environ.get("F_GRADE", "30"))     # error giving a full-strength response
SUPP = float(os.environ.get("F_SUPP", "1.0"))      # depth of suppression
SUPP_SCALE = int(os.environ.get("F_SUPP_SCALE", "1"))  # 1: scale it by the field, 0: by TH
SUB_EPS = int(os.environ.get("F_SUB_EPS", "1"))    # fractional arrivals inside eps
CENTROID = int(os.environ.get("F_CENTROID", "1"))  # bump position = centroid, not argmax
EDGE_SIGN = int(os.environ.get("F_EDGE_SIGN", "1"))  # demand flips sign along an INHIBITORY edge
INH_MAG = int(os.environ.get("F_INH_MAG", "1"))      # judge an inhibitory edge on |w|, not w
INH_REACH = int(os.environ.get("F_INH_REACH", "0"))  # let a net-inhibited time still be requestable
CLIFF_HOLD = int(os.environ.get("F_CLIFF_HOLD", "0"))  # pin the weight that broke a count,
CLIFF_ALL = int(os.environ.get("F_CLIFF_ALL", "0"))    # ...or every weight into that neuron
GUARD = float(os.environ.get("F_GUARD", "0"))        # veto a step that would CREATE a spike
GUARD_WIN = int(os.environ.get("F_GUARD_WIN", "25")) # ignore peaks this near a wanted/actual spike
MARGIN = float(os.environ.get("F_MARGIN", "0"))      # push down NEAR-MISSES that should not fire
MARGIN_FRAC = float(os.environ.get("F_MARGIN_FRAC", "0.8"))  # a peak this high counts as a near-miss
MARGIN_WIN = int(os.environ.get("F_MARGIN_WIN", "25"))       # ignore peaks this near a wanted/actual spike
SUPP_SUPPORT = int(os.environ.get("F_SUPP_SUPPORT", "1"))  # only suppress where the field speaks
SUPP_WIN = int(os.environ.get("F_SUPP_WIN", "30"))         # half-width of "speaks here"
# DENSITY MODE: no bump extraction at all on hidden neurons.  The demand IS the field --
# every positive timestep is a request weighted by its height, every negative one a
# suppression.  bumps_of()'s docstring warns that a raw density "multiplied spikes instead of
# placing them", but that predates the field rewrite AND the measured failure is now the
# opposite (a plateau asking for 2 spikes where 9 are wanted), so the old evidence does not
# carry over.  1 = raw height, 2 = each run rescaled to carry the mass of a single bump
# (its peak), which keeps the old magnitude while spreading it over the run.
DENSITY = int(os.environ.get("F_DENSITY", "0"))
DENSITY_PTS = int(os.environ.get("F_DENSITY_PTS", "9"))   # <=0 keeps the field fully dense
# Batched backward propagation (exact; verified by _verify_prop.py).  Defaults ON whenever
# DENSITY is, because that is where it matters: fully dense propagation costs 281.7s per 100
# rounds through the per-request loop and 23.5s batched, a 12x speedup that is what makes
# "every positive timestep is a request" affordable at all.  Left OFF by default on the bump
# path, where there are only ~5 requests per neuron and batching is a slight net loss (4.6s
# against 4.2s).
FASTPROP = int(os.environ.get("F_FASTPROP", "1" if DENSITY else "0"))
NO_OCC_MASK = int(os.environ.get("F_NO_OCC_MASK", "0"))   # bypass the mask, for verification
PAIR_ORDER = int(os.environ.get("F_PAIR_ORDER", "0"))  # 0 greedy-nearest (as before),
# 1 ordered at OUTPUTS only, 2 ordered everywhere
SHIFT_MODE = int(os.environ.get("F_SHIFT_MODE", "0"))  # 0 ordered pairing, 1 nearest,
# 2 SLOPE (no pairing at all -- immune to a spurious spike shifting every later match)
SHIFT_ORDER = int(os.environ.get("F_SHIFT_ORDER", "1"))  # pair spikes to peaks IN ORDER
# rather than by nearest, so a phase offset over half an ISI cannot invert the sign
SHIFT = float(os.environ.get("F_SHIFT", "0"))  # timing correction signed by WHICH SIDE of
# the spike its nearest field peak lies on, applied at the spike; density kept as a level term
PUSH_ONE = int(os.environ.get("F_PUSH_ONE", "0"))  # early spike -> negative ONLY,
# late spike -> positive ONLY, instead of both
PUSH = float(os.environ.get("F_PUSH", "0"))  # push every spike toward its nearest FIELD
# PEAK, scaled by the field height there -- a move term derived from the density itself
WIDE_EPS = int(os.environ.get("F_WIDE_EPS", "0"))  # hidden-neuron eligibility not truncated
# at the neuron's own spikes -- an upper bound on what widening eligibility support can buy
DPOW = float(os.environ.get("F_DPOW", "1.0"))
DPOW_C = float(os.environ.get("F_DPOW_C", os.environ.get("F_DPOW", "1.0")))  # same, for the
# CREATE channel; defaults to DPOW so a single knob still behaves as before  # exponent on demand AMPLITUDE.
# 1 = as-is, 0 = sign and extent only (magnitude discarded), intermediate = partial flattening
KNORM = int(os.environ.get("F_KNORM", "0"))  # 1: backward message normalised to unit MASS
# (k/sum) rather than unit PEAK (k/max), so demand is not amplified by the window width
GRADED_ERR = int(os.environ.get("F_GRADED_ERR", "0"))  # KEEP_BEST scores |Dcount|, not a
# flat 99 -- without it a single-output case that never matches its count freezes at round 1
CMASK = int(os.environ.get("F_CMASK", "0"))   # mask the CREATE channel inside a spike's
# refractory footprint, so CREATE_FLOOR only protects genuinely unserved requests
SLOPE = float(os.environ.get("F_SLOPE", "0"))     # move a matched spike UP THE FIELD SLOPE
SLOPE_WIN = int(os.environ.get("F_SLOPE_WIN", "5"))  # half-width of the slope estimate
SLOPE_NORM = int(os.environ.get("F_SLOPE_NORM", "2"))  # 0 raw, 1 per-neuron, 2 sign*u*amt, 3 sign*u
PEAK_FRAC = float(os.environ.get("F_PEAK_FRAC", "0"))  # average only the part of a run
# at or above this share of its peak.  0 = plain centroid, 1 = argmax.  A narrow peak
# sitting on a broad plateau is averaged away otherwise: on over-demand N1 the run
# 222..380 peaks at 376 (true spike 373) on a 159-sample plateau at 56% of peak, and the
# centroid lands at 303 -- 73 samples off, with F^8 weighting still only reaching 345.
SPREAD = int(os.environ.get("F_SPREAD", "0"))      # propagate a request over its WHOLE run
PAIR_WIN = int(os.environ.get("F_PAIR_WIN", "0"))  # a paired target ignores ITS OWN spike's reset
PAIR_EPS = int(os.environ.get("F_PAIR_EPS", "0"))  # ...and so does the ELIGIBILITY at that target.
# OFF: 65 -> 23/104 alone, 13/104 stacked with PAIR_WIN.  The counterfactual is only
# self-consistent for ONE spike at a time -- it removes the reset of the spike being moved
# while every other spike stays as-is, so on a neuron with several spikes the resulting
# eps is not the eligibility of any coherent configuration.  Damage scales with spike
# count: 3n D, 3n E and 4n G all go 8/8 -> 0/8.
SPREAD_PTS = int(os.environ.get("F_SPREAD_PTS", "9"))  # sample points per run
SUBSAMPLE = int(os.environ.get("F_SUBSAMPLE", "0"))  # fractional threshold-crossing times.
# OFF by default: it improves gradient ALIGNMENT on over-demand's descent path (28.9% ->
# 36.4% of steps pointing toward truth) and still costs the suite 59 -> 57/104, almost all
# of it 3n D collapsing 8/8 -> 3/8.  Alignment on one path does not predict recovery.
GN = int(os.environ.get("F_GN", "0"))              # joint solve for an OUTPUT's inputs.
# OFF: it costs 59 -> 49/104.  The solve assumes a neuron's residuals are jointly
# satisfiable BY THAT NEURON'S INCOMING WEIGHTS, and systematically they are not -- on
# over-demand, N2's third spike is late because N1 is late, and w01 is not in N2's system,
# so no (w12, w02) fixes it.  First order makes a small compromise; GN confidently solves
# an unsatisfiable system and takes a large wrong step.  Damage lands on the multi-hop
# cases (chain 8->6, 4n F 8->6, 4n G 8->4) and the one gain on 5n H (6->8), the pure
# feedforward chain where each output spike has a clean attribution.
GN_RIDGE = float(os.environ.get("F_GN_RIDGE", "1e-3"))
CREATE_FLOOR = int(os.environ.get("F_CREATE_FLOOR", "1"))  # a weight a MISSING spike needs
# increased must not come out decreased -- see gradient()
OCCLUDE = int(os.environ.get("F_OCCLUDE", "1"))    # an arrival that is swallowed earns nothing
KEEP_BEST = int(os.environ.get("F_KEEP_BEST", "1"))


def sp_frac(V, n):
    """Sub-sample threshold-crossing times: where V ACTUALLY reached TH, between samples.

    sp() reports the integer sample at which V first reaches threshold, but the crossing
    happens somewhere inside the preceding step, and linear interpolation on the two
    bracketing voltages recovers it.  This is not a smoothing heuristic -- the fractional
    time is a quantity the simulator already determines and the integer report discards.

    It matters because every timing quantity downstream is otherwise a STAIRCASE in the
    weights: as w moves continuously the true crossing slides continuously while the integer
    jumps, so the demand is piecewise-constant and its gradient carries no information about
    where inside a plateau the state sits.  Measured on over-demand along the straight line
    from a stuck point to the true weights -- a path whose output error descends
    monotonically 4.0 -> 0.0 with no wrong-count region anywhere -- the projected gradient
    was NEGATIVE on 71% of it, flipping sign at every plateau boundary.
    """
    v = np.asarray(V[:, n], float)
    out = []
    for t in np.nonzero(v >= TH)[0]:
        t = int(t)
        if t == 0:
            out.append(0.0); continue
        v0, v1 = v[t - 1], v[t]
        out.append((t - 1) + (TH - v0) / (v1 - v0) if v1 > v0 else float(t))
    return out


def elig_frac(spikes_f, T, resets_f, refrac_f):
    """eps(t) with FRACTIONAL presynaptic arrivals -- same quantity, continuous in w.

    grad_trace's eligibility lays HK[0:] down starting at the integer spike sample, so eps
    is a step function of the weights: it changes only when an integer spike time moves, and
    never in between.  Measured on over-demand along the descent path, eps changed on exactly
    23 of 240 steps -- the same 23 on which an integer spike time moved -- and ALL 12 large
    jumps in the gradient landed on them.  Sub-sampling the demand alone therefore bought
    nothing, because the demand is MULTIPLIED by this.

    HK is a smooth kernel sampled at integer offsets, so h(t - q) for fractional q is a
    linear interpolation between its neighbours.  Nothing is invented: the fractional
    crossing is already determined by the simulator.

    The reset and refractory boundaries are left HARD.  An arrival crossing a reset has its
    charge wiped and one landing in a refractory shadow is discarded outright -- those are
    genuine step functions in the dynamics, not artefacts of integer reporting, and
    smoothing them would make eps continuous but no longer the derivative of anything the
    simulator does.
    """
    e = np.zeros(T)
    R = sorted(resets_f)
    RF = sorted(refrac_f) if refrac_f is not None else R
    tt = np.arange(T, dtype=float)
    for q in spikes_f:
        arr = q + DELAY_ITERS
        if RF and any(r < arr < r + REFRAC_ITERS for r in RF):
            continue
        nxt = next((r for r in R if r > q), float(T))
        lo = int(np.ceil(q))
        hi = int(min(q + KWIN, np.floor(nxt) + 1, T))
        if hi <= lo:
            continue
        dt = tt[lo:hi] - q                      # fractional offsets
        i0 = np.floor(dt).astype(int)
        fr = dt - i0
        i1 = np.minimum(i0 + 1, KWIN - 1)
        ok = (i0 >= 0) & (i0 < KWIN)
        e[lo:hi] += np.where(ok, HK[np.clip(i0, 0, KWIN - 1)] * (1 - fr)
                             + HK[i1] * fr, 0.0)
    return e


def wlabels(C):
    """Name each weight by the edge it sits on: w01 for N0->N1, w12 for N1->N2, ...

    Positional names (w0, w1, w2) say nothing about WHICH synapse is meant and have to be
    cross-referenced against the edge list every time; wIJ is self-describing.  An
    underscore is inserted when an index reaches two digits so w1_10 stays unambiguous.
    """
    out = []
    for si in range(len(C)):
        a, b = int(C[si, 0]), int(C[si, 1])
        sep = "_" if (a > 9 or b > 9) else ""
        out.append(f"w{a}{sep}{b}")
    return out


def wstr(C, w, fmt="{:.0f}"):
    """'w01=239  w12=1583  ...' -- weights always shown WITH their edge."""
    return "  ".join(f"{lb}={fmt.format(float(x))}"
                     for lb, x in zip(wlabels(C), np.asarray(w, float)))


# ---- the local rule ----------------------------------------------------------------
def _spread_mass(dst, src):
    """Add `src` to `dst` keeping each run's TOTAL but only DENSITY_PTS samples of it.

    The upstream loop costs one plausibility solve per nonzero sample, so a dense field is
    expensive to propagate even batched.  Mass is what the upstream reads ("this span is
    wanted, this much in total"), so thinning the samples while preserving the sum keeps the
    message.  DENSITY_PTS <= 0 keeps every sample.
    """
    for sgn in (1.0, -1.0):
        m = np.nonzero(sgn * src > 0)[0]
        if not len(m):
            continue
        for r in np.split(m, np.nonzero(np.diff(m) > GAP)[0] + 1):
            if not len(r):
                continue
            idx = r if (DENSITY_PTS <= 0 or len(r) <= DENSITY_PTS) else \
                r[np.linspace(0, len(r) - 1, DENSITY_PTS).round().astype(int)]
            tot = float(src[r].sum())
            wgt = np.abs(src[idx])
            sw = float(wgt.sum())
            dst[idx] += tot * wgt / sw if sw > 0 else tot / len(idx)


def local_demand(bumps, spikes, T, field=None, spikes_f=None, volt=None, fieldc=None):
    """Signed demand for ONE neuron: what it is asked for, against what it does.

    `bumps` is [(time, strength)] -- the spikes this neuron is wanted to produce.  Pairing
    to its actual spikes is greedy-nearest, one-to-one, and has NO distance cutoff: a spike
    is surplus only once every bump has been claimed.  With one spike against five bumps
    there is nothing to delete, so a far-off spike is mistimed by definition.

        bump unclaimed   -> CREATE    positive at the bump
        spike unclaimed  -> SUPPRESS  negative at the spike
        paired           -> MOVE      positive at the bump, scaled by the error

    Only an EARLY spike also takes a negative.  Suppression removes drive, which moves a
    spike LATER: right for an early one, wrong for a late one, which is corrected by the
    positive at its bump (necessarily earlier than it).  Applying it in both directions
    inverts the gradient outright, because eligibility grows through an epoch and the lone
    negative at a late spike outweighs every positive before it.
    """
    L = np.zeros(T)
    Lc = np.zeros(T)          # the CREATE part alone
    Lp = np.zeros(T); Lpc = np.zeros(T)   # the same, SPREAD over each run
    pairof = {}               # target time -> the spike it is paired with
    sp_ = [int(q) for q in spikes if 0 <= q < T]
    # fractional time per spike, for MEASURING error; the integer stays the index
    ff = {int(q): float(fq) for q, fq in zip(spikes, spikes_f)} if spikes_f \
        else {int(q): float(q) for q in spikes}
    bm = []
    for _b in bumps:                      # (time, strength[, run])
        _q = int(_b[0])
        if 0 <= _q < T:
            bm.append((_q, float(_b[1]), _b[2] if len(_b) > 2 else None))
    # SILENCE IS NOT AN INSTRUCTION.  A neuron whose field is identically zero has received
    # no demand at all -- nothing downstream asked anything of it.  Treating that as "you
    # should not be firing" made truth a non-fixed-point: at the true weights of 4n F the
    # output matches its targets, so L[output] = 0, so the field of the neuron feeding it is
    # empty, and its perfectly placed spike was suppressed at -7.0e-03 with a gradient
    # flowing backwards from it.  No information means no update.
    if not bm and (field is None or not np.asarray(field).any()):
        return L, Lc, Lp, Lpc, pairof
    # MODES 1/2 ONLY.  This gate read `if DENSITY` and so captured every nonzero mode:
    # 3-6 returned here and never reached the bumps-for-timing code below, which made them
    # silent aliases of mode 1 (plus, before it was fixed, a broken output create rule).
    if DENSITY in (1, 2) and field is not None:
        # The demand is the field itself.  No pairing, no bump times, no suppression derived
        # from "which spike went unclaimed" -- height alone says what is wanted where, and a
        # wide positive run therefore carries proportionally more create mass, which is
        # exactly the count the run stands for.  Outputs are untouched (field is None for
        # them): they have real targets and need no inference.
        f = np.asarray(field, float).copy()
        if DENSITY == 2:
            # rescale each run to the mass of one bump, so the change is PURELY about where
            # the demand sits, not how much of it there is
            pos = np.nonzero(f > 0)[0]
            if len(pos):
                for r in np.split(pos, np.nonzero(np.diff(pos) > GAP)[0] + 1):
                    if len(r):
                        m = float(f[r].sum())
                        if m > 0:
                            f[r] *= float(f[r].max()) / m
        # HOW MUCH vs WHETHER, PER CHANNEL.  Measured: demand AMPLITUDE carries COUNT
        # information and demand SIGN+EXTENT carries TIMING information -- at DPOW=0 the dev
        # subset gives |dt| 14.59 against 17.82, while |Dcount| goes 0.00 -> 0.09.  A single
        # global exponent has to pick one.  DPOW governs the MAIN demand (which drives
        # placement) and DPOW_C the CREATE channel (which drives the count), so each can take
        # the form that suits what it is for.
        def _flatten(v, e):
            if e == 1.0:
                return v
            m = float(np.abs(v).max())
            return v if m <= 0 else np.sign(v) * m * (np.abs(v) / m) ** e
        L[:] = _flatten(f, DPOW)
        _c = np.maximum(_flatten(f, DPOW_C), 0.0)
        if CMASK:
            # A POSITIVE FIELD WHERE THE NEURON ALREADY FIRES IS NOT A CREATE.  Lc feeds
            # CREATE_FLOOR (g = max(g, gc)), which exists so a genuinely missing spike cannot
            # be cancelled by an unrelated timing correction.  Unmasked, a dense field makes
            # gc > 0 on nearly every weight, so the floor fires everywhere and overrides
            # suppression: on 4n F the correct g[w01] = -9.67e-07 became +8.68e-08 and w01
            # ran to 2407 against a true 240, with N1 firing 5 times against a true 1.
            # Modes 3-6 already mask this; modes 1/2 did not.
            for _f in sp_:
                _c[max(0, _f - REFRAC_ITERS):min(T, _f + REFRAC_ITERS + 1)] = 0.0
        Lc[:] = _c
        # PROPAGATION IS SUBSAMPLED, and it has to be.  The upstream loop runs a plausibility
        # solve per NONZERO timestep per edge; a dense field turns ~5 requests per neuron into
        # ~900, which is a 100x+ slowdown per round.  So Lp keeps SPREAD_PTS points per run
        # with the run's TOTAL MASS preserved -- the upstream still sees "this whole span is
        # wanted, this much in total", which is the part that matters, at bounded cost.
        if SHIFT > 0:
            # DIRECTION FROM THE SPIKE'S POSITION RELATIVE TO WHERE IT IS WANTED.
            # More drive makes a neuron fire EARLIER, so the sign of a timing correction
            # cannot come from the field's sign at the request -- it has to come from which
            # SIDE of the spike the request is on:
            #     spike EARLY (wanted later)  -> LESS drive  -> negative
            #     spike LATE  (wanted earlier)-> MORE drive  -> positive
            # The demand goes AT THE SPIKE, not at the peak: that is where the eligibility is
            # guaranteed nonzero, and where changing the drive actually moves this spike.
            # Measured motivation, chain seed 4: |dt| falls 55 -> 29 by LOWERING w01 while the
            # dense demand (6:1 positive, every request later than every spike) gives
            # g[w01] = +1.18e-07 and raises it.
            # The raw density is DELIBERATELY LEFT IN PLACE alongside this.  Nearest-peak
            # matching is unreliable when the spike COUNT is wrong -- a spike can be matched
            # to a peak meant for a different one -- and the density's own level term still
            # says "more wanted here / less wanted there" regardless of any pairing, so it
            # counteracts a bad match instead of compounding it.
            v = np.asarray(field, float)
            pk = np.nonzero((v[1:-1] > 0) & (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1
            if SHIFT_MODE == 2:
                # SLOPE MODE -- NO MATCHING AT ALL.  Any spike-to-peak pairing is fragile:
                # ordered pairing is thrown off by a single spurious spike (every later index
                # shifts), and nearest pairing inverts the sign once the phase offset exceeds
                # half an ISI.  The local SLOPE of the field carries the same information
                # without pairing anything:
                #     field RISING at the spike  -> wanted later  -> less drive -> negative
                #     field FALLING              -> wanted earlier-> more drive -> positive
                # Direction from the slope, MAGNITUDE from the field height nearby -- dF/dt
                # decays sharply with depth (|L| falls ~100x per hop on chain) while the field
                # height does not, so a slope-scaled term vanishes upstream.
                for _f in sp_:
                    a, b = max(0, _f - SLOPE_WIN), min(T - 1, _f + SLOPE_WIN)
                    if b <= a:
                        continue
                    d = float(v[b]) - float(v[a])
                    if d == 0.0:
                        continue
                    u = float(np.abs(v[a:b + 1]).max())
                    amt = SHIFT * u * (-1.0 if d > 0 else 1.0)
                    L[_f] += amt
                    Lp[_f] += amt
            elif len(pk):
                # ORDERED, NOT NEAREST.  Nearest-peak matching inverts the sign whenever the
                # phase offset exceeds half the inter-spike interval: on chain seed 4 the
                # peaks are [122,218,318,418] against spikes [46,146,246,346,446], every
                # spike genuinely 26 EARLY, and nearest pairs 146 with 122 (24 away, against
                # 218's 72) so four of five are judged LATE and take the wrong sign.  Both
                # sequences are ordered, so pair them in order; surplus spikes take no shift
                # term and are left to the density.
                pairs = ([(int(_f), int(_q)) for _f, _q in zip(sp_, pk)] if SHIFT_ORDER
                         else [(int(_f), int(pk[int(np.argmin(np.abs(pk - _f)))])) for _f in sp_])
                for _f, q in pairs:
                    err = abs(q - ff.get(_f, float(_f)))
                    if err <= DEAD_ZONE:
                        continue
                    amt = min(1.0, (err - DEAD_ZONE) / max(GRADE, 1e-9)) * SHIFT * float(v[q])
                    d = -amt if _f < q else amt
                    L[_f] += d
                    Lp[_f] += d
                    pairof[q] = _f
        if PUSH > 0:
            # PUSH EACH SPIKE TOWARD ITS NEAREST FIELD PEAK.  A density says how much is
            # wanted WHERE, but nothing in it says "this spike belongs over there" -- which is
            # why DENSITY fixes counts and loses timing (|dt| 7.6 -> 17.8 on the dev subset).
            # bumps_of supplies that by extracting one request per run and pairing greedily;
            # this instead lets every spike find its own nearest LOCAL MAXIMUM of the field, so
            # a wide run with several maxima can attract several spikes rather than collapsing
            # to one request.  Strength comes from the field height AT THE PEAK (how much a
            # spike is wanted there) times the usual distance grading.
            v = np.asarray(field, float)
            pk = np.nonzero((v[1:-1] > 0) & (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1
            if SHIFT_MODE == 2:
                # SLOPE MODE -- NO MATCHING AT ALL.  Any spike-to-peak pairing is fragile:
                # ordered pairing is thrown off by a single spurious spike (every later index
                # shifts), and nearest pairing inverts the sign once the phase offset exceeds
                # half an ISI.  The local SLOPE of the field carries the same information
                # without pairing anything:
                #     field RISING at the spike  -> wanted later  -> less drive -> negative
                #     field FALLING              -> wanted earlier-> more drive -> positive
                # Direction from the slope, MAGNITUDE from the field height nearby -- dF/dt
                # decays sharply with depth (|L| falls ~100x per hop on chain) while the field
                # height does not, so a slope-scaled term vanishes upstream.
                for _f in sp_:
                    a, b = max(0, _f - SLOPE_WIN), min(T - 1, _f + SLOPE_WIN)
                    if b <= a:
                        continue
                    d = float(v[b]) - float(v[a])
                    if d == 0.0:
                        continue
                    u = float(np.abs(v[a:b + 1]).max())
                    amt = SHIFT * u * (-1.0 if d > 0 else 1.0)
                    L[_f] += amt
                    Lp[_f] += amt
            elif len(pk):
                for _f in sp_:
                    q = int(pk[int(np.argmin(np.abs(pk - _f)))])
                    err = abs(q - ff.get(_f, float(_f)))
                    if err <= DEAD_ZONE:
                        continue
                    amt = min(1.0, (err - DEAD_ZONE) / max(GRADE, 1e-9)) * PUSH
                    u = float(v[q])
                    pairof[q] = _f
                    if PUSH_ONE and _f < q:
                        # ONE-SIDED.  Raising a weight makes a neuron fire EARLIER, so a
                        # positive demand at a LATER peak pushes the weight the wrong way.
                        # Measured on chain seed 4: |dt| falls 55 -> 29 by LOWERING w01, yet
                        # g[w01] = +1.18e-07 -- the positive at the later peak outvoted the
                        # negative at the early spike.  An early spike needs less drive and
                        # nothing else; a late spike needs more, at its earlier peak.
                        L[_f] -= amt * SUPP * u
                        Lp[_f] -= amt * SUPP * u
                    else:
                        L[q] += amt * u
                        Lp[q] += amt * u
                        if not PUSH_ONE and _f < q:
                            L[_f] -= amt * SUPP * u
                            Lp[_f] -= amt * SUPP * u
        _spread_mass(Lp, np.asarray(L, float)); _spread_mass(Lpc, _c)
        return L, Lc, Lp, Lpc, pairof
    # STRENGTH IS THE MESSAGE, ON BOTH SIGNS.  A bump carries its own amplitude forward
    # (create and move both scale by u), but suppression was pinned to SUPP*TH regardless of
    # how faint the field was -- so a neuron whose field is numerical dust still emitted a
    # FULL-SIZE negative, and a weak request could never outweigh it.  Scale the negative by
    # the same thing the positives are scaled by: the strongest spike this neuron is being
    # asked for.  At an output the bumps carry TH, so outputs are unchanged.
    scale = max((u for _, u, _r in bm), default=0.0) if SUPP_SCALE else TH
    pairs, claimed, used = [], set(), set()
    def _lay(i, amt, dst):
        # SPREAD A REQUEST OVER ITS WHOLE RUN for the upstream, keeping the TOTAL fixed.
        # Collapsing a run to one sample makes that single t choose the epoch bound, the
        # candidate set and other[t] for the entire upstream neuron, so a one-sample error
        # in the request can shift `lo` past a reset and change which candidates exist at
        # all -- a discontinuity with no small-error regime.  Spread, the contribution is a
        # weighted average over the run and moves continuously.  The COUNT survives because
        # it lives in the mass, not the position, which is why the sharp L is kept for the
        # LOCAL gradient (a bare density there is satisfied by firing many times).
        q, u, rr = bm[i]
        if not SPREAD or rr is None or len(rr) < 2:
            dst[q] += amt * u
            return
        idx = rr if len(rr) <= SPREAD_PTS else \
            rr[np.linspace(0, len(rr) - 1, SPREAD_PTS).round().astype(int)]
        wgt = np.maximum(field[idx], 0.0) if field is not None else np.ones(len(idx))
        tot = float(wgt.sum())
        if tot <= 0:
            dst[q] += amt * u
            return
        dst[idx] += amt * u * wgt / tot

    if PAIR_ORDER and (field is None or PAIR_ORDER == 2):
        # ORDERED PAIRING.  Greedy-nearest inverts early/late whenever the timing error
        # exceeds HALF an inter-spike interval: on chain seed 4 the output fires
        # [159,259,359,459] against targets [214,314,414,514] -- uniformly 55 EARLY, ISI 100 --
        # and nearest pairs each target with the FOLLOWING spike (45 away, "late") instead of
        # its own (55 away, "early").  Three of four then take no negative, and the leftover
        # pair is target 514 with spike 159, 355 steps apart.  The demand that reaches every
        # upstream neuron is sign-inverted as a result.
        # Applied at OUTPUTS by default (PAIR_ORDER=1), where the bumps are real targets:
        # same count as the spikes when the count is right, and in the same order, so the
        # i-th target IS the i-th spike's home.  On a hidden neuron the bumps are inferred
        # field peaks, where an ordered rule was measured wrong 59% of the time -- hence the
        # separate PAIR_ORDER=2 to opt into that.
        for i in range(min(len(bm), len(sp_))):
            claimed.add(i); used.add(sp_[i]); pairs.append((i, sp_[i]))
    else:
        for _d, i, f in sorted((abs(bm[i][0] - ff.get(f, f)), i, f)
                               for i in range(len(bm)) for f in sp_):
            if i in claimed or f in used:
                continue
            claimed.add(i); used.add(f); pairs.append((i, f))
    for i, (q, u, _rr) in enumerate(bm):
        # ONLY WHERE THE DENSITY REPLACES IT.  Outputs pass field=None and get their demand
        # from real targets, so the density branch below never runs for them -- gating this on
        # DENSITY alone deleted the create rule on outputs with nothing to take its place, and
        # an output with 0 spikes against 4 targets then generated NO demand at all (chain
        # collapsed to 0 output spikes, field identically empty).
        if i not in claimed and not (DENSITY in (3, 4, 5, 6) and field is not None):
            L[q] += u
            Lc[q] += u      # nothing fires for this demand at all
            _lay(i, 1.0, Lp); _lay(i, 1.0, Lpc)
    for f in sp_:
        if f not in used and not (DENSITY == 3 and field is not None):
            # An unclaimed spike is surplus.  Depth is the default SUPP unless the field is
            # explicitly negative here, in which case the field's own magnitude is the
            # better statement -- that is where "d must NOT cross at t" arrives from
            # downstream, and discarding it would throw away the whole negative half of the
            # signal, since bumps_of() reads positive runs only.
            mag = SUPP * scale
            if field is not None and float(field[f]) < 0.0:
                mag = max(mag, -float(field[f]))
            if SUPP_SUPPORT and field is not None:
                # SURPLUS NEEDS EVIDENCE; "no bump claimed me" is not evidence when the
                # field had only one bump to give.  Pairing is one-to-one, so a sparse field
                # marks every extra spike surplus -- and the field gets SPARSER as the
                # output gets closer, which makes this anti-convergent.  Measured on 3n L
                # seed6: the output is off by 3 on ONE spike, N1's field is nonzero on 15 of
                # 520 points with a single bump at t=6, and N1 -- firing at exactly its five
                # true times -- takes -6.62e-04 on four of them.
                # A spike outside the field's support is not surplus, it is UNMENTIONED.
                # Where the field does speak, the old verdict stands, which is what keeps
                # 4n F's N1 (5 spikes, 1 wanted, all inside the support) deletable.
                a, b = max(0, f - SUPP_WIN), min(T, f + SUPP_WIN + 1)
                if not np.asarray(field[a:b]).any():
                    mag = 0.0
            L[f] -= mag
            Lp[f] -= mag
    if MARGIN > 0 and volt is not None and scale > 0:
        # KEEP MARGIN AGAINST SPIKES THAT SHOULD NOT HAPPEN.  Demand only ever acts at times
        # something DID fire or IS wanted, so a voltage that climbs to just under threshold
        # somewhere unwanted is invisible -- until a weight rises and it crosses.  3n L sits
        # in exactly that trap: raising w02 to fix the first output spike (which N1 cannot
        # affect, its eligibility there is 0) makes N2 cross at ~167 and ~367 as well, going
        # from 3 spikes to 5, because w12 is not inhibitory enough to hold those down.  The
        # two errors cancel on the later spikes, so they read as exact and nothing objects.
        # Pushing near-misses down preserves the room for w02 to move.
        # SAFE ONLY BECAUSE OF A MEASURED GAP: across the whole suite the highest
        # sub-threshold peak AT THE TRUE WEIGHTS is 0.71*TH (3n L; over-demand 0.67, every
        # other case has none at all), while the near-misses worth suppressing sit at
        # 0.83 (3n L) and 0.94 (5n H).  A threshold at 0.8 is above every legitimate peak,
        # so truth stays a fixed point.
        v = np.asarray(volt, float)
        hi = (v[1:-1] > MARGIN_FRAC * TH) & (v[1:-1] < TH)
        pk = hi & (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])
        for t in (np.nonzero(pk)[0] + 1):
            t = int(t)
            if any(abs(t - q) <= MARGIN_WIN for q, _u, _r in bm):
                continue                      # a spike is wanted about here
            if any(abs(t - f) <= MARGIN_WIN for f in sp_):
                continue                      # it already fired about here
            L[t] -= MARGIN * scale * float(v[t]) / TH
    for i, f in pairs:
        q, u, _rr = bm[i]
        fq = ff.get(f, float(f))
        if SLOPE > 0 and field is not None:
            # MOVE THE SPIKE UP THE FIELD'S SLOPE, instead of toward an extracted point.
            # Reducing a whole positive run to one requested time is arbitrary when the run
            # is flat, and measured across the suite it usually IS: 83-84% of runs have a
            # peak barely 1.2x their own median, and the two estimators (centroid, argmax)
            # disagree by a median 11-16 samples while the true spike time sits near
            # neither.  Switching between them flips 8 cells with no shape signature
            # distinguishing the ones it helps from the ones it hurts.
            # The slope needs no extraction and is self-scaling: rising field -> the spike
            # should be LATER -> less drive -> negative; falling -> earlier -> positive; and
            # a flat run gives a small demand on its own, which is right, because on a flat
            # run the position genuinely does not matter much.
            a, b = max(0, f - SLOPE_WIN), min(T - 1, f + SLOPE_WIN)
            if b > a:
                d = (float(field[b]) - float(field[a])) / (b - a)
                if SLOPE_NORM >= 2:
                    # DIRECTION FROM THE SLOPE, MAGNITUDE FROM THE BUMP.  Everything derived
                    # from dF/dt at the spike decays with depth: each propagation convolves
                    # the field wider and flatter, so on chain |L| falls 100x per hop
                    # (|L2| 8.7e-06 one hop out, |L1| 8.2e-08 two hops out) while the
                    # point-move, whose size is the bump strength u, is preserved
                    # (4.80e-03 -> 4.65e-03).  Rescaling by the neuron's OWN largest slope
                    # does not help -- that normalises WITHIN a neuron, not across hops, and
                    # measured it made things worse still (|L1| 6.8e-09, suite 42 -> 38).
                    # u is the only depth-stable quantity here, so the slope may set the
                    # sign and nothing else.
                    amt = 1.0 if SLOPE_NORM >= 3 else min(
                        1.0, abs(q - fq) / max(GRADE, 1e-9))
                    sgn = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
                    L[f] -= SLOPE * sgn * u * amt
                    Lp[f] -= SLOPE * sgn * u * amt
                elif SLOPE_NORM == 1:
                    # SLOPE FOR DIRECTION, BUMP STRENGTH FOR MAGNITUDE.  The raw slope is
                    # MULTIPLICATIVE down a chain: dF/dt shrinks at every hop, so the demand
                    # attenuates by orders of magnitude per link.  Measured on chain at its
                    # stuck point, the field is essentially flat at the spikes (slope
                    # -9.7e-08 on N2, -9.1e-10 on N1), giving L[2] = 8.7e-06 against the
                    # point-move's 4.8e-03 (550x down) and L[1] = 8.2e-08 against 4.6e-03
                    # (56,000x down); g(w01) ends at 3.3e-12 versus 1.3e-07 and the run
                    # freezes with max|g| = 0.  Dividing by the neuron's OWN largest slope
                    # keeps "flatter here than elsewhere on me" -- which is the useful part
                    # -- while restoring a depth-independent size, taken from the bump the
                    # spike is paired with, exactly as the point-move does.
                    ii = np.arange(T)
                    aa = np.maximum(ii - SLOPE_WIN, 0)
                    bb = np.minimum(ii + SLOPE_WIN, T - 1)
                    darr = (field[bb] - field[aa]) / np.maximum(bb - aa, 1)
                    dmax = float(np.abs(darr).max())
                    rel = d / dmax if dmax > 0 else 0.0
                    L[f] -= SLOPE * rel * u
                    Lp[f] -= SLOPE * rel * u
                else:
                    L[f] -= SLOPE * d * GRADE
                    Lp[f] -= SLOPE * d * GRADE
            continue
        err = abs(q - fq)
        if err <= DEAD_ZONE:
            continue                       # close enough: contribute nothing
        amt = min(1.0, (err - DEAD_ZONE) / max(GRADE, 1e-9))
        L[q] += amt * u
        _lay(i, amt, Lp)
        pairof[q] = f            # this target is the counterfactual home of spike f
        if fq < q:
            # ONLY AN EARLY SPIKE GETS THE NEGATIVE.  Suppression removes drive, which moves
            # a spike LATER: right for an early one, actively wrong for a late one, which is
            # corrected by the positive at its bump (necessarily earlier than it).
            L[f] -= amt * SUPP * scale
            Lp[f] -= amt * SUPP * scale
    if DENSITY in (3, 4, 5, 6) and field is not None:
        # BUMPS FOR TIMING, DENSITY FOR COUNT.  The move term above is the only thing bumps
        # are still used for -- it needs a single time to move a spike TOWARD, and a density
        # cannot supply one (measured: F_DENSITY=1/2 fix the count on 14n P/Q but take |dt|
        # from 0.0 to 20-33 on the cases that were already exact).  Creation and suppression
        # need no such point, only "more wanted here" / "less", which is what height says --
        # and reading it per-sample is what lets an 878-step plateau ask for eight spikes
        # instead of one.
        # The positive half is masked inside each existing spike's REFRACTORY footprint: the
        # neuron already fires there, so that part of the run is a MOVE (already expressed
        # above), not a create.  The rest of the plateau still asks.  The negative half is
        # never masked -- negative field AT a spike is precisely "this one should not exist".
        # MODE 5 READS CREATION FROM THE CREATE-ONLY CHANNEL.  The full field's positive
        # part conflates two things: "a spike is missing here" and "the spike you already
        # have should MOVE here".  Measured on 14n Q, N2 has the right count, 242 positive
        # field samples and Fc identically zero -- every one of those samples is a move.
        # Feeding them to the create term asks for spikes that nothing actually lacks.
        _src = (np.asarray(fieldc, float) if (DENSITY == 5 and fieldc is not None)
                else np.asarray(field, float))
        dpos = np.maximum(_src, 0.0)
        for _f in sp_:
            dpos[max(0, _f - REFRAC_ITERS):min(T, _f + REFRAC_ITERS + 1)] = 0.0
        # MODE 4 KEEPS BUMP-DERIVED SUPPRESSION.  Mode 3 dropped both bump rules and
        # count agreement fell BELOW the bump baseline (|Dcount| 0.76 against 0.37) -- the
        # "unclaimed spike -> suppress" rule was the only reliable way to DELETE a spike,
        # and the density's negative half only exists where something downstream actively
        # sent negative demand, which is much rarer.  So mode 4 takes creation from the
        # density and leaves deletion to the bumps.
        if DENSITY == 6:
            # MASS-MATCH THE CREATE DENSITY TO A BUMP.  Mode 4 fixes the count but loses
            # timing (|dt| 7.6 -> 14.0), and the reason is scale, not shape: the hidden
            # create mass sums to 2.7e+05 against move terms of order 1e-2, so the move --
            # the ONLY thing that carries a spike's position -- is swamped.  Rescaling each
            # run to the mass of a single bump keeps "spikes are wanted across this span"
            # while leaving the move term audible.
            pos = np.nonzero(dpos > 0)[0]
            if len(pos):
                for r in np.split(pos, np.nonzero(np.diff(pos) > GAP)[0] + 1):
                    if len(r):
                        m = float(dpos[r].sum())
                        if m > 0:
                            dpos[r] *= float(dpos[r].max()) / m
        dens = (dpos if DENSITY in (4, 5, 6)
                else dpos + np.minimum(np.asarray(field, float), 0.0))
        L += dens
        Lc += dpos
        _spread_mass(Lp, dens); _spread_mass(Lpc, dpos)
    return L, Lc, Lp, Lpc, pairof


def bumps_of(F, gap=None, reach=None):
    """Positive runs of the field -> one requested spike each, at the run's peak.

    This is what gives the field a COUNT.  A density alone cannot express "fire once": an
    86-step-wide band is better satisfied by five spikes than by one, which is why feeding
    the raw density in as demand multiplied spikes instead of placing them.

    `reach` marks times where the neuron actually has incoming drive.  Within a run the
    request goes to the best REACHABLE time when one exists, because a request placed where
    no input has arrived multiplies zero eligibility and moves nothing: on 4n F seed 0 the
    field was one nearly-flat run over 0..223 (0.9 at t=0 against 1.0 near t=190) and the
    raw argmax landed on t=17, before the first input arrival at 19, giving max|g| = 0 at a
    state one step from correct.
    If NOTHING in the run is reachable the request still goes in, at the plain argmax.  That
    case is not noise, it is the message: masking it instead cost 16/52 -> 10/52, because a
    neuron that cannot fire where it is needed is exactly what its OWN upstream has to fix,
    and zeroing the field there empties bumps, hence L, hence the upstream field.
    """
    gap = GAP if gap is None else gap
    pos = np.nonzero(F > 0)[0]
    if not len(pos):
        return []
    out = []
    for r in np.split(pos, np.nonzero(np.diff(pos) > gap)[0] + 1):
        if not len(r):
            continue
        cand = r[reach[r]] if reach is not None and reach[r].any() else r
        if CENTROID:
            # AMPLITUDE-WEIGHTED CENTROID, not argmax.  The peak of a broad, nearly flat run
            # hops between samples on near-ties, so the requested time is a step function of
            # the field; the centroid slides continuously with it.
            # PEAK_FRAC restricts the average to the part of the run at or above that share
            # of its peak, so a sharp peak is not dragged off by a long low plateau.
            wgt = F[cand]
            if PEAK_FRAC > 0:
                keep = wgt >= PEAK_FRAC * float(wgt.max())
                if keep.any():
                    cand, wgt = cand[keep], wgt[keep]
            q = int(round(float(np.sum(cand * wgt) / max(float(wgt.sum()), 1e-30))))
            q = int(np.clip(q, cand[0], cand[-1]))
        else:
            q = int(cand[np.argmax(F[cand])])
        out.append((q, float(F[q]), r))
    return out


# ---- the field ---------------------------------------------------------------------
def _occ_mask(resets_d, T):
    """Arrival times swallowed by a refractory shadow, as a lookup over the whole timeline.

    The occlusion test inside _plausibility depends only on the ARRIVAL time and the
    downstream neuron's own spikes -- not on the request time t, the co-drive, or the weight.
    It was therefore being rebuilt identically for every request into the same neuron: a
    Python loop over ~30 resets, 3 array ops each, ~5200 times a round on 50n A.  Build it
    once per neuron and index it instead.
    """
    m = np.zeros(T + DELAY_ITERS + 1, bool)
    for r in resets_d:
        a, b = int(r) + 1, min(len(m), int(r) + REFRAC_ITERS)
        if b > a:
            m[a:b] = True
    return m


def _timing(qs, t, occ, resets_d=()):
    """(hk, ok): the half of the kernel that does NOT depend on which edge is asking.

    TIMING and OCCLUSION are properties of the downstream neuron and the request time alone;
    only the WEIGHT term varies per edge.  With fan-in 5 the same (d, t) pair was recomputing
    this five times, on arrays averaging 78 samples where numpy call overhead dominates.
    Cached by the caller, so each (d, t) pays for it once.
    """
    dts = t - qs
    hk = np.take(HK, dts, mode="clip")
    ok = (dts >= 0) & (dts < KWIN) & (hk > 0)
    if OCCLUDE and occ is not None:
        ok &= ~occ[qs + DELAY_ITERS]
    elif OCCLUDE and len(resets_d):
        arr = qs + DELAY_ITERS
        for r in resets_d:
            ok &= ~((arr > r) & (arr < r + REFRAC_ITERS))
    return hk, ok


def _weight_term(hk, ok, other_t, wsi):
    """The per-edge half.  Equivalent to the tail of _plausibility, minus the inf arithmetic:
    where `ok` is false the old code produced wmin = inf -> mis = inf -> k = 0, and the final
    mask zeroes it anyway, so the branch is dropped rather than computed."""
    if not ok.any():
        return None
    wmin = (TH - other_t) / np.where(ok, hk, 1.0)
    mis = (np.abs(np.abs(wmin) - abs(wsi)) if (INH_MAG and wsi < 0)
           else np.abs(wmin - wsi))
    mis /= max(abs(wsi), WFLOOR * W_CRIT)
    return np.where(ok, 1.0 / (1.0 + mis / max(TOL, 1e-9)), 0.0)


def _plausibility(qs, t, other_t, wsi, resets_d, occ=None):
    """K(q -> t): can a spike at q make d cross at t, at the weight we already have?

    Three factors, all necessary:
      TIMING     the PSP must be able to reach t from q at all
      OCCLUSION  the arrival must not land in the refractory shadow of a spike d REALLY
                 produced.  Without this the field's own maximum sat on a dead time: for
                 4n F's target at 290, 21 of 57 candidates were swallowed, ALL at the front
                 of the window, and the peak sat on the deadest of them.
      WEIGHT     the weight that would place the spike at t, against the weight on the edge.
                 Never allowed to reach zero -- a hard cutoff collapses the two field
                 variables into one and makes the field vanish exactly when the weight is
                 most wrong (4n F: w=25 against a needed 445 gave an identically empty
                 field, self-reinforcing, since a silent field cannot pull the weight back).
    """
    dts = t - qs
    ok = (dts >= 0) & (dts < KWIN)
    hk = np.where(ok, np.take(HK, dts, mode="clip"), 0.0)
    ok &= hk > 0
    if OCCLUDE and occ is not None:
        ok &= ~occ[qs + DELAY_ITERS]
    elif OCCLUDE and len(resets_d):
        # fallback: PAIR_WIN drops a reset for one request, so the shared mask does not apply
        arr = qs + DELAY_ITERS
        for r in resets_d:
            ok &= ~((arr > r) & (arr < r + REFRAC_ITERS))
    if not ok.any():
        return None
    need = TH - other_t
    wmin = np.where(ok, need / np.where(hk > 0, hk, 1.0), np.inf)
    if INH_MAG and wsi < 0:
        # AN INHIBITORY EDGE IS JUDGED ON |w|.  wmin = need/h is POSITIVE whenever there is
        # a deficit, so |wmin - wsi| = wmin + |wsi| for a negative weight -- it can never be
        # small however well the magnitudes match, and the edge is penalised purely for its
        # sign, which is fixed and not something the field decides.  On 3n L (wmin ~ +445,
        # wsi = -606) the signed form gives mis 1.73 -> k 0.37 against a magnitude form's
        # 0.27 -> k 0.79, so the only inhibitory edge in the suite had its whole field
        # contribution attenuated ~2x.  EDGE_SIGN already carries the DIRECTION; this is
        # about the plausibility MAGNITUDE.
        mis = np.abs(np.abs(wmin) - abs(wsi)) / max(abs(wsi), WFLOOR * W_CRIT)
    else:
        mis = np.abs(wmin - wsi) / max(abs(wsi), WFLOOR * W_CRIT)
    return np.where(ok, 1.0 / (1.0 + mis / max(TOL, 1e-9)), 0.0)


def _prop_plan(Lpd, Lpcd, resets_d, T, occ):
    """[(ts, qs, hk, ok)] per epoch -- everything in the batched propagation that is the same
    for every edge into this neuron.  Only the weight term downstream of it varies, so with
    fan-in 5 this was being rebuilt five times over identical (n_t x n_q) arrays."""
    act = np.nonzero((Lpd != 0) | (Lpcd != 0))[0]
    if not len(act):
        return []
    rs = np.asarray(sorted(resets_d), float)
    lo_of = (rs[np.searchsorted(rs, act, side="left") - 1].astype(int)
             if len(rs) else np.full(len(act), -1))
    if len(rs):
        lo_of = np.where(np.searchsorted(rs, act, side="left") == 0, -1, lo_of)
    out = []
    for lo in np.unique(lo_of):
        ts = act[lo_of == lo]
        q0 = max(0, int(lo) + 1 - DELAY_ITERS)
        q1 = int(ts.max()) - DELAY_ITERS
        if q1 < q0:
            continue
        qs = np.arange(q0, q1 + 1)
        dts = ts[:, None] - qs[None, :]
        ok = (dts >= DELAY_ITERS) & (dts < KWIN)
        hk = np.where(ok, np.take(HK, np.clip(dts, 0, KWIN - 1)), 0.0)
        ok &= hk > 0
        if OCCLUDE and occ is not None:
            ok &= ~occ[qs + DELAY_ITERS][None, :]
        elif OCCLUDE and len(resets_d):
            arr = qs + DELAY_ITERS
            qm = np.ones(len(qs), bool)
            for r in resets_d:
                qm &= ~((arr > r) & (arr < r + REFRAC_ITERS))
            ok &= qm[None, :]
        if not ok.any():
            continue
        out.append((ts, qs, hk, ok))
    return out


def _prop_edge(f, fc, Lpd, Lpcd, other, wsi, resets_d, T, sg, occ=None, plan=None):
    """One edge's whole backward contribution, in a handful of array ops instead of a
    Python loop over every requested time.

    WHY NOT A CONVOLUTION.  The obvious move is back_corr's: a sum over t of L[t]*K(q->t) is
    a correlation as soon as K depends only on t-q.  Here it does not.  K carries a WEIGHT
    MISMATCH term, 1/(1 + |need(t)/HK(t-q) - w|/TOL), in which t and t-q enter jointly and
    non-separably -- need(t) = TH - other(t) is the drive the OTHER edges already supply at t.
    Binning need(t) would recover a convolution per bin, but that is an approximation, and the
    thing that actually costs time here is the 900-iteration Python loop, not the arithmetic.

    So: batch it instead.  Group the requested times by EPOCH (between two real spikes of d),
    since `lo` -- and hence the candidate window -- is constant within one.  Then the whole
    epoch is one (n_t x n_q) array and the accumulation is a single mat-vec.  Exact, not
    approximate: verified against the loop by _verify_prop.py at max|diff| = 0.0.
    """
    for ts, qs, hk, ok in (_prop_plan(Lpd, Lpcd, resets_d, T, occ)
                           if plan is None else plan):
        need = (TH - np.asarray(other, float)[ts])[:, None]
        wmin = np.where(ok, need / np.where(hk > 0, hk, 1.0), np.inf)
        den = max(abs(wsi), WFLOOR * W_CRIT)
        mis = (np.abs(np.abs(wmin) - abs(wsi)) if (INH_MAG and wsi < 0)
               else np.abs(wmin - wsi)) / den
        k = np.where(ok, 1.0 / (1.0 + mis / max(TOL, 1e-9)), 0.0)
        mx = k.sum(axis=1) if KNORM else k.max(axis=1)
        good = mx > 0
        if not good.any():
            continue
        k = k[good] / mx[good][:, None]
        f[qs] += sg * (Lpd[ts[good]] @ k)
        fc[qs] += sg * (Lpcd[ts[good]] @ k)


def build(C, N, w, spall, T, out_targets, spall_f=None, volt=None):
    """Field and signed demand for every neuron, downstream-first, repeated to close cycles."""
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    outg = {n: np.where(C[:, 0] == n)[0] for n in range(N)}
    # PHYSICAL eligibility: every epoch resets at a spike the neuron really produced.
    if SUB_EPS and spall_f:
        ep = {(int(C[si, 0]), int(C[si, 1])):
              elig_frac(spall_f[int(C[si, 0])], T, spall_f[int(C[si, 1])],
                        spall_f[int(C[si, 1])])
              for si in range(len(C))}
    else:
        # WIDE_EPS bounds what widening eligibility can buy.  Under DPOW=0 the run WIDTH is the
        # whole signal, and ~2/3 of it multiplies zero eligibility because eligibility resets
        # at every postsynaptic spike -- so a neuron firing too often starves the gradient that
        # would fix it.  PAIR_EPS was built to relax exactly this, but it keys off the pairing
        # map, which DENSITY mode never populates (PR is empty on every hidden neuron), so it
        # cannot act where the loss is.  Dropping the truncation entirely on hidden neurons is
        # not physical, but it is an UPPER BOUND: if maximal width does not help, no gentler
        # widening will.
        def _res(n):
            return [] if (WIDE_EPS and n not in out_targets) else spall[n]
        ep = {(int(C[si, 0]), int(C[si, 1])):
              eligibility(spall[int(C[si, 0])], T, _res(int(C[si, 1])),
                          refrac_at=_res(int(C[si, 1])))
              for si in range(len(C))}
    drive = {n: (sum(float(w[si]) * ep[(int(C[si, 0]), n)] for si in inc[n])
                 if len(inc[n]) else np.zeros(T)) for n in range(N)}

    F = {n: np.zeros(T) for n in range(N)}
    L = {n: np.zeros(T) for n in range(N)}
    Lc = {n: np.zeros(T) for n in range(N)}
    Lp = {n: np.zeros(T) for n in range(N)}
    Lpc = {n: np.zeros(T) for n in range(N)}
    PR = {n: {} for n in range(N)}
    Fc = {n: np.zeros(T) for n in range(N)}   # create-only field, propagated on its own
    _occ = {}                                 # neuron -> refractory-shadow lookup, built lazily
    _tcache = {}                              # (downstream neuron, request time) -> (hk, ok)
    bumps = {n: [(int(t), TH, None) for t in sorted(out_targets[n])]
             for n in out_targets}
    for _ in range(max(1, SWEEPS)):
        _pcache = {}      # Lp changes between sweeps, so the plans do not survive one
        _tcache.clear()
        for n in range(N - 1, -1, -1):
            L[n], Lc[n], Lp[n], Lpc[n], PR[n] = local_demand(
                bumps.get(n, []), spall[n], T,
                None if n in out_targets else F[n], (spall_f or {}).get(n),
                None if volt is None else volt[:, n],
                None if n in out_targets else Fc[n])
        for n in range(N - 1, 0, -1):
            if n in out_targets or not len(outg[n]):
                # an output still gets its demand straight from its targets; a neuron that
                # feeds nothing has no field to build
                continue
            f = np.zeros(T)
            fc = np.zeros(T)          # the CREATE part, propagated on its own
            for si in outg[n]:
                d = int(C[si, 1])
                wsi = float(w[si])
                other = drive[d] - wsi * ep[(n, d)]
                resets_d = sorted(spall[d])
                occ_d = None if NO_OCC_MASK else _occ.get(d)
                if occ_d is None and not NO_OCC_MASK:
                    occ_d = _occ[d] = _occ_mask(resets_d, T)
                _sg0 = -1.0 if (EDGE_SIGN and wsi < 0) else 1.0
                if FASTPROP and not (PAIR_WIN and PR[d]):
                    _pl = _pcache.get(d)
                    if _pl is None:
                        _pl = _pcache[d] = _prop_plan(Lp[d], Lpc[d], resets_d, T, occ_d)
                    _prop_edge(f, fc, Lp[d], Lpc[d], other, wsi, resets_d, T, _sg0,
                               occ_d, _pl)
                    continue
                for t in np.union1d(np.nonzero(Lp[d])[0], np.nonzero(Lpc[d])[0]):
                    t = int(t)
                    # A PAIRED TARGET MUST NOT BE TRUNCATED BY ITS OWN SPIKE'S RESET.
                    # "This spike is 2 steps early" is a +demand at the target and a
                    # -demand at the spike, equal in size -- they should very nearly cancel,
                    # leaving a small 2-sample residue that says "shift it".  They do not,
                    # because the spike IS a reset, so the negative gets the whole epoch back
                    # to the previous reset while the positive gets only the sliver after it.
                    # Measured on over-demand: L2[218] = -4.67e-04 spread over 82 candidate
                    # samples against L2[220] = +4.67e-04 over 2 -- 41:1 in area, so a 2-step
                    # timing correction became a large-window VETO that put -3.92e-04 on N1's
                    # own true firing time.  In the counterfactual where the spike had fired
                    # at the target there is no reset at f, so drop it.
                    _rd = resets_d
                    if PAIR_WIN and t in PR[d]:
                        _f = PR[d][t]
                        _rd = [r for r in resets_d if r != _f]
                    lo = max([r for r in _rd if r < t], default=-1)
                    qs = np.arange(max(0, lo + 1 - DELAY_ITERS), t - DELAY_ITERS + 1)
                    if not len(qs):
                        continue
                    _ck = (d, t)
                    _tc = _tcache.get(_ck) if _rd is resets_d else None
                    if _tc is None:
                        _tc = _timing(qs, t, occ_d if _rd is resets_d else None, _rd)
                        if _rd is resets_d:
                            _tcache[_ck] = _tc
                    k = _weight_term(_tc[0], _tc[1], float(other[t]), wsi)
                    if k is None:
                        continue
                    # MASS-CONSERVING vs PEAK-NORMALISED.  k/max spreads a peak-height-1
                    # kernel over the whole candidate window, so one unit of downstream
                    # demand becomes ~|window| units of upstream field -- and the window
                    # averages 78 samples, so the create mass GROWS per hop (measured on
                    # 4n G: N2 2.31e-01, N1 8.59e+01, 373x one hop further back).  That is
                    # what routes an output deficit to the deepest weight in the chain.
                    # k/sum makes the backward message a distribution over which upstream
                    # time could be responsible, which is what the docstring above already
                    # claims ("a wide feasible window does not outvote a narrow one").
                    mx = float(k.sum()) if KNORM else float(k.max())
                    if mx <= 0:
                        continue
                    # L[d][t] is SIGNED, so "d must cross here" and "d must not" travel
                    # backwards through the same kernel and differ only in sign.  Normalise
                    # per request so a wide feasible window does not outvote a narrow one.
                    # DEMAND FLIPS SIGN ALONG AN INHIBITORY EDGE.  L[d][t] says what d must
                    # do; what THIS neuron must do to bring that about depends on which way
                    # the synapse pushes.  Along an excitatory edge "d must cross at t"
                    # means "fire, to help"; along an inhibitory one it means "do NOT fire,
                    # you are holding it down" -- and "d must NOT cross" inverts likewise.
                    # Without this the only inhibitory case in the suite gets its backward
                    # demand exactly inverted: on 3n L with w02 frozen at 1220, N2's two
                    # spurious spikes at 168/368 carry correct full-strength negatives, and
                    # N1 -- which suppresses them by firing -- receives an all-negative
                    # field with NO positive bump anywhere, telling it to fire less.
                    _sg = -1.0 if (EDGE_SIGN and wsi < 0) else 1.0
                    f[qs] += _sg * float(Lp[d][t]) * k / mx
                    fc[qs] += _sg * float(Lpc[d][t]) * k / mx
            if CREATE_FLOOR:
                # THE SAME RULE, ONE LEVEL UP.  F[n] sums the SIGNED L[d], so a downstream
                # creation demand and a downstream suppression annihilate in the field --
                # the missing-spike information is destroyed BEFORE it ever becomes a
                # request, and the gradient floor never sees it.  Measured across the suite,
                # 22 of 92 (hidden neuron, seed) cases lose at least one create request this
                # way, concentrated in exactly the cases that fail: chain, 3-cycle, 2-cycle
                # and 8n M, with 3n J seed2 losing all 4 and 2-cycle seed0 both of its 2.
                f = np.where(fc > 0, np.maximum(f, fc), f)
            F[n] = f
            Fc[n] = fc          # the create-only field, kept separately
            # a net-inhibited time is not automatically unreachable: the inhibition is
            # itself a weight the search can change
            _reach = (drive[n] != 0) if INH_REACH else (drive[n] > 0)
            bumps[n] = bumps_of(f, reach=_reach)
    for n in range(N):
        L[n], Lc[n], Lp[n], Lpc[n], PR[n] = local_demand(
            bumps.get(n, []), spall[n], T,
            None if n in out_targets else F[n], (spall_f or {}).get(n),
            None if volt is None else volt[:, n],
            None if n in out_targets else Fc[n])
    return F, L, Lc, ep, PR, Fc


def gradient(C, N, w, spall, T, out_targets, spall_f=None, volt=None):
    F, L, Lc, ep, PR, Fc = build(C, N, w, spall, T, out_targets, spall_f, volt)
    g = np.zeros(len(w)); gc = np.zeros(len(w))
    _cf = {}

    # A QUANTITY EVALUATED AT A TARGET MUST NOT BE TRUNCATED BY THE SPIKE THAT TARGET IS
    # TRYING TO MOVE.  eligibility() cuts each presynaptic PSP at the postsynaptic neuron's
    # next reset, and a misplaced spike IS that reset -- so the target reads zero
    # eligibility for a reason that exists only because the spike is in the wrong place.
    # Measured on over-demand at its collapse point (N1 firing 102/302/502, arriving
    # 120/320/520; N2 firing 121/321 against targets 140/220/399):
    #     eps(1,2)        as measured    counterfactual (drop the reset at 121)
    #       at 140 target   0.000e+00  ->  7.810e-06    16x the spike's own value
    #       at 220 target   0.000e+00  ->  1.570e-05    32x
    #       at 121 spike    4.931e-07  ->  4.931e-07    unchanged
    # so w12's positives all multiplied ZERO while its negatives multiplied the one
    # surviving sample: a purely negative gradient at every weight value, ratcheting w12 to
    # the clamp floor.  Self-reinforcing, too -- pushing w12 down delays the spike further
    # from its target, truncating the PSP earlier still.
    # The suppression AT the spike keeps the ordinary eligibility; that one is about the
    # epoch that really happened.
    def _eps_at(k, n, t):
        f = PR[n].get(int(t)) if PAIR_EPS else None
        if f is None:
            return ep[(k, n)][t]
        key = (k, n, int(f))
        if key not in _cf:
            rs = [r for r in spall[n] if r != f]
            _cf[key] = eligibility(spall[k], T, rs, refrac_at=rs)
        return _cf[key][t]

    for si in range(len(C)):
        k, n = int(C[si, 0]), int(C[si, 1])
        if PAIR_EPS and PR[n]:
            g[si] = float(sum(L[n][t] * _eps_at(k, n, t) for t in np.nonzero(L[n])[0]))
            gc[si] = float(sum(Lc[n][t] * _eps_at(k, n, t) for t in np.nonzero(Lc[n])[0]))
        else:
            g[si] = float(np.dot(L[n], ep[(k, n)]))
            gc[si] = float(np.dot(Lc[n], ep[(k, n)]))
    if GN:
        # SOLVE AN OUTPUT'S INPUT WEIGHTS JOINTLY, don't sum their demands.
        # V_n(t) = sum_k w_k eps_k(t) is LINEAR in the incoming weights, so the demands at
        # the times where L[n] != 0 form a least-squares system A dw ~= r with
        # A[j,k] = eps_k(t_j) and r_j = L[n](t_j).  First order takes A^T r, which SUMS
        # residuals that may pull opposite ways on a shared weight; the joint solve finds
        # the direction that satisfies them together, which is what makes the difference
        # when a single weight serves several output spikes at once.
        # Measured on over-demand with the spike count already correct
        # (N2 [136,221,406] against [140,220,399], w02=318 needing 300):
        #     t=136  "delay the first spike"   -2.076e-08
        #     t=399  "advance the last spike"  +2.550e-08
        #     sum                              +4.857e-09   -- up, when w02 must come DOWN
        # Both act mostly through w02 and the larger one simply wins.  At the TRUE weights
        # all three spikes are satisfiable at once, so the joint direction exists; a sum of
        # conflicting demands cannot find it.
        # OUTPUTS ONLY: there L is a genuine voltage residual, so driving it to zero means
        # something.  For a hidden neuron L is a timing-derived direction of arbitrary
        # scale and solving it as a residual is meaningless.
        for n in out_targets:
            syn = np.where(C[:, 1] == n)[0]
            if not len(syn):
                continue
            times = np.nonzero(L[n])[0]
            if not len(times):
                continue
            A = np.stack([ep[(int(C[si, 0]), n)][times] for si in syn], axis=1)
            r = L[n][times]
            AtA = A.T @ A
            sc = float(np.trace(AtA)) / max(len(syn), 1)
            if sc <= 0:
                continue
            try:
                dw = np.linalg.solve(AtA + GN_RIDGE * sc * np.eye(len(syn)), A.T @ r)
            except np.linalg.LinAlgError:
                continue
            if np.all(np.isfinite(dw)):
                g[syn] = dw
    if CREATE_FLOOR:
        # A MISSING SPIKE IS NOT NEGOTIABLE.  If a demand has nothing firing for it at all,
        # every weight that could produce it must go UP -- and a timing correction elsewhere
        # must not be able to cancel that, because you cannot retime a spike that does not
        # exist.  Everything is summed into one L[n] and then projected, so without this a
        # creation demand and an unrelated suppression land on the SAME weight and annihilate.
        # Measured on over-demand at its stuck point: N2 needs [140, 220, 399] and fires
        # [173, 373], so 220 has nothing serving it; the create term gives w02 = +1.67e-07
        # while the suppression from pairing 373 with 399 gives -1.70e-07, netting
        # -3.14e-09 -- the wrong sign, on a weight that a grid search says must rise
        # (250 -> 300).  Floor it at the create term instead.
        g = np.where(gc > 0, np.maximum(g, gc), g)
    return g, F, L, ep


# ---- optimiser ---------------------------------------------------------------------
def train(C, N, outs, w, T_true, params, rounds, lr=LR, cb=None, freeze=None):
    """Adam, step decay, periodic restart, trust region in spike-time units.

    Carried over from grad_trace unchanged so that a comparison between the two pathways is
    a comparison of the SIGNALS, not of their optimisers.
    """
    T = params.steps
    w = np.asarray(w, float).copy()
    out_t = {o: list(T_true[o]) for o in outs}
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    wsign = np.where(np.asarray(w, float) < 0, -1.0, 1.0)   # fixed for the whole run
    w = wsign * np.clip(np.abs(w), WMAG_MIN, WMAG_MAX)
    hold = np.zeros(len(w), int)      # iterations each weight stays pinned
    prev_cnt, prev_upd = None, None
    m = np.zeros(len(w)); v = np.zeros(len(w)); ait = 0
    best_w, best_e = w.copy(), np.inf
    _Vcarry = None                    # simulation of the current w, if a previous round made it

    def out_err(ww):
        """(mean output error, the simulation it used).

        The caller keeps that simulation: KEEP_BEST scores the weights at the END of a round,
        and the next round opens by simulating the SAME weights.  Returning it turns two full
        simulations per round into one -- 32% of runtime on 14n Q, 6% on 50n A."""
        Vh = fsim(C, N, np.asarray(ww, np.float32), params)
        tot = 0.0
        for o in outs:
            f, t = sp(Vh, o), out_t[o]
            if len(f) == len(t):
                tot += float(np.mean([abs(a - b) for a, b in zip(f, t)]))
            elif GRADED_ERR:
                # A FLAT COUNT PENALTY MAKES KEEP_BEST BLIND.  At 99.0 per mismatched output,
                # a SINGLE-output case whose count never matches scores exactly 99.0 every
                # round; best_e is set at round 1 and `e < best_e` is never true again, so
                # train() returns the weights after ONE update and discards the rest.
                # Measured on 4n G seed 0: returned w = round-2 weights (289/390/654/357)
                # while the live iterate at round 800 was (1177/1099/978/208).
                # Grade it instead -- still dominated by count, but strictly ordered, so
                # getting closer on count is visible and ties break on timing.
                d = abs(len(f) - len(t))
                near = ([abs(a - b) for a, b in zip(f, t)] if f and t else [])
                tot += 99.0 * d + (float(np.mean(near)) if near else 0.0) / 1e3
            else:
                tot += 99.0
        return tot / max(len(outs), 1), Vh

    for it in range(1, rounds + 1):
        if RESTART_EVERY > 0 and it % RESTART_EVERY == 1 and it > 1:
            m = np.zeros(len(w)); v = np.zeros(len(w)); ait = 0
        ait += 1
        if _Vcarry is not None:
            V, _Vcarry = _Vcarry, None      # already simulated at the end of the last round
        else:
            V = fsim(C, N, np.asarray(w, np.float32), params)
        spall = {p: sp(V, p) for p in range(N)}
        if CLIFF_HOLD > 0:
            # WENT OVER A CLIFF -> PIN WHAT PUSHED IT AND LET THE REST CATCH UP.
            # A count change is not a gradient event: the weight that caused it gets a
            # correctly-signed push to undo it, and the run bounces off the boundary instead
            # of crossing.  But the crossing is often the RIGHT move if something else moves
            # WITH it -- on 3n L, forcing w02 over its cliff and pinning it there let w12
            # travel 255 units to -851 and solve the case exactly, which it never does on
            # its own.  So keep the step and freeze the culprit, giving the others a window
            # to compensate in.
            cnt = {n: len(spall[n]) for n in range(N)}
            if prev_cnt is not None and prev_upd is not None:
                for n in range(N):
                    if not len(inc[n]) or cnt[n] <= prev_cnt[n]:
                        continue                      # only a spike GAINED counts as a cliff
                    if CLIFF_ALL >= 2:
                        # PAUSE EVERYTHING -- the control.  If a freeze were merely lost
                        # iterations, this would cost almost nothing; whatever it costs
                        # ABOVE this is the price of letting the other weights compensate
                        # around a pinned one.
                        hold[:] = CLIFF_HOLD
                    elif CLIFF_ALL:
                        for si in inc[n]:
                            hold[si] = CLIFF_HOLD
                    else:
                        # blame the incoming weight whose own step raised the drive most
                        best, bsi = -1.0, None
                        for si in inc[n]:
                            k = int(C[si, 0])
                            e = eligibility(spall[k], T, spall[n], refrac_at=spall[n])
                            m = float(np.abs(prev_upd[si] * e).max())
                            if m > best:
                                best, bsi = m, si
                        if bsi is not None:
                            hold[bsi] = CLIFF_HOLD
            prev_cnt = cnt
        hold = np.maximum(hold - 1, 0)
        spf = {p: sp_frac(V, p) for p in range(N)} if SUBSAMPLE else None
        g, F, L, ep = gradient(C, N, w, spall, T, out_t, spf, V)
        if float(np.abs(g).max()) == 0.0 and all(spall[o] == out_t[o] for o in outs):
            best_w, best_e = w.copy(), 0.0
            break
        if GUARD > 0:
            # PREDICT THE CROSSING, DO NOT THRESHOLD THE HEIGHT.  A sub-threshold peak that
            # must never fire is invisible to every other term -- demand only acts where
            # something DID fire or IS wanted -- so a weight can be walked straight into it.
            # 4n V settles exactly there: N2 holds 3 spikes with peaks at 0.665 of threshold,
            # w02 wants to rise for the output timing, and at w02 = 833 those peaks cross and
            # the count breaks to 5.  Deepening w12 moves that cliff (833 -> 1058 as w12 goes
            # -378 -> -600), so the two must advance together, but w12's own demand points
            # the other way (+2.2e-08, i.e. LESS inhibition, because weaker inhibition moves
            # spikes earlier and the output is late).
            # Height cannot separate this: the cliff sits at 0.665 while legitimate
            # sub-threshold peaks elsewhere in the suite reach 0.71.  What distinguishes it
            # is whether THIS step would push it over, which is predictable from the step
            # itself:  dv(t) = sum_k dw_k eps_k(t).
            # The correction needs no sign handling -- eps is the PSP SHAPE, positive on
            # every edge, so one negative demand moves an excitatory weight down and an
            # inhibitory weight more negative, which are the two ways to reduce drive there.
            _m = BETA1 * m + (1 - BETA1) * g
            _v = 0.999 * v + 0.001 * g * g
            _p = ((lr / (1.0 + DECAY * ait)) * (_m / (1 - BETA1 ** ait))
                  / (np.sqrt(_v / (1 - 0.999 ** ait)) + 1e-18))
            hit = False
            for n in range(N):
                if not len(inc[n]):
                    continue
                dv = np.zeros(T)
                for si in inc[n]:
                    dv = dv + _p[si] * ep[(int(C[si, 0]), n)]
                vv = np.asarray(V[:, n], float)
                want = (out_t[n] if n in out_t
                        else [b[0] for b in bumps_of(F[n])])
                pr = vv + dv
                cand = np.nonzero((vv[1:-1] < TH) & (pr[1:-1] > TH)
                                  & (vv[1:-1] > vv[:-2]) & (vv[1:-1] >= vv[2:]))[0] + 1
                for t in cand:
                    t = int(t)
                    if any(abs(t - q) <= GUARD_WIN for q in want):
                        continue
                    if any(abs(t - q) <= GUARD_WIN for q in spall[n]):
                        continue
                    L[n][t] -= GUARD * float(pr[t] - TH)
                    hit = True
            if hit:
                for si in range(len(C)):
                    k, nn = int(C[si, 0]), int(C[si, 1])
                    g[si] = float(np.dot(L[nn], ep[(k, nn)]))
        m = BETA1 * m + (1 - BETA1) * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - BETA1 ** ait); vh = v / (1 - 0.999 ** ait)
        step = lr / (1.0 + DECAY * ait)
        upd = step * mh / (np.sqrt(vh) + 1e-18)
        if TRUST > 0:
            worst = 0.0
            for n in range(N):
                if not spall[n] or not len(inc[n]):
                    continue
                dv_tr = np.diff(V[:, n], prepend=V[0, n])
                for s_ in spall[n]:
                    if not (0 <= s_ < T):
                        continue
                    dv = sum(upd[si] * ep[(int(C[si, 0]), n)][s_] for si in inc[n])
                    sl = max(abs(float(dv_tr[s_])), 1e-3 * TH)
                    worst = max(worst, abs(dv) / sl)
            if worst > TRUST:
                upd = upd * (TRUST / worst)
        if freeze is not None:
            upd = np.where(np.asarray(freeze, bool), 0.0, upd)
        if CLIFF_HOLD > 0:
            upd = np.where(hold > 0, 0.0, upd)
        prev_upd = upd.copy()
        if cb is not None:
            cb(it, w, upd, g, spall, F, L)
        w = wsign * np.clip(wsign * (w + upd), WMAG_MIN, WMAG_MAX)
        if KEEP_BEST:
            e, _Vcarry = out_err(w)
            if e < best_e:
                best_e, best_w = e, w.copy()
    return best_w if (KEEP_BEST and best_e < np.inf) else w
