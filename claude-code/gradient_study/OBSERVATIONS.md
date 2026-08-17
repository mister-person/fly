# What the dynamics say the gradient "is" (observations only)

Data from `inspect_neurons.py` on two tiny models. th = 0.007, delay 18, refrac 22.
No method proposed here — just what the voltages and spike times show.

## Model A — single synapse N0(driven) -> N1

Sweeping the one weight w:

- **Subthreshold voltage is exactly linear in w.** V(N1) = w·gsw·Σ_k h(t − t_k),
  where h is the LIF post-synaptic potential (PSP) kernel and t_k are N0's spikes.
  So `dV/dw` is a smooth, always-available trace: the presynaptic spike train
  convolved with h (panel A4 — the red dV/dw curve tracks the blue voltage ramp).

- **The threshold introduces exactly two nonsmoothnesses:**
  1. **Spike creation** — a true discontinuity. #spikes is a staircase in w
     (A2): 0→1 at w≈193, 1→2 at ≈445, 2→3 at ≈565. Gradient of spike *count*
     is 0 almost everywhere and undefined at the jumps. This is the part
     surrogate gradients paper over.
  2. **Spike-time shift** — once a spike exists, its time is a *smooth* function
     of w (A3: first spike slides 246→88 as w grows). The exact gradient is
        dt*/dw = −(dV/dw) / (dV/dt)  evaluated at the crossing.
     Empirically −0.70 vs predicted −0.65 (8%, discretisation). No surrogate
     needed — this is real signal.

- **The two per-neuron scalars that govern timing sensitivity:**
  slope `dV/dt` at the crossing (2.35e-3/step here) and margin `th − V` just
  below it (2.3e-5). Margin ≈ one slope-step. Small slope ⇒ large timing
  sensitivity; large margin ⇒ far from firing.

## Model B — chain N0 -> N1 -> N2

N0 spikes [1,101,201], N1 [88,188], N2 [175].

- **Upstream weight couples only through presynaptic spike *times*.** w01+20 moves
  N1 [88,188]→[79,179] and that alone moves N2 175→166. w12+20 leaves N1 fixed
  and moves N2 175→166 by scaling the PSP amplitude. Two different routes to the
  same downstream effect (panels B2–B4).

- **The backward signal is concentrated at spike events.** dV(N2)/dw01 and
  dV(N2)/dw12 (B3, B4) are near-zero between spikes and blow up at each
  crossing/reset — the loss gradient is naturally a sum of contributions injected
  at spike times, transported upstream by shifting presynaptic spike times.

## Structure this implies for the gradient

The exact gradient of a spike-time / van-Rossum loss decomposes as, per spike event:

    dL/dw  =  Σ_{spikes s}  (dL/dt_s) · (−1 / slope_s) · (dV/dw at t_s)
                                                     └ = presyn train ⊛ h

i.e. a per-neuron adjoint attached to each **spike**, converted between
voltage-space and time-space by 1/slope, and carried to upstream neurons by
perturbing their spike times. The smooth subthreshold PSP term (dV/dw) is exact
and free; the only genuinely hard object is the spike-*creation* discontinuity —
which is exactly the far-subthreshold / "dead neuron" regime where the sigmoid
surrogate's slope → 0 and where the earlier runs plateaued.

## Model C — two summing inputs A, B -> one neuron

Each input alone is subthreshold (w=300 < w*=445); only together do they cross
(coincidence detection). At the crossing t*, each input's **credit is exactly its
PSP contribution there**: `dV/dw_i = gsw · h(t* − t_i)`.

- **Coincident inputs (Δ=0) are degenerate:** creditA == creditB exactly, the two
  sensitivity columns are proportional, and only `wA+wB` is identifiable from the
  crossing. This is precisely the near-singular case that target prop's ridge term
  was added to resolve.
- **As the gap Δ grows, credit tilts to the earlier input** (0.50 → 0.67 by Δ≈60)
  because the crossing slides later and A's PSP is more developed there; past a
  coincidence window the PSPs no longer overlap enough to fire at all.
- So credit assignment among convergent inputs is well-defined and continuous
  *except* when they're simultaneous, where it is fundamentally ambiguous.

## Model D — dead neuron (single input swept through w*)

- **max V is linear in w; margin `th − maxV` is smooth and crosses 0 at w*=445.**
- **#spikes is a hard step at w*** — nothing continuous in the spikes themselves.
- **The decisive result — which losses keep a gradient while dead:**
  - a **spike-based loss** (spike-timing / count / van Rossum) has **exactly zero
    gradient** for all w<w*: no spikes ⇒ flat loss ⇒ dead neuron never revived.
  - a **voltage/margin loss** has a smooth, nonzero gradient throughout the dead
    region, pointing toward firing.
- So a silent neuron is invisible to every spike-based objective; only a signal
  that reads its subthreshold voltage/margin can bring it back.

## How C and D tie the whole picture together

- The **dead-neuron result explains the ~1e-2 plateau / local minima.** Once
  neurons go silent, spike-based losses give them zero gradient, so they stay
  dead — the optimiser is stuck in a basin it cannot feel. The `margin = th − maxV`
  is the smooth per-neuron scalar that *does* see the way out.
- It also explains **why soft homotopy escapes where the hard surrogate doesn't:**
  at low β the sigmoid lets subthreshold neurons "partially fire," manufacturing a
  nonzero gradient across the discontinuity (a smoothed margin signal); the sharp
  hard surrogate has slope→0 far from threshold, so it inherits the dead-neuron
  blind spot.
- Net: the exact gradient is (smooth PSP term) + (spike-time term); both are real
  and cheap. The one genuinely missing signal is for **spike creation in silent
  neurons**, and the data says the quantity that carries it is the subthreshold
  **margin**, not anything spike-based.

## Testing "PSP + spike-time" as an actual gradient (grad_method.py)

Built a faithful single-neuron LIF forward (validated: reproduces the JAX sim's
spikes [88,188] exactly) and the event-based gradient
`dL/dw = Σ_s (dL/dt_s)(−1/slope_s)(dV/dw at t_s)`, with `dV/dw` from a tangent
forward that respects resets.

**Where spike-time is brittle (the worry is justified — three distinct ways):**

1. **Discrete-time quantisation.** Spike times are integer steps, so any
   spike-time loss is a *piecewise-constant staircase* in the weights — its true
   finite-difference gradient is **0** almost everywhere. The `−1/slope·dV/dw`
   term is a continuous-time *relaxation*: it gives the right descent *direction*
   (cosine 0.95 vs how the integer spike actually jumps for a finite step) but the
   loss it targets doesn't move until a whole step flips.
2. **1/slope blow-up at grazing crossings.** As a spike approaches creation/
   deletion its slope dV/dt → 0; measured min slope 2.9e-6 drove the gradient norm
   to **3900× its median**. Create/destroy points are genuine discontinuities.
   Slope-clipping (floor on |slope|) tames the magnitude but not the direction.
3. **Zero signal for dead neurons.** No spikes ⇒ no terms in the sum ⇒ gradient
   exactly 0. Training a dead neuron with spike-time alone is **permanently
   stuck** (confirmed: stays at 0 spikes).

**What works — PSP/margin carries existence, spike-time carries timing:**
from a fully dead start (one output-spike target),
  - spike-time only  → stuck, 0 spikes (loss 200)
  - PSP/margin only  → revives but lands 22 steps late (loss 242)
  - **combined**      → revives *and* hits the target (loss 0.5)
The margin (voltage) term is the robust workhorse; spike-time is a *refinement*
on top, and only safe with slope-clipping.

**Coupling caveat:** when two target spikes are driven by the *same* weights, the
spike-time term thrashes (retiming one spike disturbs the other) — an
identifiability problem, not just numerics.

**Takeaway:** this is exactly why the voltage-based soft-homotopy is the reliable
method and pure spike-time schemes are fragile. A better gradient looks like
"margin/PSP for whether-and-that-it-fires + slope-clipped spike-time for when",
not spike-time alone.

## A better way to incorporate spike times (grad_voltage_target.py)

Don't differentiate the crossing *time* at all.  Put the timing target in
**voltage space**: to fire at t*, require the smooth membrane voltage to **reach
threshold AT t***.  The objective `Σ_j 0.5·relu(th − V(t*_j))²` is a smooth
function of V (linear in w), with gradient `(V(t*)−th)·dV/dw` — the PSP term,
evaluated at the target time.  The timing falls out for free: before t* the ramp
is still below threshold, so the first crossing lands at t*.

Results (from a fully DEAD start, single output neuron):

|                | single target 95 | coupled targets [90,190] | grad norm vs weight scale |
|----------------|------------------|--------------------------|---------------------------|
| spike-time     | stuck (0 spikes) | stuck (0 spikes)         | unbounded, 224× blow-up at grazing |
| voltage-target | **fires at 94**  | **fires at [89,189]**    | bounded ~1e-7, **exactly 0 at the solution** |

Why it fixes all three brittleness modes:
- **no 1/slope** — never differentiates a crossing time → gradient stays bounded
  (max ~1e-7 vs spike-time's blow-up to ~500);
- **revives dead neurons** — V(t*) is defined whether or not the neuron fires;
- **no discrete staircase** — the objective is smooth in w, with a genuine zero
  at the solution (it's 0 at 86% of the swept scales, i.e. wherever already
  satisfied) rather than the erratic never-zero spike-time gradient;
- **no thrashing on coupled spikes** — the two shared-weight targets are just two
  smooth voltage constraints; both are hit within 1 step.

This is literally target propagation's condition (V=th at each target time) solved
by gradient descent instead of least squares — which is why it inherits target
prop's robustness while staying a plain gradient.

Two honest caveats:
- **Timing precision is bounded by the crossing slope.** On a shallow ramp the
  exact step is fuzzy (±1–2), but it degrades *gracefully* here, where spike-time
  *explodes*.  A tried "don't fire early" term to sharpen it was self-defeating
  (the slow PSP can't be below th at t*−3 and at th at t*), so it was dropped.
- **Extra-spike suppression (added, grad_multi_neuron.py).** Symmetric downward
  force: at any spike with no target nearby, push V(t_s) below threshold
  `grad += λ·(V(t_s) − (1−m)·th)·dV/dw`. Confirmed: with only the "create" term an
  extra spike survives (fire-only leaves [62,254] when target is one spike); adding
  suppression removes it (→ [237]). Full objective = create (up) + suppress (down),
  visualised per neuron as a green/red force `−dL/dV` over time. Count is then
  controlled exactly; timing lands within the match window (~20 steps here) and
  would need the light slope-clipped spike-time nudge for step-exactness.
  Refractory note: pulses closer than the refractory period interfere (a spike
  clears the next pulse's drive), so this was shown with well-spaced inputs.

## Test suite (grad_test_suite.py) — 53/53 feasible cases, fails correctly on limits

Readout neuron, N independent well-spaced pulse-inputs; a case picks which pulses
should fire and an init pattern; PASS = exactly the target pulses fire (right
selection/count), checked by pulse window not exact time.

- **create** (dead init, k=1..4 targets), **suppress** (all-firing init, keep a
  subset incl. keep=0), **mixed** (random hi/lo init + random target),
  **count sweep** 0..4, **reproduce** a random feasible pattern, **many** (6
  pulses), **coincident** (2 credit-shared inputs/pulse): **53/53 PASS**.
- **Fails correctly** (validates the metric): pulses spaced 45 (< refractory
  interference) 0/3; asking one pulse to fire twice (impossible) fail.
- **Timing is coarse:** mean error ~14 steps, only 14% within 5 steps, 51% within
  15 — count/selection is exact but step-exact timing needs the slope-clipped
  spike-time nudge on top.  This is the clean split: voltage-target owns
  *whether/which/how-many*; a light spike-time refinement owns *exactly when*.

## Recurrent test (grad_recurrent.py) — layer-local target propagation

4-neuron nets, `chain` (N0→N1→N2→N3) and `feedback` (same + N2→N1 loop).  Each
iteration: simulate the whole net (JAX), then train every non-input neuron's
incoming weights *locally* with the voltage-target objective, using its
presynaptic neurons' current spikes as inputs, toward that neuron's target times.

- **The decomposition is exact.** The isolated single-neuron forward reproduces
  every hidden neuron's spikes in the full recurrent sim (validation OK on both
  nets) — so the local voltage-target solver is legitimately reusable here.
- **With oracle targets (every neuron's true times) it CAN fully reconstruct the
  net** (best count-err = 0 on both), but it is **seed-sensitive**: output-count
  recovered on 3/8 seeds (chain) and only **1/8 (feedback)**.  The feedback loop
  is the harder case — N1 needs N2's feedback spikes while N2 needs N1's, a
  chicken-and-egg that destabilises the local updates.
- **Timing stays coarse but does NOT blow up or compound.** Per-neuron error
  ~18–20 steps (the same WINDOW slop), and it does not snowball down the chain,
  because each layer re-anchors to its own oracle target — a real advantage of
  the target-prop decomposition over end-to-end spike-time backprop.
- **Output-only ≈ oracle in this tiny net** (1–3/8), so the inversion wall isn't
  sharply visible at N=4 (coarse counts, few spikes — the output sometimes hits
  the right count by chance).  The stronger evidence for it is the 50-neuron
  partial-observation experiment in [[performance-scaling-study]], where
  output-only clearly collapsed.

**Bottom line:** recurrence is architecturally *free* for this method — the local
voltage-target solver drops straight in and can reconstruct the net given targets.
The open problems are the same two as everywhere: **(1) reliability** (escaping
the initial wrong-presynaptic-spike regime, worst in feedback loops) and **(2)
assigning hidden-neuron targets** (inversion).  The gradient itself is no longer
the fragile part.

## Intermediate neurons with NO target information (grad_recurrent_hidden.py)

Chain N0→N1→N2→N3 (each reads only its upstream neuron), only N3's target given.
Three ways to handle hidden N1,N2: ORACLE (true times), NO-INFO (untrained,
frozen random), INVERTED (targets inferred as downstream-target − latency).

Latency inversion is accurate: inferred N2 target = true N2 exactly, inferred N1
= true N1 minus its last spike (latency 71 = the model's impulse-response lag).

But the training result is a cautionary surprise:

| handling of hidden neurons | output-count OK | best N3 output |
|---|---|---|
| ORACLE (true hidden times) | 2/6 | [194,294,394,494] |
| NO-INFO (untrained)        | 2/6 | [194,294,394,494] (identical) |
| INVERTED (inferred times)  | 0/6 | [318,518] (worse!) |

Three lessons:
1. **NO-INFO ties ORACLE here** — but that's a *confound, not a success*: the
   periodic input drives everything at ~100-step intervals, so the output spike
   *count* is set by the input structure, not the learned weights.  And N3 has a
   single input, so it can only *scale* it, not *retime* it — training the hidden
   neurons can neither help nor is it needed.  Count is a weak metric in a
   periodically-driven net.
2. **Naive inversion made it WORSE.** The inferred N1 target had 4 spikes vs the
   true 5, and the suppression term then killed a *load-bearing* hidden spike,
   breaking the downstream chain.  Forcing imperfect inferred targets (plus
   suppression) onto hidden neurons is more dangerous than leaving them alone.
3. So "how does it do with no-info intermediate neurons?" — **it doesn't fall
   over, but only because the input carries the structure; it also can't *use*
   hidden neurons to do real retiming, and inferred targets can backfire.**  A
   fair stress test needs a target that DEPARTS from the input-driven pattern
   (so hidden neurons must genuinely compute), and a readout with several inputs
   so retiming is even possible — that is the real open problem.

## The fair test (grad_hidden_fair.py): timing exposes the wall that count hid

Net: N0 drives a staggered hidden chain H1→H2→H3→H4, output O reads ALL four;
the output pattern [202,302] is set by *which* hidden neurons fire and *when*.

| hidden handling | output COUNT ok | best output timing |
|---|---|---|
| ORACLE (hidden = true times) | 4/6 | [188,322] — ~17 steps off target [202,302] |
| NO-INFO (hidden frozen random) | 4/6 | [141,209] — **~77 steps off** |
| READOUT-only (O sees all hidden) | 4/6 | [141,209] — same, no better |

**CORRECTED result (representative, not cherry-picked).** Comparing best-*count*
seeds earlier gave a misleading 17-vs-77; the honest per-seed spread of output
timing error is:
    ORACLE   6–17 steps  (median 12)
    NO-INFO  13–77 steps (median 16)
So a **multi-input readout largely DOES compensate** for no-info hidden neurons:
the median output timing is only modestly worse than oracle.  The per-neuron
graph (grad_hidden_fair.png) shows the hidden neurons genuinely fire at wrong
times/counts under NO-INFO (esp. H2, H3), yet the readout re-selects and combines
those 4 mis-timed hidden spikes to land the output ~16 steps off — because it has
several inputs at spread times to work with (unlike the single-input chain, where
no compensation is possible).

**Answer to "how does it do with no-info intermediate neurons?"**  It depends on
the readout's fan-in.  With a *single* input (chain) it cannot compensate at all.
With a *multi-input* readout it compensates fairly well on the median case
(~16 vs ~12 steps) but is **less reliable** — the worst seed is 77 vs oracle's 17.
So no-info hidden neurons don't break the output outright when the readout is
wide, but they cost reliability, and correct hidden-target assignment (which
naive inversion+suppression can still worsen) remains the real edge to improve.
The gradient is solid; credit-assignment reliability is the wall.

## Minimal reproduction of the stuck-spike: TWO neurons, no hidden (grad_stuck_2neuron.py)

The create/suppress "stuck at the wrong time" pathology needs no network at all —
one input pulse -> one output (single weight) reproduces it.  With the natural
crossing at t=62 and WINDOW=20, sweeping the requested target time shows a
**~45-step band of unreachable targets (40..85) that all freeze at ~62**:

- Within ±WINDOW of the natural crossing the spike "matches" the target, so all
  forces vanish — a hard **tolerance floor**.
- Just outside it, moving the crossing means changing the one weight, but the
  **create push (raise w → fire earlier) and the suppress push (lower w → kill the
  mistimed spike → later) directly oppose**, and there is a wide weight band where
  the net gradient ≈ 0 (also because after an early spike+reset, V at a later
  target time barely responds to w).  So the spike is frozen even though the
  target is perfectly reachable (here w=496 would put the crossing exactly on
  target).

**This is a flaw in the create/suppress OBJECTIVE, not the voltage-target
gradient.**  The binary "is there a spike within WINDOW / suppress spikes outside
it" design manufactures the dead band.

A continuous soft-van-Rossum force (hard spikes + surrogate slope) does remove the
dead band and gives step-exact timing on the 2-neuron sweep (grad_continuous.py:
band width 42 -> ~0, mean timing error 11.6 -> 0.0).

**But the real lesson (grad_direct_2neuron.py): for a fixed-input neuron this is a
LINEAR problem, not a gradient/annealing problem at all.**  The sub-threshold
output is `V(t) = w·gsw·h(t − t_in)`, linear in w, so the weight that puts the
crossing at t* is closed-form:
    w* = th / (gsw · h(t* − t_in))      (read off one sub-threshold probe)
One shot, no gradient, no window, no annealing: mean |achieved − target| = 0.34
steps over 35 targets.  This is exactly target propagation's 1-hop solve.

So the gradient/voltage-target machinery was solving a linear problem the hard way
(and inventing the dead-band artifact doing it).  Iteration/annealing is only
warranted when the problem is genuinely NONLINEAR — i.e. when a neuron's INPUT
spike times themselves depend on the weights (recurrence).  Even there,
target-prop iterates local *linear* solves rather than annealing.  The scope where
a smooth gradient actually earns its keep is narrower than this exploration
assumed.

## Does the direct solve generalise to HIDDEN neurons? (grad_direct_hidden.py)

The linearity is **local, and does not compose through a hidden threshold**:

- (a) `V_N1(t*)` is linear in N1's OWN incoming weight w01 — to machine precision
  (max deviation 6e-9) up to the point N1 spikes.
- (b) but the OUTPUT N2's crossing time is a **nonlinear** function of the hidden
  weight w01 (it passes through N1's threshold, with kinks where N1's spike count
  changes).  **So you cannot solve a hidden weight directly from the output
  target** — the clean `w* = th/(gsw·h)` only works one hop at a time.
- (c) it DOES generalise if you first **assign the hidden neuron a target time**:
  then two independent 1-hop solves (w01 for N1@t1*, w12 for N2@T2* given N1@t1*)
  hit the output target exactly.  And the hidden target is a *free choice* on a
  chain — t1* = 70, 80, 90 all land N2 at ~150.
- (d) **fan-out kills that freedom.** If N1 feeds two outputs wanting different
  times (N2@205 needs N1@134; N3@175 needs N1@104), the hidden target is
  over-determined — no single N1 spike time satisfies both, so no consistent
  1-hop decomposition exists.

**So the answer: it generalises to hidden neurons only as target propagation —
assign hidden targets, then local linear solves — and only cleanly for
feed-forward chains.**  Under fan-out (and recurrence, which is fan-out in time)
the hidden target is over-determined and there is no direct solve; that is the
real, nonlinear core of credit assignment, and it is exactly where every method
here — direct solve, target-prop, and the voltage-target gradient — hits its wall.
- Open extension: this is the single-neuron / given-input-spikes case.  In a
  recurrent net the hidden neurons' target *times* must be supplied or inverted —
  the same target-assignment problem seen in [[performance-scaling-study]]'s
  partial-observation experiment.

## Why a gradient (inspired by the linear method): least-squares beats the solve when the hidden neuron is over-determined (grad_overdetermined.py)

N0 -> N1(hidden) -> N2, two weights. N1's incoming weight sets ALL its spike times
together (periodic); the readout adds a constant latency. Ask N2 for 4 spikes with
IRREGULAR spacing [123,263,343,443] — no (period, latency) fits, so the hidden
neuron is over-determined.

- DIRECT solve (latency inversion): commits to individual constraints, nails the
  first (120~123) then the rigid period drifts off the rest -> RMS 27.
- GRADIENT (van-Rossum output loss, descended): finds the least-squares compromise
  [145,245,345,445], all four moderately close -> RMS 14.3.

The point: the gradient's dV/dw is the SAME exact linear PSP h the direct solve
uses — the only difference is the gradient minimises the residual across ALL
constraints jointly (least squares) instead of solving one hop exactly. That is
the freedom the direct solve lacks under over-determination/fan-out, and the
concrete reason to want a gradient here.

Design that falls out (the linear-inspired gradient):
- loss on the OUTPUT only (van-Rossum on spikes — smooth, no window, step-exact
  timing per the 2-neuron test);
- backward uses the exact linear PSP h (= dV/dw) for every synapse — each synapse's
  credit is its PSP contribution, exactly as in the direct solve;
- the one nonlinear factor (a neuron's spike time vs its own weight) is the
  surrogate/(1/slope), needed only at neurons that actually fire (robust); a
  Gauss-Newton variant can use the per-neuron linear solve as a PRECONDITIONER so
  each step is a proper 1-hop solve rather than a small nudge.

This is essentially the project's surrogate-BPTT / soft-homotopy gradient, but the
exploration pins down its leverage: joint least-squares over coupled/fan-out
constraints, which the per-neuron direct solve provably cannot do. Next test:
scale the analytic linear-PSP gradient (not finite-difference) on the recurrent
nets and compare head-to-head with iterated target-prop.

## 2-neuron breakage tests for the van-Rossum gradient (grad_2neuron_breakage.py)

Stress-testing the linear-inspired gradient on 1 input -> 1 output (single weight)
before scaling. Predictions vs results:

| case | target | achieved | result | note |
|---|---|---|---|---|
| reachable | [80] | [80] | PASS | rising phase |
| dead-start | [80] | [] then [80] | FAIL->PASS | sharp surrogate can't revive; **wide surrogate beta=4 fixes it** |
| unreachable-late | [150] | [124] | FAIL (graceful) | past PSP peak; lands on the latest reachable crossing |
| over-count | [60,90] | [88] | FAIL (graceful) | one pulse -> one spike (probed: even w=5000 gives 1); compromise |
| refractory | [60,68] | [] | FAIL (UNgraceful) | targets < refractory; drives the neuron DEAD |
| selective-sub | [80,280] | [280] | FAIL | single weight is all-or-none across pulses |
| count-3-match | [80,180,280] | [80,180,280] | PASS | consistent per-pulse timing |
| near-peak | [120] | [120] | PASS | low slope is NOT a problem (no 1/slope division) |

Findings:
- **Two break modes, both understood:** (1) dead start — the sharp surrogate has
  no revival gradient, fixed by a wider surrogate / annealing (confirmed); (2)
  physically impossible targets (past the PSP peak, more spikes than pulses,
  sub-refractory spacing, selective firing under a single all-or-none weight) —
  which *any* method must fail.
- **Failure is mostly graceful** (closest reachable crossing, or a timing
  compromise) EXCEPT the sub-refractory case, which drives the weight to no spike
  at all — an ungraceful "kill the neuron" mode worth a keep-alive/margin safeguard.
- **near-peak passes:** the low-slope region that wrecked the spike-time gradient
  (1/slope blow-up) is a non-issue here, because the van-Rossum gradient never
  divides by the slope.
- Net: on 2 neurons the method's only real weakness is revival (needs a wide
  surrogate) and ungraceful death on infeasible targets; everything reachable and
  count-consistent works, including near-peak. These are the things to guard when
  scaling.

## Revival is a loss BARRIER, not a vanishing gradient — and the linear method's objective has none (grad_principled.py)

Trying to fix dead-neuron revival WITHOUT a schedule exposed the real cause.

- A non-saturating surrogate (surrogate_slope + REV*relu((th-V)/th)) does make the
  gradient nonzero when dead — but dead-start still stalls at the firing boundary.
- Why: the TRUE van-Rossum loss for target [80] is *barriered*. Dead (w=443) loss
  5.52; first spike (w=460, born LATE at 102, far from 80) loss **9.81 (worse)**;
  good (w=550, spike 76) loss 3.64.  Creating a spike temporarily RAISES a
  spike-density loss because the new spike is born far from the target, so no
  monotone-descent method crosses it.  This is exactly why the direct/linear solve
  (which jumps straight to the answer) beats gradient descent on revival, and why
  annealing (barrier-smoothing) was the schedule-based workaround.

- **Schedule-free fix, straight from the linear method:** descend its own residual
  `(V(t*) - th)^2`.  `V(t*)` at a FIXED target time is *monotonic* in w, so this
  loss has NO barrier and no vanishing gradient, and its gradient is exactly the
  linear PSP `dV/dw`.  Verified: dead w=100 -> 525, spike at 81 (target 80), no
  schedule.

- **But voltage-target and van-Rossum are COMPLEMENTARY by direction:**
  - `(V(t*)=th)` (linear objective): revives + moves a spike EARLIER (monotonic,
    no barrier); FAILS to move a spike LATER (then V(t*) is post-reset -> pushes w
    up -> fires even earlier).
  - van-Rossum: moves a spike LATER + suppresses extras + precise timing; has the
    revival barrier and can't move a spike earlier past the barrier.
  The sign of (t_spike - t_target) decides which one is correct.  Neither alone is
  complete; a principled unified loss must switch/blend on that sign.

- **Keep-alive is principled here, not bolted on:** the revival term is gated by
  the loss residual, so a target of silence (no target spike) produces no deficit
  and the neuron is driven silent — verified (silent-target -> [] correctly).  The
  naive "always keep >=1 spike" keep-alive would instead wrongly fire a
  target-silent neuron; this one can't, because keep-alive is a *consequence* of
  the objective.

## Unified direction-free objective: sub-threshold-voltage target (grad_unified.py)

The move-earlier/move-later split came entirely from reading the POST-RESET
voltage at the target.  The SUB-THRESHOLD voltage V_sub(t) = w·gsw·h(t-t_in)
(no reset) is MONOTONIC in w, so descending (V_sub(t*) - th)^2 reaches the target
from EITHER side — no barrier, no direction switch.  This is gradient descent on
the direct-solve / linear-method residual, done per epoch (reset the h
accumulation at each target), plus a suppression term for input pulses no target
claims.  All 10 two-neuron cases:

| case | result | | case | result |
|---|---|---|---|
| reachable | PASS | count-3-match (from dead) | PASS |
| dead-start | PASS (revives, no schedule) | near-peak | PASS |
| silent-target | PASS (goes silent) | move-later | PASS |
| unreachable-late | GRACE (alive) | move-earlier | PASS |
| over-count | GRACE (alive) | refractory | GRACE (alive, not dead) |

Why it works: V_sub is monotonic in w, so the residual has a single minimum at
w = th/A (= the direct solve) reachable from both sides — this is exactly why the
barrier (revival) and the directional failure both disappear.  Notes:
- For a SINGLE neuron this converges to the direct solve (gradient descent on a
  quadratic-in-w residual); the gradient's extra value over the direct solve is
  the least-squares behaviour on OVER-DETERMINED / multi-input systems
  (grad_overdetermined.py).
- move-later needed ~1500 Adam iters (long descent, tiny gradients ~1e-7 since
  A~1e-5) — an optimiser detail, not an objective problem.
- The probe kernel must come from the REAL forward (a sub-threshold lif_tangent
  probe), NOT the hand-computed impulse response — the analytic kernel was ~20%
  off and put the crossings in the wrong place.
- Per-epoch reset needs epoch boundaries = the neuron's target times; for hidden
  neurons those are the (assigned/inverted) hidden targets — so this cleanly
  reduces the remaining problem to target assignment, nothing else.

## The hard cases: unified TP on the 50-neuron recurrent benchmark (grad_unified_recurrent.py)

Unified objective as the local least-squares solver in target propagation, ORACLE
hidden targets (every neuron given its TRUE spike times), on the 3 recurrent cases:

| case | target | best output | cnt | loss |
|---|---|---|---|---|
| recurrent_132 | 7/2/1 | 7/2/0 | 2/3 | 1.4e-2 |
| recurrent_304 | 4/2/7 | 8/2/7 | 2/3 | 1.6e-2 |
| recurrent_240 | 3/3/7 | 5/4/9 | 0/3 | 1.3e-2 |

**It lands on the exact same ~1e-2 plateau as every prior method** (soft homotopy,
hard surrogate, CMA-ES, DE, NNLS target prop), does NOT reach the 3/3 that soft
homotopy hit on cases 0/1, and does NOT solve case 240.  Note the errors are BOTH
over- and under-shoot (132: N49 missing; 304: N47 overshoots) — not just a missing
suppression term.

**Conclusion — and it's the whole project's conclusion, re-derived from the other
side:** the unified objective is an excellent *local* solver (all the isolated
2-neuron / multi-input / chain cases pass), but the local solver quality is NOT
the bottleneck for the recurrent benchmark.  The wall is the recurrent COUPLING:
each neuron's local solve uses its presynaptic neurons' CURRENT (wrong) spike
times, and the local solves do not compose to the global solution — the fixed
point of the iteration is not the true weights.  Even with oracle targets removing
the target-assignment problem entirely, the coupling alone pins every method to
~1e-2.  So the frontier is not a better per-neuron gradient (that is solved) but
the recurrent credit-assignment coupling itself.

## Unrolling the recurrence: the failure is DEPTH compounding, not the cycle

The project already had the minimal example (tp_compounding.py): a pure
FEED-FORWARD chain N0->N1->...->NL, all true w=500, each hop solved by target prop
from true inputs+targets.  Output timing error grows ~linearly with depth
(~2 steps/hop with th RHS), and the output goes SILENT by L=16 — no cycle needed.

Re-ran the same depth sweep with the UNIFIED objective (exact probe kernel,
V_sub=th, grad_unroll.py): per-hop error drops from ~2 to ~1 step/hop (the exact
kernel helps), but it STILL compounds linearly (L=2->err2, L=4->4, L=8->8).

**So: unrolling a recurrence into a deep chain still triggers the failure.  The
cycle is irrelevant — DEPTH is the cause.**  The mechanism is discretisation: even
an exact local solver leaves ~1 step of timing error per hop (integer spike times
+ threshold overshoot: the real neuron crosses at the first step where V>=th,
slightly before V_sub=th exactly).  That per-hop error accumulates linearly with
the number of hops the signal traverses.  A recurrent net unrolls to a deep chain,
so the recurrent plateau IS this compounding — and it is the same ~1 step/hop
timing floor seen everywhere, multiplied by depth.  This is why a better per-neuron
solver alone cannot fix the recurrent case: it lowers the per-hop constant but not
the linear-in-depth growth.  Fixing it needs either (a) a per-hop error that is
zero, not just small (e.g. RHS = actual crossing voltage, sub-step correction), or
(b) solving the chain jointly rather than hop-by-hop so errors cannot compound.

## The step-nudge: kill the per-hop rounding error so it can't compound (grad_unroll.py)

The per-hop error is a rounding artifact: V_sub(t*)=th aims the EDGE of the valid
weight interval (the crossing barely lands on t*), so discretisation tips it ~1
step.  Nudge to the CENTRE of the step-interval instead:
  crossing lands on t* for  w in [th/A(t*), th/A(t*-1));  aim the MIDPOINT.
Equivalently, raise the target level by half the per-step rise:
  target = th + 0.5·(V_sub(t*) - V_sub(t*-1)).

Depth-compounding chain, EDGE vs NUDGED output error:

| L (hops) | edge err | nudged err |
|---|---|---|
| 2 | 2 | 0 |
| 4 | 4 | 0 |
| 8 | 8 | 0 |
| 16 | 16 | 0 |

The nudge gives EXACT output up to the depth the chain sustains (it dies out
naturally past L~16 even at true w) — the compounding is GONE, not just smaller.
So the depth-compounding failure mode is fixable with a one-line change to the
target level (center-of-step instead of edge-of-step).  Caveat: this removes the
DISCRETISATION ingredient of the recurrent failure; the other ingredients
(correlated-input fragility, and local solves that use wrong presynaptic times)
are separate and still open — next test is whether the nudge alone moves the
50-neuron cases off the plateau.

## Nudge on the 50-neuron cases: doesn't move them — the binding ingredient is correlated-input fragility, not discretisation

Added the step-nudge to the 50-neuron unified target-prop local solver.

- Iterated (current presynaptic times): EDGE vs NUDGED essentially the same
  (case0 2/3 both; loss even rose slightly with the nudge on cases 1/2).
- Single pass with ORACLE (true) presynaptic inputs: whole-network spike-count
  match is already poor with true inputs — case0 26/50, case1 2/50, case2 8-11/50
  — and the nudge changes it negligibly (case2 8->11, others flat/worse).

So the nudge, which gave EXACT output on the single-input depth chain (16-step
error -> 0), does not help the 50-neuron networks.  Reason: those neurons are
MULTI-input with CORRELATED inputs, and the per-neuron least-squares gives a
weight split that reproduces firing on the TRUE inputs but does not generalise
(tp_compounding.py Part 2) — case1 collapsing to net=2/50 even with oracle inputs
is exactly that fragility.  The step-nudge fixes the DISCRETISATION/depth ingredient
(single-input compounding) but not the correlated-input-split ingredient, which is
what actually binds on the recurrent benchmark.

**Final decomposition of the recurrent failure:**
  (1) discretisation / depth compounding  -> FIXED by the step-nudge (chain: 16->0)
  (2) correlated-input weight-split fragility -> the binding constraint at 50 neurons;
      NOT fixed by the nudge (needs regularisation/joint solve, and even ridge caps at ~1e-2)
  (3) recurrent coupling (local solves with wrong presynaptic times) -> compounds (1)+(2)
The per-neuron gradient story is complete; the open problem is (2): choosing, among
the many weight splits that fit correlated inputs, the one that GENERALISES under the
input drift the full recurrent sim produces.

## Fix for ingredient (2): drift-robust split selection (grad_robust.py)

The fragility is that the fit is under-determined (rank-deficient) and the fitter
picks the least drift-robust corner.  Fix: among the splits that fit, minimise the
output's sensitivity to the FRAGILE drift mode.  For two inputs that mode is
average-preserving (A+d, B-d), whose per-epoch output sensitivity is
w_A·h'(t*-t_A) - w_B·h'(t*-t_B); minimise ||P w||^2 (P = derivative matrix with the
2nd input negated) subject to the fit K·||Aw-th||^2.

Result on the minimal correlated-input examples: the robust split has
drift-sensitivity **0.00 at every gap** (0..35), vs NNLS 1.00, ridge 0..-0.38,
and even the true/balanced weights -0.5.  It finds a split that fits the true
inputs AND is fully invariant to the jitter — without being told the true weights,
using only the fit plus the derivative (jitter-sensitivity) structure.

So all three ingredients of the recurrent failure now have a targeted fix:
  (1) discretisation/compounding -> step-nudge (center-of-step)  [chain: 16->0]
  (2) correlated-input fragility -> drift-robust split           [minimal: drift->0]
  (3) recurrent coupling         -> still requires iterating (1)+(2) to a fixed point
The n-input generalisation of (2): the fragile modes are the NULL SPACE of the fit
matrix A; penalise the output sensitivity (via the derivative matrix D) projected
onto that null space.  Next: wire (1)+(2) into the 50-neuron local solve and see
if targeting robustness (not just fit) finally moves off the ~1e-2 plateau.

## All three fixes on the 50-neuron cases (grad_robust_recurrent.py): coupling still dominates

Wired nudge (1) + drift-robust split (2) into the iterated 50-neuron local solve,
oracle targets, vs fit-only:

| case | target | fit-only | +nudge+robust |
|---|---|---|---|
| 132 | 7/2/1 | 7/2/0 (2/3) | 6/2/0 (1/3) |
| 304 | 4/2/7 | 4/3/8 (1/3) | 7/4/8 (0/3) |
| 240 | 3/3/7 | 7/3/11 (1/3) | 3/3/9 (2/3, loss 1.3e-2) |

Mixed and seed-noisy: robust HELPS case 240 (2/3 — and it hits N47=3, N48=3
EXACTLY, the two outputs every prior method left stuck) but regresses 304; still on
the ~1e-2 plateau overall.  So the minimal-case fixes for (1) discretisation and
(2) fragility do not by themselves crack the recurrent benchmark: ingredient
(3), the coupling, dominates — each neuron's local solve still uses its
presynaptic neurons' CURRENT (wrong) spike times, and the per-neuron fixes cannot
compensate for inputs that are themselves off.  The one encouraging signal is
case 240's N47/N48 hitting target exactly under the robust split, suggesting the
fragility fix matters where inputs are already good; making it consistent needs
(3) — inputs and weights converging together (joint/block solve or a schedule on
the presynaptic-time mismatch), which is the standing open problem.

## Loop on ingredient (3): the coupling toy DOESN'T fail; the real 50-neuron failure is OVER-FIRING

Made a minimal feedback toy N0->N1->N2->N3 with feedback N2->N1 (grad_coupling.py)
to isolate coupling.  With nudge+robust it recovers the true weights EXACTLY, both
iterating on current spikes AND one-pass on target spikes (net 4/4, w~[501,501,51,501]).
So the simple recurrent coupling is NOT the 50-neuron culprit.

Single-pass with TRUE inputs + nudge + robust on the 50-neuron cases, with a
failure breakdown: mismatches are almost entirely EXTRA spikes
(case0 extra=24, case1 extra=49, case2 extra=42; MISSING=0 everywhere).  The
neurons fit their target crossings but also fire at non-target times -> the solve
has no SUPPRESSION.  The feedback toy worked because its neurons have no spare
crossings to suppress.

Adding upper-bound suppression cuts (grad_supp.py) barely helped (net 11->12,
1->1, 2->2; extra unchanged).  The reason: the real sim resets at the EXTRA spikes,
so a target-reset V_sub model mis-locates the cuts.  tp_50neuron's proper QP upper
bounds DID suppress and reached ~1e-2 — but no further.

**Net of this iteration:** the 50-neuron plateau is not a single removable
ingredient.  The three isolated fixes (nudge, robust split) each work perfectly on
minimal examples; the coupling toy doesn't even fail; yet at 50 neurons the local
solve over-fires, proper suppression only claws back to ~1e-2, and that ~1e-2 is
the genuine non-convex floor the project identified at the start.  The per-neuron
story is complete and the failure modes are now individually understood and fixed;
the 50-neuron plateau is their *joint* interaction under correlated many-input,
must-suppress, coupled dynamics — which is exactly "the landscape is non-convex",
now with named, minimally-fixable components rather than a black box.

## The actual minimal failing example (grad_minimal_fail.py) — and the real cause

Earlier I never made a minimal example that truly reproduces the 50-neuron failure
(the feedback toy didn't fail; the "suppression" story was a guess).  Extracting a
real neuron does reproduce it:

  N19 (case 304), 4 inputs {N4,N12,N21,N37}, fed their TRUE spike times in isolation.
  true weights [77,248,57,138] -> fires exactly at target [586,756,886,993].
  reconstructed (fit) [96,370,66,14] -> fires [299,592,764,886,992] -- an EXTRA spike.

Diagnosis, step by step:
- Superposition HOLDS (V_full = V_superposition up to the first spike) -> not a
  nonlinearity.
- But the fit's own kernel gave V_sub(585)=0.00516 vs the real 0.00694 -- a ~25%
  under-estimate.  Cause: the reconstruction kernel was only 340 steps long, so it
  TRUNCATED the slow-decaying membrane tail (time constant ~200) of inputs that
  fire far before the crossing (N4@139 is 446 steps back but still ~10% present).
- With a LONG kernel (1100) and the correct crossing times, A@w_true = th at every
  target and the fit recovers [73,255,59,134] ~ true, firing exactly the target,
  NO extra spikes.

So the over-firing was NOT missing suppression or coupling -- it was a **kernel
truncation bug** in the reconstruction model, plus a 1-step misalignment between
the full JAX sim's spike times (used as targets) and the reconstruction model's
crossings.

Fixing the kernel length (grad_unified) reduces 50-neuron over-firing substantially
(case0 extra 24->17, case2 42->26) but does not by itself restore the net match,
because the target times still come from the full sim and are ~1 step off the
reconstruction model -- and that residual misalignment, compounded over 50 neurons,
is now the dominant corruptor.  Lesson (again): build the minimal reproducing
example FIRST; it found a concrete bug that all the higher-level theorising missed.

## Alignment "bug" was a rounding artifact; the kernel fix isolates the real wall

The suspected 1-step full-sim-vs-model misalignment was NOT real: with the ACTUAL
(unrounded) weights, lif_tangent matches the JAX sim step-for-step (identical
voltages, same spikes).  My earlier 586-vs-587 came from rounding the weights in an
inline check.

With the kernel-truncation fixed (long probe) the local solve is now GOOD: each
neuron reconstructed from TRUE inputs reproduces its target IN ISOLATION for
  case0 33/39, case1 29/49, case2 33/48  active neurons (67-85%).
But the COUPLED full sim still collapses (net 12/50, 1/50, 2/50).

So the kernel bug was masking the true situation.  Now cleanly separated:
  - per-neuron local solve: mostly correct (67-85% isolated) once the kernel is right;
  - the collapse is the COUPLING: reconstructed presynaptic weights differ slightly
    from true, so their spikes drift a step or two, and downstream neurons that were
    fit to the TRUE input times mis-fire on the drifted times, cascading to ~1/50.

This is ingredient (3) in its pure form, finally isolated from the model-fidelity
bug: the single-pass fixed point (fit each neuron to current inputs) is not the
true weights because every neuron's inputs drift as its upstream is retrained.
The lever is a fixed-point/iterated solve where inputs and weights co-converge, and
a reconstruction robust to the small presynaptic drift it induces (drift-robust
split) -- but crucially, the local-solve quality is no longer the bottleneck.

## Global gradient over every neuron (grad_global.py): same plateau

Tried the "ML model" route: one loss over ALL neurons (deep supervision to the true
soft van-Rossum traces), one gradient through the whole recurrent net via the
differentiable soft-sim, Adam, fixed beta=12 (no schedule).  Unlike the greedy local
solves, this gradient DOES account for coupling (backprop credits an upstream weight
for its downstream effect).

Result: net 14/50, 1/50, 2/50 ; output count 0/3 ; loss ~1e-2 -- the SAME plateau as
local TP and every other method.  Two honest points:
- This is essentially the project's own soft/hard global-gradient method, and
  supervising ALL neurons (vs just outputs) does not break the plateau -- the
  project already found N_TRAIN=50 gives the same floor.
- Fixed beta (no schedule) is notably worse than the ANNEALED soft homotopy, which
  is the project's best (exact on cases 0/1).  I.e. the one thing that reliably helps
  the global gradient escape the plateau is beta annealing -- a schedule.

So the global gradient confirms, from the ML-training side, what the per-neuron side
showed: the ~1e-2 floor is the non-convex landscape / local minima, not the credit
rule.  A single global gradient at fixed smoothness gets stuck in the same basins;
annealing the smoothness (soft->hard) is the known escape but caps at 2/3 cases and
is exactly the schedule we were trying to avoid.

## Minimal example that defeats the GLOBAL gradient: the timing fork (grad_global_minimal.py, grad_global_vt.py)

Smallest failing net: 3 neurons, N0(input)->N1->N2 plus a bypass N0->N2 (also the
4-neuron N0->N1->N2->N3 + bypass N0->N3).  A slow path and a fast bypass to the same
neuron -> two basins.  True weights use the SLOW chain (bypass weak, w=50); a wrong
basin fires via the FAST bypass.

The global gradient (deep supervision over ALL neurons, soft-sim, Adam) FAILS on it
in every variant tried:
  - van-Rossum loss, fixed beta   -> UNDER-fires: N2 collapses to 1 spike, N3 dies
    (loss ~65 all along true<-found, drops to 0 only at true: a barrier).
  - van-Rossum loss, ANNEALED beta -> also fails (collapses).
  - VOLTAGE-TARGET loss (barrier-free per neuron), fixed beta -> OVER-fires via the
    bypass (output fires ~73 instead of ~195), even with suppression LAM up to 100.

So the failure is a genuine two-basin local minimum with a spike-CREATION barrier
between them: to move from the bypass basin to the chain basin you must transiently
kill the output (add chain spikes it doesn't yet have), which every spike/voltage
loss penalises on the way.  This is the project's ~1e-2 non-convexity distilled to
3-4 neurons -- and it defeats the global gradient with any fixed-beta loss, the
per-neuron voltage-target trick, and suppression tuning.  The only thing the project
ever found to help is beta annealing (a schedule), and even that fails on this exact
net (minimal_failures.py).  This is the irreducible core: a coupled creation barrier
that neither a better per-neuron rule nor a single global gradient escapes.

## The fork is SOLVED by the direction-free latency-crediting V_sub solve (grad_fork_test.py)

The 3-neuron fork defeated every GLOBAL-gradient variant (van-Rossum, voltage-target,
annealed beta, suppression).  It is cleanly SOLVED by the local sub-threshold
voltage-target (V_sub=th) solve:

  A) N2 alone, true inputs: recovers the TRUE split chain=503 / bypass=48 and fires
     exactly [129,237,337].  No suppression, no backward flow -- the latency credit
     h(57) (chain) > h(28) (bypass) makes the fit prefer the chain by itself.
  B) whole fork, oracle N1+N2 targets, iterated V_sub solve: 4/4 seeds recover
     w~[501,503,48] (=true), N2 exact.
  C) OUTPUT-ONLY, N1's target INFERRED by the backward latency message
     (N2_target - 57): 4/4 seeds recover, N2 exact.

Why it wins where the global gradient loses:
  1. sub-threshold V_sub (no reset) is MONOTONIC -> no move-later barrier; the
     global voltage/van-Rossum losses read the POST-RESET voltage and cannot push
     an early (bypass) spike later.
  2. the fit credits each input by its PSP contribution h(t*-t_pre) -> automatically
     prefers the path whose latency matches the target (chain), which the global
     gradient's post-reset dynamics destroy.
  3. per-neuron solve -> no cross-neuron gradient to fall into the wrong basin;
     coupling handled by (mild, here) iteration.

Traced confirmation earlier: at bypass-basin points the GLOBAL gradient pushes the
bypass UP (cos -0.28 vs the true direction) exactly because of (1)+(2).  The V_sub
solve inverts that.  Big open question: this is a small fork with mild coupling;
whether the same V_sub solve (fixed kernel, no over-suppression, iterated) now moves
the 50-neuron cases -- where coupling is severe -- is the next test.

## YES: the V_sub method credit-assigns through hidden neurons with NO known data (grad_infer_test.py)

Only OUTPUT neurons get a target.  Hidden targets are INFERRED by the backward
latency message (presynaptic should fire ~LAT before each downstream target;
fan-out neurons take the union), LAT=71 measured from a single-edge probe (model
knowledge, not target data).  Then the direction-free V_sub solve + iteration.

  - CHAIN N0->N1->N2->N3, output N3 only: inferred N1,N2 targets (N2 inferred =
    true N2 EXACTLY, 3 hops) -> outputs recovered 4/4 seeds.  Inference error does
    NOT compound.
  - FANOUT N0->N1->{N2,N3}, outputs only: N1 target inferred by aggregating both
    -> 4/4.
  - HARD FANOUT N1->{N2(w500 early), N3(w200 late)}: conflicting latency demands;
    inferred N1 target is slightly WRONG (spurious 246, missing 472) -> STILL 4/4.
    The method is robust to imperfect hidden targets.

So the whole thing works as a real learning rule: output supervision only, hidden
targets inferred by the backward PSP-latency message, forward V_sub solve, iterate.
No 1/slope (threshold only crossed forward), no oracle hidden data.  Caveats: LAT is
fixed (fine when latencies are similar; the hard fanout still worked at w=200 vs
500); these are feed-forward; and it is small-scale.  Open: adaptive per-edge
latency, cycles, and whether output-only + inferred targets holds at 50 neurons.

## Breaking case: different downstream latencies defeat the fixed-latency inference

Net N0->N1->{N2, N3} with w=[500, 900, 470]: one N1 spike (72) produces N2 at
latency 38 (strong synapse) and N3 at latency 82 (near-threshold synapse).  The
fixed LAT=71 backward message splits that ONE N1 spike into TWO inferred targets:
110-71=39 (from N2) and 154-71=83 (from N3).  Inferred N1 target becomes
[39,83,139,183,...] -- 9 spikes, DOUBLED per period -- vs true N1 [72,172,272,372,472].
N1 is told to fire twice per period where it fires once -> both outputs break ->
0/4 seeds recovered.

Root cause: the per-edge latency depends on the (unknown) downstream weight
(strong synapse = short lag, near-threshold = long lag), so a single fixed LAT is
wrong for divergent-strength fan-out.  The fix is ADAPTIVE latency: measure each
edge's current lag from the running sim (downstream_spike - presynaptic_spike) and
use that in the backward message, re-estimating as weights converge.  (The method
simulates every iteration, so this data is already available -- the only reason it
broke is that the inference ignored it and used a constant.)

## Divergent-latency fan-out: gauge-anchored clustering (grad_infer_adaptive.py)

The break case: hidden N1 fans out to N2 (w900, latency 38) and N3 (w470, latency 82).
A FIXED backward latency LAT=71 infers two different N1 times (t_N2-71, t_N3-71) that
don't merge -> N1 doubled to 9 spikes -> 0/4 recovered.

Weight-derived per-edge latency was tried and rejected (each traded the uniform cases
for the break):
  strategy              break  chain  fanout-eq  fanout-hard
  fixed 71 (orig)        0/4    4/4      4/4        4/4
  measured-from-spikes   1/4    1/4      2/4        3/4
  model_lat(w) each iter 3/4    1/4      1/4        2/4
  EMA of model_lat       1/4    2/4      4/4        0/4
  converge-then-refine   0/4    3/4      4/4        0/4
Root cause: latency tied to TRANSIENT weights jitters during bootstrap and, in a
chain, the per-edge latency errors compound with depth. model_lat(w) itself is exactly
calibrated (matches the real single-edge lag 71/47/38/82) -- the instability is the
weight-dependence, not the estimator.

FIX (4/4 on ALL four cases, weight-free, schedule-free): gauge-anchored clustering.
  1. Group downstream target spikes so each group holds at most one spike per
     downstream neuron (a repeated neuron label starts a new group -- no window tuning;
     CLUSTER_WIN is only a loose guard).
  2. Each edge has ONE global latency L_d (one weight = one latency). Measure each
     edge's OFFSET relative to the most-firing downstream (median over shared groups).
  3. Choose the gauge so the LATEST edge sits at MAX_LAT (=82, the longest single-spike
     latency a firing weight can realise): L_ref = MAX_LAT - max_offset. Hidden times =
     ref_targets - L_ref.
On the break this infers N1 = [72,172,272,372,472] EXACTLY (== true), because the
gauge places the slow edge (N3) at its true 82 and the fast edge (N2) at 38. It also
resolves the lone-5th-spike inconsistency (N2 fires 5x, N3 4x): anchoring to the
single reference keeps the N1->N2 latency uniform across all spikes.

### Follow-up probes (grad_latency_probe.py)

Q1 "latency depends on the receiving neuron's voltage": TRUE but does NOT break the
method. Latency enters only as a coarse ANCHOR for the hidden target; the actual weight
fit (solve_vsub) uses real presynaptic trains and sums the kernel since the last reset,
so accumulation/voltage state is already handled. Crucially the clustering reads each
edge's latency OFFSET from the actual target times, so voltage-driven latency is
absorbed automatically.
  Q1a  N2 = chain(N1) + bypass(N0): bypass pre-charges N2 -> N1->N2 eff latency 29 (vs
       from-rest ~71).  clustered inference recovered 4/4.
  Q1b  fan-out, EQUAL weights (both 500), latencies split by VOLTAGE (bypass pre-charges
       N2 only): N1->N2 lat=29, N1->N3 lat=71.  A weight-derived model_lat would give
       BOTH 71 and miss this; clustering reads offset=42 from targets -> recovered 4/4.
  (Output-only supervision -> inferred N1 need not equal true N1; output recovery scored.)

Q2 "does a smaller learning rate fix the weight-latency jitter?": NO. Sweeping the
model_lat-every-iteration scheme over alpha=0.5/0.2/0.1 (rounds scaled up to keep total
travel equal) stays flat & noisy, never approaching 4/4:
  net           cluster  a=0.5  a=0.2  a=0.1
  chain           4/4     1/4    2/4    1/4
  fanout equal    4/4     1/4    0/4    1/4
  fanout hard     4/4     2/4    1/4    1/4
  break           4/4     3/4    3/4    3/4
=> the degradation is BIAS (moving-target coupling + depth-compounding), not step-size
variance; LR can't average it away.  Weight-free gauge-anchored clustering is 4/4 across
all cases.

## Gauge-anchored clustering on the 50-neuron recurrent benchmark: does NOT transfer

The 50-neuron RECURRENT_CASES are >50% back-edges (case0 72/132, case1 158/304,
case2 121/240) -- no topological order.
  * OUTPUT-ONLY clustered inference is a non-starter: acyclic latency-backprop from the
    3 output neurons reaches 0/46 hidden neurons on ALL three cases (nothing to fill,
    every hidden neuron sits in a cycle).
  * ORACLE-target ceiling (V_sub local solve given TRUE targets, nudged):
        case0 target 7/2/1 -> 7/2/0  (2/3 counts, loss 1.9e-2)
        case1 target 4/2/7 -> 6/4/7  (1/3 counts, loss 2.0e-2)
        case2 target 3/3/7 -> 7/3/10 (1/3 counts, loss 1.3e-2)
    Soft homotopy solves case0 & case1 EXACTLY; the local solver, even with oracle
    targets, stays at the ~1e-2 plateau below it.

Conclusion: the gauge-anchored clustering genuinely fixes divergent-latency FAN-OUT on
feedforward structure, but the 50-neuron benchmark's difficulty is RECURRENCE (coupled
spike times through cycles), which the local per-neuron V_sub solve does not resolve and
which blocks acyclic target inference outright.  Recurrence remains the open problem;
soft homotopy is still the method to beat here.

## Alternative to clustering: RELAXATION LOOP (grad_infer_relax.py)

Motivation: the clustering fix needs an acyclic backward pass -> 0 hidden reachable on
recurrent graphs.  Keep the gauge-anchored placement but drop the topological order:
re-infer EVERY hidden neuron's target each outer iteration from its downstream
neighbours' CURRENT relaxed targets (Gauss-Seidel sweeps).  On a cycle, info flows
around the loop over sweeps; hidden targets warm-start from current sim spikes.

Path to it (each grouping/anchor choice traded cases; recorded for the loop):
  - measured-latency back-map + average: break 2/4, chain 2/4 (early-bias fixed point).
  - + feasibility anchor max(t*-MAX_LAT), proximity grouping: chain/fanout 4/4, 3-cycle
    4/4, but break RE-DOUBLES (each edge sits at its own max-latency spike).
  - one-per-neuron temporal grouping + max(t*-MAX_LAT): break 4/4 but fanout-hard 0/4
    (max-latency bound wrong for a weak ACCUMULATING downstream).
  - one-per-neuron grouping + REFERENCE-OFFSET gauge (L_ref = MAX_LAT - max_offset):
    ALL feed-forward 4/4 AND 3-cycle 4/4.  <-- adopted (anchor_targets()).

Final scores (seeds/4):
  feed-forward: break 4/4, chain 4/4, fanout-eq 4/4, fanout-hard 4/4  (== clustering)
  recurrent:    3-cycle 4/4   |   2-cycle 0/4   cycle+fanout 0/4
=> The relaxation loop matches clustering on feed-forward and, unlike clustering,
recovers a real recurrent model (3-cycle).  The 2-cycle/cycle+fanout misses are an
INHERENT limit of output-only supervision, not the anchor: on the 2-cycle N1 fires 5x
but its only downstream N2 fires 4x, so N1's extra spike is invisible to any
downstream-target inference (output N3 IS recovered, N2 lands within ~11 steps).

### Relaxation loop on the 50-neuron recurrent benchmark (grad_relax_50.py): runs, does NOT solve

Output-only (3 outputs pinned), infer all hidden by relaxation sweeps, fit by solve_vsub:
  case0 target 7/2/1 -> 8/7/1  (1/3 counts, loss 3.1e-2)
  case1 target 4/2/7 -> 10/7/12 (0/3, loss 2.5e-2)
  case2 target 3/3/7 -> 9/6/14  (0/3, loss 2.4e-2)
Tuning (rounds 15->30, sweeps 10->20) does NOT help -- if anything over-firing grows
(case0 9/8/1).  So it is under-CONSTRAINT, not under-convergence.

Unlike clustering it RUNS on these graphs, but it is worse than the oracle ceiling
(2/3,1/3,1/3) and below soft homotopy (case0/1 exact).  Failure mode = hidden neurons
OVER-FIRE: with fan-in/out 6-11 and >50% back-edges, each hidden target is assembled
from many also-inferred noisy downstream targets -> too many one-per-neuron groups ->
excess spikes.  Output-only supervision is under-constrained in DENSE recurrence; the
relaxation loop unlocks small clean cycles (3-cycle 4/4) but not the 50-neuron cases.
Recurrence + density remains the open problem; soft homotopy still the method to beat.

### Isolating the 50-neuron over-firing: target assignment, not presynaptic timing

Partial-observation run (grad_relax_50, true_pre=True): feed each local solve the TRUE
presynaptic spike times, targets still inferred output-only:
  case0 7/2/1 -> 9/6/3   case1 4/2/7 -> 9/6/11   case2 3/3/7 -> 9/6/11   (all 0/3)
Perfect presynaptic timing does NOT help (still over-fires ~like sim-spikes 1/3,0/3,0/3);
oracle TRUE targets recover 2/3,1/3,1/3.  => the culprit is TARGET ASSIGNMENT.

### Minimal 4-neuron reproduction (grad_overfire_minimal.py): over-attribution -> over-fire

  N0->N1 weak(150): N1 truly NEVER fires
  N0->N2 strong(500): N2 fires 5x on its own          (output)
  N1->N2 weak(50): real but irrelevant
  N1->N3 (500): N3 truly SILENT                        (output)
Because N1 has an edge into output N2, output-only inference CREDITS N1 with all of N2's
spikes and infers N1 target = [90,190,290,390] (true N1 = []).  Forcing the silent
neuron to fire ~5x then drives N3 (fed only by N1) to spike though its target is empty ->
N3 over-fires on 4/4 seeds -> FAIL.  This is exactly the dense-recurrence failure: a
hidden neuron over-credited for downstream spikes it did not cause.  Downstream-only
credit assignment has no way to know N0 (not N1) caused N2, so it cannot silence N1.

## Partial/probabilistic credit + silence veto (grad_credit.py) -- fixes the minimal case

Idea: don't credit a presynaptic neuron with ALL of a downstream's spikes; charge it in
proportion to the share of that downstream's drive it actually supplies, and let an
EMPTY downstream target act as a veto.
  1. PARTIAL CREDIT  r[n->d] = w_nd*Sum hk(t-s) / (total drive at d's target times).
     Only downstream where n is an above-average contributor (r >= 1/fanin_d) may place
     demands on n.
  2. SILENCE VETO  a downstream with an empty target vetoes n firing, weighted by n's
     structural influence share.  If veto > positive credit -> n's target = [] (silent).
  3. Empty targets are now ACTED ON (previously skipped, so nothing ever suppressed a
     neuron): incoming weights scaled by 0.9*th/Vmax until drive is sub-threshold.
  4. Silence must not be ABSORBING (a silenced neuron has zero dynamic share and could
     never earn credit back): when d is UNDER-firing, attribute by STRUCTURAL weight
     share; use dynamic share only once d meets its target.

Minimal over-fire case (grad_overfire_minimal): 0/4 -> 4/4, N1 correctly inferred []
(true []) -- the 1.0 veto share from silent output N3 beats the 0.09 credit from N2.
ALL regressions preserved: break/chain/fanout-eq/fanout-hard 4/4, 3-cycle 4/4,
2-cycle & cycle+fanout 0/4 (the inherent output-only limit).

50-neuron recurrent, output-only: over-firing ELIMINATED but now UNDER-fires; still unsolved
  case0 target 7/2/1 -> 5/0/0  (0/3, loss 1.49e-2; was 8/7/1 at 3.10e-2)
  case1 target 4/2/7 -> 3/2/6  (1/3, loss 2.05e-2; was 10/7/12 at 2.49e-2)  <-- close
  case2 target 3/3/7 -> 0/0/5  (0/3, loss 1.70e-2; was 9/6/14 at 2.43e-2)
Loss improves on case0/case2 and case1 lands near target, but the hard cutoffs
(r >= 1/fanin, veto > credit) over-correct in dense graphs -> silence cascades.  The
structural re-entry rule (4) did not recover it (case1 3/2/6 -> 0/2/5), so the
under-firing is not merely the absorbing-silence trap.  Dense recurrence still unsolved;
the credit mechanism is the right direction (correct failure mode removed, small cases
all green) but needs a soft/graded demand rather than binary include-exclude.

## The silence VETO is too absolute -- minimal example (grad_veto_minimal.py)

Mirror of the over-fire case.  Net:
    N0->N1 (500): N1 must FIRE
    N1->N2 (500): N2 driven ONLY by N1     (output, must fire)
    N1->N3 (50) : too weak to reach th     (output, must stay SILENT)
N1's credit for N2 is 1.0, but its structural influence on silent N3 is ALSO 1.0 (sole
input), so veto >= credit -> N1 inferred [] though it truly fires [72,172,...] -> N2
loses its only driver.  Veto version 0/4; relaxation loop (no veto) 4/4 on the same net.

ROOT CAUSE: a silent downstream means "the EDGE w[n->d] must stay small", NOT "the
SOURCE n must not fire".  The veto charges the constraint to the wrong variable.  The
correct mechanism already existed: scaling the SILENT neuron's own incoming weights
sub-threshold (train()), which is exactly w[n->d].

FIX: drop the firing veto entirely; keep partial credit + edge-level suppression.
  veto minimal   0/4 -> 4/4    over-fire minimal stays 4/4 (partial credit suffices)
  feed-forward break/chain/fanout-eq/fanout-hard 4/4, 3-cycle 4/4  (all preserved)
=> the veto was doing no necessary work; strictly better design.

50-neuron (output-only) still unsolved, and shows the tension the user predicted:
                     case0 (7/2/1)      case1 (4/2/7)      case2 (3/3/7)
  no credit          8 7 1  (1/3)       10 7 12 (0/3)      9 6 14 (0/3)   over-fires
  credit + VETO      5 0 0  (0/3)       0 2 5   (1/3)      0 0 5  (0/3)   under-fires
  credit, no veto    6 2 0  (1/3,1.58e-2) 8 8 14 (0/3)     7 8 11 (0/3)   over-fires
Case0 is the closest yet (6/2/0 vs 7/2/1).  Absolute include/exclude on either side
overshoots; a GRADED demand (credit scaling how MUCH a neuron fires, not a binary
include/silence) is the remaining idea.

## Minimal example that STILL over-fires under partial credit (grad_overdemand_minimal.py)

Found by sweeping random 4-6 neuron nets for failures with found_out_count > target, then
pruning dead-end neurons.  Smallest clean case is 3 neurons, FEED-FORWARD:
    N0->N1 (250): N1 is a weak ACCUMULATOR, truly fires only 2x  [173, 373]
    N1->N2 (700), N0->N2 (300): output N2 fires 3x               [140, 220, 399]

The hidden neuron legitimately fires LESS OFTEN (2x) than the output it drives (3x) --
some of N2's spikes come from the N0 bypass, not from N1.  But anchor_targets() emits ONE
hidden spike per downstream pattern-instance, so N1 is demanded 3 spikes [58,138,317].
PARTIAL CREDIT DOES NOT FILTER THIS: N1 (w700) genuinely IS the majority contributor to
N2, so it passes r >= 1/fanin and collects every demand -- the credit fix only removes
demands from neurons that are NOT responsible, not excess demands on ones that are.

Result (4 seeds): seeds 0-1 N1 fires 5x [62,162,262,362,462] vs true 2x (reaching the
earliest demanded t=58 needs a big weight, which then fires every input period);
seeds 2-3 collapse the other way and silence N1.  Output [181,381] or [130,330] vs
target [140,220,399] -- 0/4 either way.

ROOT CAUSE: the inference cannot represent "this hidden neuron should fire LESS often
than its downstream".  One demanded spike per downstream firing is an upper bound baked
into the anchoring.  This is exactly the 50-neuron over-firing, reproduced in 3 neurons,
and it is orthogonal to the two fixes so far (partial credit = who is responsible;
edge-level silence = where the constraint applies).  The missing ingredient is deciding
HOW MANY spikes a hidden neuron owes -- i.e. splitting a downstream's spikes between its
several drivers, rather than charging all of them to each qualifying driver.

## PER-SPIKE COMPETITIVE credit -- best 50-neuron result so far

User's point: partial credit should not collapse to 0 in the over-demand case.  Correct
diagnosis -- credit was scored per downstream NEURON (one scalar over its whole train,
thresholded at r >= 1/fanin), so it could only include or exclude the ENTIRE train:
a hidden neuron owing only SOME of a downstream's spikes got all of them (over-fire) or
none (silenced).  Two steps:

  (a) per-spike credit with a hard 1/fanin threshold  -> WORSE: went to 0 on 4/4 seeds of
      grad_overdemand_minimal (exactly the "shouldn't go to 0" failure).
  (b) per-spike COMPETITIVE credit (adopted): each downstream target spike is owned by
      the presynaptic(s) with the top share (within 0.9x of max), so every spike has an
      owner and a neuron that is the best driver of some spike can never be zeroed.
      Per spike: where d already fires, attribute by who CAUSED it (kernel-weighted
      drive); where d is MISSING that spike, nobody caused it, so attribute by who should
      GROW it (structural weight share) -- this also stops silence being absorbing.

over-demand minimal: count FIXED -- demanded N1 target is now 2 spikes [138,317] (true 2,
was 3 or 0) and N1 fires 2x (was 5x or silenced).  Still 0/4 on output match: residual
error is now TIMING (anchor ~35 steps early: gauge assumes MAX_LAT=82 but the true
N1->N2 latency here is 47), a separate issue from credit.

Regressions all preserved: over-fire 4/4, veto 4/4, break/chain/fanout-eq/fanout-hard
4/4, 3-cycle 4/4, 2-cycle & cycle+fanout 0/4.

50-NEURON, output-only -- best yet:
                    case0 (7/2/1)     case1 (4/2/7)      case2 (3/3/7)
  no credit          8 7 1  (1/3)     10 7 12 (0/3)      9 6 14 (0/3)
  credit+veto        5 0 0  (0/3)      0 2 5  (1/3)      0 0 5  (0/3)
  per-neuron credit  6 2 0  (1/3)      8 8 14 (0/3)      7 8 11 (0/3)
  PER-SPIKE COMPET.  6 2 0  (1/3)      4 0 7  (2/3)      5 3 6  (1/3)
Counts are now CLOSE to target instead of wildly over/under; case1 at 2/3 exceeds even
the ORACLE-target ceiling (1/3).  Still no exact recovery, and the remaining error is
spike TIMING (the MAX_LAT gauge is wrong when the true edge latency is much shorter).

## Competitive credit breaks on MULTI-SOURCE spikes (grad_coincidence_minimal.py)

User's objection: a spike can legitimately have several sources that all deserve credit;
winner-take-all starves the minority contributor.  Confirmed, but only after a first
attempt that was NOT a valid example:

  BAD EXAMPLE (rejected): N1,N2 both fire at the SAME time, N3 needs their sum
  (w 350/150).  Competitive credit silences N2 on 3/4 seeds and the recovered weights are
  badly wrong (rel.err up to 0.87) -- BUT the OUTPUT still matches 4/4, because the solve
  compensates by growing w(1->3).  When two sources fire at the same time they are
  interchangeable: only their SUM matters, so the weights are non-identifiable from the
  outputs.  Wrong weights + right outputs = the example is not good enough, not a method
  failure.

  GOOD EXAMPLE: make the sources NON-interchangeable by putting them on different
  schedules.
      N0->N1 (500): N1 fires PERIODICALLY        [72,172,272,372,472]
      N0->N2 (300): N2 weak ACCUMULATOR, SPARSE  [140,340]
      N1->N3 (350), N2->N3 (400): output N3      [167,301,403]  ISIs [134,102] IRREGULAR
  N1 holds the larger share, so competitive credit awards it every N3 spike and N2 gets
  ZERO credit -> N2 inferred silent on 4/4 seeds -> output collapses to N1's regular
  period-100 pattern [175,275,375,475] (4 spikes, wrong times).  0/4.

  Non-compensability VERIFIED by scanning w(1->3) over its whole range 20..3000 with N2
  silenced: NO value reproduces [167,301,403] (closest [231,431]).  N1 alone is periodic
  and cannot produce irregular ISIs, so the second source is genuinely required -> this is
  a true OUTPUT failure, not a weight-degeneracy artifact.

LESSON for building these examples: an output-only method can only be faulted by an
OUTPUT error.  If wrong weights still yield right outputs, the weights are simply
unidentifiable from the data.  The multi-source concern only bites when the co-drivers
fire on DIFFERENT schedules, so the survivor cannot absorb the other's role.

=> credit must be able to SPLIT one spike across several sources (graded shares that sum
to 1), not award it to a single winner.  Both extremes now have a minimal counterexample:
all-or-nothing per-neuron credit (grad_overdemand_minimal) and winner-take-all per-spike
credit (this file).

## Making credit SPLITTABLE (grad_credit.py, SPLIT_FRAC)

Goal: one spike may have several legitimate sources, so credit must split instead of
going to a single winner.

What actually blocks the split: it is NOT the owner threshold.  Loosening
"share >= 0.9*top" to 0.4/0.25/0.15 leaves grad_coincidence_minimal at 0/4.  Measuring
the shares shows why -- at TRUE weights the co-driver N2 holds a healthy 0.23-0.40 share,
but during training its timing drifts, its realised share collapses to 0.09, its weight
then shrinks, and the share shrinks further: an ABSORBING STATE IN THE WEIGHT DOMAIN.
Magnitude of realised drive therefore cannot decide ownership -- it presupposes the state
is already right.

Adopted rule: OWNERSHIP BY TIMING FEASIBILITY, gated by STRUCTURAL weight share.
A presynaptic co-owns spike t of d if (a) it has a spike in the causal window
[t-MAX_LAT, t), and (b) its structural share sum(w_pd)/max_p is >= SPLIT_FRAC.  Timing
feasibility is state-robust (it does not collapse when weights drift); the structural
gate still excludes negligible edges (0.09 in grad_overfire_minimal); the actual DRIVE
split is left to solve_vsub fitting d's incoming weights.
  - timing feasibility ALONE regresses over-fire minimal 4/4 -> 2/4 (a negligible edge is
    credited merely for being timing-feasible), hence the structural gate.

Results: coincidence starvation FIXED -- N2 now fires the correct 2x (was 0x on 4/4
seeds) and the output goes from a full collapse [175,275,375,475] to a near-miss
[164,307,404] vs target [167,301,403].  Still 0/4: the residual error is TIMING (a few
steps), the same MAX_LAT-gauge imprecision seen in grad_overdemand_minimal, not credit.

TRADE-OFF (SPLIT_FRAC dials it; regressions are 4/4 at BOTH ends):
                       small regressions   coincidence         50-neuron
  0.25 (loose split)   all 4/4             N2 correct 2x/4     0/3, 0/3, 0/3
  0.75 (tight gate)    all 4/4             N2 starved on 2/4   0/3, 2/3, 1/3
  (winner-take-all)    all 4/4             N2 starved 4/4      1/3, 2/3, 1/3
A single global floor cannot serve both regimes: in the dense 50-neuron graphs (fan-in
6-11) admitting every timing-feasible source means nearly everyone co-owns everything ->
over-demand -> over-firing returns (found 8 8 11 vs target 4 2 7).  Splitting credit is
necessary for correctness on the minimal cases but needs to be GRADED BY NUMBER OF
CO-OWNERS in dense graphs, not a fixed share floor.  Default left at 0.75 (best overall:
keeps every regression at 4/4 and the best 50-neuron numbers).

## Gradient reformulation: BACKWARD TRACE method (grad_trace.py) -- partial

Motivation (user): rethink the timing solution "in a more gradient sort of way" -- neurons
pass information backwards only along their edges, but store extra information with them.
The timing method keeps failing because it forces COMBINATORIAL decisions (who owns a
spike, how many spikes a hidden neuron owes, fire-or-silent) and every threshold chosen
for them breaks some case.  A continuous formulation dissolves those decisions.

DESIGN.  Each neuron stores two time-varying traces; messages travel only along edges:
  forward, per neuron:  eligibility eps_k(t) = sum_s h(t-s)  = dV_n(t)/dw_kn  (local)
  backward, per neuron: learning signal L_n(t), a demand on its voltage over time
      output: L_o(t*) = th - vsub_o(t*) at targets; 0.9th - vsub at genuinely extra spikes
      hidden: L_n(t) = [sum_d w_nd * corr(L_d, h)(t)] * g_n(t),
              g_n(t) = exp(-((vsub_n(t)-th)/(SIG*th))^2)   (own near-threshold sensitivity)
  update: dw_kn = sum_t L_n(t) eps_k(t),  applied with Adam.
Credit to each source is automatically proportional to w_nd*h, so several sources SPLIT a
spike's credit with NO ownership rule, nothing can be thresholded to zero (no absorbing
state), and "how many spikes" is never decided -- it emerges from the drive.

FOUR BUGS FOUND AND FIXED (each was diagnosed from the gradient SIGN on a 1-synapse net):
  1. no-reset PSP sum accumulates over the whole run and sits far above threshold later,
     flipping the demand sign even where the neuron is under-driven -> use epoch resets.
  2. post-reset blindness: reading the raw simulated voltage at a target makes an
     over-driven neuron (fires early, already reset) look under-driven -> epoch-reset
     V_sub at the TARGET times restores monotonicity in w (this is solve_vsub's trick).
  3. epoch off-by-one: accumulation ended AT the reset, so eps was exactly 0 at every
     target -- the very times the demand is evaluated -> epoch must be (prev, reset].
  4. a merely LATE spike was counted as spurious and suppressed, cancelling the push-up
     toward its own target -> match current spikes to targets first (MATCH_WIN), suppress
     only genuinely EXTRA ones.  Plus the step-nudged kernel (HKN) for discretisation.
Also: normalising the step by max|g| discards magnitude (with one synapse every step is
then exactly +-lr, so it creeps and cannot fine-tune) -> Adam.

STATUS.  On a single synapse the gradient is now correctly SIGNED in both directions and
exactly ZERO at the true weight, and training recovers w=500 exactly from 3 of 4 inits
(300/400/700 -> OK; 900 -> 492).  On multi-neuron nets it is still 0/4 everywhere: the
hidden-neuron signal is the unresolved part -- hidden traces drift badly (coincidence:
N1 found [135,335] vs true [72,172,272,372,472]) although the OUTPUT lands close
([208,307,408] vs [167,301,403]).  The suspect is the hidden gate g_n: a magnitude-only
bump on the neuron's own V_sub, which carries no information about WHICH DIRECTION in
time its spike must move.  That is the same voltage->spike-time mapping that made the
1/slope term brittle, now reappearing in the backward pass.

### Signed timing demand (grad_trace.py) -- fixes the hidden gradient, still no exact recovery

Derivation: moving n's spike from s to s-delta changes V_d(t) by w_nd*h'(t-s)*delta, so
    tim_n(s) = sum_d w_nd * sum_t L_d(t) h'(t-s)
is positive exactly when the spike should fire EARLIER (-> more drive -> positive voltage
demand at s).  Correlating against h itself carries MAGNITUDE ONLY and cannot say which
way in time the spike must move -- that was the blocker in the previous entry.
Implemented as HKP (kernel derivative) and applied at the neuron's existing spikes.

Hidden gradient sign on the chain N0->N1->N2 (varying w(0->1), true 500):
    w=350 (2 spikes vs 5)  grad +   OK      w=600  grad -   OK
    w=450                  grad +   OK      w=750  grad -   OK
    w=500                  grad +3e-10, ~3 orders below the others -- effectively zero
Magnitudes rose from ~1e-13 to ~1e-7, i.e. the hidden signal is no longer vanishing.

CREATION TERM MUST BE OFF.  A timing-only signal says nothing about spikes a neuron is
MISSING, so a creation term (correlation with the all-positive kernel h, gated near
threshold) was added alongside.  It diverges: being all-positive it is a PERSISTENT
upward push on every hidden weight whenever downstream demand has positive mass, and it
never balances -- chain weights ran 568->1345 and 385->1161 against a true 500, while the
output layer stayed near 500.  Down-weighting (CREATE=0.2, 0.05) does not help; only
CREATE=0 does.  The signed timing term is self-limiting because h' changes sign either
side of the kernel peak.  Default set to CREATE=0.

STATUS: timing-only converges to a near-miss rather than diverging -- chain output
[211,311,411,511] vs true [214,314,414,514] (a uniform 3-step offset), weights
[706,502,453] vs [500,500,500].  Still 0/4 on every case: it settles at a nearby fixed
point where the gradient is ~0 but the discrete spikes are a few steps off.  Open
questions: (a) the uniform offset suggests a residual bias in the epoch/nudge convention
for HIDDEN neurons (the output layer alone recovers exactly, so the bias enters via the
backward message); (b) spike CREATION still has no non-divergent formulation -- it needs
a demand that is signed (deficit vs excess), not an all-positive push.

### Spike CREATION as a backward-propagating REQUEST -- the gradient method starts working

User's insight: creation is not "push the weights up".  A spike is needed AT A SPECIFIC
TIME, and the "spike needed" signal has to keep being passed backward.

Implemented exactly that, and it explains the earlier divergence.  The OUTPUT creation
demand was already deficit-based (th - vsub) so it switched off once satisfied; the HIDDEN
one was a propagated MAGNITUDE (sum_d w_nd corr(L_d,h)) with no notion of "enough", so it
pushed forever.  New scheme:
  * seed: each unmatched output target time becomes a REQUEST at that time.
  * propagate: if vsub_n(tau) < CREATE_FLOOR*th there is nothing at tau to amplify, so no
    weight can create the spike -- the request is passed back to each presynaptic at
    tau - NOM_LAT, and keeps propagating until it reaches a neuron that can satisfy it.
  * demand: L_n(tau) += th - vsub_n(tau), the DEFICIT -- vanishes once the spike exists,
    so it is self-limiting.
Plus a decaying Adam step (a fixed step made the output oscillate 209/219/215 about a
target of 214).

Divergence is gone: chain weights now stay near truth (~470-540 vs 500) instead of running
to 1345.

HARNESS BUG worth remembering: `def train(..., lr=LR)` binds the default at DEFINITION
time, so setting G.LR afterwards did nothing and every suite run silently used lr=0.5
instead of 10.  All the "0/4 everywhere" results above were that bug, not the method.

RESULTS with lr applied (4 seeds each), vs the discrete timing method:
                     timing method   trace+requests
    over-fire            4/4             4/4
    veto                 4/4             4/4
    coincidence          0/4             1/4   <-- gradient solves what timing NEVER could
    fanout equal         4/4             4/4
    chain                4/4             3/4
    fanout hard          4/4             2/4
    BREAK divergent      4/4             2/4
    over-demand          0/4             0/4
    3-cycle              4/4             0/4
    2-cycle              0/4             0/4
The headline: over-fire and veto are solved 4/4 with NO ownership rule, NO share
threshold and NO fire-or-silent decision -- the combinatorial machinery that needed a
different hand-tuned constant for every case is simply gone.  And coincidence, which
defeated every discrete variant (the co-driver was always starved to silence), is now
recovered on one seed, because credit splits automatically in proportion to w_nd*h.
Regressions to fix: 3-cycle (recurrence -- timing got 4/4) and over-demand; and the
remaining cases are seed-sensitive, i.e. an optimisation/robustness problem now rather
than a credit-assignment one.

### Close look at over-demand under the trace method (overdemand_trace.png)

Net:  N0 --250--> N1 --700--> N2(out),  N0 --300--> N2 (bypass)
TRUE: N0 [1,101,201,301,401,501] | N1 [173,373] (2 spikes) | N2 [140,220,399] (ISIs 80,179)

"0/4" badly understates it -- the behaviour is BIMODAL:
  seed0  w=[250,657,297] (true [250,700,300])  N1 [173,373] EXACT   out [141,221,399]
         vs target [140,220,399]  -> off by just 1,1,0 steps
  seed1  w=[249,865,314]                        N1 [177,377]        out [137,216,401]
  seed2  w=[235,255,275]  COLLAPSE              N1 [225] (1 spike)  out [150,280,450]
  seed3  w=[246,242,280]  COLLAPSE              N1 [222] (1 spike)  out [148,279,448]

The plot shows why.  N1 is a weak ACCUMULATOR: it charges across TWO input periods
(rising, plateauing near 0.004, then climbing to threshold at 173).  In the collapse
basin w(0->1)=235 instead of 250 charges it slightly slower, its first spike slips to
225, and after the reset it NEVER re-reaches threshold -- 1 spike instead of 2.  The
number of hidden spikes changes DISCONTINUOUSLY with w01: a bifurcation the local
gradient cannot see across.

KEY GAP: in the collapse basin the OUTPUT has the RIGHT COUNT (3 spikes) at wrong times,
so with MATCH_WIN=60 every target is matched and NO creation request is ever generated --
the mechanism is blind to the missing HIDDEN spike, because requests are seeded only from
unmatched OUTPUT targets.  A hidden neuron can have the wrong spike count while the
output count looks right.

HYPOTHESIS TESTED AND REFUTED: tightening MATCH_WIN to force requests.  It makes things
WORSE (60 -> 0/4 with two near-misses; 30 and 15 -> 0/4 with the good basin destroyed,
seed0 degrading to out [133,263,363,463] and N1 firing 5x).  Correctly-tracking spikes get
treated as unmatched and the spurious requests over-drive the hidden neuron.  So the fix
is NOT a matching-tolerance tweak: it needs a signal that the hidden STRUCTURE is wrong
even when the output count is right.

### "Move earlier" demands also raise creation requests upstream (user's idea) -- implemented

Rationale: firing earlier needs presynaptic drive EARLIER.  If that presynaptic spike does
not exist, no weight increase can help -- amplifying what is there cannot move the spike
back.  So an unmet "move earlier" demand must raise a CREATION request in the inputs.
Implemented by seeding a request at EVERY target the neuron does not already hit (HIT_TOL
= 3 steps) rather than only at UNMATCHED targets, so a merely LATE spike also propagates a
request upstream; the existing rule (nothing at tau to amplify -> ask presynaptics at
tau - NOM_LAT) then carries it back.  Output deficits are no longer double-counted.

Effect on over-demand (true N1 [173,373], target out [140,220,399]):
  NOM_LAT=71  seed0 N1[173,373] out[141,221,399] | seed1 N1[156,356] out[176,376] (lost a spike)
  NOM_LAT=47  seed0 N1[173,373] out[141,221,399] | seed1 N1[175,375] out[137,218,399]
  (seeds 2-3 still collapse to 2 output spikes in both)
So the idea helps -- with the right latency seed1 goes from losing an output spike to a
near-miss (N1 within 2 steps, output within 3) -- but it RE-EXPOSES THE LATENCY PROBLEM:
the request is propagated with a FIXED NOM_LAT, and the true N1->N2 latency here is 47,
not 71, so at 71 the request asks N1 to fire at 149 when it should fire at 173, dragging
it ~24 steps early.  This is the same fixed-latency failure that broke the discrete
inference (see the divergent-latency entries), now reappearing in the request channel.
Still 0/4 -- but 2 of 4 seeds are near-misses, and the remaining collapse is the
accumulator bifurcation (N1 losing its second spike), which is a basin problem.

NEXT: the request should carry its own latency instead of a constant -- the natural
gradient-side answer is to propagate the request to the time that MAXIMISES the
presynaptic's influence on the demanded spike (argmax of h over the causal window, or the
h'-weighted centre), which is a per-edge quantity the traces already contain.

### Requests carrying their own latency -- correct but DESTABILISING (negative result)

Derivation: if n is short by deficit D = th - vsub_n(tau) at a requested time, presynaptic
k must supply w_kn*h(dt) = D.  Taking the RISING branch (smallest such dt) puts k's
contribution BELOW D at every earlier time, so n first crosses threshold exactly at tau
rather than before.  Per-edge, per-request, no constant.  request_lat() implements it.

VALIDATION -- the formula recovers the true latencies from first principles:
    w=700 -> 47   (exactly the true N1->N2 latency in over-demand)
    w=500 -> 71   (the known nominal)
    w=250 -> None (correct: a weak accumulator edge CANNOT fire from a single spike)

BUT IT SCORES WORSE.  Suite totals (4 seeds x 10 cases):
    fixed NOM_LAT + unmatched-only seeding          20/40   <-- best
    per-request latency + late-spike seeding        15/40
    per-request latency + unmatched-only seeding    14/40   <-- ablation isolates the cause
The ablation shows the regression is the LATENCY change itself, not the seeding change.
Reason: request_lat is exact at the TRUE weights but is evaluated on the CURRENT ones,
which are wrong during training -- so requests land at wrong times and chase a moving
target.  This is the SAME lesson as the weight-derived per-edge latencies in the discrete
method (model_lat was exactly calibrated and still destabilised every uniform case): a
quantity derived from transient weights is unstable even when its formula is right, and a
stable constant beats an accurate moving one.
Default left at USE_REQ_LAT=0 / HIT_TOL=60 (the 20/40 configuration); request_lat is kept
behind the flag since it is the correct expression once weights are near-converged.

### Requests spanning the WHOLE weight/latency range (user's idea) -- recovers the loss, drops a constant

Instead of committing a request to ONE latency, make it a TRACE spread over the entire
causal window, weighted by kernel influence h(tau - s).  Rationale: choosing a single dt
requires the CURRENT weight, which is wrong during training, so the request lands at the
wrong time and chases a moving target.  If the request covers all feasible dt weighted by
h, the LOCATION comes only from the kernel -- which is FIXED -- and the weight merely
scales magnitude.  Still seeded from DEFICITS, so it vanishes once satisfied (an
undeflected magnitude is what diverged in the first creation attempt).
Implementation: R[o] seeded with th - vsub at unhit targets; the part n cannot satisfy
locally (vsub < CREATE_FLOOR*th) is passed upstream as back_corr(unmet, HK)/peak; the
resulting trace is added to L as a graded, self-limiting creation demand.

Suite totals (4 seeds x 10 cases):
    fixed NOM_LAT constant            20/40
    per-request POINT latency         14/40   (exact formula, but chases moving weights)
    WHOLE-RANGE graded request        19/40   <-- recovers nearly all the loss
Per case vs the constant baseline: fanout hard 2/4 -> 3/4, BREAK 2/4 -> 1/4, everything
else unchanged (over-fire 4/4, veto 4/4, chain 3/4, coincidence 1/4, cycles 0/4).
On over-demand the exact score is still 0/4 but the STRUCTURE improved: collapsed seeds
went from 2 to 1 (seed3 rescued, N1 back to 2 spikes [177,377] and the output to the right
count [133,229,403]).

Significance beyond the score: at parity it removes the hand-set NOM_LAT constant
entirely -- latency is no longer a parameter, it is carried by the kernel shape.  That is
the same move that worked for credit (graded shares instead of a chosen owner) now applied
to latency: whenever a quantity had to be COMMITTED to a single value, spreading it over
the feasible range and letting the traces weight it has been the stable choice.

## Close look at the 3-cycle failure (trace method 0/4, discrete timing method got 4/4)

Net:  N0 -500-> N1 -500-> N2 -500-> N3(out),  feedback N3 -60-> N1
TRUE: N1 [72,172,262,362,461] | N2 [143,243,333,433] | N3 [214,314,404,504]
The intervals are NOT uniform: N3 gaps are 100, 90, 100.  That 90 is the feedback's whole
signature -- N3's spike at 214 feeds back and advances N1's third spike from 272 to 262.
Reproducing that modulation IS the task.

FINDING 1 -- the feedback gradient is real but ~10 orders of magnitude too small.
  Holding the other weights at truth and varying w(3->1):
      w_fb= 20 grad +2.7e-16 |  60 (true) +7.8e-18 | 120 -6.9e-17 | 200 -7.5e-17 | 300 -5.5e-17
  The SIGNS are correct (positive below 60, negative above).  The magnitudes are 1e-16..
  1e-18 against ~1e-7 on the forward path.  Adam normalises PER PARAMETER, so it rescales
  this negligible signal to full-size steps and the feedback weight wanders on noise --
  learned 81 / 219 / 174 / 264 against a true 60, and seed1 reached w_fb=219 while
  producing perfectly uniform 100-gaps, i.e. no feedback signature at all.

FINDING 2 -- but the feedback edge is NOT the whole story.  Pinning w_fb=60 and training
the rest is STILL 0/4:
      seed0 w=[528,515,474] out [214,314,407,507]   seed2 w=[485,559,481] out [214,314,402,502]
      seed3 w=[489,521,510] out [211,311,399,499]   (true out [214,314,404,504])
The first two output spikes come out EXACT; only the feedback-modulated 3rd and 4th are
off, by 2-5 steps.  The forward weights converge only to ~5-12% (515, 474, 559 vs 500),
and the recurrent modulation AMPLIFIES that residual error into a few-step timing offset,
while the unmodulated early spikes absorb it.

DIAGNOSIS of the regression vs the discrete method: this is a PRECISION problem, not a
credit-assignment one.  The discrete method solved each neuron's incoming weights in
CLOSED FORM (least squares, solve_vsub) and so hit the weights exactly; the trace method
converges iteratively to a few percent, which is fine on feed-forward nets but not where a
feedback loop amplifies small errors.  Two concrete implications: (a) per-parameter step
normalisation is wrong when edge influences differ by 10 orders of magnitude -- the step
should respect relative gradient scale; (b) a hybrid is attractive -- trace gradients for
credit assignment (where they clearly beat the discrete rules) with a closed-form local
solve for the final precision.

### Fixing the gradient itself: a real VANISHING-GRADIENT bug, found by decomposing the dot product

Chasing the 3-cycle's "10 orders of magnitude too small" feedback gradient, decomposing
dw = sum_t L_1(t) eps_3(t) at the TRUE weights showed the eligibility was FINE:
    eps_3 at N1's spikes = 1.03e-05   vs   eps_0 = 1.27e-05    (comparable)
The smallness was entirely in L_1 (~1e-13 against ~1e-3 at the output), and BOTH of N1's
gradients were ~1e-18 -- so this was never a weak feedback edge, it was the hidden signal
vanishing for every edge.

CAUSE: the timing demand sum_t L_d(t) h'(t-s) has units of BENEFIT PER UNIT TIME SHIFT,
not volts.  Feeding it back as if it were a voltage demand leaves a stray factor ~w*h'
(~500 * 1e-7 ~ 5e-5) at every hop, so the signal decays about 1e5 PER LAYER.  The correct
conversion to a voltage demand is division by the local slope dV/dt at the spike
(ds/dV) -- the same 1/slope factor the project found brittle, but here it is required for
DIMENSIONAL correctness of the backward pass, and it is floored to stay stable.

TWO FIXES TRIED:
  (a) rescale each hop to a fixed reference.  Restores magnitudes (hidden gradients 1e-18
      -> 1e-8) but DESTROYS CONVERGENCE: the demand can then never decay to zero, so the
      gradient does not vanish at the solution (feedback gradient stuck at +8e-8 at the
      true w_fb) and the weights never settle.  Suite 3/20 on the first five cases.
  (b) divide by the local slope (adopted).  Dimensionally right and self-limiting:
        max|L_1| 1.07e-04 vs max|L_3| 1.63e-05  -- comparable, no vanishing
        feedback gradient +5.5e-8 (w=20) -> +1.2e-9 (w=60, true) -> -8.1e-9 (w=120)
      i.e. correctly signed AND collapsing ~40x at the solution.

END-TO-END: 17/40, against 19/40 for the whole-range baseline -- within noise, NOT an
improvement, despite the gradient being visibly healthier (coincidence 1/4 -> 0/4, chain
3/4 -> 2/4, fanout hard 2/4 -> 3/4, over-fire and veto still 4/4).  3-cycle remains 0/4.
So the vanishing-gradient bug was real and is fixed, but it was not what was capping the
score; the remaining gap is the precision issue from the previous entry (forward weights
converging only to ~5-12%, which recurrence then amplifies).

## 3-cycle re-examined with the dimensionally-corrected gradient: it is ILL-CONDITIONING

Behaviour is now split cleanly:
  seed2 (converging)  out [214,314,405,505] vs true [214,314,404,504], gaps [100,91,100]
                      vs true [100,90,100] -- ONE step off; the feedback modulation IS
                      being learned.  seed0 similar ([213,313,402,502]).
  seeds 1,3 DIVERGE   weights run away (w1 -> 1289, w_fb -> 614), output spikes lost.
In EVERY seed the feedback weight settles far above truth (92-256 vs 60) while still
producing very nearly the right 90-gap -- i.e. many w_fb values give almost the same
output.  That direction is nearly FLAT.

THE GRADIENT IS NOT BIASED -- checked directly.  Scanning the output-layer weight with
the others at truth:
    w(2->3)= 440 grad +4.8e-08 out [262,462]        | 500 (TRUE) grad +9.1e-10 out CORRECT
    w(2->3)= 470 grad +2.4e-08 out [225,325,425]    | 530 grad -2.3e-08 out [208,308,...]
The gradient is ~25x smaller at truth than just either side, flips sign exactly there, and
truth is the ONLY scanned value giving the correct spikes.  So the true weights ARE
(approximately) a stationary point and the residual 9.1e-10 is just the discretisation
remainder.  The earlier worry that truth was not stationary was unfounded.

So the failure is OPTIMISATION CONDITIONING, not the gradient:
  * step decay does NOT help -- DECAY=0.3/1.0 freeze before arriving (1-2 output spikes).
  * per-parameter Adam (default) rescales every direction to a full-size step, so the flat
    feedback direction drifts far (92-256) while sharp directions are fine.
  * GLOBAL normalisation, tried as the fix, is WORSE: flat directions then barely move at
    all (w_fb stuck at its init 31/36/65) while the dominant direction blows up
    (w(0->1) -> 1019, one seed emitting no output spikes).  Reverted to per-parameter.
Neither extreme works because the problem is genuine ill-conditioning: the feedback weight
is weakly identified by the outputs while the forward weights are sharply identified.
The honest next step is a curvature-aware or trust-region step (bound the change in
OUTPUT SPIKE TIMES per iteration rather than in weight units), not another global rescale.

### Trust region in spike-time units -- NO EFFECT (negative result)

Implemented as planned: predict each spike's shift ds = -(sum_k dw_k eps_k(s))/(dV/dt at s)
-- the same slope factor already in the backward pass -- and rescale the whole step so the
worst predicted shift stays within TRUST steps.  Rationale was that weight units are the
wrong currency under ill-conditioning.

Swept the bound over 16x on the 3-cycle (300 rounds, 4 seeds):
    TRUST= 0 (off)  mean|dt| 41.2   0/4      TRUST= 2.0   mean|dt| 41.2   0/4
    TRUST= 0.5      mean|dt| 39.2   0/4      TRUST= 8.0   mean|dt| 41.2   0/4
Results are identical to within noise -- the constraint essentially never binds, because
the decaying step is already ~0.6 weight units by iteration 300, far below any shift the
bound would clip.  It DOES stop the gross runaways seen earlier (seed1/seed3 no longer
blow up to w~1289), but that is all it buys.

More important: the residual is NOT oscillation, it is a STUCK fixed point.  Seeds 0 and 2
settle 3-4 steps off ([211,311,408,508] and [218,318,400,500] against [214,314,404,504])
and stay there regardless of the bound.  A trust region can only help when the step is
overshooting; here it is under-resolving.

REGRESSION NOTE: the dimensionally-correct slope conversion HURT this case.  Before it,
seed2 reached [214,314,405,505] (err 0.5); after it, [218,318,400,500] (err 4.0), matching
the suite drop 19/40 -> 17/40.  So dividing by dV/dt is right dimensionally but the
floored slope is evidently a poor estimator near threshold, where dV/dt is small and
noisy -- exactly the regime the project earlier flagged as making 1/slope brittle.  Worth
revisiting with a better slope estimate (e.g. fitted over a window rather than one
backward difference) before concluding the conversion is wrong.

## WHY the 3-cycle seeds are stuck: a genuine BARRIER, not slow convergence

At seed0's fixed point w=[676,496,459,91] (true [500,500,500,60]) the per-target demands
have OPPOSITE SIGNS:
    target 214 found 209 (err -5)  L_3 = -9.22e-05   (fire LATER)
    target 314 found 309 (err -5)  L_3 = -9.22e-05   (fire LATER)
    target 404 found 405 (err +1)  L_3 = +2.66e-05   (fire EARLIER)
    target 504 found 505 (err +1)  L_3 = +2.66e-05   (fire EARLIER)
Two spikes want to move later and two earlier, so the summed demand partly cancels and the
majority wins: net gradient on w(2->3) is -2.0e-09 (decrease) while truth needs +41
(459 -> 500).  Two of the four gradient components point AWAY from the true weights.
Structurally the output gaps are [100,96,100] against a true [100,90,100] -- the feedback
modulation is too weak, which no single output weight can fix.

DECISIVE TEST -- walk the straight line from the stuck point to truth:
    alpha   0.0   0.1   0.2   0.3   0.5   0.7   0.9   1.0
    mean|dt| 3.0  2.5   4.5   5.0   5.0   3.5   2.5   0.0
The error RISES to 5.0 in the middle before falling to 0 at truth.  The stuck point is a
genuine LOCAL MINIMUM separated from the solution by a RIDGE.  Local gradient descent
cannot cross it, which explains everything observed: trust regions do not help (not
overshoot), step decay does not help (it just freezes the wrong basin faster), and the
gradient at the fixed point legitimately points partly away from truth.

IMPLICATION: the remaining 3-cycle failure is a GLOBAL-OPTIMISATION problem, not a
credit-assignment or gradient-correctness one -- both of which are now verified sound
(truth is stationary, signs correct, magnitudes comparable across layers).  This is
precisely what the project's SOFT HOMOTOPY does: annealing beta smooths the landscape and
removes exactly these barriers, which is why it solves the 50-neuron case0/case1 exactly
while local methods plateau.  Schedule-free ways to cross barriers, given the stated
dislike of schedules: momentum large enough to carry through a ridge of this height,
injected noise, or multi-start with the trace gradient as the local refiner.

### Which components point the wrong way, and why (they fail for OPPOSITE reasons)

At the stuck point w=[676,496,459,91] (true [500,500,500,60], err 3.0):
    w(0->1)    grad down   truth -176 down   CORRECT
    w(1->2)    grad down   truth   +4 up     wrong, but delta is 0.8% -- already right
    w(2->3)    grad down   truth  +41 up     WRONG
    w(3->1)fb  grad up     truth  -31 down   WRONG

Probing each axis on its own is decisive -- the two failures are NOT the same kind:

w(2->3) is in a ONE-DIMENSIONAL LOCAL MINIMUM.  Both directions are worse than 3.0:
    -30 -> err 99.0   -10 -> err 99.0   (gradient's way; output loses spikes entirely)
    +10 -> err  8.5   +30 -> err 15.5   (truth's way)
There is no descent along that axis at all.  Its nonzero gradient is purely the RESIDUAL
of the cancelling demands (2 x -9.2e-05 "later" against 2 x +2.7e-05 "earlier"), and that
leftover points to the catastrophic side.  Adam then amplifies a -2e-09 residual into a
full-size step.  This is the one component the gradient genuinely mis-signs.

w(3->1) is LOCALLY CORRECT BUT GLOBALLY WRONG.  Increasing it really does help:
    +30 -> err 2.5 (BETTER than 3.0)      -10/-30 (truth's way) -> err 3.5 (worse)
It strengthens the too-weak feedback modulation (gap 96 toward the true 90).  But truth
reaches gap 90 with LESS feedback (60) plus a much smaller w(0->1) (500 vs 676).  So this
is the barrier expressed in a single coordinate: an honest local improvement that leads
away from the global optimum.

CONCLUSION: only w(2->3) is mis-signed by the gradient itself, and that traces to opposite
demands cancelling to a small residual on a locally flat/ridged axis.  The feedback weight
is doing exactly what a correct local gradient SHOULD do -- which is why no fix to the
gradient can rescue this case, and why it needs a global mechanism (smoothing/homotopy,
momentum through the ridge, or multi-start).

## EXACT SIMULATION-LEVEL CAUSE of the wrong gradients (traced end to end)

THE PARAMETERS THAT DO IT
  neuron_decay 0.99497 (tau ~199) and rise_decay 0.9803 (tau ~50) put the PSP kernel PEAK
  at dt = 110 steps, peak value 1.5746e-05 per unit weight.
  With threshold 0.007 this fixes a CRITICAL WEIGHT = th/peak = 444.5:
  below 444.5 a single presynaptic spike can NEVER cause a spike, at any timing.
  The true weights (500) sit only 12.5% ABOVE that critical value, so the whole network
  operates just above a cliff, near the flat top of the kernel.

CONSEQUENCES IN THAT REGIME (all measured)
  1. latency is hypersensitive to weight -- w=446 -> lag 103, 450 -> 96, 459 -> 88,
     470 -> 82, 500 -> 71, 550 -> 61, 676 -> 48.  A 9% weight change (459->500) moves a
     spike 17 steps.  The curve is near-vertical at the cliff because dh/dt -> 0 at the peak.
  2. dV/dt at the crossing is correspondingly TINY: 2.22e-05 at w=459, 5.26e-05 at w=500.
     The timing->voltage conversion divides by this, so 1/slope is 19,000-45,000 and
     ill-conditioned exactly at the solution.
  3. input period is 100 while the kernel peak is 110 -- comparable -- so single-spike
     triggering and cross-period ACCUMULATION are adjacent regimes with a discontinuous
     spike count between them.

HOW THIS PRODUCES THE STUCK POINT (3-cycle, stuck w=[676,496,459,91] vs true [500,500,500,60])
  measured lags:  TRUE  N1 [71,71,61,61,60]  N2 [71]*4   N3 [71]*4   (N2 fires 4x)
                  STUCK N1 [48,48,44,44,44]  N2 [72]*5   N3 [88]*4   (N2 fires 5x)
  w(0->1)=676 is 52% above critical -> steep regime, N1 fires 23 steps EARLY.
  w(2->3)=459 is  3% above critical -> flat regime, N3 takes 17 steps LONGER.
  The two errors COMPENSATE to give nearly-correct output times, and the configuration is
  self-consistent -- but N2 fires 5 times instead of 4, so the internal structure differs.

  This is exactly why w(2->3)'s gradient is unusable: the weight is 3% above the 444.5
  cliff, so a descent step of -10/-15 pushes it BELOW critical, N3 can no longer fire from
  a single N2 spike, and the error jumps to 99 (measured).  Ascent moves away from the
  compensating balance (err 8.5 at +10).  The axis is a narrow ledge between a cliff and a
  rising slope -- which is the local minimum reported earlier, now explained mechanically.

A REAL BUG FOUND ON THE WAY (fixed): SLOPE_FLOOR was 0.01, i.e. a floor of 7.0e-05, but
the actual dV/dt in the operating regime is 2.2e-05..5.3e-05 -- ENTIRELY BELOW the floor.
The 1/slope factor was therefore clipped to a constant across the whole region of
interest, destroying its weight dependence exactly where it mattered.  Lowering it to
0.002 (floor 1.4e-05) helps only marginally (mean|dt| 41.2 -> 38.6, still 0/4), so the
clipping was real but not the binding constraint; the cliff is.

BOTTOM LINE: the wrong gradients are not a flaw in the credit assignment or the backward
pass -- they are the local linearisation of a system operating 3-12% above a hard firing
cliff set by threshold/kernel-peak, where dV/dt ~ 0 and the spike count changes
discontinuously.  Any purely local method inherits this.  Mitigations are structural:
operate further from the cliff (larger weights relative to threshold), smooth the
threshold (homotopy/beta annealing, which is precisely what soft homotopy does), or
explicitly model the cliff as a constraint boundary rather than linearising across it.

### WHICH LOSS TERM points the wrong way (exact decomposition)

grad w(2->3) = sum over output targets of  L_3(t*) * eps_2(t*),  with L_3(t*) = th - vsub_3(t*):
   target 214  found 209 (err -5)  vsub 7.092e-03  L_3 -9.221e-05  eps_2 1.545e-05  term -1.425e-09  DOWN
   target 314  found 309 (err -5)  vsub 7.092e-03  L_3 -9.221e-05  eps_2 1.545e-05  term -1.425e-09  DOWN
   target 404  found 405 (err +1)  vsub 6.973e-03  L_3 +2.659e-05  eps_2 1.519e-05  term +4.040e-10  UP
   target 504  found 505 (err +1)  vsub 6.973e-03  L_3 +2.659e-05  eps_2 1.519e-05  term +4.040e-10  UP
                                                                            TOTAL   -2.042e-09  DOWN
   (th = 7.000e-03; truth needs w(2->3) to go UP, 459 -> 500)

THE OFFENDING TERM is the VOLTAGE-TARGET term th - vsub_3(t*) evaluated at the two EARLY
spikes (targets 214 and 314).  Each contributes -1.425e-09 against +4.04e-10 from the two
late ones -- 3.5x larger, so they decide the sign.

It is not miscomputed.  vsub_3(214) = 7.092e-03 vs th = 7.000e-03 is a genuine 1.3% excess
of drive at that instant, and the term correctly says "too much drive here".  The error is
ATTRIBUTION: that excess is caused by N1 firing 23 steps early (w(0->1)=676), not by
w(2->3) being too large, but the term charges it to every incoming edge in proportion to
eligibility.

AND THE ELIGIBILITY CANNOT DISCRIMINATE: eps_2 is essentially identical at all four
targets (1.545, 1.545, 1.519, 1.519 e-05).  So the gradient collapses to an almost
UNWEIGHTED SUM of the four demands, and its sign is decided purely by which timing error
is larger -- 5 steps beats 1 step.  Nothing in the eligibility separates the two
PRE-feedback spikes (set by the feedforward path alone) from the two POST-feedback ones
(set by feedforward plus the loop), which is precisely the distinction the solution needs.

=> The fix is not a better step rule but a term that can attribute excess drive to the
RIGHT edge: the demand needs to be weighted by each edge's marginal contribution to the
TIMING error (which differs between the two regimes), not by its raw PSP amplitude (which
does not).

### "If N1 fires 23 steps early, why can't the gradient push it right?"  -- it DOES; the path is blocked

Probing w(0->1) alone from the stuck 676 toward the true 500 (others held at stuck):
   w01=676  N1 [49,149,245,345]  out [209,309,405,505]  err  3.0  grad -1.8e-09
   w01=620  N1 [54,154,249,349]  out [214,314,409,509]  err  2.5  grad -1.2e-08
   w01=560  N1 [61,161,255,355]  out [221,321,415,515]  err  9.0  grad -4.1e-08
   w01=500  N1 [72,172,264,364]  out [232,332,424]      err 99.0  grad -1.3e-07
   (true N1 [72,172,262,362,461], true out [214,314,404,504])

So the gradient on w(0->1) is CORRECT and stays correct -- it points down the whole way and
at w01=500 N1 lands on essentially its true spike times.  The obstruction is that fixing N1
makes the OUTPUT far WORSE (err 3 -> 99), because w(1->2)=496 and w(2->3)=459 are tuned to
compensate for an early N1.  The stuck configuration is a COMPENSATING PAIR, and the
straight path between the two consistent configurations runs through badly inconsistent
ones.  Note the gradient MAGNITUDE grows 70x along this path (1.8e-09 -> 1.3e-07): the
objective is happy to keep going even as the true error explodes, i.e. the trace objective
and the spike-time error DISAGREE along this axis.

Does a sequential path rescue it?  From [500,496,459,91] three of four gradients point at
truth (w(1->2) UP, w(2->3) UP, w(0->1) already there); only w(3->1) is wrong (UP, truth
needs DOWN).  Holding w(0->1)=500 and training the rest, the feedback weight RUNS AWAY:
   91 -> 156 -> 241 -> 325 -> 390 -> 480,  dragging w(1->2) to 840, out never recovers (err 99).

ANSWER: the gradient does push N1 the right way; what stops it is (a) the intermediate
region between the two compensating configurations has much higher error, so joint descent
is repelled from it, and (b) the feedback weight has a persistently wrong-signed gradient
that grows to re-establish the compensation, undoing any progress on N1.  The feedback
edge is the actual saboteur -- with it pinned, the forward weights are locally correct.

### How to DETECT the misbehaving edge without knowing the true weights

Three candidate detectors, tested against a healthy feed-forward chain as control:

  1. CONSENSUS  |sum of per-target terms| / sum|terms|  -- FAILS to separate.
     w(0->1) 0.433, w(1->2) 0.626, w(2->3) 0.558, feedback 0.409.  The feedback edge is
     not distinguishable from the forward ones.

  2. GRADIENT SIGN FLIPS along a sweep of each weight (idea: a well-determined weight has
     ONE optimum, hence one flip; a pathological one is multi-modal) -- FAILS on the
     control.  3-cycle feedback shows 2 flips (+++ + + - - + + + + +), but the HEALTHY
     chain shows 2 flips on w(0->1) and 4 on w(1->2) -- more than the pathological edge.
     Discretisation makes every hidden weight multi-modal, so this says nothing.

  3. RUNAWAY RATIO  |net displacement| / total variation over training -- WORKS.
     A converging weight reverses direction as it settles, so net << total variation; a
     weight that is not being determined by the data moves one way forever, ratio -> 1.
         3-cycle:        w(0->1) 0.45   w(1->2) 1.00 <==   w(2->3) 0.09   w(3->1) 1.00 <==
         healthy chain:  w(0->1) 0.05   w(1->2) 0.69       w(2->3) 0.52
     Ratio 1.00 means the weight never once reversed over the whole run.  Both genuinely
     misbehaving edges (the feedback edge, and w(1->2) which was dragged to 840) are
     flagged; no weight in the control reaches 1.00.

USABLE SIGNAL: ratio ~1.0 sustained while the OUTPUT SPIKE ERROR fails to improve.  Both
quantities are observable without ground truth -- the error is measurable against the
given output targets, and the trajectory is free.  The interpretation is that the edge is
absorbing error rather than being constrained by it, which is exactly what the feedback
weight was doing (91 -> 480 while the error sat at 99).
Caveat: it is a trajectory diagnostic, so it needs a window of iterations; it cannot flag
the problem from a single gradient evaluation, and static tests (1 and 2) demonstrably
cannot either.

## "Rule 1: the gradient should be ZERO at the true answer" -- it wasn't.  TWO REAL BUGS FOUND

Challenged on why the gradient was nonzero at the true weights (~2e-09), the answer was
not freezing or conditioning -- it was two genuine defects in the loss:

BUG 1: THE OUTPUT LOSS WAS AN EQUALITY, NOT A HINGE.
  L(t*) = th - vsub(t*) demands the voltage EQUAL threshold at the target.  But a spike
  lands exactly on t* for a whole INTERVAL of weights: those with V(t*) >= th AND
  V(t*-1) < th.  The equality keeps pushing even when the spike timing is already perfect.
  Fixed with two one-sided terms, both zero on that interval:
      L(t*)   += max(0, th - vsub(t*))       under-driven  -> push up
      L(t*-1) += min(0, th - vsub(t*-1))     firing early  -> push down

BUG 2: THE NUDGED KERNEL BIASED THE VOLTAGE RECONSTRUCTION LOW.
  HKN = (h(dt-1)+h(dt))/2 averages a RISING kernel, so vsub sat ~0.4% below the true
  voltage.  Measured at the true weights on the chain:
      V_sim(t*-1)=6.957396e-03  V_sim(t*)=7.010012e-03   (th=7.0e-03: correct interval)
      vsub(t*-1)=6.930097e-03   vsub(t*)=6.983731e-03    <- BELOW th, so hinge_up fired
  i.e. even after Bug 1 was fixed the hinge stayed permanently active, because the
  reconstruction disagreed with the simulation.  The nudge was introduced for
  discretisation, which the hinge interval now handles properly -- it was redundant AND
  biasing.  Using the true kernel HK, the reconstruction matches the simulator to 6
  significant figures (vsub 6.957423e-03 vs V_sim 6.957396e-03).

RESULT -- the gradient is now EXACTLY ZERO at the true weights on every case tested:
    chain 0.000e+00 | 3-cycle 0.000e+00 | fanout-eq 0.000e+00 | over-fire 0.000e+00
and correct solutions are STABLE ATTRACTORS, which they never were before:
  * started AT truth on the 3-cycle: weights do not move at all over 150 rounds, output
    stays [214,314,404,504] exactly (previously it drifted off).
  * started 10% off: converges to err=0.0 by round 90 and STAYS there (rounds 120, 150
    identical), landing on w=[529,534,465,98] -- a different but equally valid solution,
    since the weights are not uniquely identifiable from the outputs.

HONEST TRADE-OFF: from the standard +-50% random init the suite scores 7/20 on the first
five cases, versus 9-10/20 with the OLD biased gradient.  The biased version explored more
(its permanent residual acted like a perturbation) and so stumbled into solutions more
often, but it could not hold them.  The corrected gradient is provably right and stable;
what remains is purely a GLOBAL/basin problem -- from a 10% start it converges perfectly,
from 50% it lands in the wrong basin.  That also retires the earlier heuristics: KEEP_BEST
now changes nothing (7/20 either way) because the optimiser no longer walks away from
solutions, and the runaway-freezing heuristic is no longer motivated.

## What the DISCRETE method actually does on the 3-cycle, and how to replicate it

Its inferred hidden targets are WRONG and it still gets 4/4:
    N1 inferred [50,150,240,340]   true [72,172,262,362,461]   (4 spikes not 5, ~22 early)
    N2 inferred [132,232,322,422]  true [143,243,333,433]      (~11 early)
    N3 = the given output target
So correct hidden TIMINGS are not the ingredient -- only mutual consistency is.

TESTED AND REJECTED: deep supervision.  Feeding those same inferred hidden targets into the
trace gradient as extra per-neuron hinge terms does NOT help (3-cycle 0/4, and the chain
drops 2/4 -> 1/4).  Having per-neuron targets is not the active mechanism either.

THE ACTUAL MECHANISM -- it JUMPS, it does not descend.  Weight trajectory, seed0:
    init   [568, 385, 270,  31]
    iter0  [656, 452, 925,  25]   <- w(2->3) moves 654 in ONE step (270 -> 925)
    iter1  [662, 466, 722,  83]      then damped-averages down...
    iter7  [667, 472, 474, 270]   out [211,311,403,503] -> target [214,314,404,504]
The closed-form solve lands on the right SCALE immediately; gradient descent has to walk
that distance in small steps and gets trapped on the way.  Note its feedback weight ALSO
runs away (31 -> 270 vs a true 60), the same pathology as the gradient method -- so that
runaway is not what separates them.

WHY IT CAN JUMP: V_sub is LINEAR in a neuron's incoming weights, so each neuron's
subproblem is exact linear least squares with a closed-form minimiser.  This is not a
better gradient, it is a SECOND-ORDER step.

HOW TO REPLICATE IT WITH GRADIENTS: use a Gauss-Newton step per neuron -- precondition the
update by the local Gram matrix, (A^T A)^-1 A^T r, where A is the per-target eligibility
matrix the traces already compute.  It is cheap precisely because the subproblem is linear,
and it reduces to the discrete method's solve in the limit.  Adam's diagonal, scale-free
step throws away exactly the curvature information that permits the jump, which is why the
first-order version cannot cross the same distance.

## The jump DOES escape the stuck point -- so the direction was always knowable (earlier barrier claim RETRACTED)

Applying the discrete closed-form solve starting FROM gradient descent's stuck point
w=[676,496,459,91]:
    iter0 [671,485,458,204]  err 3.0      iter7  [667,470,469,590]  err 0.5
    iter3 [667,473,464,568]  err 1.0      iter8  [667,470,469,543]  err 0.0  EXACT
    ...                                   iter11 [667,470,470,503]  err 0.0  EXACT, stays
It escapes and lands on w=[667,470,470,503] -- NOT the true [500,500,500,60], but an
equally valid solution reproducing the output exactly.  It got there by driving the
FEEDBACK weight 91 -> 503 while leaving w(0->1) near 667.

THIS RETRACTS THE EARLIER "BARRIER" / "WRONG DIRECTION" CONCLUSIONS.  I had judged the
feedback gradient wrong because it pointed UP while the true weight was 60.  But a valid
solution sits at w_fb ~503, and UP was correct all along -- I was measuring against the
wrong reference.  The straight-line "ridge" found earlier was the path to the TRUE
weights specifically, not to the nearest solution.

With the corrected (hinge + true-kernel) gradient, checked directly:
    true weights      [500,500,500,60]   output CORRECT   max|g| = 0.000e+00
    discrete solution [667,470,470,503]  output CORRECT   max|g| = 0.000e+00
    "stuck" point     [676,496,459,91]   output wrong     max|g| = 2.918e-09
Both valid solutions are EXACT fixed points, and the stuck point is NOT stationary -- so
it was never a local minimum of the corrected objective at all.

WHAT ACTUALLY REMAINS: pure step control.  Running the trace gradient from the stuck point
with the decay off, it moves the feedback weight the RIGHT way and passes near the
solution (at 100 rounds: [591,479,479,214], err 1.5) but then overshoots and diverges
(w_fb 421 -> 792 -> 1053 -> 1439; w(1->2) hits the 3000 clip).  The information is present
and the zeros are in the right places; the optimiser simply cannot stop on them.  That is
a far more tractable problem than a landscape barrier, and it is what the Gauss-Newton
step from the previous entry would fix -- it lands ON the local minimiser rather than
stepping past it.

### Step size on the feedback weight: NOT the problem -- w(1->2) runaway is

Measured on the 3-cycle from the stuck point (lr=10, decay off):
  * w_fb solution window (others held at the discrete solution [667,470,470]):
        correct output for w_fb in [445, 555]  ->  110 units wide
  * w_fb step size actually taken: +4.9, +4.2, +3.7, +1.4, +0.5, +4.7, +6.5, +9.8, +3.8,
        +9.1, +7.7, +6.8 per 10 iterations  =  roughly 0.5-1.0 units per iteration,
        peaking near 1.0.
  => steps are ~10x SMALLER than the window; the optimiser is not stepping over it, and it
     does land inside (rounds 80-90: w_fb 447 then 485, both in [445,555]).

But the joint state is wrong by then:
    r 40: w=[630,486,465,233]  err  1.5   <- near-solution at a DIFFERENT w_fb (233!)
    r 80: w=[644,721,589,447]  err 55.0   w_fb in window, but w(1->2)=721 vs ~470 needed
    r 90: w=[544,736,570,485]  err 43.0   w_fb in window, w(1->2)=736
    r130: w=[837,1048,438,691] err 99.0
w(1->2) rises MONOTONICALLY throughout: 457 -> 499 -> 566 -> 629 -> 721 -> 806 -> 885 ->
953 -> 1048, never once reversing -- exactly the runaway the |net|/TV detector flagged at
ratio 1.00.

Two conclusions:
  1. The solution set is a MANIFOLD, not a point: r40 shows a near-solution at w_fb=233
     with different companion weights, while the discrete solve found one at w_fb=503.
     Each combination of the other weights has its own w_fb window.
  2. The failure is therefore not step size on w_fb but the UNBOUNDED DRIFT of w(1->2),
     which slides the joint state along/off the manifold faster than w_fb can settle into
     the corresponding window.  Fixing the drift (or taking the Gauss-Newton step, which
     solves all of a neuron's incoming weights jointly and lands on the manifold rather
     than sliding along it) is the target, not the step size.

## Gauss-Newton: fixes the overshoot, but only where a real RESIDUAL exists

Implemented as predicted: vsub is linear in a neuron's incoming weights, so the demands
form A dw ~= r with A[j,k] = eps_k(t_j) (the eligibility already computed) and r_j =
L_n(t_j).  First-order uses A^T r; GN preconditions by (A^T A)^-1.

ATTEMPT 1 -- GN on ALL neurons: catastrophically unstable.  From a 10% start it jumped to
w=[2259,1084,385,3000] (the clip) in 10 rounds; every damping level tried (GN_LAM 1e-3,
0.1, 1.0, 10) gave 0/12.  CAUSE: for an OUTPUT neuron L is a genuine hinge residual in
volts, so driving it to zero is meaningful; for a HIDDEN neuron L is a timing-derived
descent DIRECTION with arbitrary scale, and "solving" A dw = r for it is meaningless.

ATTEMPT 2 -- GN restricted to neurons with real residuals (outputs), first-order for
hidden.  Now well behaved:
  * starts AT truth -> does not move (still exactly stationary)
  * from 10% off  -> err 2.0, 2.0, 1.5, 1.0, 1.0, 0.5  monotone
  * from the STUCK point -> err 3.0, 3.0, 2.5, 2.0 ... monotone, which first-order never
    managed (it oscillated and then diverged)
So GN does fix the diagnosed problem: it no longer steps past the minimiser.

BUT IT PLATEAUS.  Long run from the stuck point (alpha=1.0, 250 rounds):
    r100 [644,464,488,123] err 2.0     r250 [636,458,501,148] err 2.0
    out [212,312,406,506] vs target [214,314,404,504]
Weights essentially stop moving.  Note the residual pattern is the familiar conflicting
one -- the first two spikes 2 EARLY, the last two 2 LATE -- which no single output-layer
weight can fix.  Suite at 100 rounds: 1/12.

CONCLUSION: the overshoot diagnosis was right and GN cures it, but GN can only be applied
to the output layer, and the remaining error lives in the HIDDEN timings.  To extend GN
inward the hidden neurons need genuine residuals -- i.e. actual target TIMES -- which is
exactly what the discrete method manufactures by inferring hidden targets (and why it can
solve every neuron in closed form).  Deep supervision was tested earlier and did not help
with first-order steps; the untested combination is INFERRED HIDDEN TARGETS + GN, which is
essentially a reconstruction of the discrete solver inside the gradient framework.

## Distance-from-initial penalty: four formulations tested

1. ADDITIVE pull-back  w += PROX*(w_init - w)  -- CATASTROPHIC.  4/12 -> 0/12, and even
   PROX=0.0005 (a 7% cumulative pull over 150 iters) gave 0/12.  REASON: correct solutions
   are exact fixed points of the hinge gradient (g = 0), so at a solution the pull is the
   ONLY force acting and drags the weights straight back off it.  Any always-on additive
   penalty destroys the stationarity we had just established.

2. Same, GATED on an unsatisfied demand -- barely better (1/12).  The gate almost never
   fires: hidden neurons carry a nonzero timing demand even when the outputs are correct.

3. MULTIPLICATIVE on the step (user's suggestion: multiply the penalty by the gradient, so
   zero gradient means no pull and the sign can never flip):
       upd = upd / (1 + PROX * rel^2),   rel = |w - w_init| / w_init
   This is the right FORM -- it cannot dislodge a solution and only ever damps, never
   reverses.  But as a plain function of rel it degrades monotonically: 4/12 (off), 3/12
   (PROX=1), 1/8 (PROX=5), 0/8 (PROX=20).  REASON: init = true * U(0.5,1.5), so the true
   weights legitimately sit up to 2x the init (rel = 1.0) -- seed0 genuinely needs 1.85x
   and 1.94x moves -- and a penalty starting at rel=0 damps exactly the travel required.

4. BANDED multiplicative -- free inside the a-priori plausible range, penalised outside:
       excess = max(0, rel - 1.0);   upd = upd / (1 + PROX * excess^2)
   NEUTRAL on score (4/12 at PROX=0, 2 and 10) and it does bound the drift:
       seed1 max rel displacement 1.76 -> 1.46      seed3 1.59 -> 1.40
   and it targets the right weight -- the feedback edge is the ONLY one exceeding the band
   (the others sit at 0.15-0.66), so the penalty fires exactly where the runaway is.

CONCLUSION: the multiplicative form is correct and the banded version is harmless and
mildly effective at what it was designed for, but bounding the drift does NOT fix the
3-cycle (still 0/4).  That confirms the earlier reading that the runaway is a SYMPTOM
rather than the cause -- the weights drift because the hidden timings are unresolved, and
restraining the drift does not resolve them.

## 3-cycle re-examined after the hinge fix: it was SLOW, not stuck -- and the suite jumps to 32/40

Inspecting the settling points with the corrected gradient showed something new: the
OUTPUT-layer weight is fully converged (seed0 g(2->3) = 1.9e-14, seed2 8.3e-13) while the
HIDDEN weights still carry the largest demands.  Not a stalled optimiser -- an unfinished
one.  Following seed2 further:
    r100 [480,505,506,146] gaps [100,79,100] err 5.5
    r200 [535,546,461, 87] gaps [100,91,100] err 0.5
    r400 [529,497,485, 90] gaps [100,90,100] err 0.0  EXACT
    r500..r1000  unchanged -- weights frozen, error stays 0
So the 3-cycle IS solvable by the trace gradient; the earlier 150-round budget was simply
too short, and the corrected gradient now HOLDS the solution once found.

The other ingredient is PERIODIC ADAM RESTARTS.  My manual runs had been calling train()
in blocks, which silently reset the moment estimates; a single continuous run does not
converge.  Directly compared at 400 rounds, LR=10:
    RESTART_EVERY=0   -> 0/4        RESTART_EVERY=100 -> 2/4
(Also cost an hour to a plain harness slip: run() defaults to LR=0.5, so a test that
forgot the env var silently used a 20x smaller step and looked like a null result.)

FULL SUITE, corrected gradient + restarts, 400 rounds (4 seeds):
    chain            4/4      over-fire      4/4
    fanout equal     4/4      veto           4/4
    fanout hard      4/4      over-demand    4/4   <- was 0/4 for the whole session
    BREAK divergent  4/4      coincidence    2/4   <- defeated every discrete variant
    3-cycle          2/4      2-cycle        0/4
    TOTAL 32/40   (previous best 20/40)

Versus the discrete relaxation method, the gradient now WINS on over-demand (4/4 vs 0/4)
and coincidence (2/4 vs 0-1/4), ties on chain/fanout/BREAK/over-fire/veto, and trails only
on 3-cycle (2/4 vs 4/4) and 2-cycle (0/4 both).  The two remaining gaps are recurrent
cases, and both are now basin/seed sensitivity rather than wrong gradients: truth is an
exact stationary point, solutions are stable once reached, and seed2 demonstrably
converges to err=0 and stays.

## 2-cycle investigated: a TINY solution set, not a broken gradient

Net:  N0 -500-> N1 -500-> N2 -500-> N3(out),  feedback N2 -60-> N1
TRUE: N1 [72,169,268,368,468] (5) | N2 [143,240,339,439] (4) | N3 [214,311,410,510] (4)
Output gaps [97, 99, 100] -- a GRADUAL drift, far subtler than the 3-cycle's single
[100,90,100] modulation, so every spike needs its own precision.

THE GRADIENT IS FINE.  Scanning the feedback weight with the companions at TRUTH:
    w_fb= 20 err 2.0 grad +6.7e-09 | 40 err 0.8 +2.4e-09 | 60 err 0.0 grad +0.00e+00
    w_fb= 90 err 1.2 grad -1.7e-09 | 140 err 2.8 -3.8e-09
Textbook: positive below the optimum, exactly zero at it, negative above.
But with the companions only ~5% off ([476,504,*,542]) the gradient is NEGATIVE at every
w_fb and the error really is minimised at the clip (0.8 at w_fb=20).  So the feedback
weight is optimising correctly -- it is the companions that are slightly wrong.

BEST REACHED: err 0.25 at w=[477,515,30,525], out [214,310,410,510] vs [214,311,410,510]
-- THREE spikes exact, one off by a SINGLE step.  Then it oscillates (0.75, 3.50, 1.50,
2.75, 1.00, 3.50) over 1200 rounds without ever landing.

WHY -- the solution set is tiny.  Random local search, 1000 samples at +-8% per weight:
    around the optimiser's best point [477,515,30,525]:  0 exact solutions found
    around the TRUE weights [500,500,60,500]:            3 exact solutions found
So even AT truth only ~0.3% of a +-8% neighbourhood recovers the output exactly, and the
basin the optimiser settles in contains NONE.  This is qualitatively different from the
3-cycle, where solutions form a broad manifold (the discrete solver found one at
w_fb=503, the gradient another at [529,497,485,90]) and the only problem was patience.

CONCLUSION: 2-cycle failure is not a gradient defect or a barrier -- it is a
measure-zero-ish target.  The gaps [97,99,100] differ by 1-3 steps, so matching all four
output spikes pins the weights far more tightly than any other case in the suite.  Getting
from err 0.25 to 0 needs sub-step weight resolution, which argues for a final refinement
stage (e.g. the Gauss-Newton solve, exact for the linear subproblem) rather than more
first-order iterations.

## 50-NEURON RECURRENT CASES with the corrected trace gradient: 3/3 counts on ALL THREE

First a 9.5x speedup: back_corr's explicit 400-step loop (called per edge per sweep) was
the bottleneck at 0.86 s/round.  m[s] = sum_dt Ld[s+dt] K[dt] is a correlation, so it
equals convolve(Ld, reversed K) offset by KWIN-1 -- verified identical to the loop
(max abs diff 7e-23) and small-net results unchanged (chain 4/4, 3-cycle 2/4).
Now 0.09 s/round.

RESULTS (output-only, 200 rounds, LR=10, restarts every 100):
    case             edges  target   found   counts   loss
    recurrent_132_7_2_1  132  7 2 1   7 2 1    3/3    2.54e-2
    recurrent_304_4_2_7  304  4 2 7   4 2 7    3/3    1.45e-2
    recurrent_240_3_3_7  240  3 3 7   3 3 7    3/3    1.17e-2

COMPARISON on the same benchmark:
    discrete relaxation (output-only)   1/3, 0/3, 0/3   (counts 8/7/1, 10/7/12, 9/6/14)
    credit-weighted best (output-only)  1/3, 2/3, 1/3
    ORACLE-target V_sub (true targets
      given for EVERY neuron)           2/3, 1/3, 1/3
    trace gradient (output-only)        3/3, 3/3, 3/3   <-- all three
So the corrected gradient beats every previous output-only method on count recovery, and
also beats the ORACLE-target local solver -- which is handed the true spike times of all
50 neurons -- while using only the 3 output targets.  Loss is the best measured on cases
1 and 2 (1.45e-2, 1.17e-2 vs a 1.7-3.1e-2 range previously).

STILL NOT EXACT.  Case0 per-output offsets show why:
    N47 target [140,340,540,633,722,846,936]
        found  [222,426,548,668,787,883,989]   offsets [82,86,8,35,65,37,53]
    N48 target [798,993]  found [818,979]  offsets [20,-14]
    N49 target [951]      found [897]      offsets [-54]
Right NUMBER of spikes everywhere, wrong TIMES -- offsets of tens of steps.  The method
now reliably recovers the spike-count structure of a dense 50-neuron recurrent net from
output observations alone, but not the precise timing.  That is the same split seen on the
small cases (structure first, sub-step precision last) and points again at a final
refinement stage rather than more first-order iterations.

## Minimal example of the 50-NEURON failure mode (grad_twoloop_minimal.py)

Signature to reproduce: right spike COUNT, wrong TIMES (50-neuron case0 N47 target
[140,340,540,633,722,846,936] vs found [222,426,548,668,787,883,989]).

Found by random search over 4-6 neuron nets filtered on "count matches but mean|dt| > 10",
then edge pruning.  Smallest reproduction is 4 neurons with TWO interlocking loops:
    N0 -900-> N2 -500-> N3(out)
    N3 -900-> N2                  loop 1:  N2 <-> N3
    N3 -900-> N1 -500-> N3        loop 2:  N1 <-> N3
TRUE output [110,177,251,325,389,461], gaps [67,74,74,64,72] -- IRREGULAR, produced by the
two loops interfering, exactly like the 50-neuron targets (case0 gaps 200,200,93,89,124,90).

Result 0/4, with the 50-neuron signature: counts right, times close but never exact --
per-seed mean|dt| = 3.8, 1.3, 99.0 (one count miss), 0.3.  Best seed found
[109,177,251,325,389,462] against [110,177,251,325,389,461]: four of six spikes EXACT and
two off by one step.

HYPOTHESIS TESTED AND REFUTED along the way: that irrelevant weights degrade the
optimiser.  The pruned 5-neuron version contained a dead-end sink neuron (edges in, none
out) which cannot affect the output, yet appeared to fail worse.  Running the two versions
side by side gave IDENTICAL per-seed results ([3.8, 1.3, 99.0, 0.3] both), so dead-end
neurons cost nothing -- the earlier difference was a different weight set, not the sink.

WHAT THIS ISOLATES: the irregular inter-spike structure is what the method cannot pin
down.  Matching a COUNT only needs roughly enough total drive, which many weight
combinations supply; matching irregular GAPS requires the loops' relative timing to be
right, which is a far smaller set (cf. the 2-cycle, where a +-8% neighbourhood of truth
contained only 3 exact solutions per 1000 samples).  Both remaining failures -- this and
the 50-neuron timing error -- are the same thing at different scales.

### CORRECTION to the entry above: the 4-neuron net was NOT a valid reproduction

Checking the actual magnitudes side by side:
    4-neuron "minimal"  offsets [-1,0,0,0,0,1]                  max  1   mean 0.3
                        (other seeds max 3 and 10)              max 10   mean 3.8
    50-neuron case0 N47 offsets [82,86,8,35,65,37,53]           max 86   mean 52.3
An order of magnitude apart.  The 4-neuron net is a NEAR-MISS case like the 2-cycle (four
of six spikes exactly right), not the gross mistiming seen at 50 neurons.  My search
selected on mean|dt| > 10, the 5-neuron candidates scored 13-36, and the property did not
survive pruning to 4 neurons -- which I failed to re-verify.

VALID REPRODUCTION (grad_twoloop_minimal.py, now updated): 5 neurons, 10 edges, output N4
    N0->N2 ; N1->N2, N3->N2 ; N2->N1, N2->N3 ; N1->N4, N2->N4, N3->N4 ; N4->N1, N4->N3
The output has FAN-IN 3 and FEEDS BACK into two of its own input paths.
    TRUE out [71,129,195,252,322,387,448,508], gaps [58,66,57,70,65,61,60]
    seed0 offsets [22,2,-12,-2,-15,-20,-14,-23]  max 23  mean 13.8
    seed1 offsets [ 5,-4, -1, 4, -3,  2,  6,  9]  max  9  mean  4.2
    seed2 offsets [10,-6,-23,-4,-18,-21, -3, -3]  max 23  mean 11.0
    seed3 offsets [ 4,16, -1, 1,  4,  0, -5,  8]  max 16  mean  4.9
    -> 0/4 exact, counts correct on EVERY seed -- the 50-neuron signature.

The distinguishing structure is therefore not "two loops" but DENSE RECURRENCE WITH THE
OUTPUT FEEDING BACK INTO ITS OWN MULTIPLE INPUT PATHS, matching the 50-neuron fan-in of
6-11.  Sparse recurrence (the 4-neuron version) gives near-misses; dense recurrence with
output feedback gives gross mistiming.

### CORRECTION: this is NON-CONVERGENCE, not identifiability

I had described the 5-neuron and 50-neuron results as an identifiability problem.  That is
wrong.  Identifiability means DIFFERENT WEIGHTS, SAME OUTPUT.  Here the output is visibly
wrong (offsets up to 23 steps small, 86 steps at 50 neurons), so it is simply an incorrect
answer -- an optimisation failure.

Checked directly, and the loss knows it:
  5-neuron at its stopping point:  max|gradient| = 1.81e-06
      the output hinges are strongly ACTIVE, e.g.
        t*= 71  vsub = 1.73e-03 vs th 7.0e-03  ->  demand +5.27e-03  ("fire here")
        t*=195  vsub(t*-1) = 1.43e-02 > th     ->  demand -7.33e-03  ("firing too early")
  50-neuron case0:                 max|gradient| = 4.74e-06, 76 of 132 components nonzero
For comparison, genuinely converged points measured earlier had |g| of 1e-14 to 1e-9.  So
the gradient is live and large; the optimiser has simply not converged.

AND MORE TIME DOES NOT HELP -- it degrades.  Running the 5-neuron case to 3000 rounds:
    r 300 out 8 spikes, mean|dt| 13.8, |g| 1.8e-06
    r 600 loses a spike (7 of 8) and never recovers
    r3000 out [110,176,250,314,378,454,516] vs target [71,129,195,252,322,387,448,508]
The iterate drifts steadily later (first spike 93 -> 110, last 485 -> 516) with |g| stuck
around 5e-07.  So it is not slow convergence either (unlike the 3-cycle, which reached
err=0 at r400 and froze): the trajectory wanders without approaching the target.

REVISED PICTURE of the three remaining failures:
  2-cycle          near-miss, err 0.25, tiny solution set    -- precision limited
  3-cycle          converges to err=0 given enough rounds    -- SOLVED, just slow
  dense recurrent  large live gradient, wanders, degrades    -- genuine non-convergence
    (5-neuron minimal and the 50-neuron cases)
The dense-recurrent case is therefore its own failure mode, and the earlier "counts right,
times wrong" description was only a symptom of stopping partway.

## 3-NEURON failure sweep (grad_3neuron_failures.py) -- a unified diagnosis

Swept the entire 3-neuron edge space {0->1, 0->2, 1->2, 2->1} against a weight grid
[100..1200], output N2, 2 seeds x 150 rounds.  98 failing configs.  Four cleanest kept.

THE FIRST SURPRISE: case A is FEEDFORWARD and fails on every seed --
    N0 -900-> N1 -300-> N2(out),  N0 -200-> N2 (weak bypass)
    TRUE N1 [39,139,239,339,439]  N2 [101,254,401]  gaps [153,147]
    every seed: w(0->2) collapses 200 -> ~60, w(1->2) rises 300 -> ~420,
                out ~[110,235,410], offsets [+9,-19,+9], gaps ~[124,173]
So recurrence is NOT required to break the method -- my earlier framing of the remaining
failures as a recurrence problem was too narrow.

THE UNIFYING PATTERN across the rest: a RARELY-FIRING hidden neuron injects a LOCALISED
PERTURBATION into an otherwise periodic output, and the method converges to the smooth
periodic train and misses it.
    B  N1 fires ONCE (317)      out gaps [100,100,69,131]
       found [68,168,268,368,468] -- perfectly regular, the 341 perturbation gone
    D  N1 fires ONCE (246)      out gaps [100,100,60,40,100]
       found 5 evenly spaced spikes instead of 6
    C  N1 fires 3x (317,416,515) out [246,345,444]
       found drops to 2 spikes, or lands 62 steps off
    A  same thing one level down: N2 itself accumulates across periods, so ITS gaps
       (153,147) are the irregular structure and the method finds ~124/173 instead.
The input drives every 100 steps, so a periodic output is the "easy" nearby solution; the
information distinguishing the true answer lives entirely in the few off-period spikes
contributed by a sparse hidden neuron, and that is exactly what gets smoothed away.

NOT impossible, though: seed2 solves both B and C EXACTLY (offsets all zero).  So these
are init-dependent basins, not unreachable solutions -- consistent with the 3-cycle, which
also converges from some seeds given enough rounds.

These cases are small enough to debug by hand, which was the point of the sweep: 3 neurons,
3 edges, deterministic per-seed behaviour.

## WHY nothing pushes the rare hidden neuron to fire: two threshold blockers

Case B (N0-500->N2(out), N1-1200->N2, N2-200->N1; true N1 fires ONCE at 317, giving output
gaps [100,100,69,131]).  At a failing seed N1 is silent and the output is regular.
Measured demand on N1:  max|L[1]| = 0.000e+00 -- literally nothing asks it to fire.  Why:

  BLOCKER 1 -- the missed target is MASKED.  The output should spike at 341; it spikes at
  368, which is 27 steps away and HIT_TOL is 60, so the target counts as "hit" and NO
  creation request is raised at all.
  BLOCKER 2 -- even a raised request would not PROPAGATE.  Upstream propagation requires
  vsub < CREATE_FLOOR*th = 1.4e-3, but vsub_2(341) = 4.33e-3.  There IS drive there, so the
  rule concludes "amplify locally" and never asks N1 for a spike.

Both thresholds were introduced earlier for good reasons -- HIT_TOL to stop a merely-LATE
spike being suppressed as spurious, CREATE_FLOOR to mean "nothing here to amplify".

CONFIRMED by removing them, and they must BOTH go (neither alone does anything):
    HIT_TOL=60 CREATE_FLOOR=0.2   1/3   N1 silent on seeds 0,1
    HIT_TOL=15 CREATE_FLOOR=0.2   1/3   request raised, cannot propagate
    HIT_TOL=60 CREATE_FLOOR=1.0   0/3   propagation allowed, no request exists
    HIT_TOL=15 CREATE_FLOOR=1.0   2/3   N1 fires at 315 / 319 (true 317), output EXACT

BUT THE BLANKET FIX IS FAR TOO EXPENSIVE.  Suite (5 cases, 4 seeds):
    HIT_TOL=15 CREATE_FLOOR=0.2   17/20  (chain 4/4, fanout-eq 4/4, fanout-hard 4/4,
                                          3-cycle 1/4, over-demand 4/4)
    HIT_TOL=15 CREATE_FLOOR=1.0    8/20  (chain 2/4, fanout-eq 2/4, 3-cycle 0/4,
                                          over-demand 0/4)
Always propagating requests upstream over-drives every hidden neuron -- the original
over-demand pathology returning.  Tightening HIT_TOL alone is roughly neutral (17/20 vs
18/20 at HIT_TOL=60).

WHAT THE CONDITION SHOULD BE: propagate upstream when the deficit is not LOCALLY fixable,
i.e. when scaling this neuron's own incoming weights enough to hit this target would
break its OTHER targets.  That is checkable with quantities already computed -- the
per-target rows A[j,k] of the eligibility matrix -- rather than guessed from a drive
threshold.  A sparse perturbation is exactly the case where local scaling cannot work
(it shifts every spike, not one), which is why these cases need the upstream request and
the periodic cases do not.

## Graded (distance-scaled) suppression and creation requests -- fixes the target case, costs the rest

User's proposal: a late spike should be suppressed in proportion to HOW late, and the
creation request likewise scaled by how far the target is missed, instead of the binary
MATCH_WIN / HIT_TOL tests.  Implemented all three binary decisions as continuous:
    suppression   frac = min(1, lateness / GRADE_SCALE)          (0 when exactly on target)
    request seed  frac = min(1, miss / GRADE_SCALE)
    propagation   unmet = R * clip(1 - vsub/th, 0, 1)            (fraction still missing)
All three vanish at an exact solution, so the fixed point is preserved.

IT DOES FIX WHAT IT TARGETS.  Case B (rare hidden neuron, true N1 fires ONCE at 317):
    binary : N1 = []          out [68,168,268,368,468]  -- perfectly regular, perturbation gone
    graded : N1 = [317]       out [80,180,280,343,480]  -- N1 fires at EXACTLY the true time
                                                            and the 341 perturbation appears
The residual is now a near-uniform +8 shift instead of missing structure -- a much easier
error.  This is something the binary version could never do, because the demand on N1 was
exactly zero.

BUT IT COSTS THE REST OF THE SUITE (4 cases x 4 seeds; binary baseline 14/16):
    all three graded, scale 30            7/16
    all three graded, scale 60            7/16
    graded supp only                      7/16
    graded request only                   6/16
    graded, hard propagation gate         9/16   (so the gate is NOT the culprit)
    graded + DEAD_ZONE 5                  7/16
    graded + DEAD_ZONE 15                 7/16
    graded + DEAD_ZONE 60 (= binary tol)  9/16
3-cycle drops 2/4 -> 0/4 in EVERY graded variant; over-demand 4/4 -> 0-2/4.

TWO HYPOTHESES TESTED AND REFUTED: (a) that the graded propagation gate was to blame --
hard-gating it made things worse (9/16 vs 7/16 with a hard gate but graded elsewhere...
actually 9/16, still below binary); (b) that the binary tolerance works by providing a
DEAD ZONE that lets near-correct configurations settle -- adding a dead zone of 5, 15 or
even 60 steps did not recover the binary score.

THE REAL TENSION: the binary MATCH_WIN/HIT_TOL of 60 is extremely permissive -- it ignores
timing errors up to 60 steps entirely.  The cases that currently converge (3-cycle,
over-demand) apparently NEED that permissiveness, while the sparse-perturbation cases need
sensitivity in exactly that range.  Any increase in responsiveness to sub-60-step errors
buys the latter and loses the former, regardless of the shape used.  Resolving this
probably needs the response to depend on WHY the spike is late (a shared shift of all
spikes, versus one spike out of place) rather than on how late it is -- a uniform offset
should be corrected by one weight, a lone displaced spike by a new upstream spike.

## The right discriminator is LINEAR FEASIBILITY, not lateness or spike pattern

My proposed rule -- "a uniform shift means reweight, a lone displaced spike means create" --
is WRONG, per the user's counterexample: with two presynaptic sources, one needing its
spikes earlier and the other later, the output shows a MIXED early/late pattern that the
rule reads as "displaced spikes -> create", when the actual fix is simply moving the two
weights in opposite directions.  That is the same conflicting-demand situation seen at the
3-cycle stuck point, where reweighting was the answer.  So neither distance (all the graded
variants) nor pattern can decide this.

THE ACTUAL QUESTION is feasibility: given the presynaptic spikes that CURRENTLY EXIST, is
there ANY reweighting that fires this neuron exactly at its targets?  vsub is linear in the
incoming weights, so this is a linear feasibility problem on the eligibility matrix already
being computed:
    find w in [lo,hi]  s.t.   sum_k w_k eps_k(t*)   >= th    for every target t*
                              sum_k w_k eps_k(t*-1) <  th    (so it fires AT t*, not before)
Solved with linprog; costs nothing extra since A is the eligibility matrix.

VALIDATED on exactly the cases that matter:
    case B, N1 SILENT                         feasible = False  -> create upstream  CORRECT
    case B, N1 firing at its true time 317    feasible = True   -> just reweight    CORRECT
    over-demand at true weights, MIXED
      early/late demands (the counterexample)  feasible = True   -> just reweight    CORRECT
So it fires the creation request exactly when no reweighting can work, and stays silent
when the conflicting demands are resolvable by weights alone -- which is what both the
binary thresholds and every graded variant failed to distinguish.

CAVEAT: the LP as written constrains firing AT the targets and not at t*-1; it does not yet
forbid firing at all OTHER times, so it is a necessary-condition test rather than exact.
Adding the non-firing rows for the remaining time steps makes it exact at some extra cost.
Integrating this as the propagation gate (replacing CREATE_FLOOR) is the obvious next step
and is untested.

## What should cause SUPPRESSION -- taxonomy with evidence

Settled first: a hidden spike that is merely UNUSED should NOT be suppressed.  If its PSP
stays sub-threshold downstream it costs nothing, the true network may well contain it, and
removing it does not help weight recovery either.  Suppression must be caused by HARM.

Three causes of harm:

A. THE SPIKE CAUSES A DOWNSTREAM SPIKE THAT SHOULD NOT EXIST.
   Well-defined at the output (ground truth there: an unmatched output spike is genuinely
   spurious) and must travel inward from there.  This is the only cause the current code
   handles, and only for OUTPUT neurons -- hidden neurons have no suppression at all, which
   is why opening creation up always over-fired.

B. THE SPIKE BLOCKS A SPIKE THAT SHOULD EXIST (refractory / reset).
   MEASURED, and it is common -- fraction of missed targets that are refractory-blocked
   (neuron fired within 22 steps before t*, so it physically could not fire at t*):
       3n case B (rare neuron)   8 of 10        5n two-loop      11 of 22
       3n case D                 7 of 18        3n case A         3 of  9
       3-cycle                   0 of  4
   So roughly HALF of all missed targets across the failing cases, 80% in case B.

   AND THE DEMAND MISREADS IT.  Case D, target 233, last spike at 226 (7 steps before):
       V_sim(233) = 0.000e+00     -- neuron is reset, cannot fire at any drive
       vsub(233)  = 1.016e-02 > th -- vsub resets at TARGETS, ignoring the 226 spike
       L(233)     = +0.00e+00      -- so the hinge reports the target SATISFIED
   The only nearby demand is L(232) = -2.76e-03 (push down at t*-1), and critically
   EVERY actual spike carries ZERO demand, including the 226 spike doing the blocking:
       found spikes [38,138,226,338,426] -> L = 0 at all five
   226 is matched to 233 within MATCH_WIN=60, so the spurious-spike suppression skips it.
   => THE DEFECT: demands attach to TARGET TIMES, never to ACTUAL SPIKES.  The thing that
      needs to move is the spike, and nothing addresses it.

C. THE SPIKE IS TOO EARLY AND HAS MULTIPLE CONTRIBUTORS (user's addition).
   Suppression then has to act on the CONTRIBUTORS, not on the spike itself, and the
   fragility is the split: reducing the wrong contributor breaks the other spikes it also
   serves.  This is the credit-splitting problem from earlier in the session, where graded
   proportional splitting beat winner-take-all -- so the same resolution likely applies.

IMPLICATION FOR BALANCE: because A and B are both grounded in observable output error, and
harmless spikes are deliberately left alone, suppression can be made specific rather than
blanket.  That is what should let creation be opened up without the runaway seen when
CREATE_FLOOR was raised -- the earlier runaway came from creation having NO opposing force
on hidden neurons, not from creation being intrinsically too eager.

### CORRECTION to cause B, and the result of implementing early-spike suppression

FIRST, MY MEASUREMENT OF B WAS WRONG.  Separating "a DIFFERENT spike blocks the target"
from "the target's OWN spike arrived early and sits inside its own refractory window":
    case          missed   own-early   different-spike   not blocked
    3n B (rare)      10        8              0               2
    3n A (feedfwd)    9        3              0               6
    3n D             18        7              0              11
    5n two-loop      22        8              3              11
Genuine blocking by a different spike is RARE (3 of 59).  The dominant pattern is the
target's OWN spike arriving early -- which is the user's cause C, not B.  My earlier
"roughly half of missed targets are refractory-blocked" conflated the two.

The gap is still real: an early spike gets NO demand anywhere.  The hinge at t* reads
vsub(t*) > th and calls the target satisfied, and nothing attaches to the spike itself
(case D: target 233, spike at 226, V_sim(233)=0, L(233)=0, and L=0 at all five found
spikes).

IMPLEMENTED: a negative demand at the EARLY SPIKE'S OWN TIME, scaled by how early it is
(less drive there -> it fires later).  Results:
    suite (4 cases x 4 seeds):  EARLY_GAIN 0 -> 14/16,  EARLY_GAIN 1.0 -> 13/16
        3-cycle     2/4 -> 3/4   (improved)
        over-demand 4/4 -> 2/4   (regressed)
    the 3-neuron cases that motivated it:  2/12 at every gain (0, 0.3, 1.0)
        B rare      2/4 throughout, mean|dt| 8.6 -> 7.4 (marginal)
        A feedfwd   0/4, and WORSE with gain (10.0 -> 12.0, one seed 17.3 -> 30.0)
        D dropped   0/4 throughout (count mismatch)
Net slightly negative, so the default is left at EARLY_GAIN=0.

WHY IT DOES NOT WORK: the demand is correctly placed but cannot be satisfied
independently.  Reducing drive at t=226 to delay that one spike reduces it at every other
time too, because the same weights produce all the spikes -- so delaying the early spike
delays the correct ones with it.  This is the same non-locality that defeated the
per-target hinges, the graded variants and the feasibility test: per-spike demands are not
independently satisfiable when one weight controls the whole train.

## Cause A implemented (spurious downstream spike suppresses what caused it) -- correct but INERT

Implemented as the mirror of the creation request: seed a suppression signal S at output
spikes that are GENUINELY EXTRA (one-to-one matching leaves them with no target to claim --
the one place we have ground truth that a spike should not exist), propagate it upstream by
the same kernel route as the requests, and apply it only at hidden neurons' ACTUAL spike
times.  Merely-unused spikes are deliberately left alone, per the harm-only principle.

RESULT: exactly neutral, at every gain.
    closed creation:  SUPP_GAIN 0 -> 14/16,  1.0 -> 14/16
    open creation:    SUPP_GAIN 0 ->  7/16,  1.0 -> 7/16,  3.0 -> 7/16
Identical scores at gain 3.0 gave it away: the signal is never seeded.  Checked directly:
    over-demand   target 3 spikes, found 3   extra = 0
    3-cycle       target 4 spikes, found 4   extra = 0
    3n case D     target 6 spikes, found 5   extra = 0
and a fresh sweep of the ENTIRE 3-neuron space found ZERO configurations where the trained
output over-fires.  The 5-neuron and 50-neuron cases match or under-fire as well.

SO: the method essentially never produces spurious OUTPUT spikes.  It under-fires or gets
the count right and the times wrong.  Cause A is real in principle but has no occasion to
act anywhere in the current test set.

MORE IMPORTANTLY, THIS REFUTES MY OWN EXPLANATION OF THE OPEN-CREATION REGRESSION.  I had
assumed opening creation up "over-fires", and that suppression would rebalance it.  But the
extra spikes appear at HIDDEN neurons while the output count stays correct -- which by the
harm-only criterion we agreed is HARMLESS.  So the 14/16 -> 7/16 drop when creation is
opened is NOT an over-firing problem at all; it is the responsiveness/settling problem
found with the graded variants (demands that react to sub-60-step errors never let a
near-correct configuration settle).  Suppression was the wrong remedy for that, and adding
it cannot help.

SUPP_GAIN left at 1.0: principled, provably neutral on everything tested, and it will act
if a case that genuinely over-fires at the output ever appears -- but it is UNTESTED,
because no such case exists in the current suite.

## Detecting HARMFUL hidden spikes: counterfactual ablation works, cheap integration does not

CORRECTION FIRST: I called extra hidden spikes "harmless" because the output SPIKE COUNT
stayed right.  Too narrow -- a hidden spike can also drag a downstream spike EARLIER than
its target without adding one, which is exactly the timing error we see.  Harm must include
mistiming, not just spurious spikes.

DETECTION IS NOT THE MISSING PIECE -- the backward message already reaches hidden spikes.
Measured on the 5-neuron net (hidden counts N1 3/5, N2 5/5, N3 8/7):
    N1: 3 negative, 0 positive     N2: 4 negative, 1 positive     N3: 5 negative, 2 positive
The problem is that it is INDISCRIMINATE: N1 is UNDER-firing and every one of its spikes is
being suppressed just as hard as N3's genuine extra spike.

A DETECTOR THAT DISCRIMINATES: counterfactual ablation -- delete one hidden spike,
re-simulate, and ask whether the OUTPUT timing improves.  Validated standalone:
    N1 (under-fires 3/5):  168 neutral, 293 neutral, 407 neutral   <- correctly NOT flagged
    N2 (5/5):              1 needed, 3 HARMFUL, 1 neutral
    N3 (over-fires 8/7):   4 needed, 3 HARMFUL, 1 neutral
So it leaves the starving neuron alone and fingers the genuine offenders -- precisely the
distinction the backward message cannot make.

BUT THE CHEAP INTEGRATION FAILS.  I wired it in using vsub crossings instead of a
re-simulation, and it has ZERO effect (suite 14/16 and 5-neuron mean error 8.47, identical
at ABLATE_GAIN 0, 0.5, 2.0).  The reason:
    N4 ACTUAL spikes        [91,151,205,258,323,378,432,497]
    crossings of vsub[4]    [121,177,318,373,432,497]        (6, not 8)
    targets                 [71,129,195,252,322,387,448,508]
vsub for an OUTPUT resets at the TARGET times, so its crossings are pinned near the targets
by construction and barely move when a contribution is removed -- the test measures the
wrong trace.  It flags 4 of 21 (pairs) versus 6 genuine offenders found by re-simulation,
and none of it changes training.

NEXT STEP: run the ablation on the ACTUAL simulated trace, as the standalone test does.
Cost is one partial re-simulation per hidden spike per iteration -- affordable at 3-5
neurons, and the correctness is already demonstrated.  ABLATE_GAIN left at 0 since the
integrated version is non-functional.

## Detecting harmful spikes WITHOUT re-simulation: the pivotal-contributor test

Re-simulating per spike is too slow, and unnecessary -- the contribution of a presynaptic
spike q to a downstream spike at f is just w*h(f-q), already available.  Define q as
PIVOTAL for f if removing that one contribution drops the drive below threshold:
      drive(f) = sum over q in (prev_spike, f] of w_kd * h(f - q)
      q pivotal  <=>  drive(f) - w_kd*h(f-q) < th
and HARMFUL if additionally f is EARLY relative to its matched target.  Pure arithmetic on
the spike trains and the kernel; cost O(spikes x contributors), no simulation.

AGREES WITH THE RE-SIMULATION VERDICT on the discrimination that matters:
    N1 (UNDER-firing 3/5)   pivotal-harmful: []            re-sim: none harmful
    N2                      pivotal-harmful: [398,477]     re-sim: 3 harmful
    N3 (OVER-firing 8/7)    pivotal-harmful: [342,408,462] re-sim: 3 harmful
Both leave the starving neuron alone, which the plain backward message does not (it pushed
down on all three of N1's spikes).

WIRED IN (PIVOT_GAIN): it does exactly what it was designed to do -- hidden over-firing is
corrected on every seed:
    gain 0    N2 = 8/5, 6/5, 5/5, 5/5      N3 = 8/7, 8/7 ...   mean output err  8.47
    gain 1.0  N2 = 5/5 on ALL seeds        N3 = 7/7 or 6/7     mean output err 54.78
    gain 3.0  same counts                                       mean output err 54.56
BUT IT OVER-SUPPRESSES: 2 of 4 seeds lose an output spike entirely (err 99), and N1 --
which the detector correctly reports as having NO harmful spikes at convergence -- still
falls from 3/5 to 2/5 (1/5 at gain 3), because during TRAINING its spikes are transiently
pivotal for early output spikes even though they are needed in the final solution.
Suite is roughly neutral (14/16 -> 13/16 at gain 1, 14/16 at gain 3).

STATUS: the detector is correct, cheap and validated; the control law built on it is not.
Suppressing every transiently-pivotal spike is too aggressive -- what is pivotal for an
early spike NOW may be required once the weights move.  A usable version needs the
suppression to be conditional on the spike still being harmful after the timing demand has
had its effect, or to be applied to the WEIGHT of the pivotal edge rather than to the
presynaptic spike itself.  PIVOT_GAIN left at 0.

## Why suppressed spikes never came back: a RATCHET caused by non-exclusive request matching

Question: if suppressed spikes are needed in the final solution, why does the creation
machinery not restore them?  Answer: it never fires, because of a real bug.

The creation-request seeding tested `any(|t - q| <= HIT_TOL for q in found)` -- a
NON-EXCLUSIVE proximity test.  One spike could therefore mark arbitrarily many targets as
"hit".  Measured on the 5-neuron net with the output 2 spikes SHORT (6 found vs 8 targets):
    target  71 -> nearest found 110 (39 away)   counted HIT
    target 448 -> nearest found 464             counted HIT
    target 508 -> nearest found 464             counted HIT   <- same spike claimed twice
ALL EIGHT targets counted as hit, so ZERO creation requests were raised even though two
spikes were missing.  Meanwhile suppression acts unconditionally, so the process was a
RATCHET: spikes could be removed but never restored.  That is why the pivotal suppression
drove N1 from 3/5 to 2/5 to 1/5 as its gain rose.

FIX: one-to-one matching in the request seeding (the suppression path already did this) --
each found spike can satisfy only ONE target, so surplus targets raise requests.

RESULT on the 5-neuron net (true hidden counts 5/5/7):
    before fix, gain 0    some seeds lose output spikes        mean err  8.47
    before fix, gain 1    2 of 4 seeds lose an output spike    mean err 54.78
    after  fix, gain 0    8/8 output spikes on ALL four seeds  mean err  7.72
    after  fix, gain 1    8/8 on 3 of 4; N1 and N2 reach EXACTLY 5/5 on two seeds
                                                               mean err 31.31
So with the ratchet removed the pivotal suppression finally does what it was designed for
-- hidden spike counts land exactly right on several seeds -- rather than driving spikes to
extinction.

Suite cost: 14/16 -> 13/16 (3-cycle 2/4 -> 1/4), so the fix is not free on the small cases,
but it is unambiguously correct: a matching that lets one spike satisfy two targets cannot
be right, and it was silently suppressing the entire creation pathway.

### Close look at the 3-cycle "regression" from the one-to-one fix: it was just SLOWER, not broken

The one-to-one request matching appeared to cost 14/16 -> 13/16 (3-cycle 2/4 -> 1/4).
Tracing seed0 shows it simply needs more rounds:
    r100  w=[672,491,462, 76]  N2 4sp  out [210,310,406,506]  err 3.00  |g| 5.8e-09
    r300  w=[620,487,465,250]  N2 5sp  out [214,314,405,505]  err 0.50  |g| 4.4e-10
    r400  w=[597,492,464,275]  N2 5sp  out [215,315,404,504]  err 0.50  |g| 6.0e-10
    r600  w=[596,473,482,215]  N2 4sp  out [214,314,404,504]  err 0.00  |g| 0.0e+00
    r700/r800  unchanged, gradient exactly zero -- converged and STAYS
Re-running the suite at 800 rounds instead of 400 restores 14/16 (3-cycle 2/4).  So the
fix is correct AND free; the earlier 13/16 was an artefact of the round budget.

WHAT THE FAILING SEEDS LOOK LIKE (400 rounds):
    seed2 SUCCESS  w=[513,508,486, 76]   N2 4sp (correct)   err 0.00
    seed0 near     w=[618,485,466,249]   N2 5sp             err 0.50
    seed3          w=[509,569,504,241]   N2 5sp             err 24
    seed1          w=[619,893,492,277]   N2 5sp             err 54
The pattern is consistent: failures pair an over-large w(0->1) with an over-large feedback
weight (~250 vs a true 60) and show N2 OVER-FIRING (5 spikes vs a true 4).  Compensating
errors again -- and note seed0 reaches err 0.5 while carrying w_fb=249, i.e. a badly wrong
feedback weight is largely absorbed by the others.

Also worth noting: the converged solution at r600 is w=[596,473,482,215] against a true
[500,500,500,60].  Exact output, feedback weight 3.5x off -- another point on the solution
manifold, consistent with the earlier finding that these outputs do not pin the weights.

## Close look at a REAL failure: a 254x-per-hop amplification bug that turns out to be LOAD-BEARING

Examined 3-cycle seed1 (err 54, genuinely fails).  Trajectory: max|g| = 1.4e+07 at r100,
then w(1->2) runs 725 -> 3000 (clipped) and w_fb 87 -> 1567, output collapsing 4 spikes -> 2.

TRACED THE 1.4e+07.  Demand magnitudes per neuron: output 7.0e-03 (correct, threshold
scale) but N2 3.3e+11, N1 1.0e+12, N0 2.6e+14.  Slopes were NOT floored, so not the 1/slope
term.  The amplifier is the CREATION-REQUEST propagation:
    msg = back_corr(unmet, HK) / peak
back_corr accumulates the kernel over its whole ~400-step support, so it scales with
sum(HK) = 4.00e-03, but it was normalised by peak = 1.58e-05.  Per-hop gain = 254x.
Measured directly: seeded R = 7.0e-03 -> 1.09e+02 after one sweep -> 9.5e+11 by sweep 5.
Confirmed it is a LOOP effect: on the acyclic chain max|L| grows only linearly with sweeps
(1e-6 -> 2e-4), on the cyclic 3-cycle geometrically (1.3 -> 4.2e5 -> 1.8e10 -> 1.0e12).

THE FIX WORKS NUMERICALLY.  Normalising by kernel MASS instead of peak removes the
explosion: gradient 1.4e+07 -> ~1e-07, weights no longer reach the 3000 clip.

BUT IT COSTS ACCURACY, AND EVERY VARIANT DOES:
    peak-norm, no caps (original)            14/16   <- best, despite 1e12 demands
    peak-norm + cap R to the seed            12/16
    peak-norm + cap the timing message       13/16
    mass-norm (dimensionally correct)        11/16
    mass-norm + REQ_GAIN 10 / 50             12/16 / 12/16
Retuning the gain does NOT recover it, which shows the peak normalisation was not merely
"stronger" -- it compounds PER HOP (254^k for k hops), and that is exactly the shape needed
to offset the per-hop signal DECAY measured earlier (the backward message loses ~1e5 per
layer without the 1/slope conversion, and comparable amounts with it).  So the bug is
load-bearing: it is simultaneously an explosion on cycles and the only thing compensating
depth decay.

DEFAULTS RESTORED to the best-performing configuration (REQ_PEAKNORM=1, R_CAP=0,
LOOP_CAP=0), with the pathology documented rather than papered over.  The principled fix is
a per-hop gain that compensates the measured decay EXACTLY -- something like normalising
each hop so the message magnitude is preserved relative to its source -- rather than
relying on a normalisation mismatch that happens to be the right size on shallow acyclic
nets and diverges on cyclic ones.  That is the next thing to build.

## A better normalisation: rescale each hop by its ACTUAL gain (shape-independent)

The two obvious normalisations are each correct only for one SHAPE of signal:
    sparse input (a few isolated target times):  back_corr peaks at U*peak(HK)  -> divide by PEAK
    broad input  (smeared over the support):     back_corr accumulates the mass -> divide by MASS
R is seeded SPARSE (at target times) but is smeared over the kernel's ~400-step support
after ONE hop.  So peak-normalisation is correct on hop 1 and then amplifies by
mass/peak = 254 on every hop thereafter -- which is precisely the measured behaviour
(7e-3 -> 1.09e2 after one sweep -> 9.5e11 by sweep 5).  Mass-normalisation has the mirror
flaw: right for the broad case, it crushes the initial sparse hop by the same 254x, which
is why it cost 14/16 -> 11/16.

FIX: normalise by the operation's actual gain, so a hop preserves magnitude whatever the
shape of its input:
    msg = back_corr(unmet, HK);   msg *= max|unmet| / max|msg|
Per-hop gain is then exactly 1 by construction, for sparse and broad signals alike.

Magnitudes are bounded again (3-cycle, max|L| on N1 by sweep count):
    peak-norm   1.3  -> 4.2e5 -> 1.8e10 -> 1.0e12
    self-norm   0.71 -> 1.45  -> 6.6    -> 95
Ten orders of magnitude better.

Self-norm ALONE scores 12/16, because gain 1 per hop is too little -- the 3-cycle needs the
demand to survive two hops back from the output, and it drops to 0/4 (as it does under
mass-norm and under both caps).  Adding an explicit compensation gain restores it, and the
result is FLAT in that gain, so it is not a tuned knob:
    REQ_GAIN =   3  -> 14/16      REQ_GAIN =  30 -> 14/16
    REQ_GAIN =  10  -> 14/16      REQ_GAIN = 100 -> 14/16

NET RESULT -- same accuracy, vastly better conditioning.  On the pathological 3-cycle seed:
    peak-norm (old)          max|gradient| during training 1.44e+07
    self-norm + gain 3       max|gradient| during training 7.07e+01
a 200,000x reduction with identical hidden-demand magnitudes and identical suite score.
Wider run at the new defaults (REQ_SELFNORM=1, REQ_GAIN=3): chain 4/4, fanout-eq 4/4,
fanout-hard 4/4, BREAK 4/4, 3-cycle 2/4, over-demand 4/4 = 22/24.

The separation is the point: normalisation now handles the SHAPE (exactly, by construction)
and a single explicit constant handles the DEPTH DECAY, instead of one accidental
normalisation mismatch doing both jobs and diverging on cycles.

## A real bug in the eligibility reconstruction: input arriving during REFRACTORY is discarded

Examined 3n A (the purely FEEDFORWARD 3-neuron failure, 0/4 on every seed):
    N0 -900-> N1 -300-> N2(out),  N0 -200-> N2 (weak bypass)
    TRUE N2 [101, 254, 401], gaps [153,147] -- 3 spikes in 5 input periods
Every seed collapses the bypass w(0->2) 200 -> ~40 and raises the chain 300 -> ~435.

FIRST CLUE: at the TRUE weights, where the output is EXACTLY right, the gradient is NOT
zero -- max|g| = 7.59e-08, versus exactly 0.000e+00 for the chain and 3-cycle controls.
Scanning the bypass weight with the others at truth, the gradient's zero sits near w02~130
while the correct value is 200, and at w02=200 (err = 0) the gradient pushes DOWN.

CAUSE: the simulator discards input that arrives while a neuron is refractory --
    out = out * (refractory_timers == 0)        # voltage held at zero
    rise_values *= (refractory_timers != 1)     # accumulator WIPED at the end of it
-- but eligibility() integrated it regardless.  With delay 18 and refractory 22, a
presynaptic spike at s is lost if r <= s + 18 <= r + 22 for some reset r.  Here the output
fires at 101 and N0 also spikes at 101, arriving at 119, inside the refractory window
[101,123].  Its contribution 200*h(152) ~ 3e-3 is spurious, and shows up exactly:
    t*=254   V_sim(253) = 6.96654e-03      vsub(253) = 9.91953e-03   (42% too high)
             -> fabricated demand L(253) = -2.92e-03, "you are firing too early",
                at weights where the output was already perfect.
This is why the fit was driven off the true bypass weight: truth was not a stationary point.

FIX: skip presynaptic spikes whose ARRIVAL falls in a refractory shadow of the
postsynaptic neuron's resets.  Reconstruction becomes exact and stationarity is restored:
    REFRAC_MASK=0   vsub(253)=9.91953e-03   L(253)=-2.92e-03   max|g| at truth 7.59e-08
    REFRAC_MASK=1   vsub(253)=6.96657e-03   L(253)= 0.00e+00   max|g| at truth 0.00e+00
(V_sim(253)=6.96654e-03, so the reconstruction now matches to 5 significant figures.)

SUITE (800 rounds, 7 cases x 4 seeds):
                     mask off   mask on
    3n A feedfwd        0/4       3/4    <- the case under investigation
    2-cycle             3/4       4/4
    fanout equal        4/4       4/4
    chain               4/4       3/4
    3-cycle             2/4       1/4
    over-demand         4/4       3/4
    3n D dropped        0/4       0/4
    TOTAL              17/28     18/28
Net positive, and it fixes the two cases whose failure mode was exactly this.  The three
single-seed regressions are most likely basin effects rather than a defect, since the fix
strictly improves the reconstruction (exact vs 42% error) -- but that is untested.

### Checked: are the regressions from the refractory fix real?  (12 seeds instead of 4)

Correctness first -- gradient at the TRUE weights, all cases:
    mask OFF   chain 0.0, 3-cycle 0.0, over-demand 0.0, 2-cycle 0.0, 3n A 7.589e-08
    mask ON    ALL FIVE exactly 0.000e+00
So the mask strictly improves stationarity; no case is made worse by it.

Statistics at 12 seeds (the 4-seed run was too thin):
                      mask off   mask on
    chain                8/12      8/12    <- the 4-seed "regression" was NOISE
    3-cycle              4/12      2/12    <- real, small
    over-demand         10/12      8/12    <- real, small
    3n A feedfwd         0/12     10/12    <- the case under investigation
    2-cycle             10/12     12/12
    TOTAL               32/60     40/60
Net +8.  chain was sampling noise as suspected; 3-cycle and over-demand are genuine but
small regressions.

Since stationarity is restored everywhere and broken only WITHOUT the mask, those two
regressions are not a gradient defect.  The likely mechanism is the one already seen when
the biased (pre-hinge) gradient scored better: the phantom refractory contributions act as
a perturbation that occasionally kicks the optimiser out of a bad basin.  Not a reason to
retain a reconstruction that is 42% wrong, but it does say the remaining failures are
basin-limited rather than gradient-limited, which is consistent with everything else in
this file.

## 3n D: the request asks for a spike at a TIME, but weight increase moves spikes EARLIER

Net: N0 -200-> N1 -700-> N2(out), N0 -1200-> N2 (strong bypass).
TRUE N1 fires ONCE at [246]; TRUE N2 [33,133,233,293,333,433], gaps [100,100,60,40,100] --
the 293 spike exists ONLY because N1 fires at 246.  Gradient at truth is 0.000e+00.

FIRST, A MATCHING BUG (fixed, but not the blocker).  One-to-one matching was greedy in
TARGET order, so an early target could steal a later one's spike and cascade:
    293 -> 333 (40 away),  333 -> 433 (100 away),  433 -> None
Greedy by CLOSEST PAIR gives the obvious 293 -> None instead.  Suite effect is neutral
(39/72 vs 40/72; 3n D 0/12 -> 1/12), so the mis-assignment was not what blocked this case.

THE ACTUAL FAILURE CHAIN.  At the converged point w=[247,1213,20] (true [200,1200,700]):
    N1 fires TWICE at [183,383]   -- true is ONCE at [246]
    w(1->2) has collapsed to 20, the lower clip
Scanning w12 from there, the gradient is NEGATIVE at every w12 >= 100, i.e. its optimum has
INVERTED.  Restore w01 to its true 200 and the picture flips completely:
    w01=247: N1 [183,383]  w12 grad -1.6e-08 (100) ... -1.5e-07 (700)   -> collapses
    w01=200: N1 [246]      w12 grad +6.9e-08 (20) ... +0.0e+00 (700)    -> EXACT at 700
So w01 is the binding error, and its gradient at the converged point is +3.33e-04 --
FOUR ORDERS larger than anything else -- pushing it further ABOVE the true 200.

WHY IT POINTS THE WRONG WAY.  The missing output spike at 293 needs N1 to fire inside the
epoch (233, 293].  N1's actual spikes at 183/383 lie outside it, so eps_1(293)=0 and w12's
gradient is exactly 0 -- correctly, since w12 genuinely cannot help.  The request therefore
falls entirely on w01 as "more drive to N1".  But for an ACCUMULATING neuron more drive
makes the EXISTING spikes fire EARLIER (183 -> sooner) and eventually adds more of them; it
does not place a spike at 246.  The demand is right, the available response is backwards.

This is the same mismatch seen throughout: a request specifies a spike at a TIME, while the
only actuator is a weight that shifts the WHOLE train.  It is why "fire later" is
unreachable by weight increase, why the early-spike suppression could not be satisfied
independently, and why this case needs w01 DOWN and w12 UP simultaneously -- a coordinated
move that no single per-target demand expresses.

## 3n D continued: a direct WEIGHT demand works, but the case needs a weight to go DOWN

Two more bugs found and fixed along the way:
  * MATCHING was greedy in TARGET order, letting an early target steal a later one's spike
    and cascade (293->333, 333->433, 433->None).  Greedy by CLOSEST PAIR gives the correct
    293->None.  Suite-neutral (39/72 vs 40/72), so not the blocker.
  * The user's weight-demand term was scaled by max|g|, which VANISHES exactly when the
    ordinary gradient is frozen -- i.e. when it is needed.  Rescaled to a fixed reference.

THE WEIGHT DEMAND ITSELF WORKS.  A request cannot reach a weight through g = L . eps when
the presynaptic does not fire in the epoch, because eps is then 0.  Stating instead what
the EDGE would have to be -- w_kn >= deficit / peak(h) -- gives a path that does not
multiply through the presynaptic's firing.  From the frozen point w=[3000,1205,200]:
    WREQ_GAIN=0     w12 stays 200 forever
    WREQ_GAIN=1/10  w12 rises 200 -> 480 (its computed target ~444) and holds
So the mechanism does what it was designed to do.

BUT THE MISSING SPIKE STILL DOES NOT APPEAR, and the reason is decisive.  With w01 pinned
at the upper clip (3000), N1 fires at [24,124,224,324,424].  The needed epoch for the 293
output spike is (233,293]; N1's spikes at 224 and 324 STRADDLE it without landing inside.
No value of w12 can help -- there is nothing to multiply in that window.  What the case
needs is w01 DOWN (to ~200, so N1 fires once at 246), and nothing asks for that.

WHY w01 WENT UP TO THE CLIP: the request for the 293 spike is expressed as "more drive to
N1", and more drive makes an accumulating neuron fire EARLIER and more often, never at a
LATER time.  So the request pushed w01 monotonically to 3000 and the state froze there,
with the output hitting 5 of 6 targets EXACTLY and one permanently missing.

THE GENERAL GAP: a creation request can only ever ASK FOR MORE DRIVE.  It has no way to say
"fire LATER", which for an accumulating neuron means LESS drive.  Every failure examined
today that survives the gradient fixes has this shape -- 3n D needs w01 down and w12 up
simultaneously; the 3-cycle needed w(0->1) down with w(2->3) up; the early-spike suppression
could not be satisfied because reducing drive delays the whole train.  A request that
carries a SIGN on the drive, or that is expressed on the spike TIME rather than the drive
magnitude, is the missing primitive.

## Suppressing the BLOCKING spikes (user's call): right diagnosis, correct gradient, no outcome change

The observation was that N1's spikes at 224/324 have to go.  They do -- and the reason is
not refractory (224 is 69 steps before the needed 293) but the RESET: having fired at 224,
N1 restarts from zero and cannot re-accumulate to threshold by 293, because the input at
201 is already spent and 301 arrives too late.

IMPLEMENTED as cause B generalised to hidden neurons, which had never been done (hidden
neurons have no targets, so "blocks a needed spike" was previously undefined).  A request
at tau on n now looks at n's most recent spike q < tau and asks whether, after resetting
there, n could reach threshold by tau:
    reach = sum_k w_kn * sum{ h(tau - t) : q < t <= tau }      blocked if reach < th
and if so suppresses q.  This is checkable purely from the spike trains and the kernel.

IT WORKS AT THE GRADIENT LEVEL -- it flips the sign that was wrong:
    BLOCK_GAIN=0   demand at N1's spikes  +0.073, +0.112, +0.112   w01 grad +2.13e-06  UP
    BLOCK_GAIN=5   demand at N1's spikes  -0.292, -0.254, -0.072   w01 grad -2.11e-08  DOWN
and truth needs DOWN (3000 -> 200).  At gain 20 it also UNFREEZES the clip that nothing
else could move: w01 runs 3000 -> 1963 -> 971 -> 953 instead of sitting at 3000 forever.

BUT IT DOES NOT CHANGE OUTCOMES.  It settles at w01=953 against a true 200, N1 still fires
5 times instead of once, and the suite is identical with and without it:
    BLOCK_GAIN 0    chain 3/6, 3-cycle 0/6, 2-cycle 6/6, over-demand 4/6, 3n A 5/6, 3n D 0/6 = 18/36
    BLOCK_GAIN 20   exactly the same, 18/36
Two reasons it is weak: the corrected gradient is ~100x smaller than the wrong one it
replaces (2e-08 vs 2e-06), so it needs a large gain to be felt at all; and even with w01
falling, N1's five spikes have to collapse to ONE, which is a change of spike COUNT that a
per-spike suppression term nudges only indirectly.

Default left at BLOCK_GAIN=0.  The diagnosis and the detector are sound -- this is the
first mechanism that can express "fire later" at all -- but it is not sufficient on its own.

## Sharpening the request to ONE time: mechanically right, not yet a reliable win

Premise (user): if a request names one specific time, the neuron should converge to firing
AT that time.  The design fought this -- the request is deliberately SPREAD over all
feasible latencies, so on 3n D it is nonzero across [0,274], and a broad "fire somewhere in
here" demand is satisfied by firing EVERYWHERE (N1 fires 5 times against a true 1).

IMPLEMENTED: collapse R[n] to a single time and give the demand BOTH signs -- pull towards
the requested moment, push down on this neuron's spikes further than SHARP_WIN from it.

WHICH TIME matters enormously:
  * argmax of the spread is WRONG.  back_corr(delta at 293, HK) peaks where 293-s =
    argmax(HK) = 110, i.e. s = 183 -- the latency of maximum kernel INFLUENCE, not the one
    that causes a crossing (true latency here is 293-246 = 47).  Measured: the neuron
    converges to 356 and the extra output spike lands at 394 instead of 293.  It converges
    to A time, just not the right one.
  * the WEIGHT-IMPLIED latency (first dt with w*h(dt) >= deficit) is much better.  With it,
    3n D seed5 produces the correct STRUCTURE for the first time from a random init:
        N1 = [236]  (true [246])
        out = [32,132,232,290,332,432] vs target [33,133,233,293,333,433]
        six spikes WITH the perturbation, offsets [-1,-1,-1,-3,-1,-1]
    Previously every seed gave five evenly spaced spikes and no perturbation at all.

RESULTS ARE INCONSISTENT, though:
    thorough version (scan all nonzero request entries), 400 rounds:  13/20 vs 10/20  HELPS
    fast version (peak of each downstream request only), 800 rounds:  10/24 vs 13/24  HURTS
The thorough version had to be abandoned because it runs request_lat (400 steps) over ~275
entries per sweep per round, which is intractably slow; the peak-only approximation picks a
worse time and loses the benefit.  So the mechanism is validated in principle and produces
the first correct structure on the hardest case, but a usable version needs the right
requested time computed cheaply -- the approximation is what is failing, not the idea.
Defaults left at SHARP_GAIN=0.

## Making the sharp-request latency fast: a binary search instead of a linear scan

request_lat asks for the smallest dt with w*h(dt) >= deficit, by scanning up to KWIN=400
steps.  The accurate sharpening needs one such query per nonzero request entry (~275) per
edge per sweep per round, which made it intractable and forced a peak-only approximation
that picked worse times and lost the benefit.

HK's rising phase is MONOTONIC, so the same query is a binary search, and it vectorises
over all deficits at once:
    _HK_RISE = HK[:argmax(HK)+1]                       # increasing, searchsortable
    dt = np.searchsorted(_HK_RISE, deficits / w)       # -1 where the peak cannot cover it
Verified IDENTICAL to the scalar version on 1200 cases (0 mismatches), and
    200 x 275 queries:  scalar 0.396 s   vectorised 0.0011 s   -> 355x faster
Training cost with the thorough sharpening restored is now negligible:
    SHARP_GAIN=0  2.7 ms/round     SHARP_GAIN=1  2.8 ms/round   (4% overhead)

With the thorough version affordable again, sharpening is a modest net positive:
    800 rounds, 4 seeds:  13/24 -> 14/24   (chain 3/4 -> 4/4, 3-cycle 0/4 -> 1/4,
                                            over-demand 3/4 -> 2/4)
    400 rounds, 4 seeds:  10/20 -> 13/20   (measured earlier with the slow-but-accurate
                                            version, same behaviour)
So the idea holds up once the right requested time is computed properly: the earlier
inconsistency was the approximation, not the mechanism.  Still short of a decisive win --
3n D remains 0/4 -- but it is now cheap enough to keep on and iterate against.

## 3n D SOLVED (first time), with the fast sharp request

With the thorough sharpening restored (vectorised latency, 355x faster), 3n D over 8 seeds:
    seed6  w=[203,1191,655] (true [200,1200,700])  N1=[243] (true [246])
           out=[33,133,233,293,333,433]  ==  target        EXACT
    seed5  w=[213,1345,604]  N1=[236]  offsets [-1,-1,-1, -3,-1,-1]   6-spike structure
    seed7  w=[213,1267,743]  N1=[236]  offsets [-1,-1,-1,-13,-1,-1]   6-spike structure
    seed2  w=[165,1019,874]  N1=[355]  extra spike at 394             structure, misplaced
    seed3  w=[156,1140,754]  N1=[442]  extra spike at 486             structure, misplaced
    seed0/1  5 spikes    seed4  7 spikes
=> 1/8 EXACT and 4/8 with the correct six-spike structure, against 0/6 and NO structure at
all before sharpening.  This case had been 0/N for the whole session.

THE DETERMINING FACTOR IS w01.  Seeds that land near the true 200 (203, 213, 213) get N1
firing ONCE at 236-243 and the structure follows; seeds with w01 too low (165, 156) get N1
firing once but far too LATE (355, 442).  The misplaced spike is then matched to target 293
with a ~100-step error (closest-pair still pairs 394 with 293, since the counts work out)
and the configuration is stable there -- a wrong basin, not a wrong gradient.

So the remaining 3n D failures are basin selection on a single weight, which is a much
narrower problem than "the request cannot express what it needs".

## What the extra spikes were doing: nothing, and then the wrong thing

Asked whether the misplaced spikes are pushing on anything correctly.  Measured at 3n D
seed2's converged point (N1=[357], out=[36,136,236,336,396,436] vs target
[33,133,233,293,333,433]) and found TWO defects.

DEFECT 1 -- TWO DIFFERENT MATCHINGS.  The suppression path matched greedily over spikes in
TIME order while the request path used closest-pair, and they disagreed:
    suppression: 396 -> 433 (so never suppressed),  436 -> 293 (so pushed UP by +6.3e-03)
    requests   : 396 -> 293,                        436 -> 433
So the MISPLACED spike (396) received exactly ZERO demand while a CORRECT spike (436) was
being pushed up.  Fixed by sharing one closest-pair matching; afterwards 396 correctly gets
suppression (-7.8e-04) and 436 gets 0.

DEFECT 2 -- THE SHARPENING WAS UNSIGNED, AND THEN SWAMPED, AND THEN OVERWRITTEN.
The request asks N1 for 254 (true 246) and N1 sits at 357, so it must fire EARLIER, i.e.
needs MORE drive.  The demand there was -0.4622 (less drive -> later).  Three layers:
  a) the sharpening only ever SUPPRESSED spikes far from tau, regardless of direction;
  b) making it signed but ADDITIVE was useless -- worth |vsub-0.9th| ~ 7e-4 against a
     timing term of ~0.46 at the same spike (which carries the 1/slope factor), 650:1;
  c) it also ran BEFORE the backward relaxation, which then added that timing term on top.
Fixed by SETTING THE SIGN after the relaxation: a spike later than tau gets a positive
demand, earlier gets negative.  Demand at N1 flips -0.4622 -> +0.4629 and the w01 gradient
-1.73e-05 -> +2.20e-05, which is the direction truth needs (165 -> 200).

RESULT on 3n D (8 seeds): 1/8 exact -> 3/8 exact.  The two seeds that had the misplaced
spike are exactly the ones fixed:
    seed2  N1 355 -> 252,  out now == target      EXACT
    seed3  N1 442 -> 251,  out now == target      EXACT
    seed6  already exact (N1 243)
Suite is NEUTRAL, 12/24 either way: 3n D 0/4 -> 2/4, chain 3/4 -> 2/4, over-demand 3/4 ->
2/4.  So the mechanism does precisely what it was built for and costs about as much
elsewhere.  Default left at SHARP_GAIN=0.

The through-line for the whole session: every one of these was a case where the demand
existed but could not reach, or could not correctly sign, the quantity that had to change.

## Container restart: environment restored, DEFAULTS corrected

Reinstalled via setup_env.sh (jax 0.10.2, numpy 2.4.6, matplotlib 3.11.1, scipy 1.17.1).
Verified the gradient is still exactly 0.000e+00 at the true weights.

IMPORTANT -- the file defaults did NOT match the configuration everything was measured
under.  Several settings found HARMFUL were still on by default, and I had been overriding
them by env var on every run:
    LR             0.5  -> 10.0    every measurement used 10; 0.5 is far too small
    RESTART_EVERY  0    -> 100     periodic Adam restart, 3-cycle 0/4 -> 2/4
    GRADE_SUPP/REQ/PROP  1 -> 0    the graded variants cost 14/16 -> 7/16
    PROX           0.01 -> 0.0     additive pull-back breaks stationarity, 4/12 -> 0/12
    SUPP_GAIN      1.0  -> 0.0     inert (no case over-fires at the output, so never seeds)
A plain run with the old defaults gave 2-cycle 0/4; with the corrected ones it is 3/4.

Defaults-only suite (800 rounds, 4 seeds), no env vars:
    chain 3/4, fanout-eq 4/4, fanout-hard 4/4, BREAK 4/4, 2-cycle 3/4, 3-cycle 0/4 = 18/24

OPEN DISCREPANCY: the 3-cycle is 0/4 here against 2/4 measured earlier.  Changes since that
measurement are the refractory mask, closest-pair request matching, and sharing ONE
matching between the suppression and request paths.  The last of those was validated only
on 3n D plus a single suite run that was neutral overall, so it is the prime suspect and is
NOT properly verified.  Worth bisecting before trusting the 3-cycle number either way.

## Bisecting the 3-cycle regression: the correctness fixes cost more than they gain

Three changes were made after the 3-cycle last measured 2/4: the refractory mask, the
closest-pair REQUEST matching, and sharing that matching with the SUPPRESSION path.
Bisected on the 3-cycle (8 seeds):
    SHARED PAIR REFRAC   3-cycle
      1     1     1        1/8     (what the defaults had become)
      1     1     0        2/8
      1     0     1        2/8
      0     1     1        1/8
      0     0     0        4/8
Each of REFRAC_MASK and PAIR_MATCH costs about one seed; SHARED_MATCH alone costs nothing.

FULL COMPARISON (6 cases x 8 seeds):
    case          all-original   all-fixed   mask-only (adopted)
    chain             5/8           4/8          4/8
    3-cycle           4/8           1/8          2/8
    2-cycle           8/8           7/8          8/8
    over-demand       7/8           6/8          6/8
    3n A              0/8           7/8          7/8
    3n D              4/8           1/8          1/8
    TOTAL            28/48         26/48        28/48
Adopted the third: same total as all-original but with NO case at zero.  Reverted the two
matching changes by default (SHARED_MATCH=0, PAIR_MATCH=0) despite each fixing a real
defect, because both measurably cost recovery.

A CORRECTION TO AN EARLIER CLAIM.  I reported 3n D as "0/N for the whole session, finally
solved by sharpening (3/8)".  That was wrong: 3n D is 4/8 with NO sharpening once the
refractory mask is off.  It had been 0/N because I made that mask default-on -- so the
sharpening was clawing back damage I had introduced, and still ended below the 4/8 that
simply not masking gives.

WHY THE MASK SPLITS THE CASES.  Checked at the true weights:
    3n A  mask off: max|g| 7.59e-08, |vsub - V_sim| at targets 2.9e-03   <- real defect
          mask on : max|g| 0.00e+00,                            2.4e-08
    3n D  mask off: max|g| 0.00e+00,                            4.4e-08   <- no defect
          mask on : max|g| 0.00e+00,                            4.4e-08
3n A genuinely has a presynaptic spike landing in a refractory shadow at the solution, so
the mask is REQUIRED there (without it truth is not stationary and the case is 0/8).  3n D
has no such spike -- the mask is a no-op at its solution and only perturbs the search path
away from it, which is where the 4/8 -> 1/8 comes from.
Masking on ACTUAL spikes rather than target times (REFRAC_ACTUAL) takes 3n A to 8/8 but
wrecks over-demand (7/8 -> 1/8) and totals 21/48, so it is off.

## Close look at the CHAIN (4/8) -- a real bug in the epoch construction

The chain is the simplest case in the suite (feed-forward, uniform weights, no bypass) and
it only recovers 4/8.  Every failure has the SAME signature -- a UNIFORM offset, every
output spike early by the same amount:
    seed5  w=[591,592,451]   offsets -6 x4
    seed1  w=[640,866,452]   offsets -29 x4
    seed7  w=[875,1033,450]  offsets -43 x4
    seed4  w=[1089,873,450]  offsets -44 x4
and the trajectory moves AWAY from the solution with the gradient endorsing it:
    r400  off  -6  g=[-8.3e-07,...]   <- correct, asks for less drive
    r800  off -24  g=[+1.7e-06,...]   <- FLIPPED, now driving weights up
    r1600 off -37                     r2400 spikes lost entirely

CAUSE -- THE EPOCH BOUNDARY EXCLUDES THE DRIVING SPIKE.  vsub at target t* accumulates from
the PREVIOUS TARGET.  With the output 6 steps early, N2 fires at 213 while the epoch for
target 314 is (214, 314] -- so the spike that actually drives that target is excluded, and
the only one inside is at 313, whose PSP is ~0 one step later.  Measured:
    target 214  vsub 7.07e-03  (fine)      target 314  vsub 0.0000e+00
    target 414  vsub 0.0000e+00            target 514  vsub 0.0000e+00
    -> hinge demands the FULL deficit +7.00e-03 at three of four targets
So the reconstruction reports NO drive, the demand asks for MORE, the spikes move earlier
still, and they get excluded further: a positive feedback loop.  In the true solution N2
fires at 243, comfortably inside (214, 314], which is why this only appears once mistimed.

NOT the refractory mask -- checked, MASK=0/1 and ACTUAL=0/1 all give bit-identical
trajectories here (the chain has no spike in a refractory shadow).

FIRST FIX ATTEMPTED AND REJECTED: extend each epoch back to the neuron's real reset when
that is earlier (EPOCH_ACTUAL).  It does stop the runaway -- offsets stay -3..-8 instead of
diverging to -37, and the weights stay near truth ([491,519,508] vs [591,592,451]) -- but
it converges to a BIASED fixed point (-8, never 0) and the suite collapses to 1/48 from
26/48.  Using targets as epoch boundaries is what makes the demand monotone in w; replacing
them with actual resets destroys that, so nothing recovers exactly.
Left at EPOCH_ACTUAL=0.  The bug is precisely characterised but NOT yet fixed: the fix has
to keep the target-based monotonicity while preventing the boundary from cutting off the
spike that drives the target.

## FIXED: the epoch-boundary runaway (EPOCH_EXTEND)

The bug: vsub at target t* accumulates only over (previous target, t*].  Once the network
is mistimed the spike that actually drives t* can fall just before that boundary, leaving
an epoch whose only spike is too close to t* to contribute.  vsub reads ~0, the hinge
demands the FULL deficit, that asks for more drive, the spikes move earlier and are
excluded further -- a positive feedback (chain: -6 -> -24 -> -37 -> spikes lost).

TWO ATTEMPTS:
  1. EPOCH_ACTUAL -- replace the boundaries with the neuron's actual spike times.  Stops
     the runaway but destroys everything: the epoch for target 314 became (308, 314], six
     steps long, so nothing can be explained.  Suite 26/48 -> 1/48.  Rejected.
  2. EPOCH_EXTEND (adopted) -- leave every boundary alone EXCEPT where the epoch cannot
     reach threshold at all.  The test is on achievable DRIVE, not spike presence: the
     broken epoch is not empty, it holds a spike one step before the target whose PSP is
     ~0.  Where drive < EXTEND_FLOOR*th, pull that one boundary back to admit the previous
     presynaptic spike.

    def _drive(lo, t): return sum_k w_k * sum{ h(t-q) : lo < q <= t }
    if _drive(prev_bound, t) < EXTEND_FLOOR * th:  prev_bound = last_presyn_spike_before - 1

RESULT.  chain seed5, which used to diverge, now converges EXACTLY at r400 and stays:
    EPOCH_EXTEND=0   -6 -> -24 -> -29 -> -37 (and on to losing spikes)
    EPOCH_EXTEND=1   w=[520,523,474], out == target, offsets [0,0,0,0], stable to r1600
Suite 26/48 -> 27/48 at 800 rounds (chain 4/8 -> 5/8, over-demand 6/8 -> 7/8, but
2-cycle 7/8 -> 6/8).

THE 2-CYCLE "REGRESSION" IS NOT REAL -- it is a ROUND-BUDGET artifact.  Per-iteration
traces of the flipped seed (seed4) show both configs taking the SAME path: dive early to
off ~ -50, sit on a plateau, then escape and converge inward.  EE=0 escapes at ~it150 and
is exact by it550 (then sits there 250 iterations).  EE=1 stays on the plateau until
~it500 and at it800 is still improving monotonically -- offsets [10,-3,-18,-53] ->
[9,4,-3,-11] -> [5,3,-1,-6] -> [1,2,1,-1] -> [-1,0,0,-1].  It had not finished, not
failed.  At 1200 rounds BOTH configs are 8/8 on 2-cycle.

So 800 rounds is the wrong yardstick for this comparison.  Re-run at 1600:
    EPOCH_EXTEND=0   chain 4  3-cycle 1  2-cycle 8  over-demand 6  3n A 7  3n D 1  = 27/48
    EPOCH_EXTEND=1   chain 5  3-cycle 1  2-cycle 8  over-demand 7  3n A 7  3n D 1  = 29/48
A clean +2 with NOTHING regressing.  Note this also cost EE=0 nothing to give: the extra
budget moved it 26 -> 27 as well (2-cycle seed3).

The cost of the fix is that it lengthens the early plateau (~350 extra iterations on
2-cycle seed4).  Worth watching, but it is a convergence-RATE cost, not an accuracy one.

CRUCIALLY IT PRESERVES STATIONARITY, which is what killed attempt 1.  At the true weights
every epoch already contains its driving spike, so no extension fires and the construction
is unchanged.  Verified: max|g| at truth = 0.000e+00 on chain, 3-cycle, over-demand, 3n A
and 3n D, both with the flag on and off.

### tooling: _suite_mp.py

The suite is embarrassingly parallel (every case x seed is independent), so it now runs as
a 16-way process pool: `python3 _suite_mp.py <rounds> [seeds]`.  Uses 'spawn' so each
worker imports grad_trace fresh and picks up the env-var config from the parent, and pins
BLAS/JAX to one thread per worker so 16 processes do not each spawn 16 threads.  ~5.6x
wall-clock (both configs at 800 rounds: 220s -> 39s; at 1600 rounds: 75s).  Verified to
reproduce the serial numbers exactly, per-case and per-seed, before being used.

train() also gained an optional `cb=None` callback, invoked each iteration just before the
weight update with (it, w, upd, g, spall, vsub, L).  This matters because chunking a run
into repeated train() calls is NOT equivalent to one long call -- RESTART_EVERY and
KEEP_BEST reset per call, which silently produced a completely different (and much worse)
trajectory when trying to trace 2-cycle seed4.

## 3n D: why it is stuck at 1/8 -- the request is a PLATEAU, and the answer is a fossil

WHAT THE CASE ENCODES.  Edges 0->1 (200), 0->2 (1200), 1->2 (700), output N2.
w(0->1)=200 is BELOW the critical weight 444.5, so N1 cannot fire from a single input
spike -- it accumulates and fires EXACTLY ONCE, at 246.  w(0->2)=1200 is well above
critical, so N0 drives N2 directly every 100 steps (33,133,233,333,433).  N1's single
spike produces ONE extra output spike at 293.  That one spike is the entire observable
signature of both hidden weights.

THE REPORTED ANSWER IS A FOSSIL.  seed2 "ends" at w=[194,967,887] with every N0-driven
spike 5 late.  That is not where the optimiser is -- it is a KEEP_BEST memory of an early
iterate.  The LIVE weights end at [3000,1230,340], pinned against the clip.  Reading the
final w without the trajectory is misleading on this case.

THE GRADIENT ON THE DIRECT EDGE IS PERFECT.  Scanning w(0->2) at the stuck point: error
falls monotonically 12.5 -> 0.0 at exactly 1200, g(0->2) is positive below, EXACTLY zero
at 1200-1250, negative above.  Textbook.  It is simply never followed.

WHAT SWAMPS IT.  At seed2's start,
    g = [ 5.54e-04,  3.21e-08, -6.91e-09 ]      <- g(0->1) is 17000x g(0->2)
and the cause is the creation request.  Decomposing the hidden neuron's signal:
    REQ_GAIN=3 : L[1] has 276 nonzero, 275 POSITIVE, a FLAT +1.26e-01 plateau over
                 [0..453];  sum|L[1]| = 26.7   vs   sum|L[2]| = 0.020   (1300x)
    REQ_GAIN=0 : L[1] has ONE nonzero, at 453, NEGATIVE;  g(0->1) = -4.42e-06
So the request is literally "fire somewhere in [0,453]" -- and a broad demand is satisfied
by firing EVERYWHERE.  w(0->1) runs 152 -> 3000 (clip) and stays, its gradient positive
forever and never reversing; N1 becomes a per-cycle firer (24,124,224,324,424) instead of
a single accumulator; the case's structure is destroyed and g(0->2), g(1->2) then go to
EXACTLY zero from ~it600 on.

IT IS A DILEMMA, NOT A ONE-SIDED BUG.  Turning the request off does not fix it, it inverts
it: with REQ_GAIN=0, N1 goes SILENT (w(0->1) 152 -> 108), the 293 spike is never created,
and the gradient is identically zero everywhere by it200 -- a permanent stall at 5/6
spikes.  (Note w(0->2) does then reach 1228, essentially the true 1200: the direct edge
solves perfectly once the request stops fighting it.)  Aggregate 3n D is 1/8 either way,
which is why REQ_GAIN looked irrelevant until the trajectories were compared.

SECONDARY: THE TRUST REGION IS BLIND HERE.  At the stuck point the predicted voltage
change dv at EVERY actual N2 spike is exactly 0, and the raw slope there is 0 (floored),
because eligibility is evaluated at ACTUAL spike times while epochs are anchored at TARGET
times -- N2 fires at 38 but its epoch is (33,133] and the driving N0 spike is at 1, in the
previous epoch.  So the trust region sees only N1's single spike, whose slow accumulator
crossing has a tiny slope (2.1e-05) and therefore a predicted shift of 16.33, and scales
the WHOLE step by 0.12 to protect a spike that barely matters.  EPOCH_EXTEND does not
catch this: it tests drive at the TARGET times, where it is 5.64e-03, comfortably above
the floor.  This throttle alone is not fatal (200 iterations would still cover 967->1200)
but it is the same epoch-boundary confusion in a second place.

## Multi-start over SHARP_GAIN: 29/48 -> 36/48

SHARP_GAIN collapses the request plateau to ONE time.  With REFRAC_MASK=1 it takes 3n D
1/8 -> 3/8 and 3-cycle 1/8 -> 3/8 -- both of the cases that would NOT move for twice the
rounds -- while costing over-demand 7->5 and 2-cycle 8->7.  Net 29 -> 30.  Verified real,
not a budget artifact: 1600 and 3200 rounds give bit-identical results for both settings.

The important part is that the two settings fail on DISJOINT seeds:
    chain    sharp0 {0,2,3,5,6}          sharp1 {1,4,5,6,7}        union 8/8
    3-cycle  sharp0 {5}                  sharp1 {4,5,6}            union 3
    3n D     sharp0 {6}                  sharp1 {2,3,6}            union 3
Running BOTH and keeping the better run gives 36/48, six above either alone.  Selection is
legitimate -- it scores only mean |t_found - t_target| on the OUTPUT neurons, computable
from the given targets, the same quantity KEEP_BEST already uses; the true weights are
never consulted.  The selector is also PERFECT here: it realises the full union on every
case, with no mis-picks.  Cost is 2x compute.  This is a workaround exploiting the
complementarity, NOT a fix for the plateau.  (_suite_multi.py)

CONFIG PLUMBING TRAP.  SHARP_GAIN cannot be varied per-job via env vars in a process pool:
workers are REUSED, and grad_trace reads its config into module globals at import time, so
the second job to land on a worker finds the module already in sys.modules and silently
keeps the FIRST job's setting.  This showed up as both columns being identical, with
sharp1 scoring 29 instead of its true 30.  Set the module attribute directly.  (The
earlier one-config-per-pool runs are unaffected -- there the env var was set before the
pool was created, so every worker imported with the right value.)

## WHY sharpening hurt: it is TWO changes, and only one of them is good (29 -> 34)

Sharpening does two independent things.  Splitting them (new flag SHARP_FLIP, default 1 =
old behaviour) shows they pull in opposite directions:

  (a) COLLAPSE the request R[n] to a single time tau.
  (b) SIGN-FLIP: at every actual spike q with |q - tau| > SHARP_WIN, force the demand's
      sign -- "later than tau => needs MORE drive, earlier => less" (grad_trace.py:702).

                          chain 3-cyc 2-cyc over-dem 3n A 3n D  total
    SHARP_GAIN=0            5     1     8      7      7    1     29
    collapse + flip (old)   5     3     7      5      7    3     30
    collapse, NO flip       8     4     8      6      7    1     34   <-- best single
Stable: 1600 and 3200 rounds are identical.  Stationarity at the true weights is preserved
by both variants (max|g| = 0.000e+00 on all six cases).

(b) IS THE HARMFUL HALF.  Its sign rule silently assumes the neuron should fire ONCE, at
tau: it treats every spike further than SHARP_WIN from tau as a mistimed copy to be dragged
toward tau.  That is true of exactly one case in the suite -- 3n D, whose N1 genuinely
fires once (246) -- and it buys 3n D +2.  Everywhere else it costs: chain -3, 3-cycle -1,
over-demand -1.  Net -4 against (a) alone.

MEASURED ON over-demand seed2, where the true N1 fires TWICE (173, 373):
    SHARP_GAIN=0   N1 -> [173,373]  w -> [250,689,300]  vs true [250,700,300]   EXACT
    SHARP_GAIN=1   N1 -> [61,161,261,361,461]  w -> [555,376,20]                MISS
With tau ~ 173, the SECOND AND CORRECT spike at 373 is "later than tau", so its demand is
forced positive -- N1's input weight is driven 190 -> 555, past the critical 444.5, and the
neuron flips from a sub-critical accumulator into a per-cycle firer.  w(0->2) collapses to
the lower clip of 20.  So sharpening inflicts on over-demand precisely the pathology it was
introduced to cure in 3n D.

A TEMPTING BUT WRONG PREDICTOR: "sharpening helps iff the hidden neurons fire rarely."  It
fits the extremes (3n D fires 1x, +2; over-demand fires 2x, -2) but FAILS in the middle --
3-cycle and 2-cycle have IDENTICAL hidden structure (N1 5 spikes, N2 4 spikes) yet
sharpening gives +2 and -1 respectively.  Spike count alone does not predict it; the
flip/collapse split does.

## Multi-start now 39/48

Three starts (sharp0, collapse, collapse+flip) selected on observable output error:
    chain 8/8   3-cycle 5/8   2-cycle 8/8   over-demand 8/8   3n A 7/8   3n D 3/8  = 39/48
against 29 / 34 / 30 for the three individually.  over-demand reaches 8/8 only as a union
(sharp0 {0,1,2,3,4,6,7} + collapse {0,1,4,5,6,7}), and 3n D still needs the flip variant.
Session progression: 26 -> 27 (EPOCH_EXTEND) -> 29 (adequate rounds) -> 34 (collapse
without flip) -> 39 (3-way multi-start).

## 3n D: two plausible fixes for the sign-flip, both NEGATIVE

3n D needs the flip and nothing else supplies it.  Measured on seed2 (start [152,958,920],
true [200,1200,700]):
    SHARP_FLIP=1  N1 -> [251] (one spike, true 246), w -> [195,1201,811], EXACT by it200
    SHARP_FLIP=0  N1 -> [141,341] (TWO spikes), w(1->2) driven to the clip at 20, MISS
Without the flip nothing exerts downward pressure on the spurious second N1 spike: the
collapse names tau but only pulls, it never pushes.  With the flip, the spike below tau is
suppressed and the one above boosted, and they merge to a single spike.  So the flip is
doing the right thing for a fire-once neuron; its ONLY defect is assuming there is exactly
one tau.  Two attempts to remove that assumption:

(1) SHARP_MULTI -- let the request name a SET of times, flipping each spike against its
    NEAREST tau.  NO EFFECT WHATSOEVER: bit-identical results at both flip settings.
    Instrumented (SHARP_DEBUG=1) to find out why, and the candidate pool is the limit:
        3n D       seed2  candidates [255]        -> taus [255]   (correct, fires once)
        over-demand seed2  candidates [162]        -> taus [162]   ONLY ONE, though N1
                                                                   truly needs 173 AND 373
        over-demand seed3  candidates [155, 377]   -> taus [155]   second dropped by
                                                                   SHARP_FLOOR
    The root cause is that R[d] lists only UNMET demands.  When one of a neuron's several
    jobs is already satisfied, the request names only the OTHER time -- so no amount of
    multi-tau machinery can recover the missing one.  It is absent from the input.

(2) SHARP_PROTECT -- never flip a spike that materially drives an ON-TARGET output spike
    ("do not drag a spike that is already doing useful work").  WORSE, 30 -> 29.  With the
    on-target test at SHARP_WIN=30 it protects 3n D's HARMFUL spikes too (they still land
    within 30 of an output spike and supply >20% of its drive), costing 3n D 3 -> 1.
    Tightening to PROTECT_TOL=1..3 restores 3n D to 3/8 but over-demand STAYS at 5/8 --
    the guard buys nothing.  Tracing over-demand seed2 shows the trajectory is BIT-IDENTICAL
    to PROTECT=0: by it200 the output is already down to 2 spikes against 3 targets, so
    NOTHING is on target and nothing is ever protected.  The derailment happens before the
    guard can act.  It is a "don't break what works" rule applied to a run that is already
    broken.

CONCLUSION.  The flip's fire-once assumption cannot be repaired from the request, because
the request does not contain the information (it only lists what is missing, never what is
already correct).  Distinguishing "should fire once" from "should fire several times"
needs a signal this method does not currently have.  The multi-start keeps the benefit
without resolving it: 3n D is 3/8 in the collapse+flip start and 1/8 elsewhere.

Both flags are left in, defaulting OFF (SHARP_MULTI=0, SHARP_PROTECT=0), so defaults still
give 34/48 and stationarity at truth is still exactly 0.000e+00 on all six cases.

## The gradient on N1 (3n D's sub-critical accumulator), situation by situation

N1 has ONE incoming synapse (N0->N1), so g(0->1) = dot(L[1], eps[(0,1)]) is the whole
story, and L[1] is the downstream request (the deficit at N2's missing 293 spike propagated
back through w(1->2)) gated by N1's near-threshold sensitivity.  Evaluated at representative
weight points; "->" is the sign Adam would move w(0->1):

  TRUE  w=[200,1200,700]  N1@246           g=0, L[1] EMPTY.  Nothing to do. Correct.

  seed6 w=[204,1191,655]  N1@242, out 1 early
        g(0->1)=-6.3e-8, one small NEGATIVE demand at 242 -> lower w -> N1 later -> 292->293.
        Correct and gentle.  Identical with/without flip (242 is within SHARP_WIN of tau).

  seed2 w=[159,1139,738]  N1 LATE (@437, no 293 at all)
        At this ENDPOINT L[1] is a single POSITIVE lump at the REQUESTED creation latency
        (~248), NOT at N1's actual late spike -> "create drive near 248" by raising w.  The
        method is BLIND to the misplaced spike at 437: there is no demand there to correct.

  seed0 w=[243,940,379]  WEAK (N1@223 fires, but w(1->2)=379 sub-critical so NO output)
        The edge that must grow is w(1->2), and g(1->2)=0.  WHY: L[2] at target 293 is a
        full hinge deficit +7.0e-3, but eps[(1->2)][293]=0 -- N1@223 is one cycle EARLY, so
        it sits in the epoch BEFORE target 233 and is MASKED OUT of the 293 epoch.  Its PSP
        at 293 (unmasked w12*h(70)) never enters the gradient.  So the direct fix is
        invisible, and only the request path acts, pushing w(0->1) UP -- the WRONG edge.
        This is the epoch-boundary masking again, now hiding a weight increase.

  seed4 w=[274,1199,1019]  TWICE (N1@151,351 -> spurious outputs 186,386)
        flip=0: g=-1.35e-6, dominated by a NEGATIVE term at the 351 spike -> lower w ->
                suppress the double firing.  CORRECT.
        flip=1: the 351 term is flipped POSITIVE -> g=+2.69e-6 -> raise w -> MORE firing.
                WRONG.

WHERE THE FLIP ACTUALLY BITES (seed2 at the START, not the endpoint):
        start w=[152,958,920], N1@453 (very late)
        flip=0: L[1]@453 = -0.105  -> g(0->1)=-2.2e-6  -> lower w -> N1 later/silent (WRONG,
                N1 must come EARLIER to 246)
        flip=1: L[1]@453 = +0.105  -> g(0->1)=+6.7e-6  -> raise w -> pull N1 earlier (RIGHT)
        The flip only differs where N1's ACTUAL spike carries a demand; at the no-flip
        endpoint that spike had none, which is why the endpoint decomposed identically.

UNIFYING RULE.  The flip's "later than tau => needs more drive" is correct precisely when
the neuron should fire ONCE and is merely late (seed2 -> pull the late spike earlier), and
exactly backwards when the late spike is a spurious SECOND spike that should be removed
(seed4 -> it forces creation instead of suppression).  Same fire-once assumption, seen now
at the level of individual spikes.  This is why flip nets +2 on 3n D: it rescues the
late-single-spike seeds (2,3) and mis-signs the double-spike seeds (4,5), which fail
anyway.  The weak seeds (0,1,7) are untouched by flip because their blocker is the
epoch-masked g(1->2)=0, not N1's timing.

## Is there a principled direction?  The gradient is fine; the OBSTACLE IS THE JUMP

Two hypotheses tested and BOTH REJECTED:

(H1) "eligibility omits the derivative through its own reset times, and that missing term
     is the sign information."  eligibility() truncates at resets (e[s:hi], hi = next
     reset+1) and treats them as constants, so the term sum_s dV(t)/ds * ds/dw is absent.
     TESTED against central finite differences of the REAL simulator, hidden neurons only
     (output eps is the derivative of a COUNTERFACTUAL target-reset V_sub by design, so it
     is not supposed to match):
         smooth points (spike train unchanged): 1087 probes, MEDIAN RELATIVE ERROR 1.2e-04,
             max |fd - eps| = 4.6e-06  (vs th = 7e-3, i.e. 0.07% of threshold)
     The eligibility is ACCURATE.  There is no missing term worth having.
     (Two bugs in my own first version of this test made it look like half the gradients
     were wrong: I compared OUTPUT probes, where the mismatch is deliberate, and my hidden
     probe set was every hidden spike time -- all of which the near-spike filter removed,
     leaving ZERO hidden probes while still printing a confident total.)

(H2) "the epoch masking hides a needed weight increase on 3n D seed0."  Also FALSE.
     Finite difference at that point gives dV2(293)/dw(1->2) = 0.000e+00, exactly matching
     eps.  The code is right.

WHAT IS ACTUALLY THERE.  On seed0 (w=[243,940,379]) the edge w(1->2) is EXACTLY INERT:
scanned over its ENTIRE range 20..3000 the output train never changes, not once.  Reason,
and it is structural: N1@223 arrives at 223+18=241, N2 fires at 238 and is refractory
through 238+22=260, so 241 lies inside the shadow and the simulator discards the input at
every weight.  The TRUE solution clears it -- N1@246 arrives 264, past N2@233's shadow
[233,255].  And the escape is irreducibly JOINT: over a grid, w(1->2) stuck at 379 gives
0 exact hits for any (w01,w02), and w(0->2) stuck at 940 gives 0 exact hits for any
(w01,w12).  No single-weight move works either.

SO: all the large fd-vs-eps disagreements sit exactly where a perturbation CHANGES THE
SPIKE TRAIN (max 3.5e-03, half a threshold -- a jump, not a slope).  The difficulty is not
the sign convention, not matching, and not a missing derivative.  It is that the needed
move crosses a DISCONTINUITY across which the derivative is genuinely zero or undefined.

WHAT IS PRINCIPLED AND COMPUTABLE.  Not "is this spike good or bad" (a sign on a spike),
but WHERE THE NEAREST DISCONTINUITY IS AND WHICH CONSTRAINT BINDS.  Both are exact and
local, needing no resimulation and no matching:
  * OCCLUSION.  Edge k->n is inert for presynaptic spike q iff q + DELAY_ITERS lies in
    [s, s+REFRAC_ITERS] for some own-spike s of n.  This says the edge is DEAD, which side
    of the shadow the arrival sits on, and hence WHICH WAY the occluder or the arrival must
    move to revive it.  The direction comes from the geometry, not from a create/suppress
    dichotomy -- which is why it gives EITHER sign, as required.
  * MARGIN.  A spike exists at t iff V(t) >= th; the margin th - V(t) is smooth in w and
    measures the DISTANCE TO THE JUMP even where the jump has not happened, so it is
    informative exactly where the derivative is zero.
This covers both awkward cases with one mechanism: "suppression must move a spike earlier
to get it out of the way" is a refractory shadow whose edge must clear, and "creation must
weaken a connection so an earlier spike does not interfere" is the same constraint read
from the other side.

## OCCLUSION-AWARE DEMAND: 34 -> 38 single setting, 39 -> 43 multi-start

Implements the principle from the previous section: the obstacle is a DISCONTINUITY, and
what is exactly computable is the window an arrival must land in to serve a demand.

    q can serve a demand at t on d  <=>  q + DELAY_ITERS  in  ( last_reset_of_d_before(t), t ]
                                         and clear of d's refractory shadows

Two uses, both local, no matching and no resimulation:

  OCCL_MASK (default 1) -- the request may only ASK at feasible times, and the backward
    message's POSITIVE part is zeroed at infeasible times (its NEGATIVE part is KEPT: a
    spike that can serve nothing SHOULD be discouraged, and that is the signal that moves
    it out).  The forward pass already masks occluded spikes via REFRAC_MASK; the backward
    message did not, which was the actual asymmetry.
  OCCL_GAIN (default 1.0) -- when NOTHING can currently serve a demand, push each
    presynaptic spike TOWARD the window: below it => negative demand (fire later), above
    it => positive (fire earlier).  Direction from geometry, so it yields either sign.

                       chain 3-cyc 2-cyc over-dem 3n A 3n D  total
    before                8     4     8      6      7    1     34
    OCCL_MASK only        7     5     8      7      7    3     37
    + OCCL_GAIN           8     6     8      7      6    3     38
Identical at 1600 and 3200 rounds.  Stationarity at truth preserved: max|g| = 0.000e+00 on
all six.  3n D reaches 3/8 WITHOUT the sign-flip hack -- a principled mechanism doing what
SHARP_FLIP was faking.

THE DIAGNOSIS THAT DROVE IT (3n D seed0, w=[243,940,379]): the request sat at t=183, which
arrives at 201, BEFORE N2's reset at 238 -- its PSP is wiped at the epoch boundary, so a
spike there cannot serve target 293 for ANY weight.  Yet it supplied the DOMINANT term in
g(0->1) (+3.57e-06 at 183 vs +4.79e-07 at N1's real spike) and pushed w(0->1) the wrong
way.  The feasible window there is q in 238..274; the true N1 is at 246.

INTERACTION FOUND: the feasibility test must use the PHYSICAL resets (targets for outputs,
own spikes for hidden), NOT rst[].  EPOCH_EXTEND deliberately widens an epoch exactly when
nothing in it can reach threshold -- i.e. the occlusion case -- and that widening re-admits
the very times the mask exists to reject (window came out 227..274 instead of 238..274).

LIMIT, honestly: after masking, g(0->1) on seed0 becomes exactly 0 rather than correct.
The request moves to 238, which is ALSO unreachable -- N1 already fired at 223 (resetting)
and N0's next spike is at 301.  The occlusion structure depends on the weights that are
being solved for, so feasibility is circular, and seed0 needs a joint move that no local
rule supplies.  The gain is real but it comes from the cases where ONE edge is dead and
the rest of the configuration is sound.

MULTI-START now 43/48 (was 39): sharp0 30, collapse 38, collapse+flip 30, no-occl 34.
chain 8/8, 3-cycle 7/8, 2-cycle 8/8, over-demand 8/8, 3n A 7/8, 3n D 5/8.  Keeping a
"no-occl" start pays: 3n A is 7/8 only there.

CONFIG TRAP, AGAIN.  Every entry in STARTS must specify EVERY key any other entry varies.
Workers are reused and the config lives in module globals, so an unset key inherits the
previous job's value -- "collapse" scored 35 in the multi-start while scoring 38 standalone
because it was picking up no-occl's OCCL_MASK=0.  Same failure mode as the earlier env-var
version, one level up.

## Three new cases: 3n E / 4n F / 4n G  (grad_3nD_variants.py)

3n D's content is a RARE sub-critical accumulator (w below the 444.5 single-spike weight)
whose one spike carries the whole signature of two weights.  These vary that along the two
axes that matter:
    3n E  the accumulator fires TWICE            [[0,1],[0,2],[1,2]]      w [260,1200,950]
    4n F  the rare spike is RELAYED via a 2nd hidden  [[0,1],[1,2],[0,3],[2,3]] w [240,1200,1200,1100]
    4n G  both: two rare spikes, each relayed     same edges                w [250,500,1200,700]

    case   hidden spikes            output                                   created
    3n D   N1 [246]                 [33,133,233,293,333,433]                 [293]
    3n E   N1 [160,360]             [33,133,197,233,333,397,433]             [197,397]
    4n F   N1 [224] N2 [256]        [33,133,233,290,333,433]                 [290]
    4n G   N1 [173,373] N2[244,444] [33,133,233,291,333,433,491]             [291,491]

WELL-POSEDNESS, all four checked:
  * the accumulator fires the intended number of times, not once per input cycle;
  * every hidden spike CREATES an output spike rather than shifting one.  This killed the
    first several dozen candidates: a hidden spike whose arrival lands in the output's
    refractory shadow merely drags an existing spike earlier (333 -> 328) and tests
    nothing.  Verified by severing the hidden edge and diffing the trains.
  * the target is NOT reproducible with the hidden path silent for ANY direct
    input->output weight over 20..3000 -- so this is not a weight degeneracy.
  (Two bugs on the way: my first search conflated "hidden spike counts" with "expected
  extras" in one list so nothing ever matched, and my first F pick produced a SHIFT not a
  creation.  Both caught by making the ORIGINAL 3n D pass the same criterion first.)

RESULTS at 1600 rounds, current defaults:  3n E 6/8,  4n F 2/8,  4n G 0/8.
Suite is now 46/72.  Multi-start 56/72 (sharp0 37, collapse 46, collapse+flip 43,
no-occl 35).

WHAT THEY SHOW:
  * 3n E (fires twice) is NOT harder than 3n D -- 6/8 vs 3/8, BETTER.  Two marks give the
    method twice the evidence, and the fire-once pathologies matter less than expected.
    Note 3n E is 0/8 in the no-occl start and 6/8 in every other: it depends ENTIRELY on
    the occlusion work.
  * 4n F (relayed) is the case where the sign-flip finally earns its keep: 7/8 with
    collapse+flip against 1-2/8 everywhere else.  A demand crossing two backward hops is
    what SHARP_FLIP was actually good for.
  * 4n G (both) is 0/8 in EVERY configuration -- the first case in the suite no setting
    touches.  Failure is uniform: COUNT 5 vs 7, the two marks never created.  Two distinct
    mechanisms, both visible in the dump: on some seeds the relay w(1->2) lands below the
    444.5 critical weight so N2 is SILENT and the path is dead (seeds 2,3: N2=[]); on the
    others N2 fires but the marks still fail to appear.  Relay + rarity compound.

## EARLY_STOP: halt on a zero-gradient exact match

train() now breaks when the output matches every target AND max|g| == 0 -- a genuine fixed
point, so no later iterate can differ.  Tested against the spall/V already computed that
iteration, so it costs no extra simulation.  Results are BIT-IDENTICAL per case and per
seed (verified by diff over the full 72-cell suite); wall clock 78s -> 50s, CPU 895s ->
564s (1.6x).  It also removes the drift KEEP_BEST exists to undo.

## 4n G: the increment-vs-total clamp bug  (46 -> 51/72)

4n G was 0/8 in every configuration.  Diagnosis, seed7 (w=[269,699,1519,508], N1=[154,354],
N2=[201,401], output missing its two marks):
  * g(2->3) = 0 on ALL EIGHT seeds -- and CORRECTLY so.  N2@201's PSP is reset away at the
    233 epoch boundary, so w(2->3) genuinely cannot affect V(291).  The fix must MOVE N2's
    spikes, not reweight that edge.
  * g(1->2) = 0 on 7 of 8 seeds, and L[2] was entirely EMPTY on seeds 5,6,7 -- the demand
    died before reaching the relay.

The occlusion machinery was firing correctly: for target 291 it computed the q-window
205..272, saw N2@201 below it, and wrote L[2][201] = -0.028 ("move this spike later").
Then the backward relaxation added Ln[201] = +6.84e-03 at that SAME infeasible time on
EVERY ONE of its 6 sweeps.  Traced step by step:
    -2.80e-02 -> -2.12e-02 -> -1.43e-02 -> -7.48e-03 -> -6.42e-04 -> +6.20e-03 -> clamped 0
The occlusion term is written ONCE before the sweeps; the message re-injects its positive
six times, outvotes it, and the clamp -- which only removes positives -- then zeroed the
lot, leaving g(1->2) = 0.

FIX: clamp the INCREMENT, not the running total.
    if OCCL_MASK and n in _occl_ok:
        Ln[bad] = np.minimum(Ln[bad], 0.0)      # was: L[n][bad] = np.minimum(...)
    L[n] = L[n] + Ln
The accumulated negative now survives and g(1->2) = -2.85e-07 (weaken -> N2 fires later).

                  chain 3-cyc 2-cyc over-dem 3n A 3n D 3n E 4n F 4n G  total
    before          8     6     8      7      6    3    6    2    0     46
    after           6     6     8      7      6    7    7    3    1     51
3n D 3 -> 7 is the headline: the case that resisted everything all session.  Cost is
chain 8 -> 6.  At 3200 rounds the total is 56/72, so this is budget-limited again, not
converged.

OPEN: 3n E is the FIRST case in the suite where truth is NOT a stationary point --
max|g| = 4.95e-06 at the true weights, all of it on w(0->1), the accumulator weight.
This is NOT caused by the occlusion work: it is identical (4.95e-06 / 5.33e-06) across all
four OCCL_MASK x OCCL_GAIN settings including both off.  It is a property of the case --
a sub-critical accumulator firing TWICE -- and it means the optimiser will drift off the
correct answer there.  Worth chasing: it is the cleanest known violation of the stationarity
property the whole method depends on.

## ROOT CAUSE of the 3n E stationarity break: the refractory window was off by one AT BOTH ENDS

At the TRUE weights 3n E reproduces its target exactly, yet g = [4.95e-06, 0, 0].  Tracing
it: vsub[2] reads 0.0000e+00 at targets 233 and 433, while the SIMULATOR has
V(233) = 7.056840e-03 -- byte-identical to the unobstructed V(33).  A perfectly placed
target was being read as completely undriven, fabricating a full-deficit hinge of +7e-3
that propagated to the accumulator weight.

WHY.  N2 fires at 197; its next input N0@201 arrives at 201+18 = 219 = 197 + REFRAC_ITERS.
The mask discarded it because it used
        r <= arrival <= r + REFRAC_ITERS         [r, r+22], 23 steps
but jax_spiking_model.py:61 sets the timer to refractory_iters+1 on firing and decrements
immediately, so it runs 22,21,...,1 over steps r..r+21 and is 0 at r+22; and `out` at step
r itself is gated by the OLD timer, still 0.  The true discarded window is therefore
        r < arrival < r + REFRAC_ITERS           (r, r+22), i.e. r+1..r+21
The mask was one step too wide at BOTH ends.  Fixed at all three sites that encode it
(eligibility, the occlusion _live test, the occlusion mask's shadow removal).

    stationarity at truth, max|g|:  3n E 4.95e-06 -> 0.000e+00
    ALL NINE cases now exactly 0.000e+00
    suite 51 -> 54/72 at 1600 rounds;  56 -> 58/72 at 3200
    3n E 7 -> 8/8,  4n F 4 -> 8/8 (at 3200),  chain 6 -> 7/8

This is the same class of bug as the epoch-boundary one, and worth stating plainly: THREE
separate off-by-one/boundary errors in the reset-and-refractory bookkeeping have each cost
several cases (EPOCH_EXTEND, the increment-vs-total clamp, and now this).  The forward
simulator is the authority; anything that re-derives its masking by hand must be checked
against it numerically, not by reading.

## What happened to chain: OCCL_MASK, one seed, a sub-critical trap

chain was 8/8 before the occlusion work, 6/8 after the increment-clamp fix, 7/8 after the
refractory fix.  Isolated at 3200 rounds:
    OCCL_MASK=1 OCCL_GAIN=1   7/8      OCCL_MASK=0 OCCL_GAIN=1   8/8
    OCCL_MASK=1 OCCL_GAIN=0   7/8      OCCL_MASK=0 OCCL_GAIN=0   8/8
So it is OCCL_MASK, not the move-demand, and it costs exactly ONE seed (seed2).

seed2 parks at w = [456, 415, 626] with w(1->2) = 415 just BELOW the critical single-spike
weight 444.5, so N2 fires only twice (accumulating) instead of once per N1 spike, and the
output has 2 spikes against 4 targets.  It oscillates in 409..420 for 1400 iterations.
At that point g(1->2) is NEGATIVE in BOTH configurations (-4.1e-07 masked, -3.7e-07
unmasked), i.e. pushing the relay weight further DOWN and away from the critical value it
must cross -- so the mask is not uniquely responsible for the sign at the trap, it changes
which trajectory arrives there.  The trap itself is the sub-critical barrier: below 444.5
the relay under-fires, and the demand generated by an under-firing relay does not push it
back up over the barrier.

Net: OCCL_MASK costs chain 1 seed and is worth far more elsewhere (it is what takes 3n D,
3n E, 4n F and 4n G off the floor), so it stays on.  The sub-critical barrier is the
remaining structural issue, and it is the SAME phenomenon as 4n G's dead relay -- a weight
that must cross 444.5 to change the spike COUNT, with no gradient pointing across.

## CORRECTION: it is not a "sub-critical barrier", it is a COMPENSATION TRAP (2 weights)

I claimed the remaining failure was "a weight that must cross 444.5 to change the spike
COUNT, with no gradient pointing across".  That is WRONG, tested three ways:

  * 2n  N0->N1 (barrier on an OBSERVED neuron): 8/8, including random starts at 381 and
    293 which climb back to ~500.
  * 3n  N0->N1->N2, barrier one hop from the output: every sub-critical seed recovers.
  * FORCED sub-critical values, not left to chance -- w(1->2) started at 120/200/280/360/420
    with the others at truth, at BOTH depths (out one hop away, and out two hops away):
    10/10 RECOVER.  From as low as 120.
A single sub-critical weight is not a trap at any depth.  The gradient crosses the barrier.

WHAT chain seed2 ACTUALLY IS.  It parks at [456, 415, 626].  Restarting from there and from
each single-weight repair:
    [456, 415, 626]  as-is                    -> STUCK   out [263,463] vs 4 targets
    [500, 415, 626]  w(0->1) restored         -> STUCK   out [254,454]
    [456, 500, 626]  w(1->2) restored         -> OK
    [456, 415, 500]  w(2->3) restored         -> OK
    [500, 415, 500]  ONLY w(1->2) sub-crit    -> OK
So the trap requires a PAIR: w(1->2) sub-critical (415) AND w(2->3) over-strong (626).
Either one alone is harmless; together they stick.  The mechanism is COMPENSATION -- the
downstream weight has grown to partly cover for the under-firing upstream, N2 fires twice
instead of four times but the inflated w(2->3) still drives N3, so the residual output
error is too small to demand more from N2, and the thing that would fix the upstream is
exactly what the compensation has removed.

This also explains why every single-weight probe recovered: there is no compensator, so the
demand stays pointed at the one wrong weight.

SIMPLEST KNOWN INSTANCE: the 4-neuron chain N0->N1->N2->N3, started at w = [456, 415, 626]
(or anything with w(1->2) below 444.5 and w(2->3) well above 500).  It is deterministic and
needs no seed hunting.  Whether a 3-neuron version exists is untested.

This is a different and harder target than a barrier: the failure is not a missing gradient
across a discontinuity but a REAL local minimum in a 2-weight subspace, reached because one
weight absorbed the error of another.  Note the earlier finding that 4n G's escape is
irreducibly joint is the same shape.

## Why the "tell N2 to fire earlier" chain never happens in the trap

Traced hop by hop at w=[456,415,626] (N1 91,191,291,391,491 / N2 212,412 / N3 264,464 vs
target 214,314,414,514):

1. THE MATCHING IS CORRECT.  264 is exactly 50 from BOTH 214 and 314 (a tie), and the
   closest-pair matching picks 264<-214 and 464<-414, i.e. "must fire EARLIER" -- the right
   side of the tie.  Unmatched targets 314 and 514.

2. THAT INFORMATION IS THEN DISCARDED.  L[3] is nonzero ONLY at the four TARGET times
   (214,314,414,514), every one a full-deficit creation hinge of +7.0e-3.  There is NO
   entry at 264 or 464 -- N3's own spikes carry ZERO demand.  The suppression term is
   `frac = 0.0 if d <= MATCH_WIN else 1.0` with d=50 and MATCH_WIN=60, so a spike 50 steps
   late gets exactly nothing at its own location.  Lateness is measured, then used only to
   decide a suppression strength that is zero inside the window.  So what travels backward
   is "CREATE a spike at 214", never "MOVE your spike at 264 earlier".

3. AT N2 THE CREATION DEMANDS CONFLICT AND CANCEL.  N2 has one spike at 212 and is asked to
   serve two targets with it: the window to serve 214 wants q <= 196 (fire EARLIER, +mag)
   while the window to serve 314 wants q in 218..296 (fire LATER, -mag).  Net L[2][212] =
   -2.4e-02 -- pushing it LATER, the wrong way, while true N2 is at 143 (69 steps EARLIER).

4. AND EVEN A PERFECT TIMING CHAIN COULD NOT FIX THIS.  The deficit is in COUNT: N2 fires
   2x and must fire 4x.  Moving two spikes earlier still leaves two spikes.  The timing
   chain the intuition describes is necessary but not sufficient here.

    g = [-2.74e-06, -4.12e-07, 0.0]   w0 needs UP gets DOWN, w1 needs UP gets DOWN,
                                      w2 needs DOWN gets NOTHING.  All three wrong.

ACTIONABLE: (2) is a real gap independent of this case -- an output spike matched to a
target within MATCH_WIN=60 produces no demand at its own time whatsoever, so "you are 50
late" is never expressed as a timing demand.  GRADE_SUPP=1 grades it but only as
SUPPRESSION (push down), which is not the same as "shift earlier".

## MOVE_GAIN: a signed timing demand at matched-but-mistimed output spikes (58 -> 60/72)

Fills the gap found above: a spike matched within MATCH_WIN carried NO demand at its own
time, so "you are 50 late" was measured and dropped.  Added in the output seeding, beside
the suppression term:

    off = t - claimed[t]                      # >0 late, <0 early
    amt = min(1, max(0, |off| - DEAD_ZONE) / GRADE_SCALE)
    L[o][t] += MOVE_GAIN * sign(off) * amt * TH

More drive => fires sooner, so LATE gets a POSITIVE demand and EARLY a NEGATIVE one (the
SHARP_FLIP convention).  Graded from DEAD_ZONE, so it vanishes at perfect timing.

    MOVE_GAIN   0     0.25   0.5    1.0
    total/72    58    60     60     51
Stationarity at truth preserved exactly (0.000e+00 on all nine) at 0.25 and 0.5 -- the
grading is what guarantees it.  Default set to 0.25, which keeps 3n A, 3n E and 4n F all at
8/8; 0.5 trades those for 3n D 8/8.  At 1.0 the timing term swamps the creation hinge.

    per case at 0.25:  chain 7  3-cyc 7  2-cyc 8  over-dem 5  3n A 8  3n D 6  3n E 8
                       4n F 8  4n G 3     = 60/72
3n A 6->8, 3-cycle 6->7 and 4n G 2->3 are the gains; over-demand 7->5 is the cost.

IT DOES NOT FIX THE TRAP IT WAS DESIGNED FOR, and the reason is worth recording.  The
demand IS now written (L[3][264] = +7.0e-3), but g(2->3) stays exactly 0 because
eps[(2,3)][264] = 0: N2's spike at 212 falls BEFORE the target-anchored reset at 214, so
its PSP is truncated out of the epoch containing 264.  The demand lands where the
eligibility is zero.  This is the counterfactual-epoch mismatch again -- demands are placed
at ACTUAL spike times while eligibility is built on TARGET-anchored epochs, and the two
disagree exactly when a spike is far enough from its target to sit in a different epoch,
which is precisely when the timing demand is most needed.  Fixing the trap needs
eligibility on the ACTUAL reset structure for this term, which is a deeper change.

## Dual eligibility for the timing term (MOVE_ACT): IMPLEMENTED, MEASURED, NEGATIVE

The diagnosis was right and the machinery does exactly what it was meant to.  A second
eligibility built on the ACTUAL resets, eps_act = eligibility(spall[k], T, spall[n],
refrac_at=spall[n]), is computed for output neurons; the MOVE term is accumulated into a
separate Lmove and the output-edge gradient becomes
      dot(L - Lmove, eps)  +  dot(Lmove, eps_act)
so the creation hinge keeps the counterfactual target-anchored epochs (which is what makes
truth a fixed point) while the timing demand is scored where the spike is actually caused.

It works as designed.  On the chain trap:
      eps    [(2,3)] nonzero count 0        <- counterfactual: N2@212 arrives 230, inside
                                               the refractory shadow of the TARGET spike
                                               at 214, so it is discarded entirely
      eps_act[(2,3)] nonzero count 68, and eps_act[(2,3)][264] = 1.1228e-05 = HK[52] EXACTLY
      g(2->3):  0.0  ->  +3.93e-08
The demand now reaches the weight.  Stationarity at truth is still exactly 0.000e+00 on all
nine cases.

BUT IT IS WORSE, AND NOT MARGINALLY:
      MOVE_ACT=0 (counterfactual eps, MOVE_GAIN 0.25)   60/72
      MOVE_ACT=1  MOVE_GAIN 0.25                        49/72   chain 7->4, 3-cyc 7->4,
                                                                3n A 8->2
      MOVE_ACT=1  MOVE_GAIN 0.10                        56/72
      MOVE_ACT=1  MOVE_GAIN 0.05                        58/72   (-> the MOVE_GAIN=0 baseline)
Lowering the gain only walks back toward doing nothing; there is no setting where the
actual-reset scoring beats the counterfactual one.  Default MOVE_ACT=0.

AND ON THE TRAP ITSELF THE NEW GRADIENT POINTS THE WRONG WAY -- correctly.  g(2->3) is now
POSITIVE, i.e. "raise w2 so the spike at 264 comes earlier", which is a faithful answer to
the demand that was placed.  But the true fix is w2 DOWN (626 -> 500): the deficit is in
COUNT (N2 fires 2x, must fire 4x), not timing.  Moving two spikes earlier still leaves two
spikes.  So the timing chain was never going to fix this case, exactly as flagged before
implementing it; what the experiment settles is that the missing plumbing was NOT the
obstacle.  The obstacle is that a count deficit has no timing expression at all.

## MOVE_COHERE: gate the timing demand on agreement (60 -> 62/72, over-demand restored)

MOVE_GAIN cost over-demand 7/8 -> 5/8.  Both lost seeds (0 and 6) fail identically: the
output collapses to a 2-spike PERIODIC train ([174,374], [166,366]) instead of the
irregular 3-spike target [140,220,399], and the weights barely move from their start --
seed0 goes [284,539,162] -> [281,518,183], where with the term OFF it travels properly to
[250,685,300].  It FREEZES.

WHY.  With 2 output spikes against 3 targets the matching pairs them with widely separated
targets: 174<-140 (34 LATE, wants earlier) and 374<-399 (25 EARLY, wants later).  A weight
change shifts ALL of a neuron's spikes the same way, so those two demands are contradictory
and cancel.  The chain trap is the opposite case: 264<-214 and 464<-414, both +50, both
wanting earlier -- coherent, and it escapes.

So the disagreement is itself the signal that the deficit is a COUNT problem rather than a
timing one, and creation should be left to act alone.  Scale the timing term by

    coherence = | sum over matched spikes of sign(t - claimed[t]) | / count

1.0 when all agree, 0 when evenly split, graded in between.  Only spikes beyond DEAD_ZONE
are counted, so perfect timing is unaffected and truth stays stationary (verified
0.000e+00 on all nine).

                  chain 3-cyc 2-cyc over-dem 3n A 3n D 3n E 4n F 4n G  total
    MOVE_COHERE=0   7     7     8      5      8    6    8    8    3     60
    MOVE_COHERE=1   7     7     8      7      8    6    8    8    3     62
Every other case is IDENTICAL, per seed -- over-demand 5->7 is the whole difference, and
chain seed2 (the compensation trap) still passes.  Default on.

## 4n G: the dead relay is an ABSORBING STATE, and the demand drives it there

Two corrections to what I said before looking at the live trajectory.

1. NOT ATTENUATION ACROSS HOPS.  The backward signal AMPLIFIES: max|L| per layer on the
   failing seeds gives L2/L3 = 4-16x and L1/L2 up to 22x, i.e. 7.0e-03 at the output
   becoming 0.628 at N1, ~90x over two hops.  And 4n F has the IDENTICAL topology and depth
   and is 8/8.  Depth is not the discriminator.

2. NOT THE TRUST REGION.  Scale factors at the failing seeds are 0.70 and 0.33 -- a real
   throttle but nowhere near a freeze, and the implied travel times are 6-100 iterations.

3. THE REPORTED ENDPOINT IS A KEEP_BEST FOSSIL AGAIN (third time this session).  seed5
   RETURNS [301,628,1218,550] but the LIVE weights are [187,440,1232,532]; seed7 returns
   [269,686,1519,508] while live is [185,496,1244,495].  My "the weights barely move"
   reading was taken from the returned value and is wrong -- w(0->1) actually travels
   281 -> 185.  Any analysis of a failing run MUST use the cb hook, not the return value.

WHAT ACTUALLY HAPPENS.  From the live trace, |upd| = 0.000e+00 at EVERY checkpoint from
it400 to it3200 -- the update is exactly zero, not small.  And at that live point seed5 has
N2 = [] : the relay is DEAD.  The sequence is

    the demand correctly says "N2 fires too early, fire later" -> lower w(1->2)
    -> w(1->2) crosses BELOW the critical 444.5 (440 on seed5, 496 then lower on seed7)
    -> N2 stops firing entirely
    -> with no N2 spike there is no eligibility on the 2->3 edge and nothing for the
       demand to attach to, so g == 0 on every edge
    -> FROZEN PERMANENTLY.  Absorbing.

So all five failures are one mechanism.  Seeds 2 and 3 START in the dead-relay state; seeds
0, 5 and 7 WALK INTO it.  The locally-correct timing demand ("fire later" = less drive) is
precisely what pushes w(1->2) through the barrier that annihilates the spike.

This is the same discontinuity family as everything else this session: lowering a weight
past the critical value does not DELAY the spike, it DELETES it, and the local gradient
cannot see the difference.  A guard would need to know that a proposed decrease crosses
w_crit for a neuron whose spike is load-bearing -- which is computable, since w_crit =
th/max(HK) is a constant and the drive decomposition is already available.

## The silent relay: gradient SHOULD flow through it, and two separate things stop it

Checked at 4n G seed5's LIVE frozen point w=[187,440,1232,532], N2 = [] (silent):

  * eps[(1,2)] is FINE -- 174 nonzero points on [346..519].  Eligibility is built from the
    PRESYNAPTIC train, so a silent postsynaptic neuron does NOT zero it, exactly as
    expected.  The dead end is not there.
  * the near-threshold sensitivity gate is NOT the cause either: N2's vsub peaks at
    6.9284e-03 against th = 7.0000e-03, only 1% short, so exp(-((vsub-th)/(SIG*th))^2)
    = 0.9996.  Wide open.

TWO STACKED CAUSES, both of which have to go:

 1. OCCL_MASK ZEROES THE WHOLE DEMAND.  max|L2| = 0.00e+00 with the mask on, 1.20e-01 with
    it off.  The mask judges every time infeasible for the silent neuron and removes the
    positive demand everywhere, so nothing at all reaches N2.  The feasibility test is
    computed from the CURRENT spike structure, and for a neuron that currently does not
    fire, that structure is precisely what has to change -- the same circularity already
    recorded for 3n D seed0, but here it is fatal rather than merely unhelpful.

 2. EVEN UNMASKED, THE DEMAND LANDS WHERE THE PRESYNAPTIC CANNOT ACT.  With OCCL_MASK=0,
    L[2] is nonzero but g(1->2) is STILL exactly 0: L[2]'s support and eps[(1,2)]'s support
    ([346..519]) do not overlap.  The demand asks N2 to fire early enough to mark target
    291, while N1's only spike is at 327 and cannot cause an N2 spike before 346.
    The correct response is to forward the demand to N1 ("fire earlier so N2 can"), and
    that does not happen -- max|L1| = 0.00e+00 in BOTH mask settings.

So the request stops dead at the first neuron that cannot satisfy it locally, instead of
propagating upstream.  That is the mechanism the creation-request design was supposed to
provide ("if the neuron has no drive to amplify at tau, the request PROPAGATES backward to
its presynaptics at tau - latency") and it is not firing here.  That, not the barrier and
not the trust region, is what makes the dead relay absorbing.

## PROP_SILENT: a silent neuron always forwards the request (62 -> 63/72)

The gate that stopped the demand is grad_trace.py:532

    unmet = R[n] * (vsub[n] < CREATE_FLOOR * TH)      # GRADE_PROP=0, CREATE_FLOOR=0.2
    if unmet.max() <= 0: continue

"only the part n cannot satisfy locally is passed on", implemented as "only where n has
almost NO drive".  For 4n G's dead relay that is exactly backwards: N2's vsub peaks at
0.99*th, so the floor concludes N2 can serve the request itself and forwards NOTHING --
while N2 in fact fires not at all, and the 1% it is short of is precisely what its own input
weight has to supply.  A neuron with no spikes has demonstrably NOT satisfied the request,
whatever its vsub says.

GRADE_PROP=1 (the documented graded alternative) DOES fix the propagation -- max|L1| goes
0.00e+00 -> 7.12e-04 -- but applied everywhere it costs 62 -> 52/72 (3-cycle 7->3,
4n F 8->2).  Restricting the graded form to SILENT neurons keeps the gain and none of the
cost:

                  chain 3-cyc 2-cyc over-dem 3n A 3n D 3n E 4n F 4n G  total
    PROP_SILENT=0   7     7     8      7      8    6    8    8    3     62
    PROP_SILENT=1   7     7     8      7      8    6    8    8    4     63
Stationarity at truth preserved (0.000e+00 on all nine): at truth no neuron that owes a
spike is silent, so the branch never fires.

PARTIAL, NOT SOLVED.  The demand now reaches N1, but g(0->1) at the frozen point is STILL
0 -- L[1]'s support does not overlap eps[(0,1)].  The request has moved one hop further
back and stalled again for the same reason it stalled at N2: it names a time the upstream
cannot serve.  Same disease, one layer up.

## 4n G seeds 4 and 5: the barrier walk, and BARRIER_CLAMP (NEGATIVE)

Both seeds fail by the SAME route, visible in the live trace (the return value is a
KEEP_BEST fossil, as usual):
    seed4  start w(1->2)=506 (supra-critical), N2=[198,398] -- the CORRECT count, ~46 early
    seed5  start w(1->2)=654 (supra-critical), N2=[184,384] -- likewise
The demand correctly says "N2 fires too early, fire later" => lower w(1->2), and that walks
506->420 / 654->411 straight through W_CRIT = 444.54.  N2 collapses 2 spikes -> 1 and the
run freezes at it301 with g = [0,0,0,0].  So these are not a tuning problem: the optimiser
starts with the right hidden spike COUNT and destroys it by following a correct local
demand.

Note the graded propagation IS working now: max|L1| = 5.1e-03 at the frozen point where it
used to be 0.00e+00.  The demand reaches N1 and still cannot act, because L[1]'s support
does not overlap eps[(0,1)].

BARRIER_CLAMP -- refuse to step a weight DOWN through W_CRIT in one update -- was the
obvious guard and it does NOT work.  Two rounds:
  * first version clamped to exactly W_CRIT, which made the next iteration's `w > W_CRIT`
    test FALSE so the following step walked through unimpeded -- w(1->2) still ended at 422
    with the clamp nominally "on".  My bug; fixed by landing at W_CRIT*1.002 and testing >=.
  * with that fixed the clamp holds mechanically (w(1->2) rests at 445) and STILL FAILS:
    N2 fires ONCE (seed4 N2=[341], seed5 N2=[331]) rather than twice.  Sitting exactly on
    the barrier is a degenerate state -- at w = W_CRIT a single spike's peak drive EQUALS
    threshold, so firing is knife-edge and decay decides it.  The clamp trades an absorbing
    dead state for a marginal one.
  Sweep: BARRIER_CLAMP=1 is equal or worse at every setting and drops out of the top rows
  entirely (best 6/8 is BARRIER_CLAMP=0).  Default 0.

STATE OF 4n G: 0/8 at the start of this thread -> 6/8, at REQ_GAIN 0.1-0.3, TRUST 5-20,
LR 10-20, MOVE_GAIN 0.25, OCCL_MASK 1 (OCCL_MASK is load-bearing: 0/8 without it at every
setting tried).  The direction of every gain is "turn the request DOWN and let the step
run" -- REQ_GAIN 3.0 -> 0.1-0.3 while TRUST 2 -> 5-20 -- consistent with the measured 4-16x
per-hop amplification.  Several distinct configs reach 6/8 recovering DIFFERENT seed sets,
which is why hyperparameters alone will not close it.

## Where 4n G rests, and why nothing pushes back  (OCCL_RELOC: NEGATIVE)

seed4 rests at w = [289, 420, 1262, 538]; N1=[144,344] N2=[376] N3=[33,133,233,333,422]
against target [33,133,233,291,333,433,491] -- marks 291, 433, 491 all missing.

THE CORRECT MOVE IS SMALL AND VISIBLE.  To mark 291, N2 must fire near 244; N1@144 gives
dt=100 and 420*HK[100] = 6.574e-03 against th 7.000e-03 -- short by 4.3e-04.  w(1->2) needs
447.2 (it is at 420, true 500).  And eps[(1,2)][244] = 1.5653e-05 is NONZERO, so a demand
placed at 244 WOULD act.

WHAT ACTUALLY BLOCKS IT IS OCCL_MASK, ALONE:
    OCCL_MASK=1   L[2] EMPTY            g = [0, 0, 0, 0]
    OCCL_MASK=0   L[2] at [228, 376]    g = [9.91e-06, +2.26e-06, 0, 0]
i.e. with the mask off, g(1->2) is POSITIVE -- exactly the direction that re-crosses the
barrier.  The mask is deleting the only gradient that escapes.  And the mask is not simply
wrong to reject: the demand sits at 228 while the feasible window for marking 291 is
q in [237,273], so 228 genuinely cannot serve it.  The fault is that rejection DELETES the
demand rather than moving it to a time that can.

OCCL_RELOC -- carry each rejected request entry to the NEAREST feasible time instead of
zeroing it -- was the obvious repair and it is NEGATIVE: 6/8 -> 4/8 on 4n G.  So "the
request is at the wrong time" is NOT the binding constraint at this resting point either;
moving it to a feasible time does not make the run escape.  Default OCCL_RELOC=0.

Three guards have now been tried against this same resting state and all three fail:
BARRIER_CLAMP (prevent the crossing), PROP_SILENT/GRADE_PROP (forward the demand further
back), OCCL_RELOC (retime the demand).  Each does what it says mechanically and none
recovers the case.  The one measurement that keeps coming back is that OCCL_MASK=0 gives
the RIGHT gradient here (+2.26e-06) but 0/8 overall, while OCCL_MASK=1 gives 6/8 overall
and zero here.  That tension -- the same mechanism being load-bearing globally and fatal
locally -- is the thing to resolve, and none of the three guards touches it.

CORRECTION: OCCL_RELOC did not even fire at the resting point -- L[2] is still EMPTY with
RELOC=1 (L[2][244] = +0.00e+00, g(1->2) = 0).  It relocates R[n], but what empties L[2]
here is the OTHER half of OCCL_MASK: the post-relaxation clamp

    if OCCL_MASK and n in _occl_ok:  Ln[bad] = np.minimum(Ln[bad], 0.0)

which zeroes the POSITIVE part of the propagated message at infeasible times, every sweep.
So the -4/8 measured for OCCL_RELOC is the cost of relocating requests elsewhere in the
suite, NOT a test of the resting-point hypothesis -- that hypothesis is still untested.
The right place to try relocation is the Ln clamp, not the R mask.

## Accounting for the 4n G thread: what each change cost, measured in isolation

I had been introducing changes in bundles and measuring only the union, which is how a -7
sat unnoticed for several rounds.  Isolated on the full suite at 3200 rounds:

    GRADE_PROP  LN_RELOC  OCCL_FROMDEMAND   total
        0           0            0           62      <- pre-thread state (63 with PROP_SILENT)
        0           0            1           55      (-7)
        0           1            0           57      (-5)
        1           0            0           52      (-10)
        1           1            1           49      (current before revert)

THREE INDEPENDENT COSTS, roughly additive:
  * GRADE_PROP=1 (unconditional graded propagation) -10.  Correct in principle -- a neuron
    that fires once early has the same problem a silent one does, so gating on "silent" was
    wrong -- but forwarding the graded remainder from EVERY neuron floods the upstream.
  * OCCL_FROMDEMAND=1 (derive the feasibility window from downstream L when R is empty) -7.
    This was PLUMBING I added to run an ablation and never measured.  It makes the mask
    process neurons it used to skip, so the Ln positive-clamp fires on far more of them.
    AND IT BUYS NOTHING: 4n G is 8/8 with it either on or off.  Free to remove.
  * LN_RELOC=1 (retime the rejected propagated demand) -5.  The change that actually fixed
    4n G, and the CHEAPEST of the three.

NO FREE LUNCH ON 4n G.  With GRADE_PROP=0 and OCCL_FROMDEMAND=0, LN_RELOC alone moves 4n G
only 3/8 -> 4/8 while costing 3-cycle 7->4 and 4n F 8->5 (net 62 -> 57).  4n G's 8/8 needs
GRADE_PROP=1 AND LN_RELOC=1 AND REQ_GAIN~0.3, which lands the suite at ~48-49/72.  So the
4n G fix genuinely costs ~14 cases elsewhere; it is not a plumbing artefact.

DEFAULTS RESTORED to the best measured configuration: GRADE_PROP=0, OCCL_FROMDEMAND=0,
LN_RELOC=0 -> 62/72 with 4n G at 3/8.  The 8/8 config is documented above and available by
env var if 4n G is the priority.

METHOD NOTE.  A POINT ABLATION IS NOT A TRAJECTORY TEST.  At 4n G's resting point turning
REQ_GAIN off barely changed g(1->2) (+4.196e-07 -> +4.552e-07), which I read as "the request
machinery contributes essentially nothing".  Running it says the opposite: MINIMAL (request
off) is 0/8 while the full config is 8/8.  The request is what NAVIGATES the trajectory into
the region where the retiming can act; it does little AT the resting point because by then
the trajectory is already stuck.  Same error shape as reading a static gradient instead of
running the dynamics, which has now bitten three times today.

The 8/8 components are irreducibly conjunctive: REQ_GAIN + SHARP_GAIN + OCCL_GAIN together
give 8/8, every PAIR gives <= 2/8, and each alone gives 0-2/8.  MOVE_GAIN is redundant there
(8/8 with or without), which does drop MOVE_GAIN/MOVE_COHERE/MOVE_ACT from what 4n G needs.

## A DIRECT DEMAND algorithm (NEW_DEMAND) -- prototype, promising, not yet working

Motivation: the request path is five stages that each patch the previous one's failure --
seed, propagate (back_corr + self-norm), reject (feasibility mask), concentrate (sharpen),
rescue (retime).  That is why they are CONJUNCTIVE (REQ+SHARP+OCCL all three give 8/8 on
4n G, every PAIR <= 2/8) and why it needs ~10 interacting gains.  But the measurements say
a demand is determined by exactly two computable quantities:
    FEASIBLE   q + DELAY in ( last_reset_of_d_before(t), t ]  minus refractory shadows
    REACHABLE  vsub_n(q) > 0  -- there must be drive at q for a weight to amplify
So build the demand DIRECTLY as a density over feasible-and-reachable times, weighted by
what a spike at q would deliver at t (w*h(t-q)) times how close n already is to firing.
One function, `demand_direct`, replaces all five stages: no seeding constant, no per-hop
gain, no rejection step, nothing to retime -- an infeasible time simply gets zero mass
rather than being placed and then removed.

WHAT IT GETS RIGHT ALREADY:
  * at 4n G's resting point [289,420,1262,538], where the OLD path gives g = [0,0,0,0],
    it gives g(1->2) = +8.47e-06 -- correctly signed (w1 420 -> true 500 needs UP).
  * stationarity at truth is preserved with NO dead-zone or grading: at the true weights
    every downstream demand is zero, so the density is empty and g = [0,0,0,0] exactly.
    The old path needed DEAD_ZONE/GRADE_* tuning for this.

WHAT IS STILL WRONG: per-hop magnitude.  A neuron serving several downstream demand times
accumulates a full-height density for each, so the signal grows with depth -- measured
g(0->1) = 1.05e-03 against g(1->2) = 8.47e-06, a 124x imbalance two hops out, which drives
the upstream weight wildly.  Adding a magnitude-preserving rescale (each neuron's demand
scaled to the largest downstream demand that produced it -- the one thing REQ_SELFNORM was
for) cuts it to 1.00e-04, still 12x.  4n G remains 0/8 at NEW_GAIN 1/3/10.

So: the construction is sound in DIRECTION and in its fixed point, and wrong in RELATIVE
SCALE ACROSS LAYERS.  That is a narrower problem than the five-stage pipeline had, and it
is the same per-hop-gain issue the old path spent REQ_SELFNORM/REQ_PEAKNORM/R_CAP/LOOP_CAP
on -- so it is not obviously easier, but it is now ONE problem instead of five.
Left behind NEW_DEMAND=0; defaults unchanged at 62/72.

## Where the hidden-neuron problem actually is (oracle-bounded)

ORACLE TEST.  Hand every hidden neuron its TRUE spike times and seed them through the SAME
hinge/matching/suppression path the outputs use, then measure sign agreement of the weight
gradient against (true - w) on the hidden edges:
                        inferred        ORACLE
        3n D             81.6%          100.0%
        4n G             71.9%           90.0%
        4n F             71.1%           77.5%
So the machinery that turns a demand into a weight DIRECTION is fine -- it was never the
bottleneck, despite a full day spent instrumenting eligibility, epochs, occlusion and
retiming.  What is broken is choosing WHERE the hidden neuron should fire.

DECOMPOSED, that is two separate defects:

 (1) WHICH downstream deficits are mine.  The inference asks n to fire for every target
     where TH - vsub[d][t] > 0, so when a DIRECT edge is weak, vsub falls short everywhere
     and the hidden neuron is blamed for deficits belonging to another edge.  3n D's true
     N1 = [246] came out as [5,105,305,405] -- the latencies to the N0-DRIVEN targets.
     THIS ONE SELF-CORRECTS as the other paths converge (predicted, then measured: with
     w(0->2) at truth the [5,105,305,405] pattern vanishes entirely).  Subtracting n's own
     share from the deficit -- it is self-referential otherwise -- improves it further
     (4n G right-count 2/24 -> 7/24).

 (2) WHEN to fire to serve target t.  Does NOT self-correct: with the direct edge at TRUTH,
     3n D still placed N1 at 170 and [197,257] against 246.  request_lat_vec returns the
     first dt at which w*h(dt) ALONE covers the deficit -- the earliest marginal crossing --
     while the truth sits well up the rising phase where the PSP COMBINES with co-drivers.
     Replaced with an exact 1-D solve (`solve_latency`): scan admissible q for the first
     tau with other(tau) + w*h(tau-q) >= TH equal to t, take the latest such q.
     IT WORKS IN ISOLATION: on 3n D, whenever the COUNT is right the TIME is now right too
     (within-20 1/3 -> 3/3 of right-count cases; w=[223,1797,619] gives N1=[240] vs a true
     246, 6 steps out, where it previously gave [180,240]).

BUT SIGN ACCURACY IS UNCHANGED: 71.6 / 68.4 / 80.0 with explicit targets vs 71.9 / 71.1 /
81.6 inferred.  A target train with the wrong NUMBER of spikes gives bad directions however
well each time is placed, and the count is still wrong in ~4 of 5 cases.  So the binding
constraint is (1), not (2) -- and (1) is exactly the "which presynaptic OWNS a downstream
spike" question this method was built to avoid, and which grad_credit.py solved explicitly
with per-spike competitive credit.  The claim in the module docstring that credit "splits
automatically ... and how many spikes is never decided" is the thing that does not hold up:
the deficit signal carries no information about WHICH edge should close it.

All of this is left behind HID_TARGETS=0; defaults unchanged at 62/72.

## NEGATIVE RESULTS: four hidden-demand constructions, and a BAD METRIC

Scored on actual OUTPUT RECOVERY (matched 800 rounds unless noted):
    baseline (existing request path)            54/72   (62/72 at 3200)
    HID_TARGETS (explicit targets + exact latency solve)  33/72
    FIELD, flat weighting                       17/72
    FIELD, concentrated (POW=16)                17/72
    NEW_DEMAND (direct feasible+reachable density)  18/72 at 3200
    ORACLE, true hidden spike times *** CHEATING ***    43/72

THE METRIC I INTRODUCED WAS WRONG, and it steered four experiments.  I proposed sign
agreement of the gradient against (true - w) as "the right success criterion".  It does not
predict recovery:
    FIELD POW=16   sign 71.0/66.7/72.5%   recovery 17/72
    baseline       sign 71.9/71.1/81.6%   recovery 54/72
    ORACLE         sign 90.0/77.5/100%    recovery 43/72   <-- BEST sign, WORSE recovery
Concentration lifted sign accuracy 62->72% and moved recovery not at all (17->17).
Sign agreement measures convergence to the TRUE WEIGHTS; the task is reproducing the
OUTPUT, and many weight configurations do the latter without approaching the former.

THE ORACLE RESULT IS THE IMPORTANT ONE.  Perfect hidden spike times make things WORSE
overall (62 -> 43), collapsing 2-cycle 8->0, 4n F 8->1, 3-cycle 7->2, while helping only
3n D (6->8) and 4n G (3->4).  Pinning a hidden neuron to its true trajectory forces the
weights to satisfy the hidden times AND the output, which selects the unique true solution
instead of any output-correct one.  The request path's demand is not a poor approximation
of the true hidden times -- its LOOSENESS IS LOAD-BEARING, and that is why every sharper
construction lost.

COROLLARY FOR 4n G: the oracle gives it only 4/8.  So 4n G's problem is NOT hidden-target
inference at all -- perfect targets barely help.  Whatever blocks it is downstream of
knowing when N1 and N2 should fire.

## KICK_GAIN: escape the frozen-and-wrong state  (62 -> 66/72, and 3n D solved)

The oracle result said 4n G's blocker is NOT hidden-target inference (perfect targets give
only 4/8), so the remaining problem is that there is no gradient to follow at all.  That
state is EXACTLY detectable without knowing anything about targets:

    max|g| == 0 on EVERY edge  AND  the output is still wrong

That is not a solution, it is an absorbing state -- on 4n G the relay N2 falls below the
single-spike weight, stops firing, and with no N2 spike there is no eligibility on the 2->3
edge for any demand to attach to, so the run sits there for ~2900 iterations.  The only
neurons that can be responsible are the hidden ones firing LEAST, and a silent or
near-silent hidden neuron the output still needs can only be revived by more drive.  So set
g on their incoming edges to a positive constant and let Adam do the rest.

                  chain 3-cyc 2-cyc over-dem 3n A 3n D 3n E 4n F 4n G  total
    KICK_GAIN=0     7     7     8      7      8    6    8    8    3     62
    KICK_GAIN=10    7     7     8      7      8    8    8    8    5     66
3n D SOLVED (6 -> 8/8, the case that resisted everything all session) and 4n G 3 -> 5/8.
Nothing regressed.  Insensitive to the gain: 10 / 30 / 100 / 300 all give 4n G 5/8, and 10
vs 100 both give 66/72 -- it only has to move.  Stationarity at truth preserved exactly
(0.000e+00, all nine): at truth the output is CORRECT, so the "and wrong" clause is false
and the kick never fires.  Default 10.0.

WHY THIS WORKED WHEN FIVE DEMAND CONSTRUCTIONS DID NOT.  Every previous attempt tried to
manufacture a BETTER GRADIENT for the dead state.  There is no gradient there to improve --
the eligibility is genuinely zero, as the finite-difference check confirmed hours earlier.
The fix is not a better demand but a NON-GRADIENT move, justified structurally rather than
by differentiation, and gated on a condition that is false everywhere except the pathology.

## KICK_STALL: stalling, not freezing  (66 -> 67/72)

With the freeze fixed, 4n G's remaining seeds MOVE FREELY (0/3200 zero-update iterations)
and still converge to N1=1,N2=1,N3=5 against a true 2,2,7 -- w(0->1) driven DOWN to ~190-205
(true 250) until the accumulator is too slow to fire twice.  seed5 VISITS (2,2,5) and
(2,2,6) and drifts away, so the right count is reachable and actively abandoned.  There is
a second count boundary here (fires-twice vs fires-once) distinct from W_CRIT, and g is
nonzero throughout, so the frozen test never fires.

g == 0 turns out to be the extreme case of a more general condition: seed3 travels 0.10
weight units over its last 500 iterations without ever hitting g == 0.  Gate on the WEIGHTS
NOT MOVING instead, which covers both.  Suite 66 -> 67/72, stationarity still exactly
0.000e+00 (at truth the output is correct so the "and wrong" clause is false).

REJECTED: gating on "the output has too few spikes" instead.  Observable and false at
truth, but true for most of TRAINING, so the kick fires constantly and swamps the real
gradient: 4n G 5/8 -> 0/8 and the suite 66 -> 24/72.  The frozen/stalled tests work BECAUSE
THEY ARE RARE -- that is the actual design constraint on a non-gradient move, not
observability.

COMPLEMENTARITY: KICK_STALL settings recover DISJOINT 4n G seeds --
    KICK_STALL=0    ok {0,1,4,6,7}
    KICK_STALL=10   ok {1,2,3,5,6}   (recovers exactly the stuck seeds 2 and 3)
whose union is all eight.  Both give 67/72 on the suite.  So 4n G 8/8 is available to a
multi-start over the stall threshold, in the same way sharp0/collapse were complementary
earlier.

## 4n G SOLVED: 8/8.  It was a BUDGET problem, and the ordering happens by itself

The three seeds left at 3200 rounds all sat in the same state: N1 and N2 firing the CORRECT
2 spikes each (seed7 spends 2631 of 3200 iterations there), output stuck at 5 spikes, marks
291 and 491 missing.  The blocker was w(2->3), the edge carrying N2 to the output:
    seed2  385 (true 700, need +315)   g = 0.00e+00
    seed3  423 (true 700, need +277)   g = 0.00e+00
    seed7  420 (true 700, need +280)   g = 0.00e+00
All below W_CRIT = 444.5, so N2's spike CANNOT fire N3 at any latency (peak PSP 6.06-6.66e-03
vs th 7.00e-03, 5-13% short), so no mark appears, so there is no demand at 291 to attach
to, so no gradient on the edge that would fix it.  In seeds 3 and 7 the arrival is ALSO
occluded: N2@232 arrives 250, inside the shadow of the TARGET spike at 233 (to 255), so
even a sufficient weight would be discarded -- N2 must ALSO move ~12 steps later.

THE HYPOTHESIS WAS RIGHT AND THE OPTIMISER ALREADY DOES IT.  Raising w(2->3) before N2's
timing is right would place the mark ~50 steps early, and that spurious spike is then
suppressed, pushing the weight back down -- so it must stay sub-critical until the timing
lands.  Traced on seed2, that is exactly the observed order: N1 124 -> 166 and N2 208 -> 239
FIRST, and only then w(2->3) 385 -> 604.  It just needs the iterations.

    4n G by budget:  3200 -> 5/8    10000 -> 8/8    25000 -> 8/8
    SUITE at 10000:  70/72
    chain 7  3-cycle 7  2-cycle 8  over-demand 8  3n A 8  3n D 8  3n E 8  4n F 8  4n G 8

So 4n G was never structurally broken at the current defaults; what the two kicks bought was
making those extra iterations PRODUCTIVE instead of frozen (it was 3/8 at 3200 before them,
and no amount of extra budget helps a run whose update is identically zero).  The remaining
2 failures are one seed each on chain and 3-cycle.

## WHY 4n G needs 10k iterations: the hidden edges OSCILLATE (and momentum trades, not fixes)

It is not step size and not the trust region.  Measured on seed2: median |update| is 0.76
weight units/iteration and the total travel required is ~450 units, so a straight run would
take ~590 iterations.  It takes 10,000 because the trajectory wanders.  Net displacement
over total path length, per edge:

    w(0->1)   path 3567   net   64    EFFICIENCY  1.8%
    w(1->2)   path 2194   net   96                4.4%
    w(2->3)   path  750   net  190               25.3%
    w(0->3)   path  664   net  370               55.7%   <- the DIRECT edge, nearly straight

The two hidden edges cover ~50x more distance than they need; the direct edge does not.
That is the 71% hidden-edge sign accuracy showing up as zigzag -- roughly three steps
forward, one back, and worse once magnitudes vary.  The 10k iterations are ~600 iterations
of progress buried in ~9400 of oscillation.

MOMENTUM CONFIRMS THE DIAGNOSIS AND IS NOT A FREE FIX.  Raising Adam's beta1 averages the
reversals out and takes 4n G from 5/8 to 8/8 AT 3200 ROUNDS (a 3x speedup on the case), but
it costs elsewhere:
    BETA1   4n G@3200   suite@3200   suite@10000
    0.90      5/8          67/72        70/72
    0.97      8/8          61/72        63/72
    0.99      8/8          60/72          -
The damage is concentrated in 4n F (8 -> 3) and 3-cycle (7 -> 4), and MORE BUDGET DOES NOT
RECOVER IT (63/72 at 10k), so it is genuine overshoot on the well-conditioned cases rather
than slowness.  Left at BETA1=0.9; 0.97 is available per-case and is the right setting if
4n G is the priority.

The real fix is not a smoothing constant but the 71% -- a hidden demand that reverses a
third of the time is what makes the trajectory 50x longer than it needs to be.

## 50-neuron RECURRENT_CASES at current defaults: COUNT right, TIMING wrong, and PLATEAUED

    case0 (132 edges)  N47 7/7 spikes mean|dt|=44.1   N48 2/2 dt=11.5   N49 1/1 dt=24.0
    case1 (304 edges)  N47 4/4 dt= 5.8               N48 2/2 dt=12.0   N49 7/7 dt= 7.1
    case2 (240 edges)  N47 3/3 dt=35.0               N48 3/3 dt=33.3   N49 7/7 dt=17.3
    EXACT 0/18 (2 seeds); 17 of 18 outputs have the CORRECT SPIKE COUNT.

800 rounds gives BIT-IDENTICAL errors to 300 -- every mean|dt| unchanged -- so it converges
early and plateaus.  Stuck, not slow, unlike 4n G.

This is the cleanest statement of where the method stands at scale: it reliably recovers
HOW MANY spikes each output fires and does not recover WHEN.  That matches the small-case
diagnosis -- the direct edges converge efficiently (55.7% path efficiency) while the hidden
edges oscillate at 1.8-4.4% because their demand reverses ~29% of the time.  At 50 neurons
almost every edge is hidden, so almost nothing converges in timing.

## Two new case families: 4-WAY COINCIDENCE (8n K) and NEGATIVE weights (3n L)

  8n K  N0 fans out to N1..N4 at four DIFFERENT weights (spreading the hidden spikes in
        time), and N1..N4 each feed all three outputs N5,N6,N7.  Every FAN-IN weight is
        100-160, i.e. 0.22-0.36x threshold, so no single connection can fire an output and
        roughly four coincident arrivals are needed.  Three outputs over 16 edges keeps it
        from being badly underdetermined.  Fan-in weights differ per output, which matters:
        with identical weights all three outputs produce the SAME train and two thirds of
        the supervision is wasted.
            N5=[145,294,445]  N6=[130,230,330,430]  N7=[132,232,332,432]   no redundant edges
  3n L  One inhibitory edge, w(1->2) = -700, turning N2's regular period-100 train into
        [33, 237, 437].

    8n K  3/8      3n L  0/8      suite now 83/104 at 3200 rounds

3n L IS BROKEN BY A ONE-LINE ASSUMPTION, not by anything subtle.  grad_trace.py:1552 is

    w = np.clip(w + upd, 20, 3000)

so the weight is clipped to a MINIMUM OF +20 and a negative weight cannot be represented at
all.  Traced: seed0 starts at w = [795.9, 923.7, -378.7] -- correctly negative, since the
initialisation is true*U(0.5,1.5) -- and after one train call the live weights are
[841.6, 268.1, 95.0].  The inhibitory edge is clipped positive on the FIRST UPDATE and can
never come back.  0/8 is therefore not a gradient failure; the search space excludes the
answer.

That clip has to go before any of the deeper sign questions can even be asked -- and those
are real too: every deficit, hinge and creation-request argument assumes more weight =>
more drive => earlier spike, which reverses for w < 0, and W_CRIT (th/max(HK)) is defined
for excitation only.

## 8n K IS NOT A 4-WAY COINCIDENCE TEST (and the fix hits a real tension)

The output voltages LOOK marginal -- every output peaks at 1.001-1.005x threshold -- but
that is the RESET clipping the trace at the crossing, not the actual margin.  Free-running
(no reset, no refractory):

    N5  V at spike 7.015e-03 = 1.002x th    FREE-RUN peak 2.017e-02 = 2.88x th
    N6              7.007e-03 = 1.001x                    2.224e-02 = 3.18x
    N7              7.037e-03 = 1.005x                    2.188e-02 = 3.13x

At that free-run peak each source is worth 0.60-0.97x threshold on its own, so TWO of four
sources suffice.  The 4-way property holds ONLY for the FIRST spike of each output
(145/130/132), before anything has accumulated.  Cause: the kernel's support is 400 steps
against an input period of 100, so each hidden neuron has 4-5 spikes whose tails overlap and
a "0.27x threshold" edge is worth ~0.8x by t=431.

TRYING TO FIX IT EXPOSES A FUNDAMENTAL TENSION.  Making the fan-out sub-critical so the
hidden neurons fire SPARSELY (no tails to pile up) does give genuine 4-way coincidence:
    fan=[300,260,230,210] k=110 (0.25x th/edge): hidden [2,2,1,1], sources needed [4,4,4],
    free-run peak only 1.14x th   -- N1=[140,340] N2=[160,360] N3=[228] N4=[238]
But each output then fires exactly ONCE (all three at 385), because needing four coincident
arrivals means the total drive barely clears threshold, which by construction happens
rarely.  That is 3 output spikes of supervision for 16 weights -- badly underdetermined,
the thing the case was supposed to avoid.

So "no single connection can fire a neuron" and "enough output spikes to constrain the
weights" pull against each other in this model: coincidence requires drive that barely
crosses, and drive that barely crosses produces few spikes.  A usable case has to buy
constraints some other way -- more outputs, or fan-in weights that DIFFER per output so the
same hidden spikes produce different trains.

## 8n M: an IRREDUCIBLE coincidence case (8n K was underdetermined)

8n K does not test coincidence.  At a stuck point the optimiser reaches 10 of 11 output
spikes on target with weights nowhere near the truth --
    found [2517,456,422,151, 302,178,20,20, 20,879,23,20, 20,819,20,151]
    true  [ 900,700,550,460, 120,120,120,120, 150,130,110,140, 100,160,140,120]
-- 8 of 12 fan-in weights collapsed to the clip floor of 20 and N4 died entirely, while N6
and N7 stayed 4/4 on target driven by ONE surviving edge each (879 and 819, ~7x their true
values).  The cause: all four hidden neurons fire on the same period-100 rhythm, so the
outputs are period-100 trains and a single edge at the right phase reproduces them.  The
output supervision cannot distinguish "four weak coincident inputs" from "one strong input".

FIX: give the hidden layer DIFFERENT RATES -- two supra-critical (fire every cycle) and two
SUB-critical accumulators (fire rarely).  The coincidence pattern is then irregular and no
single-rate source can match it.

    8n M  fan-out [900, 500, 250, 200]   fan-in 135-180 (0.30-0.40x threshold)
          N1=[39,139,239,339,439]  N2=[72,172,272,372,472]   (every cycle)
          N3=[173,373]             N4=[246]                  (rare accumulators)
          N5=[194,343,497]  N6=[190,342,491]  N7=[195,336,494]   ISIs 149/154, 152/149, 141/158

    IRREDUCIBILITY VERIFIED: for every output, NO subset of <= 3 of its four incoming edges
    reproduces the target train at any weights on a 20..3000 grid.  All four are required.

That is the property 8n K was supposed to have and did not.

8n K REMOVED from the suite.  Its 3/8 measures a degeneracy rather than a capability -- the
optimiser reaches near-perfect output with one strong edge per output and the rest clipped
to the floor -- so any future tuning that tried to "improve 8n K" would be chasing an
artefact.  8n M supersedes it.

## 8n M at 1040 steps: the rare accumulators DIE and the case collapses to single-rate

Doubling the sim runtime doubles the supervision (9 output spikes -> 18, for 16 weights) and
makes recovery WORSE, not better:
    steps=520    EXACT 1/8   wrong-count outputs 0/24   mean|dt| 2.75
    steps=1040   EXACT 0/8   wrong-count outputs 1/24   mean|dt| 15.54
At 520 six of eight seeds land within ~1 step per spike (several at 0.0/0.0/0.3), so the
case is much closer to solved than 1/8 suggests -- the failures are near-misses, not
structural.

THE FIGURES SHOW WHY.  At the true weights every output ramp is a multi-stage staircase and
the output ISIs are irregular (149/154/119/178/149).  At the 1040 stuck point (live weights,
seed0) N3 and N4 -- the two RARE ACCUMULATORS that make the case irreducible -- are BOTH
DEAD (found []), and the outputs collapse to what N1/N2 alone can produce:
    N6 found [148,248,348,...,948]   9 spikes, PERIOD 100   target 6 spikes, irregular
    N7 found [151,251,351,...,951]   9 spikes, PERIOD 100
    N5 found [171,306,471,606,771,906]  ISIs 135/165 alternating
That is exactly the 8n K degeneracy reappearing inside the irreducible case: kill the rare
sources and the survivors produce a regular train.  Irreducibility guarantees no SUBSET
reproduces the TARGET -- it does not stop the optimiser from walking into a subset solution
that produces a DIFFERENT train and sitting there.

LIVE vs RETURNED DISAGREE SHARPLY HERE (KEEP_BEST fossil again, 4th time today): the
returned weights give count-correct outputs with mean|dt| ~20, while the LIVE weights at the
same seed have N6/N7 firing 9 times against a target of 6.  The optimiser's best-ever
iterate is much better than where it actually ends up, so any figure or diagnosis must say
which one it is using.

## 8n M runs at 1040 steps (per-case simulation length)

8n M is essentially SOLVED at the default 520: 6 of 8 seeds land within ~1 step per spike
(several at 0.0/0.0/0.3), counts correct on all 24 outputs, mean|dt| 2.75.  The
discriminating regime is the LONGER run -- 18 output spikes instead of 9 for 16 weights --
where the case actually breaks: mean|dt| 15.54, and at the live weights the two rare
accumulators N3/N4 die entirely and the outputs collapse to the period-100 rhythm N1/N2
alone can produce.

So the case now carries its own length.  _diag.py defines

    CASE_STEPS = {"8n M": 1040};  DEFAULT_STEPS = 520;  steps_for(name)

and the suite harness, the diagnostic dump and both plot scripts all take their step count
from it.  Everything else stays at 520.

WHAT THE RASTER FIGURES SHOW (colour-coded arrival ticks per hidden source, under each
output trace).  At the true weights, each output crossing sits on a cluster of arrivals and
the RARE sources are visibly the trigger -- N4 fires once, and that single violet tick is
what makes the second output spike happen.  At a 520 stuck point every source is mistimed
AND THE ORDER IS INVERTED (N2 now arrives before N1: 58 vs 76, against a true 72 vs 39),
N3 moves 173->145 and N4 246->233, yet the outputs land at 195/345/497 against 194/343/497.
The method converges to a DIFFERENT hidden configuration that happens to coincide at the
right moments; the per-source errors cancel in the sum.  N6 is the residue at 1/3 on target,
4-5 steps late, because the cancellation is exact for two outputs and not the third.
That also explains the 1040 result: the compensation has to hold at every output at once,
and with twice the spikes it cannot, so the optimiser kills N3/N4 instead.

## CREATE: the real path for gradient through a SILENT neuron (the kick was a patch)

KICK_DEAD -- a per-neuron non-gradient nudge for a silent hidden neuron -- worked but does
not generalise: any gate of that shape has cases it misses.  What is actually needed is for
a demand to REACH a neuron that has not spiked.  That path already exists in the backward
relaxation:

    vol = sum_d  w[n->d] * back_corr(L[d], HK)
    Ln  = vol * exp(-((vsub[n]-TH)/(SIG*TH))^2) * CREATE

None of it requires n's OWN spikes -- vol is downstream demand, the gate is n's sub-threshold
voltage.  The TIMING term, by contrast, is added AT the neuron's own spikes, so a silent
neuron gets nothing from it.

AND CREATE DEFAULTED TO 0.0, so the whole path was switched off.  Measured on 8n M's dead
N3/N4: vol = 2.485e-06, gate = 0.836 (vsub 5.52e-03 vs th 7.00e-03 -- plainly recruitable),
CREATE_FLOOR passing 95% of timesteps, vol*gate = 2.06e-06 ... and then multiplied by zero,
giving L[3] = L[4] = 0.000e+00 exactly and no gradient on their inputs.

    CREATE   0.0    0.03   0.3    1.0    10.0
    suite     76     77     80     73      57       (KICK_DEAD=1)
    CREATE=0.3 with KICK_DEAD=0 also gives 80 -- THE KICK IS NOW REDUNDANT.

At 0.3 the silent neurons get correctly-signed gradients (+1.41e-07, +2.03e-07 where both
need to rise).  Gains are broad rather than case-specific: 5n H 5->7 (the deep chain) and
4n G 5->7, both cases where a hidden neuron must START firing somewhere in a sequence --
exactly what the kick could not generalise to.  Stationarity at truth still 0.000e+00.

Larger CREATE is much worse (1.0 -> 73, 10.0 -> 57): the same per-hop amplification this
project has hit repeatedly, since vol accumulates over every downstream edge.

DEFAULTS NOW: CREATE=0.3, KICK_DEAD=0 (kept for comparison), KICK_GAIN=10, KICK_STALL=2.

## 8n M after CREATE: rates ARE recoverable, PHASE is not (and momentum still costs)

(a) WHAT THE REVIVED ACCUMULATORS FIRE.  True structure is N3 5x at period 200 and N4 3x at
period 300 -- genuinely different rates.  With CREATE on:
    seed3   N3 5x period 200, N4 3x period 300   BOTH RATES EXACTLY RIGHT
            but mean|dt| 45-53 (N3) and 22-23 (N4) -- pure PHASE offset
    seed1   N3 3x period 300, N4 5x period 200   THE ROLES ARE SWAPPED
    seed0   N3 and N4 in lockstep at period 500
So "they converge to a common rhythm" was wrong for the good seeds: the rates are
recoverable, and what is left is phase.  The cause is visible in the weights -- N3 fires at
126/326 against a true 173/373 (~46 EARLY) and w(0->3) converges to 284-357 against a true
250.  A heavier accumulator crosses sooner, so the weight OVERSHOOT IS the phase error.
The swap in seed1 is a real ambiguity: N3 and N4 both feed all three outputs, so either rate
can be assigned to either neuron.

(b) THE LEVERS ARE COUPLED AND NONE IS DECISIVE.  On 8n M (mean output error):
    BETA1 0.97 / CREATE 0.3 / LR 10   17.53     BETA1 0.9 / CREATE 0.3 / LR 10   20.21
    BETA1 0.97 / CREATE 1.0 / LR  2   22.10     BETA1 0.9 / CREATE 0.3 / LR  2   42.97
Momentum gives a small consistent gain; step size and CREATE TRADE OFF (smaller LR is worse
at CREATE=0.3 and better at CREATE=1.0), consistent with the product mattering rather than
either alone.  And w(0->4) now reaches 210-253 against a true 200 in most configs, so the
weak-signal problem CREATE was fixing is gone -- the residual is not magnitude.

BETA1=0.97 still costs the suite badly, 80 -> 72/104, exactly as it did before CREATE.
Keeping BETA1=0.9.  Defaults stay CREATE=0.3, LR=10, BETA1=0.9 -> 80/104.

NEXT: the accumulator weight overshoots (284-357 vs 250) and that overshoot IS the phase
error.  The creation demand is supposed to be self-limiting -- seeded from a deficit that
vanishes once the spike exists -- so the thing to check is whether CREATE's contribution
actually turns off once the neuron starts firing, or keeps pushing past the target.

---

## Urgency scored by weight mismatch: the field finally becomes phase-aware

The urgency score has been through three definitions.  All three ask "how good a place to
fire is q?" for a downstream demand at t, and all three start from the same quantity,

    wmin = (TH - other[t]) / h(t - q)     the outgoing weight at which a spike at q
                                          puts d exactly over threshold at t

(1) `room = 1 - wmin/wmax`, the relative WIDTH of the admissible weight interval.  Peaks
    where wmax blows up, i.e. at the window boundary -- the least plausible place to fire.
    On 3-cycle the peaks sat ~60 steps AFTER the wanted times (195/295/385 vs 143/243/333),
    nearer the NEXT interval than the current one, and were a few samples wide against
    ~100-step intervals.
(2) `deliver = h(t-q)/max(h)`.  Broad (55% of samples above half peak, up from 1%) and
    every plausible time scored well.  But it is a property of the KERNEL ALONE, so its
    argmax is pinned at dt=110 for every neuron, every target, at every weight.  It never
    referenced the state of the network, so it could not move as w moved.
(3) `sc = max(0, 1 - |wmin - w| / w)`.  Maximum where the weight ALREADY ON THE EDGE is the
    weight that lands the downstream spike on time -- no weight change is being asked for as
    the price of the timing -- falling off with the mismatch, relative to w so the scale is
    the same on a 100-weight edge and a 1000-weight one.

Measured on 3-cycle AT THE TRUE WEIGHTS, urgency argmax offset from each true spike time:

    room       peaks ~+60 steps off, a few samples wide
    deliver    N1 -39 -39 -36 -39     score at the true times 0.43
    mismatch   N1   0   0   0   0     score at the true times 1.00
               N2   0   0   0   0     score at the true times 1.00

and breadth is kept: 265/520 samples above half peak.  This is the first version whose
maximum is ON the answer rather than at a fixed kernel offset from it.

## Suppression is now the same construction with the sign flipped

Previously: a flat -thr stamped over a +-20 window around any spike with no positive field
nearby.  A stamp is not a gradient -- it has the same depth wherever it lands and says
nothing about which weight would remove the spike.

Now a spike of d that no demand time claims is a NEGATIVE JOB running through identical
code: same wmin, same mismatch score, same lay-down, sign flipped.  So suppression is
strongest exactly where the CURRENT weight is what produces the unwanted spike, which is
the case where changing this weight actually removes it.  `other[t] >= TH` skips the job --
d crosses at t whether or not n helps, so n is not to blame.

Two consequences at the true weights of 3-cycle:
  - N1 and N2 go to ZERO negative points (was: every correct spike stamped, N2's four
    perfect spikes each taking a full -1.70e-03).
  - N3 keeps a graded negative bowl over 340-430, reaching -1.0 at 404.  This is NOT a bug
    in the construction, it is a bad input: N1 fires 5x but only 4 crossings are found
    (the 5th, 461, is too near the end of the 520-step window), so N1@461 looks unclaimed
    and N3@404 is blamed for driving it.  Demand-time lists truncate at the window edge.

FIELD_FLAT now defaults to 1 = the drives-nothing case ONLY (an arrival every downstream
discards, which has no t to mirror and no wmin to score).  FIELD_FLAT=2 restores the old
blanket test, which double-counts: it stamped -0.22 on N1's perfectly placed spike at 461.

STILL SELF-SATISFIED AT THE STUCK POINT.  At seed 0's stuck weights N1 and N2's peaks sit on
their CURRENT spikes, not their true ones -- which is what the definition says they should
do, since the current weight IS consistent with the current timing.  The field reports the
one real error correctly (a raised region over 295-345 on N2, asking for the spike that
would serve the missing output at 404) but that spike already exists at 314 and is discarded
into N3@315's refractory shadow.  A field keyed to the current weight cannot flag a timing
that is self-consistent; that needs the count/refractory information, not more urgency.

## ... but as a TRAINING signal the mismatch score is much worse, and tautologically so

Suite at 3200 rounds, 13 cases x 8 seeds:

    default (field inert)   80/104
    FIELD_XING = 1.0        44/104     (3-cycle 0/8, 4n F 0/8, 2-cycle 1/8, 4n G 1/8)
    FIELD = 1.0             11/104     (only 3n A 7/8 and 5n H 4/8 survive)

The reason is visible once the right question is asked.  "argmax offset 0 from the true
times at the TRUE weights" looked like a triumph, but it is close to a tautology: at the
true weights the current spikes ARE the true spikes, so a field that peaks on the current
spikes scores perfectly.  The test that matters is where the peaks sit when the weights are
WRONG.  Measured over 147 field peaks at stuck points, 9 cases x 3 seeds:

    median distance to nearest CURRENT spike     2 steps
    median distance to nearest TRUE spike       14 steps
    peaks closer to CURRENT than to TRUE      99/147
    peaks landing ON a current spike (<=2)    79/147
    peaks landing ON a true spike (<=2)       14/147

So the field points at where the neuron ALREADY fires.  This follows directly from the
definition: if n fires at q and d responds, then wmin ~= w at q by construction, mismatch
~= 0, and the score is maximal there.  A score keyed to the current weight is maximal at
whatever the current weight is already doing -- it certifies the status quo instead of
pointing away from it.  That is the same self-satisfaction the FIELD_XING crossings had
(OBSERVATIONS above), arrived at from the other direction.

The earlier `deliver` score was worse as a DESCRIPTION (argmax pinned at dt=110, 0.43 at
the true times) but that fixed kernel offset at least DISAGREED with the current state, so
it produced motion.  This is the tension: a score that references w is phase-aware but
fixed-point-preserving; a score that ignores w moves but does not know where to.

WHAT THIS RULES OUT.  Any urgency of the form f(|wmin - w|) has the stuck state as a local
maximum, so no amount of tuning FIELD_TOL rescues it.  To push off a self-consistent wrong
state the signal must come from something the current weights CANNOT explain -- the output
count/refractory mismatch (3-cycle's discarded N2@314 is exactly that) rather than from the
weight-consistency of an existing spike.

PERF.  demand_field vectorized over candidate times: 3-cycle 336 -> 27 ms, 8n M 1914 ->
192 ms (10x), cost relative to traces() 56x -> 4.5x.  Output byte-identical.  The old
Python-loop version made the field-on suite cost ~1.7 h per 8n M seed.

## The hard clip was a category error; fixing it fixes the FIELD but not the RESULT

Urgency was `max(0, 1 - mis)` with `mis = |wmin - w|/w`.  That reaches zero at mis >= 1 and
STAYS there, which collapses the two field variables back into one: urgency answers "do we
want a spike here", implied_w answers "and what would the weight have to be", and the second
is not licensed to SILENCE the first.

4n F seed 2, stuck at w = [543, 3000, 1200, 25] (true [240,1200,1200,1100]), N3 missing its
target at 290.  w(N2->N3) = 25 while the cheapest workable weight is 445, so mis = 16.8 and
BOTH hidden neurons had an identically zero field -- no urgency, no implied_w, no crossings,
at exactly the moment a signal was needed.  Self-reinforcing: once an edge drifts out of
band the field goes silent for it, so nothing pulls it back.

Soft falloff `1/(1 + mis/TOL)` (FIELD_HARD=1 restores the clip).  It also UNIFIES the two
earlier scores rather than choosing: when w is far below any workable weight the ranking is
dominated by wmin being smallest, i.e. max drive = the kernel peak, which is the right
answer for a too-weak weight; as w approaches a workable value the maximum slides onto the
q where wmin == w, the phase-correct answer.

Effect on the FIELD at that stuck point -- N2 goes from empty to 183/520 nonzero, urgency
over q=160..275 (the window that could serve N3@290) max 7.00e-03 at q=216, and implied_w
there reads 445: "fire about here, and the weight needs to be 445 not 25".  Correct.

Effect on RECOVERY -- none:
    4n F   baseline 8/8   SOFT novel-demand 6/8   HARD novel-demand 6/8   SOFT full 0/8
The two novel-demand runs recover the IDENTICAL seed set [0,1,3,4,5,7].

WHY: FIELD_XING consumes only field_crossings(), i.e. the LOCATIONS where implied_w meets w.
It never reads urgency at all.  So every improvement to urgency is discarded before it can
reach the gradient.  And the crossing detector requires the two samples straddling the sign
change to be ADJACENT (`b == a + 1`), so crossings that occur across a NaN gap in implied_w
are dropped: at 4n F's stuck point IW[2] has 403 finite points and 2 genuine sign changes
against w=25, and field_crossings returns {}.  N1's demand times come only from N2's
crossings, so N1 starves -- which is why N1 stayed empty even after the soft falloff.

## Which half of FIELD_XING is the poison (4n F, 8 seeds, 3200 rounds)

    baseline (field off)            8/8
    full FIELD_XING                 0/8
    demand at crossings only        1/8      <- the poison
    spike-move only                 7/8      <- nearly inert
    novel crossings, demand+move    2/8
    novel crossings, demand only    6/8

The spike-move is harmless because it is INERT: crossings sit a median 2 steps from current
spikes, so the push rounds to nothing.  Point it at novel crossings instead and it drops to
2/8.  The crossing DEMAND is the ratchet.  Suite screen at 800 rounds x 4 seeds:
baseline 29/52, full FIELD_XING 17/52, demand-only 22/52.

NEXT: the consumer is the problem, not the field.  A use that reads URGENCY (a density over
times) and IMPLIED_W (a weight target) directly -- rather than the crossing locations -- is
the only way the improved field can matter.

## Urgency used DIRECTLY as a density: 0/8, and it is a type error not a gain

FIELD_ADD adds the urgency density to the working demand (the crossings path reads only
field_crossings() and never touches urgency, which is why four successive improvements to
urgency left recovery untouched).  4n F, 8 seeds, 3200 rounds:

    baseline (field off)      8/8
    urgency add 0.25          0/8
    urgency add 1.0           0/8
    urgency add 4.0           0/8
    urgency PURE (FIELD=1)    0/8
    best crossings consumer   5/8

Mechanism, N1 spike count over training (TRUE count is 1):
    baseline          2 -> 2 -> 2 -> 1   by it800, the surplus spike is DELETED
    FIELD_ADD=0.25    2 -> 2 -> 2 -> 2   deletion blocked
    FIELD_ADD=1.0     2 -> 2 -> 2 -> 5   spikes MULTIPLIED

And the reason, measured AT THE TRUE WEIGHTS:
    N1 urgency 92 nonzero pts, width-at-half-peak 86 steps, integral 4.21e-01  (true count 1)
    N2 urgency 35 nonzero pts, width-at-half-peak 29 steps, integral 1.72e-01  (true count 1)

An 86-step-wide "fire in here" band is better satisfied by firing five times than once.
This is the request-plateau failure (grad_trace.py:564, "fire somewhere in here is satisfied
by firing EVERYWHERE") returning in a new form.  Gain does not fix it -- 0.25 is already
enough to cancel the working demand's ability to delete a surplus spike.

BOTH CONSUMERS ARE NOW RULED OUT, FOR OPPOSITE REASONS:
  crossings  -- a discrete set of points, so it CAN express a count, but the points are
                self-certifying (79/147 land on a current spike), frequently empty, and
                structurally undefined at a stuck point because implied_w STEPS over w
                rather than crossing it.  4n F 5-6/8.
  density    -- broad and well-shaped and correct about WHERE, but carries no count, and
                the count is what every hard case turns on.  4n F 0/8.

Urgency is a "where" signal with no "how many".  Anything built on it needs the count fixed
by something else before the density can be safely consumed.

## Two field-correctness fixes that are RIGHT but did not help recovery

(a) PHYSICAL RESETS (FIELD_PHYS).  vsub/eps reset an output at its TARGET times; the field
    asks a forward question and wants the drive since the output ACTUALLY last reset.  Using
    the counterfactual epochs imports an EPOCH_EXTEND bug: bounds[i-1] is BOTH the lower
    boundary of epoch i AND the target/upper boundary of epoch i-1, so widening the starved
    epoch for 4n F's target 290 moved the boundary at 233 back to 204 -- and 233, a target
    N3 HITS with 1.018*TH of drive in its own epoch, dropped out of the boundary list
    entirely and read a full deficit.  That phantom deficit laid a spurious urgency bump
    over q=116..215.  NOTE: this bug is in the DEFAULT path, where vsub/eps feed the working
    gradient.  Untested there.
(b) OCCLUSION MASK (FIELD_OCCL).  A candidate q whose arrival lands in the refractory shadow
    of a spike d ACTUALLY produced contributes nothing at any weight.  For 4n F target 290,
    21 of the 57 candidates were dead, all at the FRONT of the window -- and the field's own
    maximum sat on the deadest one (q=216, arrival 234, inside N3@233's shadow 233..255).
    Masking leaves q=237..272, which contains the true N2 spike at 256.

Both verifiably improve the field (spurious bump gone, 183 -> 70 nonzero pts; peak moves
216 -> 237).  4n F recovery: neither 6/8, OCCL only 5/8, PHYS+OCCL 5/8, PHYS only 1/8.
Field quality and recovery are decoupled while the consumer is the crossings.

## Propagate via urgency BUMPS, not implied_w crossings

Crossings were load-bearing in TWO places -- the final consumer AND the backward recursion
that gives an upstream neuron its demand times.  Replacing only the consumer (FIELD_ADD)
left the second hop starved: 4n F's N2 (one hop from an output, reads out_targets directly)
had a clean correct field while N1 had an identically EMPTY one, because N2's crossings were
[] so _demand_t[2] was never set.

FIELD_PROP=1: each contiguous run of positive urgency is ONE requested spike, its argmax is
where that spike is wanted.  This restores a COUNT, which the raw density cannot express.

    4n F           bumps -> requested times        truth
    at TRUE w      N1: 1 bump -> [225]             1 spike at 224   EXACT
                   N2: 1 bump -> [257]             1 spike at 256   EXACT
    at STUCK       N1: 2 bumps -> [167, 214]       1 spike at 224
                   N2: 1 bump -> [237]             1 spike at 256   count right

## Last counterfactual-vs-physical site: the `blocked` test

The FIELD_FLAT residual tested a candidate arrival against out_targets[d], so 4n F's N2@285
(arrival 303) was "blocked" by N3's TARGET at 290 -- a spike N3 does not fire -- and took a
flat -thr over 265..305.  It was being penalised for colliding with the very target the
field was asking it to serve.  Refractoriness is PHYSICAL; a missed target casts no shadow.
Fixed under FIELD_PHYS: N2's negative points 6 -> 0.  (N1's negatives are NOT this stamp --
toggling FIELD_FLAT leaves N1 at 232 negative pts either way; those are the mirrored jobs.)

## Surplus spikes that cost nothing NOW, and the default-suppression hack

4n F stuck at w4 = 20: N2 fires 5x, 1 is wanted, and the 4 surplus spikes generate NO
negative field -- correctly, since they drive nothing at that weight.  Raising w4 alone,
N2's train held fixed:

    w4=  20  N3 [33,133,233,333,433]  spurious []  missing [290]
    w4= 445  N3 [33,127,227,327,427]  spurious []  missing [290]
    w4=1100  N3 [33,123,223,323,423]  spurious []  missing [290]

So the surplus spikes never create SPURIOUS output spikes; they drag the MATCHED ones
progressively EARLY, and no w4 ever reaches 290 (that needs an N2 spike in 237..272, and
N2's arrivals are 112/212/312/412/512).  The mirrored negative jobs fire only on UNMATCHED
downstream spikes, so they are structurally blind to this.

FIELD_BASE, a constant tax on every existing spike (the deliberately hacky version):
    base 0.0  0/8    base 0.05  0/8    base 0.15  3/8    base 0.4  2/8    base 1.0  0/8
First thing to make the urgency consumer nonzero at all, and it recovers seed 2, which no
field config had ever solved.  But the narrow optimum is the tell -- below it the surplus
spikes are never deleted, above it the WANTED spike is suppressed too.  Baseline is 8/8.

PRINCIPLED VERSION (not implemented): a negative job for MATCHED-BUT-MISTIMED downstream
spikes, graded by how far each is dragged off its target.  That flags exactly N2's four
surplus spikes, needs no constant, and self-extinguishes when the timing is right.
FIELD_BASE is a uniform tax standing in for a signal that can be computed.

## FIELD_LOCAL: the weight decision is LOCAL to the neuron

demand_field uses the downstream to BUILD F[n].  Once it exists, "where should this neuron's
weights move" is a local question: F's positive bumps are the spikes it wants and their peaks
are when, so the decision is a comparison of two spike trains on the SAME neuron.

    bump with no spike    -> CREATE    positive at the bump
    spike with no bump    -> SUPPRESS  negative at the spike
    bump paired to spike  -> MOVE      positive at the bump, scaled by the error

Pairing is greedy-nearest, ONE-TO-ONE, and has NO distance cutoff.  That is what MATCH_WIN
got wrong.  Measured on 4n F seed 3 at iteration 5 (a near-solved state: N1 and N2 each
firing once, all output counts right, w0=156 vs a true 240):

    N1 bumps [73, 173, 220, 273, 370]        spike [442]   true 224  (220 is right)
    N2 bumps [11, 111, 211, 258, 311, 411]   spike [481]   true 256  (258 is right)

The FIELD HAD THE ANSWER.  But 442 is 72 from the nearest bump and 481 is 70, both just past
MATCH_WIN=60, so each was classified "unwanted outright" and given a full-strength deletion
job -- asking to delete the neuron's ONLY spike.  With one spike against five demands there
is nothing to delete; a far-off spike is mistimed, not surplus.  That flipped g(w0) from
+4.6e-06 (correct, raise it) to -5.6e-04, 122x larger and opposite, and N1 died at iteration
11.  Ablation at that state: no field +4.57e-06, FIELD_NEG=0 -2.35e-05, FIELD_BASE=0
-4.38e-04, as configured -5.57e-04.  96% of the gradient came from L1 < 0.

ONLY AN EARLY SPIKE GETS THE NEGATIVE.  Suppression removes drive and moves a spike LATER --
right for an early one, wrong for a late one.  Applying it to both directions inverted the
local rule outright: eligibility grows through an epoch, so N1's lone negative at 442
multiplied a far larger eps than all five positives at 73..370 combined, giving
g(w0) = -1.7e-05.  Restricting it to f < q gives +1.36e-06.

    4n F, 8 seeds, 3200 rounds
      baseline (field off)      8/8
      LOCAL 1.0                 5/8  [1,4,5,6,7]
      LOCAL 0.3                 5/8  [1,4,5,6,7]   identical -- GAIN-INSENSITIVE
      LOCAL 3.0                 1/8
      LOCAL 1.0 + BASE 0.15     5/8  [1,4,5,6,7]   identical -- FIELD_BASE now REDUNDANT
      prev best (ADD + BASE)    5/8  [0,1,5,6,7]

Two structural wins at equal score: FIELD_BASE is subsumed (the surplus rule does what the
uniform tax stood in for), and the result is flat over a 3x gain range where FIELD_BASE had
a narrow peak (0/8 at 0.05, 3/8 at 0.15, 2/8 at 0.4, 0/8 at 1.0).  Insensitivity to gain is
the signature of a signal rather than a fudge.  Different seeds though -- LOCAL gets 4 not 0,
the old path gets 0 not 4 -- so it is a different trade, not a strict improvement.

OPEN: at that state g(w1) is exactly ZERO under the local rule.  N2's demand spans 11..411
while N1's only spike is at 442, so eps(1,2) starts at 460 and never overlaps any demand.
w1 cannot move until N1 fires earlier, so the fix has to sequence through w0.

## CORRECTION: the FIELD_LOCAL result above was measuring the wrong thing

The 5/8 recorded for FIELD_LOCAL is WRONG and the conclusions drawn from it are withdrawn.

The replacement `L[n] = FIELD_LOCAL * local_demand(...)` was placed inside the per-neuron
loop, but the BACKWARD RELAXATION runs afterwards and does `L[n] = L[n] + Ln`.  Its own
comment calls Ln "the large timing term" (~0.46, it carries a 1/slope factor).  Measured on
4n F seed 3 at the stuck point:

    local_demand alone     pos +7.03e-03   neg -2.33e-04
    final L[2] as used     pos +7.32e-03   neg -4.74e-01     2000x larger

So the field was BURIED, not consulted, and no ablation of PIVOT_GAIN / OCCL_GAIN /
SHARP_GAIN / MOVE_GAIN moved the number because none of them was the source.

That also explains the w1 limit cycle exactly.  The huge negative was NOT at N2's spike --
it was a ramp starting at t=255 while N2 fired at 402.  w1 climbed while N2 was silent
(small positive field gradient, +1.3e-07), N2 fired once, the relaxation's timing term
delivered a negative 50x the whole field demand (-4.18e-01 vs +8.3e-03, g flipping to
-9.7e-06), and w1 crashed back.  A perfect cycle, driven entirely by the term that was
supposed to have been replaced.

Moving the replacement to the END of L construction (and zeroing Lmove, since the timing
term is part of what is being replaced) gives L[2] neg -2.33e-04 at t=402 ONLY, and
g(w1) = +1.03e-07 -- still positive in the iteration that used to slam it down.

    4n F, 8 seeds, 3200 rounds, field ACTUALLY driving the update
      baseline (field off)   8/8
      LOCAL 1.0              1/8  [3]
      LOCAL 0.3              1/8  [3]
      LOCAL 3.0              1/8  [3]

WITHDRAWN: "LOCAL matches the previous best", "FIELD_BASE is redundant", and
"gain-insensitivity is the signature of a signal".  Gain-insensitivity was never evidence:
before the fix 0.3 and 1.0 agreed because the field term was negligible; after the fix they
agree because L[n] is the ONLY term and Adam normalises a uniformly-scaled gradient, so the
gain is close to a no-op by construction.  And the one surviving seed is 3, the seed this
entire investigation was spent debugging.

THE REAL LESSON, WHICH IS METHODOLOGICAL.  Every field result in this log -- FIELD_XING,
FIELD_ADD, FIELD_BASE, FIELD_MIST -- was measured as an ADDITION to a signal that dominates
it by ~1000x.  None of them told us what the field can do; they told us how much noise the
relaxation tolerates.  This is the first measurement where the field alone sets the
gradient, and the answer is 1/8 against a baseline 8/8.  Any future field change must be
measured with the relaxation OFF, or the number means nothing.

## THE FIELD IS NOW A SEPARATE PATHWAY: field_trace.py

grad_trace's relaxation and the field cannot be compared while they share a gradient (the
relaxation outweighs the field ~2000:1, see the correction above).  field_trace.py is
standalone: it shares the SIMULATOR, kernel, eligibility and constants, and none of the
method.  _field_suite.py prints in the same format as _suite_mp.py so numbers compare.

    bumps(o) = the known target times, for an output
    L[n]     = local_demand(bumps(n), spikes(n))       signed, PURELY LOCAL
    F[n][q]  = sum_d sum_t L[d][t] * K(q -> t | d)     backward through plausibility
    bumps(n) = the positive peaks of F[n]

The signed L[d] propagates backwards through one kernel, so "d must cross here" and "d must
NOT" travel the same path and differ only in sign.  Matching happens only WITHIN a neuron.
K = timing feasibility x occlusion x weight-plausibility 1/(1+|wmin-w|/w).

TRUTH IS AN EXACT FIXED POINT: max|g| = 0.00e+00 at the true weights on 4n F, 3n D, chain.
grad_trace never had this -- its gradient does not vanish at the solution, which is the
entire reason KEEP_BEST exists.

Three bugs found by making the field actually drive the update:

(a) SILENCE IS NOT AN INSTRUCTION.  At truth the output matches, so L[output] = 0, so the
    feeding neuron's field is empty -- and its perfectly placed spike was suppressed at
    -7.0e-03.  An empty field means no information.  Also bumps_of() reads POSITIVE runs
    only, so the negative half of the field was being discarded entirely.
(b) DEAD_ZONE MUST BE 0.  Recovery is judged on exact spike times, so any dead zone is a
    basin the field calls finished and the scorer calls wrong.  3n D halted at
    N2 = [38,138,238,289,338,438] vs targets [33,133,233,293,333,433] -- every spike
    present, ~5 steps off, field empty, gradient identically zero.
        suite: 0/52 at DEAD=5,  1/52 at 2,  3/52 at 1,  16/52 at 0
(c) REACHABILITY IS A PREFERENCE, NOT A MASK.  4n F seed 0 sat at max|g| = 0 one step from
    correct (N1 at 223 against a true 224) because every request was placed before its own
    inputs could arrive: N1 asked for t=17 (first arrival 19), N2 for 237 (N1@223 arrives
    241), N3 for 290 (N2@277 arrives 295).  Each demand multiplied zero eligibility.
    Cause: the field was one nearly-flat positive run over 0..223 (0.9 at t=0 vs 1.0 near
    190) and the raw argmax landed at the useless end.
    Placing the request at the best REACHABLE time in each run fixes it.  MASKING the field
    by reachability does not: it is physically correct (N2 truly cannot fire at 256 while N1
    fires at 181) but it deletes the message, since a neuron that cannot fire where it is
    needed is exactly what its OWN upstream must fix -- zeroing the field there empties
    bumps, hence L, hence the upstream field.
        suite: 16/52 no reachability,  10/52 hard mask,  18/52 preference

    FIELD PATHWAY 18/52 at 800 rounds x 4 seeds   (grad_trace baseline 29/52)
      3n D 4/4   3n E 4/4   3n J 4/4   3n A 2/4   4n G 2/4   chain 1/4   4n F 1/4
      3-cycle 0/4  2-cycle 0/4  over-demand 0/4  5n H 0/4  3n L 0/4  8n M 0/4

Nothing here is tuned beyond the dead zone.  8n M, 3n L and over-demand have never been
looked at under this pathway at all.

## Strength must propagate on BOTH signs, and 4 seeds cannot measure it

A bump carries its own amplitude forward -- create and move both scale by u = F[q] -- but
suppression was pinned to SUPP*TH regardless of how faint the field was.  So a neuron whose
field was numerical dust still emitted a FULL-SIZE negative, and a weak request could never
outweigh it.  Scaling the negative by the same quantity the positives use (the strongest
spike the neuron is being asked for) makes the two sides symmetric.  At an OUTPUT the bumps
carry TH, so outputs are unchanged either way.

    F_SUPP_SCALE=1 (field-scaled)   17/52 at 4 seeds    41/104 at 8 seeds
    F_SUPP_SCALE=0 (fixed TH)       18/52 at 4 seeds    36/104 at 8 seeds

+5/104, and at 4 seeds it read as -1/52 -- the WRONG SIGN.  Four cases flipped between the
two 4-seed runs (chain 1->0, 3n A 2->0, 2-cycle 0->1, 5n H 0->1) on a change that should be
small, which was the tell.

METHODOLOGICAL: 4 seeds cannot resolve changes of this size on this suite, and a good deal
of the fast A/B work in this session was done at 4 seeds or on a SINGLE CASE (the FIELD_BASE
sweep, the reachability comparisons, every 4n F-only number).  Those all deserve the same
scepticism as the 17-vs-18 reading.  Use 8 seeds.

## What the two plotted series actually are

    F[n]   FIELD    what arrives from downstream: a density over times, "how much would a
                    spike here help (+) or hurt (-)"
    bumps           the positive RUNS of F, one requested spike each, at the best reachable
                    point of the run.  This is what gives the field a COUNT -- a density
                    alone cannot say "fire once"
    L[n]   DEMAND   those requests compared against n's ACTUAL spikes; the only thing that
                    touches the weights:  g[k->n] = sum_t L[n](t) * eps_k(t)

F[n] is built by propagating the SIGNED L[d] backwards, which is why "must fire here" and
"must not" share one path and one kernel.  Outputs get no field -- they are seeded straight
from their known targets -- so an output panel showing "asks for []" is a plotting artifact.

## Weight SIGN is given, not learned; and weights are named by their edge

SIGN.  A synapse is excitatory or inhibitory and does not change type, so the sign of each
weight is part of the problem statement rather than something to search over.  field_trace
now fixes wsign at the initial value and clamps only |w| to [20, 3000].  Two bugs are ruled
out at once:
  - a hard [20, 3000] clamp makes an INHIBITORY edge unrepresentable.  3n L's true w12 is
    -700; every seed initialises correctly negative (-378..-920) and the FIRST clip snaps it
    to +20.  grad_trace has the same clamp (grad_trace.py:2106) and also scores 3n L 0/8, so
    that case has never been solvable by either pathway -- both totals carried 8 dead cells.
  - opening the clamp to [-3000, 3000] instead lets a POSITIVE weight collapse through zero
    and sit there: over-demand seed 0 ended at w12 = -2 against a true +700.  (Not caused by
    the change -- with the old clamp the same seed ended at w12 = 20 -- but equally broken.)

    [20, 3000]         41/104 at 8 seeds, 3n L 0/8
    [-3000, 3000]      42/104,            3n L 2/8
    sign-fixed         41/104,            3n L 2/8      <- kept: same score, sound
NOTE the same fix is untested on grad_trace, where it is a one-constant change.

NAMING.  Weights are now wIJ for the edge N_I -> N_J (field_trace.wlabels / wstr), used in
every plot title and in prose.  Positional names (w0, w1, w2) have to be cross-referenced
against the edge list every time.  An underscore is inserted at two digits, so w1_10 stays
unambiguous.  over-demand's stuck state reads w01=252 w12=832 w02=250 against a true
w01=250 w12=700 w02=300 -- immediately locating the wrong synapse.

## over-demand: two independent stalls, and a rule the rewrite dropped

Representative failure (seed 1), max|g| = 6.98e-09, fully stuck:
    w01=252 w12=832 w02=250   true w01=250 w12=700 w02=300
    N1 fires [170, 370]   true [173, 373]           -- 3 steps early, nearly right
    N2 fires [174, 374]   target [140, 220, 399]    -- TWO spikes where three are needed

(a) w12 HAS EXACTLY ZERO GRADIENT.  Every demand time contributes 0:
        t=140  L=+7.00e-03 * eps=0        t=220  L=+7.00e-03 * eps=0
        t=373  L=-6.07e-03 * eps=0        t=399  L=+6.07e-03 * eps=0
    N1 fires at 170/370, arriving at 188/388; N2 fires at 173/373 with refractory shadows
    (173,195) and (373,395).  BOTH arrivals land inside and are discarded by the simulator,
    so N1 contributes nothing to N2 and its edge has no derivative at all.  N1 fires just
    BEFORE N2 every cycle and is permanently invisible to it.
(b) w02 CANCELS: +1.633e-07 for the missing spike at 140 against -1.699e-07 for the unwanted
    spike at 373, netting -3.1e-09.

(a) is a rule the rewrite dropped.  field_trace masks occluded CANDIDATE times -- it will not
request a spike whose arrival would be swallowed -- but it has no way to say "your EXISTING
spike is being swallowed, move it".  grad_trace had exactly that (FIELD_REFRAC's `blocked`
test, which pushed a spike whose arrival every downstream discards).  The field is asking N1
for a spike at 119, which would arrive at 137 safely before N2@173, so the DEMAND is right;
it simply cannot reach w12, and w01 is left to act on it alone against a near-zero signal.

## A MISSING SPIKE IS NOT NEGOTIABLE  (+12/104, the largest single gain)

Everything is summed into one L[n] and then projected onto the weights, so a CREATION demand
and an unrelated SUPPRESSION land on the same weight and annihilate.  Measured on
over-demand at its stuck point (w01=252 w12=832 w02=250, true 250/700/300; N2 needs
[140,220,399] and fires [173,373], so 220 has nothing serving it):

    create term at 220                       w02 = +1.67e-07
    suppression from pairing 373 with 399    w02 = -1.70e-07
    net                                      w02 = -3.14e-09   WRONG SIGN

and a 2-D grid over N2's two input weights (w01 frozen, everything enumerable) says w02 must
RISE 250 -> 300 and w12 FALL 832 -> 640, taking the output from the wrong spike COUNT to a
mean timing error of 0.67 steps.  The landscape is smooth, single-basined, and the stuck
point sits 5 units below the correct-count band -- so the improvement is not just reachable,
it is trivially reachable, and the method was pointing away from it.

RULE: if a demand has nothing firing for it at all, every weight that could produce it must
go UP, and no timing correction elsewhere may cancel that.  You cannot retime a spike that
does not exist, so a COUNT error takes precedence over a TIMING error on the same weight.
    g = np.where(g_create > 0, np.maximum(g, g_create), g)
This is general -- it does not care how many input neurons there are.  (The narrower rule
"no negatives while the spike count is short" was rejected as too specific: it breaks with
more input neurons, where a neuron can be short of spikes AND have a genuinely surplus one.)

    field pathway 800 rounds x 8 seeds
      before          41/104
      create floor    53/104
        4n F  3/8 -> 8/8 (perfect)   chain 2/8 -> 5/8   4n G 3/8 -> 6/8
        over-demand 0/8 -> 1/8 (first ever)   nothing regressed
Beats the window curriculum (50/104) and is a real fix rather than an easier problem, so the
two are independent and may stack.

STILL OPEN, and not addressable by any demand rule:
  - w12 on over-demand has eps(1,2) = 0 at EVERY demand time.  N1 fires 170/370, arrives
    188/388, and N2's own refractory shadows (173,195) and (373,395) swallow both.  The
    eligibility is empty, so the edge has no derivative at all.  field_trace masks occluded
    CANDIDATE times but cannot say "your EXISTING spike is being swallowed, move it" --
    grad_trace had exactly that (FIELD_REFRAC's `blocked` test) and the rewrite dropped it.
  - 3-cycle 0/8, 2-cycle 0/8, 8n M 0/8.  2-cycle regressed 2 -> 0 earlier and has not been
    looked at.

METHOD NOTE: the 2-D slice enumeration (_plot_wslice.py) answers a question no gradient can
-- does an improvement EXIST nearby, and does the method point at it.  Freezing all but two
weights makes the subproblem exhaustible.  Worth doing before theorising about any stuck point.

## The create floor must ALSO be applied to PROPAGATION  (+6/104)

The gradient floor already covers hidden neurons -- Lc is built for every n and the
max(g, gc) applies to every edge -- but the information can be destroyed BEFORE it arrives.
F[n] = sum_d sum_t L[d][t] * K sums the SIGNED demand, so a downstream creation and a
downstream suppression annihilate in the FIELD, one level above where the gradient floor
operates.  Measured across the suite, 22 of 92 (hidden neuron, seed) cases lose at least one
create request that way -- concentrated in exactly the failing cases (chain, 3-cycle,
2-cycle, 8n M; 3n J seed2 lost all 4 of its requests, 2-cycle seed0 both of its 2).

Propagating the create part on its own and flooring the field with it is the same rule
restated at the propagation step:  F = np.where(Fc > 0, np.maximum(F, Fc), F)

    41/104  no floors
    53/104  gradient floor
    59/104  + propagation floor
      chain 5/8 -> 8/8   4n G 6/8 -> 8/8   3-cycle 0/8 -> 1/8   5n H 5/8 -> 6/8
      over-demand 1/8 -> 0/8 (lost its single seed)
    five cases now 8/8: chain, 3n D, 3n E, 4n F, 4n G, 3n J

## Gauss-Newton on an output's inputs: REJECTED, 59 -> 49/104

Reasoning that led to it: with the spike count already correct, over-demand still stalls
because timing demands CONFLICT on a shared weight --
    t=136 "delay the first spike"   -2.076e-08
    t=399 "advance the last spike"  +2.550e-08
    sum                             +4.857e-09   -- up, when w02 must come DOWN (318 -> 300)
V_n is linear in the incoming weights, so A dw ~= r is well posed and should find the joint
direction instead of summing conflicting pulls.

It does not, because the residuals are NOT jointly satisfiable by that neuron's inputs.  At
that point the GN step is w12 = +30.4, w02 = +6.4 -- both UP, both needing to come down --
and it is solving the system correctly: N2's third spike is late because N1 is late (381 vs
373, w01 = 247), and w01 is not in N2's system at all.  eps(1,2) is 0 at 399 as well, so
within N2's incoming weights the only lever is w02, and raising it advances the third spike
while making the first worse.
    chain 8->6   4n F 8->6   4n G 8->4   3n A 2->1   3n D 8->7   3n L 2->1   3-cycle 1->0
    5n H 6->8 is the ONLY gain -- the pure feedforward chain, where each output spike has a
    clean attribution and the local system really is satisfiable.
First order makes a small compromise; GN confidently solves an unsatisfiable system and
takes a large wrong step.  Kept behind F_GN=0.

NOTE the conflicting-timing-demand problem is REAL and still unsolved; this was just the
wrong fix for it.

## The 50n numbers are KEEP_BEST harvesting a random walk (`_plot_rounds.py`)

More rounds do NOT help on the sub-critical 50-neuron nets.  Sampling the LIVE iterate every
100 rounds over 6400 rounds, 4 seeds of 50n A:

    seed 0: init ok=0.50   best 0.50 @round 1     rounds 4400+ mean ok=0.020
    seed 1: init 0.10      best 0.20 @round 100   mean 0.025
    seed 2: init 0.30      best 0.40 @round 100   mean 0.025
    seed 3: init 0.60      best 0.60 @round 1     mean 0.015

Count agreement collapses to ~0 by round ~1800 and stays there for the remaining 4600.  On two
of four seeds the INITIALISATION (truth x U(0.5,1.5)) is never beaten.  The lr is a sawtooth --
DECAY is indexed by `ait`, which RESTART_EVERY resets -- so this is not a stalled anneal.

Yet KEEP_BEST over 800 rounds returns an iterate with 7/10 counts right on seed 0, better than
anything the every-100 sampling saw.  Both are true: the good states are TRANSIENT SPIKES
between samples, not a basin the optimiser settles into.  So the published 50n figures
(count-ok 57%) are a best-of-800 draw from a walk that drifts the wrong way, not convergence.
Any future 50n number must be reported alongside the live-iterate curve.

MECHANISM, from the same run's third panel: mean |w| / mean |w_true| starts at 1.0 by
construction, drops to 0.66-0.74 within ~1000 rounds, recovers only to ~0.83 and plateaus.  The
first thing training does is shrink every weight by a quarter to a third, and it never comes
back.  That predicts exactly what the rasters show -- output spikes late (mean +15.6 / +18.7 /
+15.3 on A / B / C) and 30 targets missed against 2 spurious across the three nets.  Under-driven
neurons reach threshold late, and the marginal ones not at all.

So the field's aggregate demand on these nets is net NEGATIVE.  On the small suite this never
showed because CREATE_FLOOR could always find the one missing spike; with 232 weights and 98%
of them sub-critical, the shrink is spread thin and the floor never fires.

## The 50n suppression bias is the HIDDEN-NEURON TARGET GUESS (`_demand_sign.py`, `_eps_support.py`)

Tested the "aggregate demand is net negative" claim above.  It is true but badly localised, and
the mechanism I first proposed for it was wrong.

Demand mass at the initial weights (truth x U(0.5,1.5)), split by whether the neuron has a
ground-truth target:

    case        L+/L- overall   OUTPUTS   HIDDEN   hidden share of all L-
    50n A s0        0.94         2.97      0.74            91%
    50n A s1        0.32         1.79      0.20            93%
    50n B s0        0.48         2.95      0.34            95%
    50n C s0        0.30         4.51      0.20            98%
    4n F  s0        3.15         9.40      1.64            81%
    4n F  s1        4.72         6.00      3.52            51%

The OUTPUTS -- the only neurons with a real target -- ask for the right thing everywhere
(L+/L- 1.15-4.51, create-dominated).  Every bit of the suppression bias sits on the 40 HIDDEN
neurons, whose spike counts are GUESSED from bumps, and where the guess reads 0.13-0.74: fire
less, almost everywhere.  With 40 hidden against 10 outputs the guess outvotes the evidence
about 9:1, which is why both weight populations erode toward zero (`upd exc` -0.05..-0.13,
`upd inh` +0.11..+0.14 -- opposite signs, both meaning "toward zero"; final |w| exc 0.79-0.94,
inh 0.22-0.46).  Inhibition collapses hardest.

The small suite hides this because it has 2-3 hidden neurons, not 40, AND because its hidden
guess is create-dominated anyway (1.55-3.52).  The bias is not in the demand rule; it is in
what the rule is applied to when most of the network is unobserved.

WRONG HYPOTHESIS, recorded so it is not retried: I expected eligibility truncation (the
`_eps_at` comment at field_trace.py:621) to be the culprit -- create requests landing where the
PSP was cut by a misplaced reset, multiplying to zero.  It is real and costs the positive share
a further 1.5-5x on EVERY case, but it does not distinguish these nets: the fraction of create
mass landing on dead eligibility is 21-39% on the 50n nets and 41-76% on the small cases where
the field works.  It is a universal tax, not the differentiator.

NOTE `mean |w|` alone is ambiguous as a diagnostic and the earlier entry leans on it too hard:
weakening an inhibitory weight also lowers mean |w| while RAISING drive.  Report exc and inh
separately, or report in drive units (upd > 0 always means more drive, since eps > 0 on every
edge and upd = +step*Adam(g)).

## `14n P` / `14n Q` -- a 9x faster reproduction of the 50n failure (`_minirepro.py`)

Two ingredients are needed, and the first one alone is NOT enough:

1. a HIDDEN-NEURON GUESS that is suppression-dominated, and
2. enough CONSTRAINT DENSITY (target output spikes per weight) that the resulting erosion
   cannot be absorbed.

First search, N=10-12 with 1-2 outputs: 26 of 50 usable nets matched the demand signature
exactly -- hidden L+/L- 0.41-0.89, hidden share of L- 89-100%, outputs still create-dominated
at 3.0-6.3 -- and the weights DID erode (|w| exc 0.83-0.95, inh 0.51-0.87).  count-ok was
nevertheless 1.00 on every one.  With 24 edges and one output the system is underdetermined
enough that many weight settings reproduce the target counts, so a 15% erosion costs nothing.

Holding the bias fixed and raising density (N=14-26, 4-6 outputs, 1040 steps) reproduces it:

    case    hid L+/L-  hid share  out L+/L-   exact  count-ok  |dt|
    14n P      0.08        88%      1.21       0/8      81%    35.7
    14n Q      0.27        96%      3.66       0/8      47%    25.3
    50n A/B/C  0.13-0.74   90-98%   1.15-4.51  0/24     57%    26.8

14n Q is the closer match on both metrics.  14 neurons, 36 edges, 4 outputs, 24 target output
spikes, 94% sub-critical, 3 inhibitory neurons; truth verified as an exact fixed point
(max|g| = 0).  800 rounds takes 21s against 191s for 50n A -- 9x -- so the defect is now
iterable.  Both installed in _diag.CASES and _suite_mp.CASES with CASE_STEPS 1040.

CAUTION when using these: 0/8 exact means the graded metric is the only signal, and both cases
were SELECTED for a demand signature rather than for failing.  A fix that moves count-ok on
14n Q without moving hidden L+/L- has probably not addressed the mechanism.

## Reading a case in causal order (`_order.py`, `_plot_topo.py`, `F_ORDER=auto`)

`min_fas_order()` solves minimum feedback arc set EXACTLY by subset DP (O(2^N * N)), which is
cheap at N <= 26 and means the backward-edge count is a proved minimum rather than whatever a
greedy pass happened to give.  `_plot_topo.py` draws the result as an arc diagram -- forward
edges below the line, backward above -- so the objective is literally the number of arcs on
top.  `F_ORDER=auto` puts _plot_field_case.py's panels in the same order; `F_ORDER=a,b,c`
overrides.

On 14n Q the minimum is ONE backward edge (N13->N5, inhibitory).  That is not the DP being
clever: _bignets builds a DAG and then adds a few feedback edges, so numeric order is already
near-optimal and the DP only found the 11/12 swap that removes the second one.  Expect the
same on any generated net; the tool earns its keep on cases with real recurrence.

WHAT THE ORDERED FIGURE SHOWS on 14n Q seed 0 (ft_14n_Q_stuck0.png):

  - the four OUTPUTS carry no field at all, only demand.  Correct by construction -- field is
    propagated backward FROM outputs, so outputs are sources of it, not carriers.
  - peak |demand| is 1.1e-2 to 5.1e-2 on the hidden neurons and exactly 7.0e-3 on every one of
    the four outputs.  The largest hidden demands are ~7x the outputs', and there are 9 hidden
    against 4 outputs.  This is the domination finding visible directly rather than as a
    summed ratio.
  - N13 is DEAD: 0 spikes against 3 targets.  Its create demand is present (three positive
    spikes at 328/585/872) but it asks for NOTHING -- bumps_of() returns [] -- so the request
    never becomes a count.  A dead output is the case CREATE_FLOOR exists for, and on this
    case it is not reaching it.  Worth chasing next.
  - in causal order the drift compounds visibly: N1/N2 sit directly on the input and are off
    by only -2 / +19 steps, and each subsequent layer is further out.

## bumps_of() gives ONE request per positive run, however wide (`_bumpdeficit.py`, `_plot_runs.py`)

The count the field hands downstream is the number of positive RUNS of the field, not the
number of spikes those runs stand for.  bumps_of()'s own docstring states the problem -- "an
86-step-wide band is better satisfied by five spikes than by one" -- but nothing in it acts on
the width; each run contributes exactly one request, at its centroid.

14n Q seed 0, hidden neurons, at the stuck weights:

    neuron  runs  asks  true | run widths                true spikes inside each run
        N7     1     1     5 | [853]                     [5]
        N9     2     2     9 | [878, 99]                 [8, 1]
        N1     4     4    10 | [97, 23, 54, 103]         [1, 0, 1, 1]
        N4     5     5     6 | [130, 6, 17, 36, 63]      [0, 0, 0, 0, 1]

N9's field is positive across 0..878 -- one run, 8 of its 9 true spikes inside it, ONE request
at t=355.  N7 is the same shape: 853 steps, 5 true spikes, one request.  REFRAC=22, so these
spans could hold dozens of spikes; nothing about the width is being read.

TWO OPPOSITE ERRORS, and they must not be netted:

    UNDER-ask (wide runs holding several spikes)  11 spikes never requested
    OVER-ask  (runs holding no true spike at all) 19 spurious requests

Netted these give -8, which reads as "the field roughly over-counts by 27%" and is meaningless.
The truth is that both halves are badly wrong: the neurons that need many spikes ask for one,
while a scatter of narrow runs asks for spikes that should not exist.  Any future metric on
bump counts must report the two separately.

This is a plausible upstream cause of the hidden-neuron suppression bias: a hidden neuron whose
true count is 9 but which asks for 2 will be told it is firing too much for the rest of the run,
which is exactly the "fire less" guess measured in the section above.  Deriving the count from
the field's AREA (or placing a request per expected inter-spike interval within a run) rather
than from the run count is the obvious thing to try -- NOT yet tested.

## Would smoothing the field fix the under-count?  NO -- and no extraction rule does
(`_counting_rules.py`)

STALENESS FIRST.  Two things in the tree are called convolution and only one is live:

  - `back_corr` (grad_trace.py:302) -- m[s] = sum_dt Ld[s+dt] K[dt] as convolve(Ld, K[::-1]).
    LIVE, called from the trace gradient, but a pure vectorisation of a loop.  Nothing to do
    with counting.
  - `FIELD_SMOOTH` (grad_trace.py:868) -- Gaussian-convolves the urgency density so it says
    "a spike is wanted around here" rather than "at exactly this sample".  This is the one
    that sounds relevant, and it is STALE: default 0.0, it lives inside demand_field(), and
    demand_field() is only reached when FIELD / FIELD_XING / FIELD_ADD / FIELD_LOCAL is
    non-zero -- all four default to 0, so it never runs.  field_trace.py has no smoothing at
    all.  Anything built on it would be built on an untested path.

MEASURED on 14n Q seed 0, same fields, rules scored against each other.  UNDER = true spikes
with no request within REFRAC of them; OVER = requests with no true spike within REFRAC.  The
9 hidden neurons carry 74 true spikes.

    rule                    requests  UNDER  OVER   N9 asks (needs 9)
    A current (1 per run)         38     64    28   2
    B smoothed sigma=5            33     63    22   2
    B smoothed sigma=10           30     63    19   2
    B smoothed sigma=20           25     64    15   2
    B smoothed sigma=40           19     66    11   1
    C peak-pick >= REFRAC         76     53    55   15
    D smooth 10 + peak-pick       40     61    27   7

Smoothing moves OVER 28 -> 11 and leaves UNDER at 63-66.  That is structural, not a tuning
failure: convolution can only MERGE positive runs, never split one, so it cannot add a request
inside N9's 878-step plateau.  It attacks the spurious-narrow-run half only.

Peak-picking is the only rule that moves UNDER (64 -> 53) and it doubles OVER (28 -> 55): on N9
it finds 15 maxima where 9 spikes are wanted.  Composing the two just interpolates between
them; there is no setting where both halves improve.

CONCLUSION: the count is not recoverable from this field by ANY post-hoc extraction.  Every
rule leaves at least 53 of 74 hidden spikes unrequested, because N9's field is a plateau rather
than nine bumps -- the temporal structure is not in the signal to be extracted.  The fix has to
be upstream, in how the field is built, not in how bumps are read off it.  Consistent with the
project's pattern: the interventions that ever gained were structural invariants, not better
readers of an existing quantity.

## DENSITY mode: the field AS the demand, no bump extraction (`F_DENSITY`)

Proposed fix for the one-request-per-run under-count: drop bump extraction on hidden neurons
entirely and let the demand BE the field -- every positive timestep a request weighted by its
height, every negative one a suppression.  Outputs keep their real targets.  `F_DENSITY=1` uses
raw height (so a wide run carries proportionally more create mass -- the point); `F_DENSITY=2`
rescales each run to the mass of one bump, isolating WHERE the demand sits from HOW MUCH there
is.  On 14n Q's N9 the demand goes from 5 nonzero samples to 1015.

Suite, 18 cases x 8 seeds, 800 rounds:

    mode              exact    |Dcount|  count-ok   |dt|
    0 bumps (base)   86/144      0.37      88%      8.06
    1 density raw    49/144      0.17      94%     14.96
    2 density mass   44/144      0.13      95%     15.46

THE TWO METRICS MOVE IN OPPOSITE DIRECTIONS, and per case it is very clean:

    COUNT FIXED                      TIMING DESTROYED
    14n Q  ok 47% -> 94% -> 100%     4n F   exact 8 -> 0   |dt| 0.0 -> 32.8
    14n P  ok 81% -> 97% -> 100%     4n G   exact 8 -> 0   |dt| 0.0 ->  1.8
    5n H   exact 5 -> 8, ok 62->100  chain  exact 8 -> 3   |dt| 0.0 -> 21.1
    3n A   exact 3 -> 8, |dt| 4.2->0 3n D   exact 8 -> 2   |dt| 0.0 -> 19.3
    4n V   ok 88% -> 100%            3n J   exact 8 -> 4

So the diagnosis was right and the fix is real: on the two cases built to expose the
under-count, density takes count agreement to 100%, which no bump rule reached.  It also
confirms the count really was the binding constraint there.  But a density cannot SAY WHEN --
the cases already solved exactly lose their placement, |dt| going 0.0 -> 20-33.  This is the
same trade bumps_of()'s docstring describes, now measured from the other side.

The two are complementary rather than competing: density carries the COUNT, bumps carry the
PLACEMENT.  A hybrid (density to fix the count, then bumps to place; or density restricted to
neurons whose field is a wide plateau) is the obvious next thing and is NOT yet tested.

## FASTPROP: exact batched backward propagation

The propagation loop ran one plausibility solve per requested time per edge, which is fine for
~5 bumps and fatal for ~900 dense samples: 281.7s per 100 rounds against 4.2s for bumps.

Not a convolution.  back_corr's trick needs K(q->t) to depend only on t-q; this kernel carries
1/(1 + |need(t)/HK(t-q) - w|/TOL) in which t and t-q enter jointly and non-separably, since
need(t) = TH - other(t) is the drive the OTHER edges supply at t.  Binning need(t) recovers a
convolution per bin but is an approximation, and with KWIN=400 the arithmetic was never the
bottleneck -- the Python loop was.

Batched instead, and exactly: requests grouped by EPOCH (between two real spikes of the
downstream neuron), where the candidate window `lo` is constant, so each epoch is one
(n_t x n_q) array and the accumulation is a single mat-vec.

    dense, per-request loop   281.7 s / 100 rounds
    dense, batched             23.5 s / 100 rounds     12x

_verify_prop.py checks it against the loop over 5 cases x 2 density modes x 2 seeds:
worst |dF| = 4.5e-13 on fields of order 1e-2, worst |dg| = 1.1e-16.  Summation order only.
FASTPROP defaults ON with DENSITY and OFF on the bump path, where batching loses slightly
(4.6s vs 4.2s) because there is nothing to batch.

## Splitting the roles: bumps for timing, density for create/suppress (`F_DENSITY=3`) -- WORSE THAN BOTH

Suite, 18 cases x 8 seeds, 800 rounds:

    mode                          exact    |Dcount|  count-ok   |dt|
    0 bumps (base)               86/144      0.37      88%      8.06
    1 density raw                49/144      0.17      94%     14.96
    2 density mass-normalised    44/144      0.13      95%     15.46
    3 bumps=timing, density=both 38/144      0.76      65%     13.02

Mode 3 does not interpolate between its parents -- it is worse than EITHER on count, which is
the one thing the density was reliably good at.  chain goes 8/8 -> 0/8 with |Dcount| 1.88.

CAUSE: dropping the bump rules dropped "unclaimed spike -> SUPPRESS" with them, and that was
the only reliable way to DELETE a spike.  The density's negative half exists only where
something downstream actively sent negative demand, which is far rarer than "no bump claimed
this spike".  The signature fits exactly: |dt| lands between the two parents (13.0 against 8.1
and 15.0) because the timing signal survived, while |Dcount| blows past both because deletion
did not.

Also settled: FULL DENSITY IS NOT WORTH IT over the mass-preserving 9-point subsample --
dense 51/144 and 45/144 against subsampled 49 and 44, count-ok identical, |dt| within 0.8.
The FASTPROP 12x buys the ability to run the experiment, not a better answer.

NEXT (running): F_DENSITY=4, creation from the density and BOTH deletion and timing from the
bumps.  If the diagnosis above is right this should recover the count.

## CORRECTION: modes 3-6 were dead code, and the split-role hybrid (F_DENSITY=4) is the best yet

TWO BUGS made every earlier mode-3/4/5/6 number meaningless.  Recorded because both are easy
to reintroduce:

  1. the mode-1/2 early return read `if DENSITY and field is not None:` -- truthy for EVERY
     nonzero mode, so 3-6 returned there and never reached the bumps-for-timing code.  They
     were silent aliases of mode 1.  Byte-identical suite output across three "different"
     modes is what exposed it; identical results are a bug signal, not a null result.
  2. the bump-derived create rule was gated on DENSITY alone, but it runs for OUTPUTS too,
     and outputs pass field=None so the density never replaces it.  Modes 3/4 therefore
     deleted the output create rule with nothing in its place: chain's output sat at 0 spikes
     against 4 targets with an identically empty field.

So the earlier readings map as: "mode 3" = mode 1 + broken outputs; "mode 4" (pre-fix) = the
same; "mode 4 solves the count" = mode 1, measured on the dev subset.  The conclusions drawn
from modes 5 and 6 (create-only channel adds nothing; scale swamps the move term) are
UNSUPPORTED -- those paths never executed.

DEV SUBSET (chain, 3n A, 4n F, 5n H, 14n Q x 4 seeds, 800 rounds; `_field_suite.py 800 4 dev`),
all three now genuinely distinct:

    mode                exact   |Dcount|  count-ok   |dt|
    0 bumps             9/20      1.38      56%      7.58
    1 density          10/20      0.06      97%     14.01
    4 HYBRID           11/20      0.09      91%      9.06

Mode 4 = creation from the density, deletion and timing from the bumps.  It keeps essentially
all of the count recovery while giving back most of the timing density costs, and takes the
best exact score.  Per case:

    14n Q   |Dcount| 2.25 -> 0.00, count-ok 31% -> 100%   (better than density's 0.12 / 94%)
    5n H    1/4 -> 4/4        3n A  0/4 -> 4/4
    chain   4/4 -> 3/4        4n F  4/4 -> 0/4, count-ok 100% -> 25%   <-- the one regression

4n F is where splitting the roles actively hurts: bumps solve it exactly, density holds its
count, the hybrid loses the count.  That is the next thing to look at, and it is now a 2-minute
test.

NOT YET VALIDATED ON THE FULL SUITE -- the dev subset was chosen for where the pathways
disagree, so it over-represents both the wins and the losses.

## Full 21-case baseline (mode 0), for comparison against any variant

    python3 _field_suite.py 800 8          # all 21 cases, ~53 min

    TOTAL exact 86/168   |Dcount| 0.61   count-ok 72%   |dt| mean 15.95 median 9.8 max 158

    50n A  0/8  |Dcount| 0.74  count-ok 64%  |dt| 29.8
    50n B  0/8  |Dcount| 0.85  count-ok 54%  |dt| 22.2
    50n C  0/8  |Dcount| 0.90  count-ok 55%  |dt| 27.8
    14n P  0/8  |Dcount| 0.38  count-ok 81%  |dt| 35.7
    14n Q  0/8  |Dcount| 1.81  count-ok 47%  |dt| 25.3

The 50n figures reproduce the originals exactly (64/54/55%), so none of the DENSITY / FASTPROP
work has perturbed the default path -- worth having as a regression check.

COST: mode 0 over all 21 cases x 8 seeds. NOTE the "53 min" figure here was INFERRED from
when a checkpoint happened to be observed, not measured -- mode 0 had merely finished by then.
The first measured value is 9:39 wall clock, after the optimisation pass below; per-case
speedups of 1.6-2.1x put the pre-optimisation time near 16-20 min, so 53 min was wrong.  A density mode runs
~3.6x slower per round (14n Q: 4.2 vs 15.0 s per 100 rounds) and the 50n cases are likely worse
still, so a three-mode comparison at 8 seeds is a 5-7 hour job.  Trim to 4 seeds, or run the 18
small cases and the three 50n cases as separate jobs.

SETUP MISTAKE worth not repeating: piping each mode through `tail -24` means nothing prints
until that mode finishes, so a run killed part-way through leaves NO partial results and no way
to watch progress.  Write per-case lines as they complete instead.

## Optimisation pass: 2.1x on the bump path, 1.8x on the density path, bit-identical output

Profiling first, on 50n A (which dominates suite cost): `_plausibility` was 60% of runtime.
Three changes, each verified before the next.

1. PRECOMPUTED REFRACTORY-SHADOW MASK (`_occ_mask`).  The occlusion test looped over every
   reset of the downstream neuron -- ~30 iterations, 3 array ops each -- inside every call.
   It depends only on the ARRIVAL time and that neuron's own spikes, not on the request time,
   the co-drive or the weight, so it was being rebuilt identically ~5200 times a round.  Built
   once per neuron and indexed.  1.3-1.5x.

2. SPLIT THE KERNEL, CACHE THE EDGE-INDEPENDENT HALF (`_timing` / `_weight_term`).  TIMING and
   OCCLUSION belong to (downstream neuron, request time); only the WEIGHT term varies per
   edge.  With fan-in 5 the same arrays were built five times.  Cached per (d, t), cleared
   each sweep.  Measured reuse: 6621 `_timing` calls against 31260 `_weight_term` calls.
   The same split applied to the batched path as `_prop_plan`, cached per neuron.

3. CARRY THE SIMULATION FORWARD.  KEEP_BEST scores the weights at the END of a round via
   out_err(), and the next round opened by simulating the SAME weights again -- two full
   simulations per round where one suffices.  The simulator is 32% of runtime on 14n Q and 6%
   on 50n A.

MEASURED (s per 100 rounds, before -> after):

    case     mode 0                 mode 4 (density)
    50n A    23.39 -> 11.04  2.12x   37.03 -> 20.50  1.81x
    14n Q     2.61 ->  1.65  1.58x    4.13 ->  3.11  1.33x
    8n M      1.53 ->  0.90  1.70x
    4n F      0.25 ->  0.21  1.19x

CORRECTNESS.  Not assumed -- checked at three levels:
  - `_verify_occ.py`: the mask against the loop it replaces, on 300 random spike trains
    (identical), then fields and gradients end to end over 6 cases x 4 mode configurations
    x 2 seeds: worst difference 0.000e+00.
  - `_verify_kernel.py`: `_timing` + `_weight_term` against the original `_plausibility` over
    4000 random requests: worst |diff| 0.000e+00, no None-disagreements.
  - the dev subset reproduces every published figure to the digit, per case and in total
    (mode 0 9/20 |dc| 1.38 ok 56% |dt| 7.58; mode 4 11/20 |dc| 0.09 ok 91% |dt| 9.06).

MEASURED BEFORE IMPLEMENTING, AND REJECTED: trimming `qs` to the KWIN window looks like a
large win (candidate windows can span a whole epoch while only the last 400 samples can be
nonzero) but mean |qs| is 78 against 75.3 useful -- 3.4% waste.  Not worth it; the arrays are
small enough that numpy CALL COUNT, not array size, is what costs.  That is also why FASTPROP
still loses on the bump path at 50n scale (27.8 vs 23.4 before this pass): batching helps only
when there are many requests per epoch, and the bump path has ~5 per neuron.

FULL-SUITE REGRESSION after the optimisation pass: all 21 cases x 8 seeds reproduce the saved
baseline EXACTLY -- 86/168, |Dcount| 0.61, count-ok 72%, |dt| 15.95, and every per-case line
identical including 50n A/B/C at 64/54/55%.  Wall clock 9:39 on 16 cores (`time
F_DENSITY=0 python3 _field_suite.py 800 8`), which is the first properly measured figure for
this suite.

## Why a strongly negative field does nothing (4n F, mode 1) -- two causes, both measured

At mode 1's stuck point on 4n F seed 0 (w01=2407 against a TRUE 240, N1 and N2 each firing 5
times against a true 1) the field on N2 is large and negative over most of the run, and max|g|
is 1.44e-07.  Two independent reasons it cannot act:

1. IT MOSTLY MULTIPLIES ZERO.  eps support is 35 of 520 samples on both w01 and w12, and only
   5.1% / 11.9% of the negative demand MASS lands where eps > 0.  Eligibility truncates at
   every postsynaptic reset, so a neuron firing 5x too often has ~7-sample epochs.  This is
   self-reinforcing: over-firing shortens the epochs, which starves the gradient that would
   correct the over-firing.

2. WHAT SURVIVES IS SIGN-FLIPPED BY CREATE_FLOOR.

       g[w01]  CREATE_FLOOR=1  +8.68e-08     CREATE_FLOOR=0  -9.67e-07
       g[w12]  CREATE_FLOOR=1  +1.11e-08     CREATE_FLOOR=0  -1.34e-07

   The floor takes g = max(g, gc) wherever the create-only gradient is positive.  It was built
   for SPARSE bump-derived creates ("a missing spike is not negotiable"); in density mode the
   create channel is positive at nearly every sample, so the floor fires nearly everywhere and
   systematically overrides suppression.  That is how w01 reached 10x its true value.

CONFIRMED on 4n F, 8 seeds:

    mode  CREATE_FLOOR   exact  count-ok  |dt|
       0             1     8/8      100%   0.0     <- the bump path NEEDS the floor
       0             0     0/8       62%   9.0
       1             1     0/8      100%  32.8
       1             0     3/8       50%   7.2     <- 4.5x better timing without it
       4             1     0/8       25%  15.6
       4             0     0/8       25%  15.8     <- mode 4 is UNAFFECTED

So the floor explains mode 1's 4n F failure and NOT mode 4's -- the two are different bugs.
And it is not a free fix: count-ok falls 100% -> 50% on mode 1, and mode 0 collapses without it.

NEXT (untested): gate CREATE_FLOOR to fire only where the create demand is genuinely UNSERVED
(no spike within REFRAC), instead of wherever gc > 0.  That keeps its purpose on sparse bump
requests while stopping it from overriding suppression at every sample under a dense field.

## Mode 1 on the full 21-case suite: the 50n prediction holds

    TOTAL exact 49/168   |Dcount| 0.23   count-ok 84%   |dt| 19.29     (mode 0: 86/168, 0.61, 72%, 15.95)

    case    count-ok 0 -> 1     |Dcount| 0 -> 1
    50n A      64% -> 74%         0.74 -> 0.33
    50n B      54% -> 69%         0.85 -> 0.38
    50n C      55% -> 86%         0.90 -> 0.17
    14n P      81% -> 97%         0.38 -> 0.03
    14n Q      47% -> 94%         1.81 -> 0.16

|Dcount| roughly halves on all three 50n nets.  The chain from the original output raster
(systematic late/missing bias) through the hidden-neuron suppression measurement to the
one-request-per-run under-count PREDICTED this, and it holds on the cases the prediction was
made about.  The cost is timing: exact 86 -> 49 and |dt| 15.95 -> 19.29, with chain 8/8->3/8,
3n D 8/8->2/8, 4n F 8/8->0/8, 4n G 8/8->0/8 all losing placement while keeping count-ok 100%.

## FULL 21-CASE THREE-WAY COMPARISON (8 seeds, 800 rounds) -- the dev subset did NOT generalise

    mode              exact     |Dcount|  count-ok   |dt|      wall
    0 bumps          86/168       0.61      72%      15.95    10:18
    1 density        49/168       0.23      84%      19.29    21:40
    4 hybrid         56/168       0.20      86%      16.13    20:20

On the 5-case dev subset the hybrid led on exact (11/20 against 9/20).  On the full suite
mode 0 wins 86 to 56.  The subset was chosen for maximum disagreement between the pathways and
flattered the hybrid; treat any dev-subset ranking as a direction to investigate, never as a
result.

WHAT IS REAL: the hybrid dominates density on every axis, and against the bump baseline it
matches TIMING (16.13 vs 15.95) with 3x better COUNT (0.20 vs 0.61).  It cannot convert that
into exact recoveries because "exact" needs both at once.

The split is by case size, and it is clean:

    HYBRID WINS (the hard cases)          HYBRID LOSES (small, already exact under bumps)
    14n P  |Dc| 0.38 -> 0.00, ok 81->100%   4n F   8/8 -> 0/8
    14n Q  |Dc| 1.81 -> 0.00, ok 47->100%   4n G   8/8 -> 0/8
    50n A  ok 64% -> 78%                    3n D   8/8 -> 2/8
    50n B  ok 54% -> 79%                    3n R   8/8 -> 2/8
    50n C  ok 55% -> 82%                    chain  8/8 -> 4/8
    3n L   2/8 -> 7/8                       3-cycle 3/8 -> 0/8
    3n A   3/8 -> 8/8                       over-demand 2/8 -> 0/8
    5n H   5/8 -> 8/8

3n L reaching 7/8 is worth noting: that case was unsolvable by EITHER pathway for most of this
project's history (see the sign-preserving clamp entry).

CONCLUSION: neither mode is strictly better.  Density-derived creation is right where the count
is the binding constraint (many hidden neurons, sub-critical weights, wide plateau fields) and
wrong where the bump path already places spikes exactly.  That argues for selecting per neuron
by field shape -- e.g. density only where a positive run is wide relative to the neuron's own
ISI -- rather than a global mode.  NOT yet tested.

## FIX: mask the CREATE channel inside a spike's refractory footprint (`F_CMASK`)

The 4n F diagnosis was wrong in one respect worth recording: the field and the eligibility are
BOTH FINE.  At the stuck point L[N1] is negative on the three spurious spikes (-1.3e-02,
-3.3e-02, -3.3e-02), POSITIVE on the one nearest the true spike at 224, and eps is nonzero
(3.2e-06) at every one of them.  The unfloored gradient is -9.67e-07 -- exactly right.  The
earlier "88-95% of the mass multiplies zero eligibility" is true of the mass in the TROUGHS,
but that mass is irrelevant: suppression only matters where something fires, and there the
signal is correct and reaches the weight.

The whole failure was CREATE_FLOOR.  Lc in modes 1/2 was max(F, 0) with NO refractory masking,
so it claimed creation even at times the neuron already fires, gc > 0 on nearly every weight,
and g = max(g, gc) overrode suppression everywhere.  Modes 3-6 already masked it; 1/2 did not.

F_CMASK=1 applies the same masking to modes 1/2.  A positive field where the neuron already
fires is a MOVE, not a create.

    4n F seed 0        w01    w12    w03   w23   N1 spikes  N2
    CMASK=0           2407   2416   1024   521      5        5
    CMASK=1            277    920    653   564      2        2
    TRUE               240   1200   1200  1100      1        1

w01 goes from 10x its true value to 1.15x, and the gradient from +8.68e-08 to the correct
-9.67e-07.  Note CMASK=1 lands on EXACTLY mode 4's weights, which is why mode 4 was immune to
CREATE_FLOOR -- it already had this fix.

DEV SUBSET, mode 1, 8 seeds:

                  exact   |Dcount|  count-ok   |dt|
    CMASK=0       19/40     0.08      97%     16.85
    CMASK=1       19/40     0.09      91%     13.80

Timing improves 18% and 14n Q reaches perfect count (0.16 -> 0.00, 94% -> 100%).  Exact is
unchanged.  4n F's count-ok falls 100% -> 25%, which is NOT a regression: the 100% was a
compensating error (hidden neurons firing 5x with w23 at half value still yield six output
spikes, at wrong times -- |dt| 32.8).  Removing the runaway exposes the real count failure
underneath, which is the same one mode 4 has: w03 and w23 stuck near half truth.

DEFAULT LEFT AT 0 pending a full-suite run -- the dev subset over-represents the cases this
touches, and that lesson has already been learned once this session.

## Where 4n G gets stuck after the CMASK fix: OVER-CREATION on already-correct neurons

`ft_4n_G_stuck0_m1cm.png` (mode 1 + CMASK, seed 0).  w01=289 w12=390 w03=654 w23=357 against
a true 250/500/1200/700; max|g| = 6.74e-04, so this is NOT a zero-gradient stall -- there is
gradient and it points the wrong way.

    N1  fires [144, 344]   true [173, 373]   asks for [102, 402]
    N2  fires [381]        true [244, 444]   asks for [5, 82, 182, 276, 448]
    N3  fires [51,151,251,351,431]  target [33,133,233,291,333,433]

N1 has the CORRECT COUNT (2), both spikes 29 steps early -- and the field asks for two MORE,
at times earlier still than the spikes it already has.  Acting on that raises w01 and drags
the spikes further from their targets.

Across all 8 seeds, 13 of 16 hidden neuron-seeds have the right count and are still asked for
more: 37 surplus requests over 16 neuron-seeds.  The single seed whose neurons ask for nothing
(seed 3) is the one that solves.

MECHANISM.  N1's spike sits at 144 and the request lands at 102 -- 42 samples away.  CMASK
masks only the +-REFRAC_ITERS (22) footprint, so a request that far out survives.  The create
channel is inheriting a downstream MISTIMING and reading it as "another spike is wanted here":
the same move/create conflation measured on 14n Q's N2 (242 positive field samples with Fc
identically zero).  So the two halves of the count error are now cleanly separated -- CMASK
fixed the under-suppression half, this is the over-creation half.

NEXT (untested): widen the create mask from REFRAC_ITERS to the neuron's own inter-spike
interval, or mask the WHOLE positive run containing an existing spike rather than a fixed
window.  4n G is a 2-second case, so this is cheap to try.

## KEEP_BEST FREEZES AT ROUND 1 when a single-output case never matches its count

out_err() charges a FLAT 99.0 per count-mismatched output.  With ONE output whose count never
matches during training, e = 99.0 on every round; best_e is set at round 1 and `e < best_e` is
never true again, so train() returns the round-2 weights and discards everything after.

    4n G seed 0 (mode 1 + CMASK)
      initial            w01=284  w12=385  w03=649  w23=362
      RETURNED by train  w01=289  w12=390  w03=654  w23=357   <- round 2
      live at round 800  w01=1177 w12=1099 w03=978  w23=208

798 of 800 rounds thrown away.  Multi-output cases (14n, 50n) degrade gracefully because a
partial match still varies the mean; SINGLE-output cases that never match are frozen outright.

CONSEQUENCE FOR EARLIER ENTRIES: any "stuck point" reported for a single-output case whose
count never matched is ONE GRADIENT STEP, not a converged state.  That includes the 4n G
over-creation table above (13 of 16 neuron-seeds asking for more than they lack) and the
42-sample request offset.  The 4n F CREATE_FLOOR result is NOT affected -- it was measured on
the gradient at fixed weights, not on a trained endpoint.

AND 4n G IS NOT A CLIFF.  Sweeping w12 with the others fixed: crossing W_CRIT=444.5 at w12=445
takes N2 from 1 spike to its true 2.  But out_err is 99.00 at EVERY w12 from 390 to 600,
because N3 still has 5 spikes against 7 targets and the count penalty is flat.  The reward for
crossing exists physically and is invisible to the scorer.

FIX `F_GRADED_ERR=1`: score 99*|Dcount| + timing/1e3 instead of a flat 99.  Still dominated by
count, but strictly ordered.

    4n G seed 0   GE=0  w=(289,390,654,357)  out 5/7      GE=1  w=(1199,941,1020,748)  out 6/7
    4n F seed 0   GE=0  w=(277,920,653,564)  out 5/6      GE=1  w=(973,894,1192,33)    out 5/6

NOT A CLEAR WIN.  Mode 0 dev subset: |Dcount| 1.03 -> 0.64, |dt| 9.41 -> 12.56, exact 24/40
unchanged.  Mode 1+CMASK dev subset: unchanged, because those five cases either already match
their counts (no freeze) or keep the same |Dcount| either way -- the identical suite numbers
there are the metric being coarse, not the flag failing; the weights differ substantially.
Default left at 0 pending a full-suite run.

## What drives w01 the wrong way on 4n G: CREATE DEMAND MISATTRIBUTED BY DEPTH

With the freeze fixed (CMASK + GRADED_ERR), 4n G seed 0 ends at w01=1199 against a true 250.
The ENDPOINT gradient is fine -- g[w01] = -3.67e-06, correctly pulling down (positive demand
contributes +4.22e-06, negative -7.90e-06).  Nothing is pulling it wrong there.  The damage is
done in the first ~150 rounds and never undone:

    round    w01     g[w01]     N1 spikes (true 2)   N3 spikes (target 7)
        1    284   +6.90e-04           2  correct            5
       50    413   +3.45e-05           2  correct            5
      150    578   +1.64e-05           5  BROKEN             5
      400   1257   +1.58e-06           5                     5
      550   1060   -4.95e-06           5                     5

The corrective negative gradient (~-5e-06) is TWO ORDERS OF MAGNITUDE weaker than the +6.9e-04
that caused the problem, so it never recovers; w01 just oscillates between 770 and 1422.

AT ROUND 1, before anything is wrong:

    N1 fires 2 / true 2   <- ALREADY CORRECT
    N2 fires 1 / true 2
    N3 fires 5 / true 7   <- the real deficit

    N1 positive field: 8.593e+01, of which 100% is CREATE
    N2 positive field: 8.065e-01, of which  29% is CREATE

    g = w01 +6.90e-04   w12 +5.05e-06   w03 +9.64e-08   w23 -4.85e-09

w01 gets 136x the gradient of w12 and 7000x that of w03 -- and w01 is the weight that least
needs to move.  The genuine deficit is on the OUTPUT PATH: w03 is 649 against a true 1200 and
w23 is 362 against 700, both needing to roughly double.  They receive almost nothing.

MECHANISM: create mass GROWS PER HOP going backward -- N2 carries 2.31e-01 and N1 carries
8.59e+01, 373x larger one hop further from the output.  So an output deficit is routed to the
DEEPEST weight in the chain rather than the shallowest, and lands on a neuron whose count was
already right.  This is the "fire more times" vs "fire harder" conflation again, but with a
specific and damaging bias: unable to tell them apart, the propagation always prefers "more
spikes, further upstream".

That predicts the failure signature seen on 4n F and 4n G both: hidden neurons driven to 2-5x
their true spike counts while the output-side weights sit at roughly HALF their true values.

## YES, IT NORMALISES: unit-MASS backward message (`F_KNORM=1`) -> perfect counts on the dev subset

The backward message was normalised to unit PEAK (k / k.max()), so each request spread a
peak-height-1 kernel over its whole candidate window (~78 samples).  Mass therefore grew by
roughly the window width at every hop.  Unit MASS (k / k.sum()) makes it a distribution over
which upstream time could be responsible -- which is what the propagation docstring already
claimed ("a wide feasible window does not outvote a narrow one"), and which /max does not do.

    4n G seed 0, round 1        KNORM=0            KNORM=1
    create mass N2 -> N1        372.8x per hop     1.9x per hop
    g[w01] / g[w03]             7156x              2x
    g                w01 +6.90e-04 w12 +5.05e-06   w01 +1.53e-07 w12 +2.02e-07
                     w03 +9.64e-08 w23 -4.85e-09   w03 +9.64e-08 w23 -4.85e-09

DEV SUBSET (mode 1 + CMASK + GRADED_ERR, 8 seeds):

                    exact   |Dcount|  count-ok   |dt|
    KNORM=0 (peak)  19/40     0.09       91%    13.80
    KNORM=1 (mass)  19/40     0.00      100%    17.82
    (mode 0 base)   24/40     1.03       69%     9.41

|Dcount| = 0.00 and count-ok = 100% across all 40 output-seeds -- the FIRST configuration in
this investigation to get every count right.  4n F, which resisted everything else, goes from
0.75 / 25% to 0.00 / 100%.

COSTS: |dt| 13.80 -> 17.82 (4n F's own |dt| 15.9 -> 33.4), and exact is unchanged at 19/40,
still under the bump baseline's 24/40.  Perfect counts do not become exact recoveries without
the timing.

AND IT DOES NOT FIX 4n G (still 1/8, w01 -> 1035 against a true 250).  The depth bias is gone
as a RATIO, but g[w01] stays positive because N1's positive field is still 100% CREATE while
its count is already correct.  Normalisation fixes how far the demand travels, not whether it
should have been created at all.

Default left at 0 pending a full-suite run.

## 4n G with all three fixes: a GENUINE local optimum, not a defect

`ft_4n_G_stuck0_knorm.png` (mode 1 + CMASK + GRADED_ERR + KNORM, seed 0).
w01=1035 w12=1202 w03=1124 w23=741 against a true 250/500/1200/700; max|g| = 1.16e-07.

THE NORMALISATION FIXED THE OUTPUT PATH.  w03 is now 1124/1200 (94%) and w23 741/700 (106%),
where before both sat near HALF their true values.  That is the depth misattribution repaired:
the shallow weights finally get their share of the demand.

THE FIELD IS CORRECT.  N1 asks for 2 spikes at [195, 395] against a true [173, 373] -- the
right count.  N2 asks at [241, 441] against a true [244, 444] -- within 3 steps.  Clean
positive humps at those times, deep negative troughs elsewhere.

BUT LOWERING w01 MAKES THINGS WORSE (others held at their final values):

     w01    N1 spikes (true 2)   N3 spikes (target 7)
     250            2  correct            5
     400            2  correct            5
     600            5                     5
    1035            5                     6   <- where it sits
    1500            5                     6

At the TRUE w01 the hidden neuron is exactly right and the OUTPUT gets worse (6 -> 5), so the
scorer correctly prefers 1035.  Gradients at that point are 1e-09..1e-07: converged, not
starved.  w01 and w12 (1202 against a true 500) have to come down TOGETHER -- the same
coordinated-move structure as 3n L's w02/w12 cliff.

STATUS PROGRESSION for this case across the session:
  1. frozen at round 1 by the flat count penalty          (scorer artifact)
  2. runaway w01 from depth-misattributed create demand   (propagation bug)
  3. genuine local optimum needing a two-weight move      (property of the loss surface)

Still 1/8 exact, but the obstacle is no longer a defect in the method.

## Running 4n G forward from its converged point: a LIMIT CYCLE, not a local optimum

Correcting the entry above.  Resuming training from w01=1035 w12=1202 w03=1124 w23=741:

  - THE GRADIENT POINTS THE RIGHT WAY.  g[w01] = -3.58e-09, g[w12] = -7.96e-09, both negative,
    i.e. down toward the true 250 and 500.
  - THE COORDINATED MOVE HAPPENS.  w01 and w12 descend TOGETHER (1035->880, 1202->1033 over 30
    rounds), which is exactly what escaping this configuration requires.
  - THE CLIFF IS CROSSED, NOT RETREATED FROM.  At round 22 N3 drops 6 -> 5 spikes and out_err
    doubles (99.04 -> 198.03).  The live iterate keeps descending straight through; only
    KEEP_BEST stops recording, having set its best at round 1.
  - IT STALLS ~200 UNITS SHORT.  The descent bottoms out at w01 ~ 790 around round 100, then
    REVERSES.  Over 3000 further rounds it oscillates between 790 and 1480 and never goes
    lower.  N1 and N2 sit at 5 spikes throughout.

From the w01 sweep, N1 fires 2 at w01 <= 400 and 5 at w01 >= 600, so the basin boundary is in
400-600.  The trajectory never gets within 200 of it.

MECHANISM OF THE REVERSAL -- a feedback trap: lowering the hidden weights costs the output a
spike (N3 6 -> 5), which INCREASES the create demand, which pushes the hidden weights back up.
The output is under-served either way (5 or 6 against 7 targets) so the create pressure never
relents.  The lever that would actually fix it is w03 -> 1200, and that just oscillates between
744 and 1221.

So the scored surface has a local optimum at w01 ~ 1035, but the DYNAMICS do not sit in it --
they orbit it.  Any fix aimed at "escaping a local optimum" (restarts, momentum, annealing) is
aimed at the wrong thing; the trajectory already leaves and comes back.

## The far side of the 4n G cliff is where the SIGNAL DIES (`ft_4n_G_stuck0_pastcliff.png`)

Plotted at the deepest point of the descent (round 90): w01=791 w12=846 w03=985 w23=507,
N3 down to 5 spikes against 7 targets.  The positive requests SURVIVE the crossing intact --
N1 still asks [202, 403] against a true [173, 373], N2 asks [245, 445] against [244, 444] --
but the suppression collapses:

                    N1 +demand   N1 -demand   ratio +/-   g[w01]
    pre-cliff  N3=6  1.195e-02    1.044e-02      1.14     -2.82e-09
    post-cliff N3=5  1.367e-02    4.265e-03      3.20     +6.93e-11

N1's negative demand falls 59% while its positive demand RISES, taking the balance from 1.14
to 3.20 and flipping g[w01] from negative to positive.  N2 does the same (0.84 -> 2.34).

THE DESCENT IS SELF-DEFEATING.  It is driven by suppression that exists only while the output
holds 6 spikes.  The moment the move succeeds far enough to cost the output its sixth spike,
the output is FURTHER from its 7 targets, create demand rises, suppression collapses, and the
gradient doing the work reverses.  That is the whole limit cycle in one measurement, and it
explains why crossing the cliff does not help even though the direction is right.

A fix has to SUSTAIN hidden-neuron suppression through a transient loss of output spikes --
i.e. stop the output's count deficit from dominating the demand while a hidden correction is
mid-flight.  CLIFF_HOLD (freeze the weight that broke a count, let the others compensate) is
already in the codebase for approximately this reason, defaults to 0, and has NEVER been
tested in density mode.

## Sign-only demand (`F_DPOW=0`): amplitude carries COUNT, sign+extent carries TIMING

4n G's limit cycle is driven by AMPLITUDE alone -- crossing the output cliff collapses N1's
suppression 59% in mass while its EXTENT grows (199 -> 229 negative samples).  So a sign-only
gradient survives the crossing where the magnitude-weighted one stalls:

                        g (magnitude)   g (sign only)
    N1 pre-cliff          -2.82e-09       -6.62e-05
    N1 POST-cliff         -1.01e-09       -1.58e-04    <- 3x weaker vs 2.4x STRONGER
    N2 POST-cliff         -9.37e-09       -3.31e-04    <- 7x stronger

F_DPOW is the exponent on demand amplitude (1 = as-is, 0 = sign and extent only, scaled by the
run peak so overall magnitude stays comparable).

DEV SUBSET (mode 1 + CMASK + GRADED_ERR + KNORM, 8 seeds):

                    exact   |Dcount|  count-ok   |dt|
    DPOW=1 (mag)    19/40     0.00      100%    17.82
    DPOW=0 (sign)   19/40     0.09       91%    14.59

A CLEAN TRADE, opposite in direction to everything the density work has done:
  - AMPLITUDE carries COUNT information.  Discarding it costs counts, which is unsurprising --
    the count IS "how much demand is here".  4n F, which KNORM had just brought to |Dcount|
    0.00 / 100%, falls back to 0.75 / 25%.
  - SIGN + EXTENT carry TIMING information.  Keeping only those improves |dt| 17.82 -> 14.59.

AND IT DOES NOT FIX 4n G (still 1/8; w01 1035 -> 911 against a true 250, and w23 COLLAPSES to
35 against a true 700 -- with amplitude gone, nothing distinguishes an important weight from an
unimportant one, so w23 runs away).  Surviving the cliff is necessary but not sufficient: the
trajectory still never reaches the basin where N1 fires 2.

DPOW=0.5 was intermediate on 4n G (w01=962, w23=54) and has NOT been swept on the suite.

## Trying to USE the count/timing split (`F_DPOW` / `F_DPOW_C`) -- the channels do not separate

The finding "amplitude carries count, sign+extent carries timing" suggests giving each channel
the form that suits it: amplitude for CREATE (count), sign-only for the MAIN demand (timing).
F_DPOW now governs the main demand and F_DPOW_C the create channel (defaults to F_DPOW, and
the refactor is an exact no-op at defaults -- verified 19/40, 0.00, 100%, 17.82 unchanged).

DEV SUBSET (mode 1 + CMASK + GRADED_ERR + KNORM, 8 seeds):

    DPOW  DPOW_C   exact   |Dcount|  count-ok   |dt|
     1.0     1.0   19/40     0.00      100%    17.82
     0.5     1.0   19/40     0.09       91%    14.65
     0.0     1.0   19/40     0.09       91%    13.85
     0.0     0.0   19/40     0.09       91%    14.59

TWO THINGS THIS SETTLES:

  1. COUNT COLLAPSES AS A STEP FUNCTION, not a gradient.  DPOW=0.5 behaves exactly like
     DPOW=0 on count (0.09 / 91%); only full amplitude gives 0.00 / 100%.  There is no
     intermediate setting that buys timing without paying the whole count price.

  2. THE SPLIT DOES NOT WORK, and the reason is structural: Lc only feeds CREATE_FLOOR, which
     is a FLOOR (g = max(g, gc)).  The count is carried by the MAIN demand L, so flattening L
     loses it no matter what the create channel does.  The count/timing split is a true
     description of what amplitude encodes, but it does not lie along the channel boundary.

AND EXACT IS 19/40 IN EVERY CONFIGURATION -- the headline metric does not move at all, so the
|dt| gain from 17.82 to 13.85 buys nothing.  Best setting remains DPOW=1 (perfect counts).

## COUNT-OK IS A BAD PROXY -- mapped around 4n G's post-cliff state (`_plot_cliffmap.py`)

Scanning the two hidden weights over 100-1500 x 100-1400 (18471 cells), scoring each cell the
way the suite does (count first, timing only where the count matches):

    w03/w23 frozen at the POST-CLIFF values (985/507):   0 of 18471 cells count-correct
    w03/w23 frozen at TRUE (1200/700):                  48 of 18471 cells (0.26%)

PAST THE CLIFF THE COUNT IS UNREACHABLE IN THE WHOLE HIDDEN-WEIGHT PLANE.  Not rare -- absent.
So in that region count-ok carries NO information about the hidden weights; it is fixed by the
output-path weights alone.  That is exactly why the trajectory can wander w01 between 790 and
1480 with the metric never responding, and it is the missing piece of the limit-cycle story.

EVEN WITH TRUE OUTPUT WEIGHTS the count-correct set is 0.26% of the plane, in ~4 DISCONNECTED
slivers, and the timing over it is mostly hopeless:

    |dt| over count-correct cells:  min 0.0   median 14.3   max 27.1
    both count-correct AND |dt| <= 2:  24 cells (0.13% of the grid), all beside the true point
    worst count-correct cells: w01 = 1470-1500 (true 250, ~6x off) with |dt| = 27.1

So a correct count is reachable ~6x away from the true weights through a completely different
firing pattern whose timing can never be fixed.  Half of all count-correct configurations here
have |dt| >= 14.

CONSEQUENCE FOR EVERY count-ok / |Dcount| FIGURE IN THIS FILE.  They are weaker evidence than
they read as.  In particular the KNORM result (|Dcount| 0.00, count-ok 100% on the dev subset)
is suspect on exactly this ground: 4n F reached "perfect counts" while its |dt| ROSE to 33.4,
which is the signature of correct counts via a structurally wrong configuration.  |Dcount| and
count-ok should be read as necessary-not-sufficient, and any claim resting on them alone needs
a timing figure beside it.

## Sign-only at 4n G's STUCK POINT: does not escape, and the earlier reasoning was wrong

Resuming training FROM the stuck point (1035/1202/1124/741), 1500 rounds:

    CREATE_FLOOR  DPOW   min w01 reached   rounds with N1 == 2 (correct)
         1        1.0          786                  0 / 1500
         1        0.0          767                  0 / 1500
         0        1.0          750                  0 / 1500
         0        0.0          587                  0 / 1500   <- deepest, still no escape

Neither sign-only nor any intermediate DPOW (0.25/0.5/0.75 all bottom at 763-772) escapes.
The two interventions INTERACT -- alone they buy almost nothing (786 -> 767, 786 -> 750),
together 786 -> 587, a 25% deeper descent -- consistent with the floor pinning the sign and
DPOW only mattering once it is released.  N1 never fires its correct 2 spikes in any config.

TWO CORRECTIONS TO THE EARLIER ENTRY:

  1. "Sign-only survives the cliff (-1.58e-04 and strengthening)" used the RAW DOT PRODUCT,
     before CREATE_FLOOR.  The gradient that actually drives training has the floor applied,
     and it keeps g[w01] POSITIVE at every DPOW: +6.9e-11 (DPOW=1), +1.7e-10 (0.5),
     +2.3e-09 (0).  Sign-only never flips the sign on the weight that matters.
  2. ADAM IS PER-WEIGHT SCALE-INVARIANT (update = m_hat / sqrt(v_hat)), so multiplying every
     gradient by a constant cancels exactly.  The "10^5x stronger" sign-only gradient is
     therefore irrelevant on its own; only the SIGN and the RELATIVE magnitude across weights
     survive.  DPOW does change the relative weighting -- at DPOW=0 w12 becomes the dominant
     term (-1.00 normalised, against -0.14 at DPOW=1) -- just not on w01.

GENERAL LESSON: measure gradients AFTER every term that modifies them (floor, trust region,
freeze), and remember the optimiser discards absolute scale.  A raw-dot-product argument about
gradient magnitude predicts nothing here.

## Widening eligibility (`F_WIDE_EPS`) -- an UPPER BOUND, and it does not unlock 4n G

Under DPOW=0 the run WIDTH is the whole signal, and ~2/3 of it multiplies zero eligibility
(w01 at the post-cliff point: 46 of 122 positive samples and 74 of 229 negative have eps > 0).
PAIR_EPS was built to relax exactly this truncation -- but it keys off the pairing map PR, and
DENSITY mode never populates it (PR is EMPTY on every hidden neuron; only the output has
entries), so it cannot act where the loss is.

F_WIDE_EPS drops the truncation entirely on hidden neurons.  Not physical, but it is the
MAXIMUM possible widening, so it bounds the whole direction:

    eps(0->1) support        120/520  ->  500/520
    positive width counting   46/122  ->  122/122  (100%)
    negative width counting   74/229  ->  209/229  ( 91%)
    g[w01]                  +2.27e-09 -> +5.16e-06   <- FURTHER the wrong way

Training from the stuck point, 1500 rounds:

    WIDE_EPS  FLOOR   min w01   rounds with N1 == 2
        0       1       767            0/1500
        0       0       587            0/1500
        1       1      1035            0/1500   <- never descends at all
        1       0       545            0/1500   <- deepest ever, still no escape

WHY IT FAILS: widening recovers positive width COMPLETELY (100%) and negative only partially
(91%), and the create humps sit near the eligibility peak while the suppression troughs are
spread across the epoch.  So any UNIFORM widening favours creation.  To help suppression it
would have to be widened ASYMMETRICALLY, which is no longer an eligibility -- it is a
re-weighting of the demand under another name.

EVERYTHING TRIED ON THIS STUCK POINT (none escapes; N1 never fires its correct 2 spikes):

    baseline                                786
    DPOW=0                                  767
    FLOOR=0                                 750
    DPOW=0 + FLOOR=0                        587
    DPOW=0 + FLOOR=0 + WIDE_EPS=1           545
    DPOW=0 + FLOOR=1 + WIDE_EPS=1          1035  (no descent)

Since WIDE_EPS is maximal, no gentler widening scheme can beat 545.  The direction is closed.

## PUSH: every spike moved toward its NEAREST FIELD PEAK (`F_PUSH`) -- best result of the session

A move term derived from the density itself, replacing what bumps_of provided.  A density says
how much is wanted WHERE but nothing in it says "this spike belongs over there", which is why
DENSITY fixes counts and loses timing.  For every actual spike, find the nearest LOCAL MAXIMUM
of the field, demand positive there scaled by the field height AT THE PEAK times the usual
distance grading, and (for an EARLY spike only) negative at the spike.  Unlike bumps_of this
does not collapse a run to one request, so a wide run with several maxima can attract several
spikes.

IT ONLY WORKS WITH CREATE_FLOOR OFF -- and the two together are a large gain.

    4n G from scratch, 8 seeds        exact   count-ok   |dt|
      density baseline (no push)       1/8      --        --
      PUSH=4 FLOOR=1                   0/8      2/8      1.8
      PUSH=4 FLOOR=0                   4/8      6/8      1.6
      PUSH=2 FLOOR=0                   5/8      6/8      0.1

    DEV SUBSET, 8 seeds               exact   |Dcount|  count-ok   |dt|
      mode 0 bumps (baseline)         24/40     1.03      69%      9.41
      density, previous best          19/40     0.00     100%     17.82
      PUSH=2.0 + FLOOR=0              27/40     0.08      92%      9.16
      PUSH=4.0 + FLOOR=0              26/40     0.09      91%     10.30

PUSH=2 beats the bump baseline on ALL THREE axes at once -- the first configuration in this
investigation to do so.  Per case against mode 0: 4n F 0/8 -> 8/8, 3n A 3/8 -> 8/8,
chain 8/8 -> 6/8 (the only regression), 5n H and 14n Q unchanged.

ALSO SETTLES AN EARLIER WORRY.  Resuming 4n G's stuck point with PUSH=4/FLOOR=0 produces an
output that is EXACTLY right -- all 7 spikes on target, |dt| = 0 -- via a genuinely different
path: w03=1207 (true 1200) and N2 firing its correct 2, but N1 firing 5 instead of 2 with
w01=626 (true 250) and w12=247 (true 500).  N1's extra spikes are absorbed by a weaker w12.
So a structurally different configuration CAN reach correct timing; the earlier concern that
count-correct wrong paths can never be timed correctly does not hold in general.

NOT YET VALIDATED ON THE FULL SUITE -- the dev subset over-represents these cases and has
already misled once this session (the hybrid's 11/20 did not survive).  Full run in progress.

## Where chain gets stuck under PUSH: the DENSITY demand is wrong-signed for an early network

chain seed 4, PUSH=2 + FLOOR=0: w01=729 w12=514 w23=730 (true 500/500/500), max|g| = 1.18e-07.
Everything is uniformly EARLY and the error compounds down the chain:

    N1 fires [46,146,246,346,446]   true [72,172,272,372,472]   -26
    N2 fires [114,214,314,414,514]  true [143,243,343,443]      -29, ONE EXTRA spike
    N3 fires [159,259,359,459]      target [214,314,414,514]     -55   (count CORRECT)

A LARGE IMPROVEMENT IS RIGHT THERE, on a smooth monotone surface with no cliff:

    w01   500(true)  560   620   680   729(stuck)  780   840
    |dt|     29.0   40.0  47.0  52.0     55.0     57.0  60.0

...and the gradient points AWAY from it: g[w01] = +1.18e-07, g[w12] = +9.98e-08 (both asking
to RAISE, which makes things worse); only g[w23] = -6.78e-08 is correct.

FIRST DIAGNOSIS WAS WRONG.  I blamed the PUSH term's positive-at-the-later-peak for outvoting
its negative-at-the-early-spike, and added F_PUSH_ONE (early spike -> negative only, late spike
-> positive only).  It changes almost nothing: g[w01] +1.18e-07 -> +1.14e-07, a 3% shift,
because the push is only ~6% of the demand (N1 positive mass 3.82e-02 with it, 3.60e-02
without).  chain stays 6/8, |dt| 13.8, identical.

THE REAL CAUSE IS STRUCTURAL AND IS IN THE DENSITY ITSELF.  Raising a weight makes a neuron
fire EARLIER, but the density expresses "a spike is wanted later" as POSITIVE DEMAND AT A LATER
TIME -- which raises the weight.  N1's demand here is 6:1 positive (3.6e-02 vs 6.1e-03) with
every request later than every spike.  For a uniformly-early network the dense main demand is
systematically wrong-signed.

The bump path never hits this because it has an explicit rule for exactly this asymmetry -- the
negative at an EARLY spike, documented as "suppression removes drive, which moves a spike
LATER: right for an early one, actively wrong for a late one".  The dense main demand has no
equivalent; it only says "positive here" and lets the projection decide the sign.

## SHIFT: timing correction signed by WHICH SIDE the wanted spike is on (`F_SHIFT`)

More drive makes a neuron fire EARLIER, so the sign of a timing correction cannot come from the
field's sign at the request -- it has to come from the request's position relative to the spike:

    spike EARLY (wanted later)  -> LESS drive -> negative AT THE SPIKE
    spike LATE  (wanted earlier)-> MORE drive -> positive AT THE SPIKE

Applied at the SPIKE, not at the peak: that is where eligibility is guaranteed nonzero and
where changing drive actually moves this spike.  The raw density is deliberately KEPT as a
level term beside it, since peak matching is unreliable when the count is wrong.

ORDERED MATCHING IS ESSENTIAL, and this is the failure the user predicted.  Nearest-peak
matching inverts the sign whenever the phase offset exceeds half an inter-spike interval.
chain seed 4: peaks [122,218,318,418] against spikes [46,146,246,346,446], every spike
genuinely 26 EARLY -- but 146 is 24 from peak 122 and 72 from peak 218, so nearest pairs it
backwards and FOUR OF FIVE spikes are judged LATE and take the wrong sign.  Pairing the i-th
spike with the i-th peak fixes it (46->122, 146->218, ...), all correctly early.

    g[w01] at chain seed 4's stuck point (lowering w01 takes |dt| 55 -> 29, so NEGATIVE is right)
      SHIFT=0                       +1.02e-07
      SHIFT=4, nearest matching     +1.61e-07   <- worse than no shift at all
      SHIFT=4, ordered              +3.47e-08
      SHIFT=8, ordered              -2.33e-08   <- sign finally correct on all three weights
      SHIFT=16, ordered             -1.11e-07

    chain, 8 seeds        exact   count-ok   |dt|
      density alone        3/8      6/8      22.8
      density + PUSH=2     6/8       --      13.8
      density + SHIFT=8    7/8      7/8       0.0
      density + SHIFT=16   6/8      6/8       0.0
      (bump baseline)      8/8       --       0.0

SHIFT=8 needs to be strong enough to outvote the density's own positive bias -- at 4 it only
halves the wrong-signed gradient, at 8 it flips it.

## FULL SUITE, PUSH=2 + CREATE_FLOOR=0: 97/168 -- BEATS THE BUMP BASELINE, and it generalised

    config                        exact     |Dcount|  count-ok   |dt|
    mode 0 bumps (baseline)      86/168       0.61      72%     15.95
    density (mode 1)             49/168       0.23      84%     19.29
    hybrid (mode 4)              56/168       0.20      86%     16.13
    PUSH=2 + FLOOR=0             97/168       0.21      81%     15.84

+11 exact over the bump baseline with 3x better count agreement and equal timing.  Unlike the
mode-4 hybrid, whose dev-subset lead vanished on the full suite, this held: 27/40 -> 97/168.

    GAINS                          REGRESSIONS
    over-demand  2/8 -> 8/8        3n D   8/8 -> 5/8
    3n A         3/8 -> 8/8        4n G   8/8 -> 5/8
    2-cycle      4/8 -> 7/8        chain  8/8 -> 6/8
    4n V         3/8 -> 6/8        4n S   8/8 -> 6/8
    3n L         2/8 -> 5/8        3n E   8/8 -> 7/8
    3-cycle      3/8 -> 4/8
    8n M         0/8 -> 1/8        <- first ever non-zero on this case, by ANY method

50n cases still 0/8 exact but count-ok improves: 69/78/68% against the baseline's 64/54/55%.

8n M SEED 4, RASTER (bigraster_8n_M_s4.png): 18 of 18 output spikes EXACT across all three
outputs -- zero adrift, zero missed, zero spurious.  A genuine solve, not a near miss.

CONFIG: F_DENSITY=1 F_CMASK=1 F_GRADED_ERR=1 F_KNORM=1 F_PUSH=2.0 F_CREATE_FLOOR=0.
CREATE_FLOOR=0 is REQUIRED -- with the floor on, PUSH collapses (4n G 4/8 -> 0/8).

## SHIFT's sign rule is WRONG MORE OFTEN THAN RIGHT -- and still improves timing

Checking the ordered pairing's assigned direction against the true direction each spike must
move (dev cases, 4 seeds, 400 rounds):

                     sign right   sign wrong   % wrong
    count CORRECT         69          99         59%
    count WRONG           69          77         53%

The concern that motivated ordered matching -- a spurious spike shifting every later pairing --
is NOT the differentiator: the sign is near-random whether or not the count is right.  It
varies by CASE instead (5n H 86% right, 14n Q 62% wrong).

Yet SHIFT=8 ordered gives the BEST |dt| of any configuration measured:

    config                        exact   |Dcount|  count-ok   |dt|
    mode 0 bumps                  24/40     1.03      69%      9.41
    PUSH=2                        27/40     0.08      92%      9.16
    SHIFT=8 ordered               24/40     0.22      81%      8.37
    SHIFT=8 slope (no matching)   22/40     0.39      72%     11.24
    SHIFT=16 slope                19/40     0.48      66%     10.95

SO THE STATED MECHANISM IS NOT SUPPORTED.  The |dt| gain is real and reproducible, but it
cannot be coming from getting the direction right, since the direction is wrong 59% of the
time.  Untested alternative: the term places demand AT THE SPIKES, where eligibility is
guaranteed nonzero, so the benefit may be one of placement rather than sign.

ALSO NOTE the matching-free slope variant, which WOULD answer the fragility objection, is worse
on the suite (22/40 against 24/40) despite giving a stronger correctly-signed gradient on chain
seed 4 (-4.50e-08 against ordered's -2.33e-08).  Its median |dt| is better (3.2 vs 4.8) and its
mean worse (11.24 vs 8.37) -- bimodal, as a purely local direction estimate would be.

TREAT SHIFT AS AN EMPIRICAL RESULT WITH AN UNKNOWN MECHANISM.  PUSH=2 + CREATE_FLOOR=0
(97/168 on the full suite) remains the configuration to build on; it is validated and its
gain is not explained by a rule that turns out to be near-random.

## Big-net rasters under PUSH: the late bias is corrected on 2 of 3 nets

seed 0, 800 rounds, config F_DENSITY=1 F_CMASK=1 F_GRADED_ERR=1 F_KNORM=1 F_PUSH=2
F_CREATE_FLOOR=0.  Figures bigraster_{50n_A,50n_B,50n_C,14n_Q}_s0.png.

    case    paired  late/early  mean off  (baseline)  mean |off|  (baseline)
    50n A     96      39/55       -4.8      +15.6        24.3       25.8
    50n B     84      31/50       -7.8      +18.7        20.3       31.4
    50n C     72      51/21      +12.7      +15.3        32.6       40.3
    14n Q     24      11/13       -1.6        --         11.5        --

The systematic LATE bias that opened this whole investigation -- outputs firing late because the
weights were under-driven -- is gone on A and B (sign flipped, late/early 65/25 -> 39/55 on A).
50n C KEEPS IT (+12.7, still 51 late to 21 early), so the correction is not universal.

Tallies (against the baseline rasters):
    50n A   exact 1->2, near 8->13, adrift 86->81, missed 2->1, spurious 2
    50n B   exact 3, near 13, adrift 68, missed 12->2, spurious 0
    50n C   exact 0, near 6, adrift 66, missed 2, spurious 1
    14n Q   exact 0, near 7, adrift 17, MISSED 0, SPURIOUS 0  <- every target produced

The count problem is essentially solved on 14n Q (all 24 targets produced across 4 outputs) and
much improved on 50n B (12 missed -> 2).  What remains on the big nets is TIMING SCATTER of
20-33 steps with no directional bias -- a different failure from the one this session started
with, and one nothing tried so far has touched.  All three 50n cases remain 0/8 exact.

## WHY only the first epoch has negative demand: the OUTPUT's greedy pairing inverts early/late

chain seed 4, PUSH config.  N2's density demand is purely positive in four of its five epochs,
which is what makes g[w12] point the wrong way.  The cause is upstream, on the OUTPUT, and it
is in the ORIGINAL BUMP PAIRING -- not in any of the new terms.

    N3 fires [159, 259, 359, 459]   targets [214, 314, 414, 514]   -- uniformly 55 EARLY

    greedy nearest pairing:
      target 214 <- spike 259  (dist  45)  -> "LATE",  no negative
      target 314 <- spike 359  (dist  45)  -> "LATE",  no negative
      target 414 <- spike 459  (dist  45)  -> "LATE",  no negative
      target 514 <- spike 159  (dist 355)  -> "EARLY", the ONLY negative

So L[3] is four positives at the targets and exactly ONE negative, at t=159.  That lone
negative is the entire source of N2's first-epoch -2.32e-08; every later epoch inherits pure
positive demand because there is nothing negative left to propagate.

THE HALF-ISI INVERSION.  The inter-spike interval is 100, so half is 50, and the error is 55.
Once a timing error exceeds half an ISI, nearest-matching pairs each target with the FOLLOWING
spike and reports LATE for a network that is uniformly EARLY.  No spurious spike is needed --
a uniform phase error is enough.  The leftover pairing is self-evidently wrong: target 514
matched to spike 159, 355 steps away.

This is the same failure predicted for the SHIFT term's nearest matching, occurring in the
mature local_demand path that everything else is built on.  It means "the density is
wrong-signed" (the earlier entry) is a SYMPTOM: the density faithfully propagates an output
demand whose sign was already inverted by the pairing.

NOT YET FIXED.  An ordered pairing at the output would give 214<-159, 314<-259, 414<-359,
514<-459, all correctly EARLY -- but ordered pairing has its own fragility, and the sign-rule
audit showed ordered matching is wrong 59% of the time on hidden neurons.  The output case is
different (real targets, not inferred peaks) so it may be safe there; untested.

## FIX: ordered pairing at the OUTPUT (`F_PAIR_ORDER=1`) -- large gain for BOTH pathways

Greedy-nearest pairing inverts early/late whenever the timing error exceeds HALF an ISI.  On
chain seed 4 the output is uniformly 55 early with an ISI of 100, so each target paired with
the FOLLOWING spike and three of four took no negative.  Ordered pairing at outputs -- where
the bumps are REAL TARGETS, same count and same order as the spikes when the count is right --
fixes it:

    N3 demand, greedy:   159:-7e-3  214:+7e-3  314:+7e-3  414:+7e-3  514:+7e-3   (ONE negative)
    N3 demand, ordered:  159:-7e-3  214:+7e-3  259:-7e-3  314:+7e-3  359:-7e-3
                         414:+7e-3  459:-7e-3  514:+7e-3                        (all four)

    g at chain seed 4's stuck point      w01         w12         w23
      greedy                          +1.18e-07   +9.98e-08   -6.78e-08
      ordered                         +6.75e-08   -3.11e-08   -2.71e-07
    (lowering all three is correct, so ordered fixes w12 and strengthens w23 4x)

DEV SUBSET, 8 seeds:

    config                          exact   |Dcount|  count-ok   |dt|
    mode 0 bumps (greedy)           24/40     1.03      69%      9.41
    mode 0 bumps + ORDERED          26/40     0.84      72%      7.48
    PUSH (greedy)                   27/40     0.08      92%      9.16
    PUSH + ORDERED                  32/40     0.00     100%      6.37

    PUSH + ORDERED per case: chain 8/8, 3n A 8/8, 4n F 8/8, 5n H 8/8 -- ALL with |dt| = 0.0 --
    and 14n Q 0/8 but count-ok 100%, |dt| 12.7.

FOUR OF FIVE DEV CASES FULLY SOLVED with perfect counts everywhere.  It helps the BUMP
BASELINE too (24 -> 26), so this is a defect in the shared pairing that was costing both
pathways, not a density-specific patch.

Applied at outputs only by default.  On hidden neurons the "bumps" are inferred field peaks,
where the sign-rule audit measured ordered matching wrong 59% of the time -- F_PAIR_ORDER=2
opts into that and is untested.

FULL-SUITE VALIDATION IN PROGRESS for both pathways.
