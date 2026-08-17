# Partial-observation target propagation

**Question:** can target prop recover the *output* behaviour of a 50-neuron
recurrent network when we only observe the spike times of a fraction of neurons?

**Setup** (`../tp_partial_obs.py`, 3 recurrent cases, 8 seeds):
the 3 output neurons `[47,48,49]` are always observed; we additionally observe a
random fraction `f` of the 46 hidden neurons. TP trains the incoming weights of
observed neurons only (target = their true spike times); unobserved neurons keep
their initial weights and never learn. Presynaptic inputs come from a forward
sim with the current weights (a second "oracle-input" variant feeds true spikes
to isolate the effect of the withheld *targets* from the effect of wrong hidden
*dynamics*). Two starts:

- **cold** — random init `w = w_true · U(0.5,1.5)`
- **warm** — soft-homotopy solution (`best_weights_caseX.npy`); TP *refines* it

Metrics: output MSE loss at `[47,48,49]`, output spike-count match (/3), and
whole-network spike-count match (how many of all 50 neurons fire the right number
of times — a breadth-of-recovery measure).

## Result: you cannot get away with observing only the outputs

| | cold start | warm start |
|---|---|---|
| observe **outputs only** (f=0) | ≈ random floor, 0–1 of 3 outputs matched | **degrades** the warm solution — in the recurrent-heavy case 304 the network collapses to 1/50 neurons matched |
| observe **25–50%** of hidden | first real gains; net recovery ~0.4–0.55 | clear improvement, outputs start matching |
| observe **all** (f=1) | still near floor (cold TP is unstable) | best: case 304 reaches **3/3 outputs and loss 9.8e-3 < soft homotopy 15.6e-3** |

Concrete (warm start, case `recurrent_304_4_2_7`, target N47=4 N48=2 N49=7):

| f observed | output loss | out match | net match /50 |
|---|---|---|---|
| 0.00 (outputs only) | 2.04e-2 | 0/3 | 1 |
| 0.25 | 1.12e-2 | 1.6/3 | 20 |
| 0.50 | 1.19e-2 | 1.2/3 | 17 |
| 1.00 | **9.8e-3** | **3/3** | 27 |
| — soft homotopy ref | 1.56e-2 | — | — |
| — random floor | 2.25e-2 | — | 7 |

## Interpretation

1. **Observing spike times does the heavy lifting.** Network recovery rises with
   the fraction observed; from only the 3 outputs, TP has essentially no signal for
   the 46 hidden neurons and cannot reconstruct the dynamics that drive those outputs.

2. **In a recurrent net, missing targets *cascade*.** An unobserved neuron gets no
   target, keeps a wrong weight, fires wrongly, and corrupts the presynaptic input
   of the observed neurons you *are* trying to fit — so re-fitting the outputs alone
   can make things worse (case 304, f=0). This is why cold-start partial-obs TP is
   unstable and barely beats random.

3. **TP works with hidden neurons only as a refiner + with broad observation.**
   Warm-started on another method's solution and observing most neurons, TP
   genuinely improves output recovery (case 304: exact 3/3, below the soft-homotopy
   loss). It is not a standalone partial-observation learner here.

4. **The missing piece is target *inversion*.** What TP fundamentally needs is a
   target for every neuron that matters to the output. `../target_prop.py` shows
   these can be *inferred* from downstream targets (Tj_pre = Tj_post − latency) for a
   feed-forward chain — no observation needed. Extending that backward-inversion to
   the recurrent, fan-out case is the way to make partial-observation TP work from
   scratch; without it, you must observe (not infer) the hidden neurons.

**Bottom line:** yes, some neurons can go unobserved, but only a minority and only
when TP refines a decent existing solution. Observing just the outputs does not
work in these recurrent networks — the hidden neurons need either observed or
inverted targets, and only the feed-forward-chain inversion is currently solved.

Files: `../tp_partial_obs.py` (experiment), `partial_obs_{random,found}.json`
(data), `plot_partial.py` → `partial_obs.png` (figure).
