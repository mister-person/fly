# Performance scaling study

How the wall-clock cost of the 5 weight-recovery methods scales with **number of
neurons** and **simulation length (steps)**, plus an extrapolation to 20,000 and
150,000 neurons.

Methods studied (see `../SESSION_NOTES.txt` / `../RESULTS.txt`):

| method | inner-loop primitive measured | scaling driver |
|---|---|---|
| Soft homotopy   | one soft `value_and_grad` step (`soft_grad`) | fwd+bwd over ~N synapses/step |
| Hard surrogate  | one manual-BPTT step (`hard_grad`)           | fwd+bwd over ~N synapses/step |
| CMA-ES          | batched forward objective (`batch_forward`)  | forward + O(n_syn²) covariance |
| Diff. Evolution | batched forward objective (`batch_forward`)  | forward × 5·n_syn population |
| Target prop     | one per-neuron NNLS/QP recovery (`tp_pass`)  | scipy per-neuron solves + forward |

## Files

- `bench.py` — builds synthetic random-recurrent networks (fixed average in-degree,
  so synapses grow linearly with neurons, like a fixed-fan-in connectome) and times
  each method's core primitive across a neuron sweep and a step sweep. Writes
  `results.json`.
- `analyze.py` — fits power laws `t = a·N^b` and `t = a·steps^b`, extrapolates each
  method's full-run time to the target sizes, and estimates memory. Writes `REPORT.md`.
- `plot.py` — log-log scaling figure. Writes `scaling.png`.
- `results.json`, `REPORT.md`, `scaling.png` — generated outputs.

## Partial-observation target prop (follow-up experiment)

`../tp_partial_obs.py` asks whether target prop can recover output behaviour when
only a *fraction* of neurons' spike times are observed. See `REPORT_partial_obs.md`
and `partial_obs.png`. Short answer: observing only the outputs fails in these
recurrent nets (missing hidden targets cascade through the recurrence); TP helps
only as a refiner of an existing solution and only when most neurons are observed.

```bash
INIT=random python3 tp_partial_obs.py   # cold start
INIT=found  python3 tp_partial_obs.py   # warm start (refines soft-homotopy weights)
python3 performance/plot_partial.py     # -> partial_obs.png
```

## Reproduce

```bash
cd /workspace/project
python3 performance/bench.py      # ~6 min on 16-core CPU; writes results.json
python3 performance/analyze.py    # writes REPORT.md
python3 performance/plot.py       # writes scaling.png
```

Knobs (env vars for `bench.py`): `DEGREE` (avg in-degree, default 5), `REPS`
(timed reps, default 5), `N_SWEEP`, `STEP_SWEEP`, `STEP_N`.

## Headline results

- **Per neuron:** all learning primitives grow ~`N^1.1` (near-linear; synapses ∝ N).
  Black-box per-member eval grows faster (`N^1.34`).
- **Per step:** every method grows **near-quadratically** in `steps` (`b ≈ 1.8–2.0`),
  even the plain forward pass — so simulation length is the more expensive axis.
- **Extrapolation (steps=1000, same iteration budget):** gradient methods reach
  ~days at N=20,000 and ~weeks at N=150,000; target prop is cheapest on raw compute;
  **CMA-ES and Diff. Evolution are memory-infeasible** (O(n_syn²) covariance /
  O(N²·steps) population batch) well before then.

See `REPORT.md` for the full tables and caveats.
