# Performance scaling of the 5 weight-recovery methods

JAX 0.10.2, CPU.  Synthetic random-recurrent networks, fixed average in-degree = 5 (so synapses grow ~linearly with neurons, as in a fixed-fan-in connectome).  Times are per-call medians of the method's core inner-loop primitive.

## 1. Per-call compute cost vs number of neurons (steps = 1000)

| N | synapses | spikes | forward (ms) | soft_grad (ms) | hard_grad (ms) | tp_pass (ms) |
|---|---|---|---|---|---|---|
| 50 | 244 | 526 | 4.21 | 19.22 | 18.88 | 193.3 |
| 100 | 498 | 918 | 8.03 | 56.09 | 55.17 | 279.0 |
| 200 | 992 | 2077 | 23.10 | 149.70 | 147.15 | 661.2 |
| 400 | 1997 | 4139 | 31.78 | 240.72 | 240.62 | 1708.7 |
| 800 | 3996 | 8612 | 55.07 | 385.98 | 413.21 | 2862.0 |
| 1600 | 7996 | 17674 | 127.02 | 1178.36 | 1183.95 | 6877.7 |

Power-law fits  t = a * N^b  (b = scaling exponent):

| primitive | exponent b | R^2 | interpretation |
|---|---|---|---|
| forward | 0.95 | 0.983 | forward sim |
| soft_grad | 1.11 | 0.978 | soft homotopy step |
| hard_grad | 1.12 | 0.982 | hard surrogate step |
| tp_pass | 1.06 | 0.989 | target-prop recovery pass |
| batch_forward | 1.34 | 0.983 | black-box per-member eval |

## 2. Per-call compute cost vs simulation length (N = 200)

| steps | spikes | forward (ms) | soft_grad (ms) | hard_grad (ms) | tp_pass (ms) |
|---|---|---|---|---|---|
| 250 | 64 | 1.56 | 5.62 | 5.44 | 14.8 |
| 500 | 653 | 4.39 | 33.20 | 29.42 | 124.6 |
| 1000 | 2077 | 26.27 | 158.00 | 157.40 | 732.5 |
| 2000 | 4940 | 62.88 | 461.70 | 464.72 | 2356.9 |
| 4000 | 10607 | 197.52 | 1461.92 | 1575.14 | 4744.9 |

Power-law fits  t = a * steps^b:

| primitive | exponent b | R^2 |
|---|---|---|
| forward | 1.78 | 0.989 |
| soft_grad | 1.98 | 0.988 |
| hard_grad | 2.03 | 0.990 |
| tp_pass | 2.09 | 0.961 |
| batch_forward | 2.41 | 0.994 |

## 3. Extrapolated full-run wall time

Each method's 50-neuron full-run wall time is scaled by the growth of its dominant primitive (fixed step budget, steps=1000).  Assumes the same number of optimizer steps / evaluations / passes at every size — i.e. this is the *compute* growth, before any convergence-difficulty penalty.

| Method | anchor @ N=50 | exponent b | @ N=20,000 | @ N=150,000 | anchor source |
|---|---|---|---|---|---|
| Soft homotopy | 6.2 min | 1.11 | 3.2 days | 30.2 days | RESULTS Run 4 avg (397/365/352) |
| Hard surrogate | 3.8 min | 1.12 | 2.2 days | 21.1 days | RESULTS Run 4 avg (239/226/218) |
| Target prop | 4.0 s | 1.06 | 38.4 min | 5.5 h | 20 x (tp_pass+forward) measured at N=50 |
| CMA-ES | 17.5 min | 1.34 | 38.0 days | 1.6 yr | RESULTS Run 6 avg (921/1115/1113) |
| Diff. Evolution | 1.6 h | 1.34 | 205.5 days | 8.4 yr | RESULTS Run 7 avg (4356/6565/6107) |

*CMA-ES and Diff. Evolution additionally carry population/covariance overhead that grows faster than the forward eval (see memory section) and become infeasible well before these compute numbers would be reached.*

## 4. Memory (the real wall for large N)

Activation storage is dense in time: the forward pass keeps voltages and rise values of shape (steps, N), and the backward pass keeps the voltage/refractory history.  With float32:

| N | activations (steps=1000) | CMA-ES covariance (n_syn^2) | DE population batch |
|---|---|---|---|
| 50 | 781.2 KB | 488.3 KB | 238.4 MB |
| 1,600 | 24.4 MB | 488.3 MB | 238.4 GB |
| 20,000 | 305.2 MB | 74.5 GB | 36.4 TB |
| 150,000 | 2.2 GB | 4.1 TB | 2.0 PB |

- **Activations** grow linearly in both N and steps (O(steps*N)); manageable into the tens-of-GB range at N=150,000 with steps=1000 and gradient checkpointing.
- **CMA-ES** stores a dense n_syn x n_syn covariance and does an O(n_syn^2) update per generation.  At N=20,000 (n_syn~100k) that is ~80 GB and O(10^10) flops/gen — infeasible.  CMA-ES does not scale past a few thousand synapses.
- **Diff. Evolution** evaluates a population of 5*n_syn candidates at once; the vmapped forward batch alone is petabyte-scale at N=20,000.  Infeasible.

## 5. Bottom line

Ordering by how well each method scales to large recurrent networks:

1. **Hard surrogate** and **Soft homotopy** — gradient methods; cost grows ~N^1.1-1.1 (super-linear mainly from the scatter/gather over ~N synapses per step plus XLA overhead).  These are the only methods with any hope at N>=20,000, and only with activation checkpointing to bound the O(steps*N) memory.
2. **Target prop** — per-pass cost grows ~N^1.1 and it needs very few passes, so raw compute is cheap, but it is pure-Python/scipy per-neuron solves and would need vectorising to be practical at scale; still, no quadratic blow-up.
3. **CMA-ES** — O(n_syn^2) covariance kills it past a few thousand synapses.
4. **Diff. Evolution** — 5*n_syn population makes each generation O(N^2*steps); worst scaling of all, infeasible earliest.

Simulation length (`steps`) is *more* expensive than neuron count: every method scales near-quadratically in `steps` (forward b=1.78, soft_grad b=1.98, hard_grad b=2.03). This holds for the plain forward pass too (b~1.78, independent of spike count), so it is a property of stepping the (steps, N) voltage-history array through the time loop, not of the learning rule.  Memory in `steps` is only linear, but the super-linear compute means keeping simulations short is the single biggest lever on runtime — halving `steps` is closer to a 4x speedup than 2x.
