# SNN Learning Method Performance Benchmarks

**JAX backend:** cpu  **Device:** cpu

**Config:** sizes=[10, 25, 50, 100, 200], steps=[500], nopt=20, p_connect=0.1

## Measured Times

| N | Steps | Method | Time |
|---|-------|--------|------|
| 10 | 500 | soft_homotopy | 28.1 ms |
| 10 | 500 | hard_surrogate | 96.1 ms |
| 10 | 500 | target_prop | 25 μs |
| 10 | 500 | blackbox | 181 μs |
| 25 | 500 | soft_homotopy | 38.6 ms |
| 25 | 500 | hard_surrogate | 115.8 ms |
| 25 | 500 | target_prop | 126 μs |
| 25 | 500 | blackbox | 819 μs |
| 50 | 500 | soft_homotopy | 31.6 ms |
| 50 | 500 | hard_surrogate | 105.8 ms |
| 50 | 500 | target_prop | 96 μs |
| 50 | 500 | blackbox | 3.9 ms |
| 100 | 500 | soft_homotopy | 42.3 ms |
| 100 | 500 | hard_surrogate | 115.1 ms |
| 100 | 500 | target_prop | 250 μs |
| 100 | 500 | blackbox | 9.1 ms |
| 200 | 500 | soft_homotopy | 67.7 ms |
| 200 | 500 | hard_surrogate | 147.4 ms |
| 200 | 500 | target_prop | 674 μs |
| 200 | 500 | blackbox | 18.8 ms |

## Scaling Fits (normalized to steps=1000)

| Method | a | p (N scaling) | R² | q (steps scaling) |
|--------|---|---------------|-----|-------------------|
| soft_homotopy | 3.0278e-02 | 0.249 | 0.7332 | 1.000 |
| hard_surrogate | 1.4717e-01 | 0.115 | 0.7177 | 1.000 |
| target_prop | 5.9140e-06 | 0.994 | 0.9084 | 1.000 |
| blackbox | 1.0691e-05 | 1.591 | 0.9819 | 1.000 |

## Extrapolated Times

|---------|-------|--------|----------------|------|
| 20,000 | 1,000 | soft_homotopy | 356.5 ms | R²=0.733, p=0.25, q=1.00 |
| 20,000 | 1,000 | hard_surrogate | 460.4 ms | R²=0.718, p=0.12, q=1.00 |
| 20,000 | 1,000 | target_prop | 112.0 ms | R²=0.908, p=0.99, q=1.00 |
| 20,000 | 1,000 | blackbox | 1.2 min | R²=0.982, p=1.59, q=1.00 |
| 150,000 | 1,000 | soft_homotopy | 588.7 ms | R²=0.733, p=0.25, q=1.00 |
| 150,000 | 1,000 | hard_surrogate | 580.6 ms | R²=0.718, p=0.12, q=1.00 |
| 150,000 | 1,000 | target_prop | 830.7 ms | R²=0.908, p=0.99, q=1.00 |
| 150,000 | 1,000 | blackbox | 30.5 min | R²=0.982, p=1.59, q=1.00 |
| 20,000 | 5,000 | soft_homotopy | 1.78 s | R²=0.733, p=0.25, q=1.00 |
| 20,000 | 5,000 | hard_surrogate | 2.30 s | R²=0.718, p=0.12, q=1.00 |
| 20,000 | 5,000 | target_prop | 560.0 ms | R²=0.908, p=0.99, q=1.00 |
| 20,000 | 5,000 | blackbox | 6.2 min | R²=0.982, p=1.59, q=1.00 |
| 150,000 | 5,000 | soft_homotopy | 2.94 s | R²=0.733, p=0.25, q=1.00 |
| 150,000 | 5,000 | hard_surrogate | 2.90 s | R²=0.718, p=0.12, q=1.00 |
| 150,000 | 5,000 | target_prop | 4.15 s | R²=0.908, p=0.99, q=1.00 |
| 150,000 | 5,000 | blackbox | 2.5 h | R²=0.982, p=1.59, q=1.00 |

## Model

Time model: `t(N, steps) = a × N^p × (steps/1000)^q`

- **Soft homotopy**: sigmoid forward + manual BPTT backward. Each Adam step does 1 fwd + 1 bwd through the full sim.
- **Hard surrogate**: hard forward + narrow-surrogate BPTT. 1 fwd + 1 bwd per Adam step.
- **Target propagation**: analytical weight recovery via impulse response. O(outputs × fan_in²) + 1 hard forward.
- **Black-box**: batched hard forward evaluations. Per-eval cost ≈ 1 hard forward. Population ≈ 10 × n_syn.
