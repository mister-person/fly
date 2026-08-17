"""Fit scaling laws to bench.py output and extrapolate to large networks.

For each core primitive we fit a power law   t = a * N^b   (and t = a * steps^b)
by least squares on log-log data.  We then project each method's full-run wall
time to 20,000 and 150,000 neurons by scaling a measured/known 50-neuron anchor
by the growth of that method's dominant primitive.

We also estimate memory, which for these dense-in-time simulators grows as
O(steps * N) for activations plus method-specific terms (CMA-ES covariance is
O(n_syn^2)), and flag the sizes at which each method becomes infeasible.
"""

import json, os, sys
import numpy as np

RES = os.environ.get("OUT", "/workspace/project/performance/results.json")
REPORT = "/workspace/project/performance/REPORT.md"

TARGETS = [20000, 150000]

# Representative full-run wall times at N=50 (steps=1000, NOPT=600, NR=8),
# averaged over the 3 recurrent cases in RESULTS.txt / SESSION_NOTES.txt.
#   soft/hard: Run 4 (spike-timing).   cmaes/de: Runs 6/7.
# target prop has no RESULTS entry; its anchor is built bottom-up from the
# measured tp_pass primitive x the pass count of a full tp_50neuron run.
ANCHORS_50 = {
    "soft":  ("Soft homotopy",   371.0, "RESULTS Run 4 avg (397/365/352)"),
    "hard":  ("Hard surrogate",  228.0, "RESULTS Run 4 avg (239/226/218)"),
    "cmaes": ("CMA-ES",         1050.0, "RESULTS Run 6 avg (921/1115/1113)"),
    "de":    ("Diff. Evolution",5676.0, "RESULTS Run 7 avg (4356/6565/6107)"),
    # tp: built below from primitive.
}

# Which measured primitive dominates each method's per-iteration cost.
DOMINANT = {
    "soft":  "soft_grad",
    "hard":  "hard_grad",
    "cmaes": "batch_forward",   # per-member forward eval
    "de":    "batch_forward",
    "tp":    "tp_pass",
}

# Number of TP weight-recovery passes in a full tp_50neuron run
# (7 margin sweep + 3 passes + 10 iterations ~ 20), each paired with a forward.
TP_PASSES = 20


def powfit(xs, ys):
    """Fit y = a * x^b in log-log.  Returns (a, b, r2)."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    lx, ly = np.log(xs), np.log(ys)
    b, la = np.polyfit(lx, ly, 1)
    a = np.exp(la)
    pred = la + b * lx
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def get_call(rec, prim):
    d = rec[prim]
    return d.get("per_member", d["call"])


def main():
    with open(RES) as f:
        data = json.load(f)

    nsweep = data["neuron_sweep"]
    ssweep = data["step_sweep"]
    degree = data["degree"]

    Ns = [r["N"] for r in nsweep]
    prims = ["forward", "soft_grad", "hard_grad", "tp_pass", "batch_forward"]

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("# Performance scaling of the 5 weight-recovery methods")
    P()
    P(f"JAX {data.get('jax_version','?')}, CPU.  Synthetic random-recurrent networks, "
      f"fixed average in-degree = {degree} (so synapses grow ~linearly with neurons, "
      f"as in a fixed-fan-in connectome).  Times are per-call medians of the method's "
      f"core inner-loop primitive.")
    P()

    # ---- neuron scaling fits ----
    P("## 1. Per-call compute cost vs number of neurons (steps = 1000)")
    P()
    P("| N | synapses | spikes | forward (ms) | soft_grad (ms) | hard_grad (ms) | tp_pass (ms) |")
    P("|---|---|---|---|---|---|---|")
    for r in nsweep:
        P(f"| {r['N']} | {r['n_syn']} | {r['total_spikes']} | "
          f"{r['forward']['call']*1e3:.2f} | {r['soft_grad']['call']*1e3:.2f} | "
          f"{r['hard_grad']['call']*1e3:.2f} | {r['tp_pass']['call']*1e3:.1f} |")
    P()

    fits_N = {}
    P("Power-law fits  t = a * N^b  (b = scaling exponent):")
    P()
    P("| primitive | exponent b | R^2 | interpretation |")
    P("|---|---|---|---|")
    interp = {"forward": "forward sim", "soft_grad": "soft homotopy step",
              "hard_grad": "hard surrogate step", "tp_pass": "target-prop recovery pass",
              "batch_forward": "black-box per-member eval"}
    for prim in prims:
        ys = [get_call(r, prim) for r in nsweep]
        a, b, r2 = powfit(Ns, ys)
        fits_N[prim] = (a, b, r2)
        P(f"| {prim} | {b:.2f} | {r2:.3f} | {interp[prim]} |")
    P()

    # ---- step scaling fits ----
    P("## 2. Per-call compute cost vs simulation length (N = "
      f"{data['step_fixed_N']})")
    P()
    P("| steps | spikes | forward (ms) | soft_grad (ms) | hard_grad (ms) | tp_pass (ms) |")
    P("|---|---|---|---|---|---|")
    for r in ssweep:
        P(f"| {r['steps']} | {r['total_spikes']} | {r['forward']['call']*1e3:.2f} | "
          f"{r['soft_grad']['call']*1e3:.2f} | {r['hard_grad']['call']*1e3:.2f} | "
          f"{r['tp_pass']['call']*1e3:.1f} |")
    P()
    Ss = [r["steps"] for r in ssweep]
    fits_S = {}
    P("Power-law fits  t = a * steps^b:")
    P()
    P("| primitive | exponent b | R^2 |")
    P("|---|---|---|")
    for prim in prims:
        ys = [get_call(r, prim) for r in ssweep]
        a, b, r2 = powfit(Ss, ys)
        fits_S[prim] = (a, b, r2)
        P(f"| {prim} | {b:.2f} | {r2:.3f} |")
    P()

    # ---- build TP anchor from primitive ----
    rec50 = next(r for r in nsweep if r["N"] == 50)
    tp50 = rec50["tp_pass"]["call"] + rec50["forward"]["call"]
    tp_anchor = TP_PASSES * tp50
    ANCHORS_50["tp"] = ("Target prop", tp_anchor,
                        f"{TP_PASSES} x (tp_pass+forward) measured at N=50")

    # ---- extrapolation ----
    P("## 3. Extrapolated full-run wall time")
    P()
    P("Each method's 50-neuron full-run wall time is scaled by the growth of its "
      "dominant primitive (fixed step budget, steps=1000).  Assumes the same number "
      "of optimizer steps / evaluations / passes at every size — i.e. this is the "
      "*compute* growth, before any convergence-difficulty penalty.")
    P()
    P("| Method | anchor @ N=50 | exponent b | @ N=20,000 | @ N=150,000 | anchor source |")
    P("|---|---|---|---|---|---|")

    def fmt_time(s):
        if s < 60: return f"{s:.1f} s"
        if s < 3600: return f"{s/60:.1f} min"
        if s < 86400: return f"{s/3600:.1f} h"
        if s < 86400*365: return f"{s/86400:.1f} days"
        return f"{s/86400/365:.1f} yr"

    proj = {}
    for key in ["soft", "hard", "tp", "cmaes", "de"]:
        label, anchor, src = ANCHORS_50[key]
        prim = DOMINANT[key]
        _, b, _ = fits_N[prim]
        row = {}
        cells = []
        for T in TARGETS:
            factor = (T / 50.0) ** b
            row[T] = anchor * factor
            cells.append(fmt_time(row[T]))
        proj[key] = (label, b, row)
        P(f"| {label} | {fmt_time(anchor)} | {b:.2f} | {cells[0]} | {cells[1]} | {src} |")
    P()
    P("*CMA-ES and Diff. Evolution additionally carry population/covariance overhead "
      "that grows faster than the forward eval (see memory section) and become "
      "infeasible well before these compute numbers would be reached.*")
    P()

    # ---- memory ----
    P("## 4. Memory (the real wall for large N)")
    P()
    P("Activation storage is dense in time: the forward pass keeps voltages and rise "
      "values of shape (steps, N), and the backward pass keeps the voltage/refractory "
      "history.  With float32:")
    P()
    P("| N | activations (steps=1000) | CMA-ES covariance (n_syn^2) | DE population batch |")
    P("|---|---|---|---|")
    steps = 1000
    for N in [50, 1600] + TARGETS:
        nsyn = degree * N
        # ~4 dense (steps,N) arrays live simultaneously (V, rise, and bwd copies)
        act_bytes = 4 * steps * N * 4
        cma_bytes = (nsyn ** 2) * 8            # float64 covariance
        de_pop = 5 * nsyn
        de_bytes = de_pop * steps * N * 4      # vmap batch of forward states
        def hb(b):
            for u in ["B","KB","MB","GB","TB","PB"]:
                if b < 1024: return f"{b:.1f} {u}"
                b /= 1024
            return f"{b:.1f} EB"
        P(f"| {N:,} | {hb(act_bytes)} | {hb(cma_bytes)} | {hb(de_bytes)} |")
    P()
    P("- **Activations** grow linearly in both N and steps (O(steps*N)); manageable "
      "into the tens-of-GB range at N=150,000 with steps=1000 and gradient checkpointing.")
    P("- **CMA-ES** stores a dense n_syn x n_syn covariance and does an O(n_syn^2) "
      "update per generation.  At N=20,000 (n_syn~100k) that is ~80 GB and O(10^10) "
      "flops/gen — infeasible.  CMA-ES does not scale past a few thousand synapses.")
    P("- **Diff. Evolution** evaluates a population of 5*n_syn candidates at once; the "
      "vmapped forward batch alone is petabyte-scale at N=20,000.  Infeasible.")
    P()

    # ---- feasibility summary ----
    P("## 5. Bottom line")
    P()
    P("Ordering by how well each method scales to large recurrent networks:")
    P()
    P("1. **Hard surrogate** and **Soft homotopy** — gradient methods; cost grows "
      f"~N^{fits_N['hard_grad'][1]:.1f}-{fits_N['soft_grad'][1]:.1f} "
      "(super-linear mainly from the scatter/gather over ~N synapses per step plus "
      "XLA overhead).  These are the only methods with any hope at N>=20,000, and "
      "only with activation checkpointing to bound the O(steps*N) memory.")
    P("2. **Target prop** — per-pass cost grows "
      f"~N^{fits_N['tp_pass'][1]:.1f} and it needs very few passes, so raw compute is "
      "cheap, but it is pure-Python/scipy per-neuron solves and would need "
      "vectorising to be practical at scale; still, no quadratic blow-up.")
    P("3. **CMA-ES** — O(n_syn^2) covariance kills it past a few thousand synapses.")
    P("4. **Diff. Evolution** — 5*n_syn population makes each generation O(N^2*steps); "
      "worst scaling of all, infeasible earliest.")
    P()
    P("Simulation length (`steps`) is *more* expensive than neuron count: every method "
      f"scales near-quadratically in `steps` (forward b={fits_S['forward'][1]:.2f}, "
      f"soft_grad b={fits_S['soft_grad'][1]:.2f}, hard_grad b={fits_S['hard_grad'][1]:.2f}). "
      "This holds for the plain forward pass too (b~1.78, independent of spike count), "
      "so it is a property of stepping the (steps, N) voltage-history array through the "
      "time loop, not of the learning rule.  Memory in `steps` is only linear, but the "
      "super-linear compute means keeping simulations short is the single biggest lever "
      "on runtime — halving `steps` is closer to a 4x speedup than 2x.")

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
