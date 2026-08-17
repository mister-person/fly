"""Log-log scaling plots from bench.py results."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.environ.get("OUT", "/workspace/project/performance/results.json")
with open(RES) as f:
    data = json.load(f)

prims = [("forward", "forward sim"), ("soft_grad", "soft homotopy step"),
         ("hard_grad", "hard surrogate step"), ("tp_pass", "target-prop pass")]
colors = dict(zip([p[0] for p in prims], ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]))


def fit(xs, ys):
    b, la = np.polyfit(np.log(xs), np.log(ys), 1)
    return np.exp(la), b


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ── neurons ──
ns = data["neuron_sweep"]
Ns = np.array([r["N"] for r in ns])
for prim, lab in prims:
    ys = np.array([r[prim]["call"] * 1e3 for r in ns])
    a, b = fit(Ns, ys)
    ax1.loglog(Ns, ys, "o", color=colors[prim])
    xf = np.array([Ns.min(), Ns.max()])
    ax1.loglog(xf, a * xf ** b, "-", color=colors[prim], alpha=.6,
               label=f"{lab}  (N^{b:.2f})")
ax1.set_xlabel("number of neurons  (synapses = 5·N)")
ax1.set_ylabel("per-call time (ms)")
ax1.set_title(f"Compute vs network size (steps=1000)")
ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=.25)

# ── steps ──
ss = data["step_sweep"]
Ss = np.array([r["steps"] for r in ss])
for prim, lab in prims:
    ys = np.array([r[prim]["call"] * 1e3 for r in ss])
    a, b = fit(Ss, ys)
    ax2.loglog(Ss, ys, "o", color=colors[prim])
    xf = np.array([Ss.min(), Ss.max()])
    ax2.loglog(xf, a * xf ** b, "-", color=colors[prim], alpha=.6,
               label=f"{lab}  (steps^{b:.2f})")
ax2.set_xlabel("simulation length (steps)")
ax2.set_ylabel("per-call time (ms)")
ax2.set_title(f"Compute vs simulation length (N={data['step_fixed_N']})")
ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=.25)

fig.suptitle("Spiking-network weight-recovery: per-call compute scaling", fontsize=13)
fig.tight_layout()
out = "/workspace/project/performance/scaling.png"
fig.savefig(out, dpi=120)
print("Wrote", out)
