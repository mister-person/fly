"""Plot partial-observation TP results: output loss and network recovery vs
fraction of neurons observed, for cold (random) and warm (found) starts."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/workspace/project/performance"
runs = {}
for init in ("random", "found"):
    p = f"{D}/partial_obs_{init}.json"
    if os.path.exists(p):
        runs[init] = json.load(open(p))

if not runs:
    raise SystemExit("no partial_obs_*.json found")

cases = list(next(iter(runs.values()))["results"].keys())
fig, axes = plt.subplots(2, len(cases), figsize=(5 * len(cases), 8.5))
if len(cases) == 1:
    axes = axes.reshape(2, 1)

colors = {"random": "#d62728", "found": "#1f77b4"}
labels = {"random": "cold start (random init)", "found": "warm start (soft-homotopy init)"}

for ci, cname in enumerate(cases):
    ax_loss, ax_net = axes[0, ci], axes[1, ci]
    for init, run in runs.items():
        r = run["results"][cname]
        rows = r["rows"]
        fs = [x["f"] for x in rows]
        loss = [x["self_mean"] for x in rows]
        lstd = [x["self_std"] for x in rows]
        net = [x["net_match"] for x in rows]
        ntrue = r["n_true_active"]
        ax_loss.errorbar(fs, loss, yerr=lstd, marker="o", color=colors[init],
                         label=labels[init], capsize=3)
        ax_net.plot(fs, np.array(net) / ntrue, marker="o", color=colors[init],
                    label=labels[init])
    # references
    r0 = next(iter(runs.values()))["results"][cname]
    ax_loss.axhline(r0["rand_loss"], ls=":", color="gray", label="random-init floor")
    if r0["soft_loss"] is not None:
        ax_loss.axhline(r0["soft_loss"], ls="--", color="green", label="soft homotopy")
    ax_net.axhline(r0["rand_netmatch"] / r0["n_true_active"], ls=":", color="gray",
                   label="random-init floor")

    ax_loss.set_title(cname)
    ax_loss.set_ylabel("output MSE loss (lower=better)")
    ax_loss.set_xlabel("fraction of hidden neurons observed")
    ax_loss.legend(fontsize=7)
    ax_loss.grid(alpha=.25)
    ax_net.set_ylabel("network spike-count recovery\n(frac of active neurons matched)")
    ax_net.set_xlabel("fraction of hidden neurons observed")
    ax_net.set_ylim(0, 1)
    ax_net.legend(fontsize=7)
    ax_net.grid(alpha=.25)

fig.suptitle("Partial-observation target propagation\n(outputs always observed; "
             "x = fraction of the 46 hidden neurons also observed)", fontsize=12)
fig.tight_layout()
out = f"{D}/partial_obs.png"
fig.savefig(out, dpi=120)
print("Wrote", out)
