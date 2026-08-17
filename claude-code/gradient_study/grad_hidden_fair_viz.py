"""Visualise the fair hidden-credit test: per-neuron voltage under ORACLE
(hidden get true targets) vs NO-INFO (hidden frozen random).  Shows that with no
hidden information the hidden neurons fire at the wrong times, so the output
lands far from its target even though the count matches.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from grad_hidden_fair import (full_sim, spikes_of, train, N, OUTPUT, w_true, T)
from grad_method import TH, DIR


def representative_run(targets, train_ns, t_out, seeds=6):
    """Return a MEDIAN-timing (count-ok) run as a fair example, plus the timing
    errors across all count-ok seeds (to show reliability, not cherry-pick)."""
    runs = []      # count-ok: (terr, V, sp)
    allruns = []   # every seed: (V, sp)
    for seed in range(seeds):
        w = train(targets, train_ns, seed, iters=60, inner=30)
        V = full_sim(w)
        sp = spikes_of(V, OUTPUT)
        allruns.append((V, sp))
        if len(sp) == len(t_out):
            terr = float(np.mean([abs(a - b) for a, b in zip(sorted(sp), sorted(t_out))]))
            runs.append((terr, V, sp))
    if not runs:                                   # no count match -> just show seed 0
        V, sp = allruns[0]
        return V, sp, []
    runs.sort(key=lambda r: r[0])
    terrs = [r[0] for r in runs]
    med = runs[len(runs) // 2]     # median-timing representative
    return med[1], med[2], terrs


def main():
    Vt = full_sim(w_true)
    T_true = {n: spikes_of(Vt, n) for n in range(N)}
    t_out = T_true[OUTPUT]
    hidden = [1, 2, 3, 4]

    print("training ORACLE ...")
    V_or, sp_or, te_or = representative_run(
        {**{h: T_true[h] for h in hidden}, OUTPUT: t_out}, hidden + [OUTPUT], t_out)
    print(f"  oracle output {sp_or}  timing errs across seeds: {np.round(te_or,0)}")
    print("training NO-INFO ...")
    V_ni, sp_ni, te_ni = representative_run({OUTPUT: t_out}, [OUTPUT], t_out)
    print(f"  no-info output {sp_ni}  timing errs across seeds: {np.round(te_ni,0)}")

    def rng_str(te):
        return f"{min(te):.0f}–{max(te):.0f} (med {sorted(te)[len(te)//2]:.0f})" if te else "n/a"

    labels = {1: "H1", 2: "H2", 3: "H3", 4: "H4", 5: "OUTPUT"}
    rows = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 10), sharex=True)
    for r, n in enumerate(rows):
        ax = axes[r]
        ax.plot(V_or[:, n], color="C0", lw=1.6, label="ORACLE hidden")
        ax.plot(V_ni[:, n], color="C3", lw=1.4, ls="--", label="NO-INFO hidden")
        ax.axhline(TH, color="k", ls="--", lw=1)
        for tt in T_true[n]:
            ax.axvline(tt, color="green", ls=":", lw=1)
        so = spikes_of(V_or, n); sn = spikes_of(V_ni, n)
        ax.plot(so, [TH * 1.05] * len(so), "v", color="C0", ms=7)
        ax.plot(sn, [TH * 1.18] * len(sn), "v", color="C3", ms=7)
        ax.set_ylim(0, TH * 1.4); ax.set_ylabel(labels[n])
        emph = "  <-- target " + str(t_out) if n == OUTPUT else ""
        ax.set_title(f"{labels[n]}: true times {T_true[n]}   "
                     f"oracle {so}  vs  no-info {sn}{emph}",
                     fontsize=9, color=("C0" if n != OUTPUT else "k"),
                     fontweight=("bold" if n == OUTPUT else "normal"))
        if r == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("t")
    fig.suptitle("Fair hidden-credit test — voltage per neuron: ORACLE vs NO-INFO hidden\n"
                 "(green ⋮ = true target times; ▼ = spikes).  Output timing error across "
                 f"seeds:  ORACLE {rng_str(te_or)}   vs   NO-INFO {rng_str(te_ni)} steps.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{DIR}/grad_hidden_fair.png", dpi=120)
    print(f"wrote {DIR}/grad_hidden_fair.png")


if __name__ == "__main__":
    main()
