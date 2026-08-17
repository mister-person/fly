"""Does the field keep improving with more rounds, or does it stall?

    python3 _plot_rounds.py "50n A" 6400 [seeds...]

Runs ONE long training per seed and samples the graded metric from inside the loop via `cb=`,
which is both cheaper than N separate runs and honest about the trajectory -- `train()` returns
the best-ever iterate, so comparing end-of-run values across separate runs would smuggle in a
max over rounds and make any curve look monotone.

The learning rate is a sawtooth (decay is indexed by `ait`, which RESTART_EVERY resets), so a
longer run is genuinely more search rather than a deeper anneal.  That is what makes the
question well posed.

Three panels share the round axis: count agreement, timing error on the count-matched neurons,
and the mean weight magnitude against truth -- the last tests whether the systematic LATE bias
seen in the raster is a uniform magnitude deficit that more rounds would close.
"""
import os, sys, json
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
EVERY = 100


def run(args):
    name, seed, rounds = args
    import numpy as np
    import field_trace as F
    from _diag import CASES, steps_for
    E, N, outs, Wl = CASES[name]
    C = np.array(E, np.int32)
    p = F.mkparams(steps_for(name))
    W = np.array(Wl, np.float32)
    T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
    wtrue = np.abs(np.asarray(Wl, float))
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    hist = []

    def score(it, w):
        V = F.fsim(C, N, np.asarray(w, np.float32), p)
        dcs, dts, late = [], [], []
        for o in outs:
            f, t = list(F.sp(V, o)), list(T[o])
            dcs.append(abs(len(f) - len(t)))
            if len(f) == len(t):
                dts += [abs(a - b) for a, b in zip(f, t)]
                late += [a - b for a, b in zip(f, t)]
        hist.append(dict(it=it, dcount=float(np.mean(dcs)),
                         ok=float(np.mean([d == 0 for d in dcs])),
                         dt=float(np.mean(dts)) if dts else float("nan"),
                         bias=float(np.mean(late)) if late else float("nan"),
                         wmag=float(np.mean(np.abs(w)) / np.mean(wtrue))))

    def cb(it, w, upd, g, spall, Ff, L):
        if it % EVERY == 0 or it == 1:
            score(it, w)

    F.train(C, N, outs, w0.copy(), T, p, rounds=rounds, lr=F.LR, cb=cb)
    return seed, hist


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "50n A"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6400
    seeds = [int(x) for x in sys.argv[3:]] or [0, 1, 2, 3]
    with mp.Pool(len(seeds)) as pool:
        res = pool.map(run, [(name, s, rounds) for s in seeds])
    tag = name.replace(" ", "_")
    json.dump({str(s): h for s, h in res}, open(f"{HERE}/rounds_{tag}.json", "w"))

    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
    HUES = ["#2a78d6", "#eb6834", "#2f9e6b", "#8f5fd0"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), facecolor=SURFACE)
    panels = [("ok", "outputs with correct spike count", "fraction"),
              ("dt", "|Δt| on count-matched outputs", "timesteps"),
              ("wmag", "mean |w| ÷ mean |w| of truth", "ratio")]
    for ax, (key, ttl, ylab) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, (s, h) in enumerate(res):
            x = [r["it"] for r in h]
            ax.plot(x, [r[key] for r in h], "-", lw=1.6, color=HUES[i % len(HUES)],
                    label=f"seed {s}")
        if key == "wmag":
            ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_title(ttl, fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlabel("round", fontsize=9, color=INK2)
        ax.set_ylabel(ylab, fontsize=9, color=INK2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("bottom", "left"):
            ax.spines[s_].set_color("#dcdcd6")
        ax.tick_params(colors=INK2, labelsize=8.2, length=3)
        ax.grid(axis="y", color="#ececE6", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.4, labelcolor=INK2, loc="lower right")
    fig.suptitle(f"{name} — does more training keep helping?  ({rounds} rounds, "
                 f"live iterate sampled every {EVERY})",
                 x=0.006, y=0.985, ha="left", fontsize=12, color=INK)
    fig.subplots_adjust(top=0.79, left=0.055, right=0.99, bottom=0.15, wspace=0.28)
    out = f"{HERE}/rounds_{tag}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)
    for s, h in res:
        q = [h[0]] + [h[len(h) * k // 4 - 1] for k in (1, 2, 3, 4)]
        print(f"  seed {s}: " + "  ".join(
            f"@{r['it']} ok={r['ok']:.2f} dt={r['dt']:.1f} bias={r['bias']:+.1f}" for r in q))
