"""3-cycle with FIELD_XING on: voltage + both field variables, at the stuck point and at truth.

    python3 _plot_3cycle_field.py [seed]

Per neuron: a voltage panel, then (for hidden neurons) urgency and implied_w.  Separate
panels rather than twin axes -- volts, volts and weight are three different units.  On the
implied_w panel the neuron's current mean outgoing weight is a dashed rule, so the CROSSINGS
(where implied_w meets it) are readable, and they are marked.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import grad_trace as G
from _diag import CASES, steps_for

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
C_V, C_URG, C_IW = "#3f3f3c", "#2a78d6", "#eb6834"

name = sys.argv[1] if len(sys.argv) > 1 else "3-cycle"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
G.FIELD_XING = float(os.environ.get("FIELD_XING", "1.0"))
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
params = G.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
hidden = [n for n in range(N) if n not in outs and n != 0 and len(inc[n])]
t = np.arange(params.steps)

w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
live = {}
G.train(C, N, outs, w0.copy(), T, params, rounds=int(os.environ.get("ROUNDS","800")), lr=G.LR,
        cb=lambda it, w, *a: live.update(w=w.copy()))


def draw(w, tag, fname):
    w = np.asarray(w, float)
    V = G.fsim(C, N, np.asarray(w, np.float32), params)
    s = {n: G.sp(V, n) for n in range(N)}
    eps, L, vsub, wr = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)
    U, IW = G.demand_field(C, N, w, s, params.steps, {o: T[o] for o in outs}, vsub, eps)
    X = G.field_crossings(C, N, w, U, IW, params.steps)
    # ONE PANEL PER NEURON, three series INDEXED TO A COMMON BASE.  Volts, volts and
    # weight cannot share a raw axis (that would be a triple-axis chart), but each has a
    # natural reference, and indexing makes 1.0 mean something for all three at once:
    #    V / threshold        = 1  ->  the neuron fires
    #    implied_w / w_now    = 1  ->  the crossing condition
    #    urgency / peak            ->  +-1, sign = wanted / not wanted
    rows = list(range(1, N))   # N3 now has a field too
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.6 * len(rows) + 2.2),
                             sharex=True, facecolor=SURFACE,
                             gridspec_kw=dict(hspace=0.42))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, rows):
        ax.set_facecolor(SURFACE)
        ax.axhline(1.0, color=MUTED, lw=1.1, ls=(0, (5, 4)), zorder=2)
        ax.axhline(0.0, color="#e4e4df", lw=1.0, zorder=1)
        v = np.asarray(V[:, n], float) / G.TH
        ax.plot(t, v, color=C_V, lw=1.6, label="V / threshold", zorder=3)
        ax.plot(s[n], [v[q] for q in s[n]], "o", ms=7, mfc=C_V, mec=SURFACE, mew=1.6,
                zorder=6)
        if U[n].any():
            u = U[n]
            pk = max(float(np.abs(u).max()), 1e-30)
            ax.plot(t, u / pk, color=C_URG, lw=1.6, label=f"urgency / {pk:.1e}", zorder=4)
            iw = IW[n]
            fin = np.isfinite(iw)
            outs_n = np.where(C[:, 0] == n)[0]
            wn = float(np.mean([abs(w[k]) for k in outs_n])) if len(outs_n) else 1.0
            ax.plot(t[fin], np.clip(iw[fin] / max(wn, 1e-9), -1.0, 3.0), color=C_IW,
                    lw=1.6, label=f"implied_w / {wn:.0f}", zorder=5)
            xs = [q for q, _ in X.get(n, [])]
            if xs:
                ax.plot(xs, [1.0] * len(xs), "o", ms=8.5, mfc=C_IW, mec=SURFACE, mew=1.8,
                        zorder=7)
        ax.set_ylim(-1.35, 3.15)
        mark = -1.2
        ref = T[n] if n in T else []
        ax.plot(ref, [mark] * len(ref), "o", ms=7.5, mfc="none", mec=MUTED, mew=1.4,
                clip_on=False, zorder=8)
        ax.plot(s[n], [mark] * len(s[n]), "o", ms=5.5, mfc=INK2, mec=SURFACE, mew=1.2,
                clip_on=False, zorder=9)
        lbl = "OUTPUT" if n in outs else "hidden"
        head = (f"N{n} ({lbl})   fires {s[n][:6]}   ·   "
                f"{'target' if n in outs else 'true'} {T[n][:6]}")
        if U[n].any():
            head += f"   ·   crossings {[q for q,_ in X.get(n,[])][:6]}"
        ax.set_title(head, loc="left", fontsize=9, color=INK, pad=5)
        ax.set_ylabel("indexed", fontsize=8.5, color=INK2)
        ax.legend(frameon=False, fontsize=7.8, loc="upper right", ncol=3,
                  labelcolor=INK2, handlelength=1.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        for sp_ in ("left", "bottom"):
            ax.spines[sp_].set_color("#dcdcd6")
        ax.tick_params(colors=INK2, labelsize=8, length=3)
    axes[-1].set_xlabel("timestep", fontsize=9, color=INK2)
    fig.suptitle(f"{name} {tag} — voltage and both field variables   "
                 f"w = {[int(round(x)) for x in w]}",
                 x=0.008, y=0.988, ha="left", fontsize=11.5, color=INK)
    fig.text(0.008, 0.955, "dashed 1.0 = threshold for V AND the crossing condition for implied_w   ·   hollow ring = true/target spike, filled = current", ha="left", fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.905, left=0.105, right=0.985, bottom=0.055)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)
    for n in hidden:
        print(f"   N{n}: fires {s[n][:6]}  true {T[n][:6]}  crossings "
              f"{[q for q,_ in X.get(n,[])][:6]}")


draw(live["w"], f"STUCK (seed {seed}, live)", f"casefield_{name.replace(chr(32),chr(95))}_stuck{seed}.png")
draw(list(Wl), "at the TRUE weights", f"casefield_{name.replace(chr(32),chr(95))}_true.png")
