"""Voltage, field and local demand per neuron, for the FIELD pathway (field_trace).

    python3 _plot_field_case.py "4n F" [seed] [rounds]

One panel per neuron, three series on a COMMON INDEXED base -- volts, volts and volts-of-
demand are not the same units, but each has a natural reference and indexing makes 1.0 mean
something for all three at once:

    V / threshold   = 1  -> the neuron fires
    F / peak|F|          -> +-1, sign = wanted / not wanted here
    L / peak|L|          -> +-1, the signed demand actually driving the weights

Markers: hollow ring = true/target spike, filled = current, and the field's BUMPS (the spikes
it is requesting, one per positive run) are drawn on the field itself, since "how many and
when" is the thing to read.  Palette matches the other scripts in this directory
(validated all-pairs light: worst CVD dE 9.2, normal-vision 24.0).
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import field_trace as F
from _diag import CASES, steps_for

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
C_V, C_F, C_L = "#3f3f3c", "#2a78d6", "#eb6834"

name = sys.argv[1] if len(sys.argv) > 1 else "4n F"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
DENSE_N = int(os.environ.get("F_DENSE_N", "60"))   # above this many nonzero demand samples,
# draw the demand as an envelope rather than one stem per sample

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
params = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, params), n) for n in range(N)}
w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)

# F_W lets the plot be drawn at a GIVEN weight vector instead of one this run reached --
# needed to ask "what would a different setting say at the SAME state?", which training to
# its own stuck point cannot answer, since the setting changes where it gets stuck.
_wov = os.environ.get("F_W")
live = {}
if _wov:
    live["w"] = np.array([float(x) for x in _wov.split(",")])
else:
    # USE THE RETURNED WEIGHTS, NOT THE LIVE ONES.  train() keeps the best-ever iterate
    # (KEEP_BEST) and that is what the suite scores; the callback sees the last iterate,
    # which can differ a lot -- on 8n M seed5 the live weights give 5 output spikes where
    # the scored ones give 6.  Plotting the live weights makes the figure disagree with the
    # number it is supposed to explain.  F_LIVE=1 restores the last iterate deliberately.
    _ret = F.train(C, N, outs, w0.copy(), T, params, rounds=ROUNDS, lr=F.LR,
                   cb=lambda it, w, *a: live.update(live_w=np.asarray(w, float).copy()))
    live["w"] = live["live_w"] if os.environ.get("F_LIVE") else np.asarray(_ret, float)
t = np.arange(params.steps)


def draw(w, tag, fname):
    w = np.asarray(w, float)
    V = F.fsim(C, N, np.asarray(w, np.float32), params)
    s = {n: F.sp(V, n) for n in range(N)}
    g, Fl, L, ep = F.gradient(C, N, w, s, params.steps, {o: T[o] for o in outs})
    # ROW ORDER.  Numeric order is meaningless for reading causality; F_ORDER=auto solves the
    # minimum-feedback-arc-set exactly (subset DP, see _order.py) so each neuron appears after
    # as many of its drivers as any ordering can manage, and a panel can be read top-down as
    # cause-then-effect.  A comma list overrides it.
    _ord = os.environ.get("F_ORDER")
    if _ord == "auto":
        from _order import min_fas_order
        _o, _nb = min_fas_order(E, N, fixed_first=0)
        rows = [n for n in _o if n != 0]
        print(f"   order (min feedback arc set, {_nb} backward): "
              + " -> ".join(f"N{n}" for n in _o))
    elif _ord:
        rows = [int(x) for x in _ord.split(",") if int(x) != 0]
    else:
        rows = list(range(1, N))
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.7 * len(rows) + 2.3),
                             sharex=True, facecolor=SURFACE,
                             gridspec_kw=dict(hspace=0.45))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, rows):
        ax.set_facecolor(SURFACE)
        ax.axhline(1.0, color=MUTED, lw=1.1, ls=(0, (5, 4)), zorder=2)
        ax.axhline(0.0, color="#e4e4df", lw=1.0, zorder=1)
        v = np.asarray(V[:, n], float) / F.TH
        ax.plot(t, v, color=C_V, lw=1.6, label="V / threshold", zorder=3)
        ax.plot(s[n], [v[q] for q in s[n]], "o", ms=7, mfc=C_V, mec=SURFACE, mew=1.6,
                zorder=7)
        bumps = F.bumps_of(Fl[n]) if Fl[n].any() else []
        if Fl[n].any():
            pk = max(float(np.abs(Fl[n]).max()), 1e-30)
            ax.plot(t, Fl[n] / pk, color=C_F, lw=1.6, label=f"field / {pk:.1e}", zorder=4)
            if bumps:
                ax.plot([b[0] for b in bumps], [Fl[n][b[0]] / pk for b in bumps], "o",
                        ms=8.5, mfc=C_F, mec=SURFACE, mew=1.8, zorder=8)
        if L[n].any():
            pl = max(float(np.abs(L[n]).max()), 1e-30)
            nz = np.nonzero(L[n])[0]
            if len(nz) > DENSE_N:
                # A DENSE DEMAND IS A CURVE, NOT A SET OF EVENTS.  In DENSITY mode L is
                # nonzero at nearly every sample, and one stem-plus-marker per sample renders
                # as a solid block that hides its own shape (and the field underneath it).
                # Draw the envelope instead; the sparse form stays for bump-derived demands,
                # where each stem really is one request.
                ax.fill_between(t, 0, L[n] / pl, color=C_L, alpha=0.30, lw=0, zorder=5)
                ax.plot(t, L[n] / pl, color=C_L, lw=1.1, zorder=6,
                        label=f"demand / {pl:.1e}  ({len(nz)} samples)")
            else:
                ax.vlines(nz, 0, L[n][nz] / pl, color=C_L, lw=2.0, zorder=5)
                ax.plot(nz, L[n][nz] / pl, "o", ms=5.5, mfc=C_L, mec=SURFACE, mew=1.2,
                        zorder=6, label=f"demand / {pl:.1e}")
        ax.set_ylim(-1.45, 2.2)
        mark = -1.28
        ref = T[n] if n in T else []
        ax.plot(ref, [mark] * len(ref), "o", ms=7.5, mfc="none", mec=MUTED, mew=1.4,
                clip_on=False, zorder=9)
        ax.plot(s[n], [mark] * len(s[n]), "o", ms=5.5, mfc=INK2, mec=SURFACE, mew=1.2,
                clip_on=False, zorder=10)
        lbl = "OUTPUT" if n in outs else "hidden"
        # SHOW THE COUNTS.  The lists are truncated to fit, and a truncated list reads as a
        # complete one: 4n G's output has SEVEN targets and displayed six, which is exactly
        # the quantity the case is failing on.
        def _tr(v):
            return f"{list(v)[:6]}" + (f"+{len(v)-6} more" if len(v) > 6 else "")
        head = (f"N{n} ({lbl})   fires {len(s[n])}: {_tr(s[n])}   ·   "
                f"{'target' if n in outs else 'true'} {len(T[n])}: {_tr(T[n])}   ·   "
                f"asks {len(bumps)}: {_tr([b[0] for b in bumps])}")
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
    # a 16-weight case overflows a one-line title, so wrap the weight list
    _ws = F.wstr(C, w).split("  ")
    _per = 8
    _lines = ["  ".join(_ws[i:i + _per]) for i in range(0, len(_ws), _per)]
    fig.suptitle(f"{name} {tag} — FIELD pathway", x=0.008, y=0.994, ha="left",
                 fontsize=11.5, color=INK)
    for _li, _ln in enumerate(_lines):
        fig.text(0.008, 0.976 - 0.014 * _li, _ln, ha="left", fontsize=8.6, color=INK2)
    fig.text(0.008, 0.976 - 0.014 * len(_lines) - 0.004,
             "hollow ring = true/target spike, filled = current   ·   "
             f"large dots on the field = the spikes it REQUESTS   ·   "
             f"max|g| = {np.abs(g).max():.2e}",
             ha="left", fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.905, left=0.095, right=0.985, bottom=0.055)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)
    for n in rows:
        print(f"   N{n}: fires {s[n][:6]}  true {T[n][:6]}  "
              f"asks {[b[0] for b in F.bumps_of(Fl[n])][:6]}")


tagn = name.replace(" ", "_")
draw(live["w"], os.environ.get("F_TAG", f"STUCK (seed {seed}, live)"),
     f"ft_{tagn}_stuck{seed}{os.environ.get('F_SUF','')}.png")
draw(np.array(Wl, float), "at the TRUE weights", f"ft_{tagn}_true.png")
