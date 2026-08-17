"""Graph the hidden-neuron demand: the broad CREATE density vs the sharp TIM correction.

    python3 _plot_field.py "3n D" [seed] [iteration]

One panel per hidden neuron.  Two series -- the CREATE channel (a density over times) and
the TIM channel (sharp, at the neuron's own spikes) -- so a legend is required and both are
direct-labelled.  Reference marks: the neuron's TRUE spike times (hollow rings on the zero
line) and its CURRENT spikes (filled).  Palette slots 1-2, validated all-pairs light
(worst CVD dE 9.2, normal-vision 24.0).
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
C_CREATE, C_TIM = "#2a78d6", "#eb6834"

name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
AT = int(sys.argv[3]) if len(sys.argv) > 3 else 50
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
params = G.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
hidden = [n for n in range(N) if n not in outs and n != 0 and len(inc[n])]

w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
grab = {}
G.train(C, N, outs, w0.copy(), T, params, rounds=max(AT, 1), lr=G.LR,
        cb=lambda it, w, *a: grab.update(w=w.copy()) if it == AT or "w" not in grab else None)
w = grab["w"]
V = G.fsim(C, N, np.asarray(w, np.float32), params)
s = {n: G.sp(V, n) for n in range(N)}


def demand(over):
    sv = {k: getattr(G, k) for k in ("TIM_GAIN", "CREATE")}
    for k, v in over.items():
        setattr(G, k, v)
    eps, L, vs, wr = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)
    out = {n: L[n].copy() for n in range(N)}
    for k, v in sv.items():
        setattr(G, k, v)
    return out


Lc = demand(dict(TIM_GAIN=0.0))     # CREATE channel only
Lt = demand(dict(CREATE=0.0))       # TIM channel only
t = np.arange(params.steps)

# one panel per (neuron, channel): the two differ by up to 2000x in magnitude
# (chain: CREATE 1.3e-05 vs TIM 2.9e-02), so a shared y-scale hides one completely.
rows = [(n, ch) for n in hidden for ch in ("CREATE", "TIM")]
fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.9 * len(rows) + 1.9), sharex=True,
                         facecolor=SURFACE, gridspec_kw=dict(hspace=0.55))
axes = np.atleast_1d(axes)
for ax, (n, ch) in zip(axes, rows):
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color="#dcdcd6", lw=1.0, zorder=1)
    if ch == "CREATE":
        y, col = Lc[n], C_CREATE
        ax.plot(t, y, color=col, lw=1.8, zorder=3)
        lbl = f"N{n}  CREATE channel — broad density, {int((y!=0).sum())} points"
    else:
        y, col = Lt[n], C_TIM
        nz = np.nonzero(y)[0]
        ax.vlines(nz, 0, y[nz], color=col, lw=2.4, zorder=3)
        ax.plot(nz, y[nz], "o", ms=7.5, mfc=col, mec=SURFACE, mew=1.6, zorder=4)
        lbl = f"N{n}  TIM channel — sharp, at its own spikes, {len(nz)} points"
    span = max(float(np.abs(y).max()), 1e-30)
    ax.set_ylim(-1.35 * span, 1.35 * span)
    mark = -1.18 * span
    ax.plot(T[n], [mark] * len(T[n]), "o", ms=8, mfc="none", mec=MUTED, mew=1.5,
            clip_on=False, zorder=6)
    ax.plot(s[n], [mark] * len(s[n]), "o", ms=6, mfc=INK2, mec=SURFACE, mew=1.3,
            clip_on=False, zorder=7)
    ax.set_title(lbl + f"   ·   peak {span:.2e}", loc="left", fontsize=9.5,
                 color=INK, pad=6)
    ax.set_ylabel("demand", fontsize=8.5, color=INK2)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_color(INK2)
    ax.yaxis.get_offset_text().set_fontsize(7.5)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_color("#dcdcd6")
    ax.tick_params(colors=INK2, labelsize=8, length=3)
axes[-1].set_xlabel("timestep", fontsize=9, color=INK2)
fig.suptitle(f"{name} — what reaches the hidden neurons at iteration {AT}",
             x=0.008, y=0.992, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.955, "hollow ring = TRUE spike time,  filled = current spike "
         "(on the lower rule of each panel)", ha="left", fontsize=8.5, color=INK2)
fig.subplots_adjust(top=0.90, left=0.10, right=0.985, bottom=0.075)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"field_{name.replace(' ','')}_it{AT}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
for n in hidden:
    print(f"  N{n}: CREATE {int((Lc[n]!=0).sum())} pts peak {np.abs(Lc[n]).max():.2e}"
          f"   TIM {int((Lt[n]!=0).sum())} pts peak {np.abs(Lt[n]).max():.2e}"
          f"   fires {s[n][:5]}  true {T[n][:5]}")
