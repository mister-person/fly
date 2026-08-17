"""Continuous timing-aware objective, replacing the binary create/suppress.

Instead of "is there a spike within WINDOW / suppress spikes outside it", use one
smooth loss: a soft (sigmoid) spike train, exponentially smoothed, matched to a
target train.  There is no hard window, so the soft spike slides continuously all
the way to the target — no dead band, and (with a sharp enough kernel) step-exact.

  s(t)   = sigmoid(beta*(V(t)/th - 1))          soft spike
  S(t)   = expconv(s, tau)                       van-Rossum smoothing
  S*(t)  = expconv(delta at each target, tau)
  L      = sum (S - S*)^2
  dL/dw  = [revconv(2(S-S*)) * beta/th * s(1-s)]  .  dV/dw     (dV/dw from tangent)

We re-run the 2-neuron target sweep and confirm the unreachable band disappears.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from grad_method import lif_tangent, TH, DIR
from grad_multi_neuron import voltage_grad, WINDOW


def expconv(x, decay):
    S = np.zeros_like(x, dtype=float); acc = 0.0
    for i in range(len(x)):
        acc = acc * decay + x[i]; S[i] = acc
    return S


def revconv(x, decay):
    R = np.zeros_like(x, dtype=float); acc = 0.0
    for i in range(len(x) - 1, -1, -1):
        acc = acc * decay + x[i]; R[i] = acc
    return R


def continuous_grad(w, ia, targets, T, beta_surr=30.0, tau=10.0):
    """Van-Rossum spike-timing loss on HARD spikes (delta-based on both sides, so
    widths match — no shrink-to-death), gradient via a narrow surrogate slope.
    One continuous force, no window."""
    V, spikes, dV = lif_tangent(w, ia, T)
    decay = np.exp(-1.0 / tau)
    found = (V >= TH).astype(float)
    S = expconv(found, decay)
    tgt = np.zeros(T)
    for t in targets:
        if 0 <= t < T:
            tgt[t] = 1.0
    Stgt = expconv(tgt, decay)
    err = S - Stgt
    d_found = revconv(2.0 * err, decay)                # dL/d(found spike)
    a = 1.0 / (1.0 + np.exp(-beta_surr * (V / TH - 1.0)))
    slope = a * (1.0 - a) * (beta_surr / TH)           # narrow surrogate
    dLdV = d_found * slope
    g = dLdV @ dV
    return g, spikes


# ── 2-neuron setup (same as grad_stuck_2neuron) ──────────────────────────────
T = 220
ia = np.zeros((1, T), bool); ia[0, 15] = True


def crossing(w):
    sp = lif_tangent(np.array([float(w)]), ia, T)[1]
    return sp[0] if sp else None


def train(w0, target, grad_fn, iters=400, lr=4.0):
    """Adam so the step slows near the solution (unit-norm overshoots below firing)."""
    w = np.array([float(w0)]); m = np.zeros_like(w); v = np.zeros_like(w)
    for t in range(1, iters + 1):
        g = np.nan_to_num(grad_fn(w, target))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
        w = np.clip(w - lr * mh / (np.sqrt(vh) + 1e-12), 20, 3000)
    sp = lif_tangent(w, ia, T)[1]
    return sp[0] if sp else np.nan


def main():
    c0 = crossing(700.0)
    print(f"natural crossing at w=700: {c0};  WINDOW={WINDOW}")
    targets = list(range(c0 - 40, c0 + 45, 3))

    fin_binary, fin_cont = [], []
    for tg in targets:
        fin_binary.append(train(700.0, tg,
                                lambda w, t: voltage_grad(w, ia, [t], T, suppress=True)[0]))
        fin_cont.append(train(700.0, tg, lambda w, t: continuous_grad(w, ia, [t], T)[0]))

    def band(finals):
        stuck = [tg for tg, fn in zip(targets, finals) if abs(fn - c0) <= 3]
        return (min(stuck), max(stuck), max(stuck) - min(stuck)) if stuck else None

    bb, cb = band(fin_binary), band(fin_cont)
    print(f"binary     unreachable band: {bb}")
    print(f"continuous unreachable band: {cb}")
    err_b = np.nanmean([abs(f - t) for f, t in zip(fin_binary, targets)])
    err_c = np.nanmean([abs(f - t) for f, t in zip(fin_cont, targets)])
    print(f"mean |final-target|:  binary={err_b:.1f}   continuous={err_c:.1f}")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ax.plot(targets, fin_binary, "o-", color="C3", ms=4, label="binary create/suppress")
    ax.plot(targets, fin_cont, "s-", color="C0", ms=4, label="continuous (soft van-Rossum)")
    ax.plot(targets, targets, color="gray", ls="--", label="ideal (final=target)")
    ax.axvspan(c0 - WINDOW, c0 + WINDOW, color="orange", alpha=.15, label="±WINDOW")
    ax.set_title("2-neuron target sweep: continuous force removes the dead band")
    ax.set_xlabel("requested target time"); ax.set_ylabel("achieved spike time")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{DIR}/grad_continuous_2neuron.png", dpi=120)
    print(f"wrote {DIR}/grad_continuous_2neuron.png")


def combined_grad(w, ia, targets, T):
    """Revive with the create-margin push while under-firing, then switch to the
    continuous ST loss for timing + suppression."""
    V, sp, dV = lif_tangent(w, ia, T)
    missing = [tt for tt in targets if not any(abs(s - tt) <= 40 for s in sp)]
    if missing and len(sp) < len(targets):
        return voltage_grad(w, ia, targets, T, suppress=False)[0], sp   # create-only
    return continuous_grad(w, ia, targets, T)


def train_neuron(w0, iavar, targets, Tvar, grad_fn, iters=400, lr=4.0, unitnorm=False):
    w = np.array(w0, float); m = np.zeros_like(w); v = np.zeros_like(w)
    for t in range(1, iters + 1):
        g = np.nan_to_num(grad_fn(w))
        if unitnorm:
            gn = np.linalg.norm(g)
            if gn > 1e-30:
                w = np.clip(w - lr * g / gn, 20, 3000)
        else:
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
            w = np.clip(w - lr * mh / (np.sqrt(vh) + 1e-12), 20, 3000)
    return w


def suite():
    """Compare binary create/suppress vs continuous(combined) on pulse cases:
    pulse-selection PASS and TIMING error."""
    import grad_test_suite as S
    from grad_multi_neuron import voltage_grad as binvg
    rng = np.random.default_rng(0)
    cats = []
    # (n_pulses, target_pulses, init) generator per category
    def gen():
        for k in range(1, 5):        # create from dead
            cats.append(("create", 4, sorted(rng.choice(4, k, False).tolist()), [100.]*4))
        for k in range(1, 4):        # suppress
            cats.append(("suppress", 4, sorted(rng.choice(4, k, False).tolist()), [700.]*4))
        for _ in range(4):           # mixed
            cats.append(("mixed", 4, sorted(rng.choice(4, rng.integers(1,4), False).tolist()),
                         [rng.choice([100.,700.]) for _ in range(4)]))
    gen()

    print(f"{'category':10s} {'method':12s} {'sel PASS':>9s} {'mean timing err':>16s}")
    for method, gfac in [("binary", None), ("continuous", None)]:
        npass = 0; ntot = 0; terrs = []
        for cat, npul, tp, init in cats:
            onsets = S.onsets_for(npul)
            Tc = npul * S.SPACING + 120
            ia = np.zeros((npul, Tc), bool)
            for i, o in enumerate(onsets):
                ia[i, o] = True
            targets = [S.crossing_time(onsets, i, Tc) for i in tp]
            gbin = lambda w: binvg(w, ia, targets, Tc, suppress=True)[0]
            if method == "binary":
                w = train_neuron(init, ia, targets, Tc, gbin, iters=350, lr=4.0, unitnorm=True)
            else:
                # single ANNEALED continuous-ST objective: wide surrogate (revive) ->
                # sharp (precise timing).  Adam.
                w = np.array(init, float); m = np.zeros_like(w); v = np.zeros_like(w)
                ITER = 500
                for t in range(1, ITER + 1):
                    beta_surr = 3.0 + (30.0 - 3.0) * (t / ITER)
                    g = np.nan_to_num(continuous_grad(w, ia, targets, Tc, beta_surr=beta_surr)[0])
                    m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
                    mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
                    w = np.clip(w - 4.0 * mh / (np.sqrt(vh) + 1e-12), 20, 3000)
            sp = lif_tangent(w, ia, Tc)[1]
            ok, _ = S.evaluate(sp, tp, onsets)
            npass += int(ok); ntot += 1
            if len(sp) == len(targets) and targets:
                terrs.append(np.mean([abs(a-b) for a,b in zip(sorted(sp), sorted(targets))]))
        mt = f"{np.mean(terrs):.1f}" if terrs else "-"
        print(f"{'ALL':10s} {method:12s} {npass:>4d}/{ntot:<4d} {mt:>16s}")


if __name__ == "__main__":
    main()
    print()
    suite()
