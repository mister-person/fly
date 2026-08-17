"""Large sparse networks in the SUB-CRITICAL regime, with inhibitory neurons.

    python3 _bignets.py            # search seeds and report the usable ones

Every case in the hand-built suite has weights above W_CRIT = 444.5, so a single presynaptic
spike crosses threshold on its own and the network is a chain of one-to-one triggers.  These
nets are the opposite: apart from a few STARTER edges off the input, every weight is drawn
below W_CRIT, so no neuron can be fired by one arrival and all activity is genuine
coincidence detection.

Neurons, not edges, carry the sign (Dale's law): an inhibitory neuron's outgoing weights are
ALL negative.  That is both the biological convention and a harder test than a scattering of
negative edges, since suppression then arrives in correlated bundles.

generate() is deterministic in `seed`, so a case is stored as its parameters rather than a
literal edge list.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import field_trace as F

W_CRIT = float(F.W_CRIT)


def generate(seed, N=50, n_start=4, n_out=10, fan_in=5, frac_inh=0.2,
             w_lo=60.0, w_hi=380.0, w_start=(650.0, 950.0)):
    """(edges, N, outs, W).  Neuron 0 is the input; it fires on its own schedule."""
    rng = np.random.default_rng(seed)
    inh = set(rng.choice(np.arange(1, N), size=int(frac_inh * (N - 1)),
                         replace=False).tolist())
    start = list(range(1, n_start + 1))                 # driven directly, super-critical
    outs = list(range(N - n_out, N))                    # the last few are the outputs
    E, W = [], []
    for n in start:
        E.append([0, n]); W.append(float(rng.uniform(*w_start)))
    for n in range(n_start + 1, N):
        # draw presynaptic partners from EARLIER neurons, so the graph is a DAG plus the
        # explicit feedback added below -- otherwise most seeds are either silent or saturated
        pool = [k for k in range(1, n) if k != n]
        if not pool:
            continue
        src = rng.choice(pool, size=min(fan_in, len(pool)), replace=False)
        for k in src:
            w = float(rng.uniform(w_lo, w_hi))
            E.append([int(k), n]); W.append(-w if int(k) in inh else w)
    # a few feedback edges, so it is not purely feedforward
    for _ in range(max(2, N // 12)):
        a = int(rng.integers(n_start + 1, N)); b = int(rng.integers(1, a))
        w = float(rng.uniform(w_lo, w_hi))
        E.append([a, b]); W.append(-w if a in inh else w)
    return E, N, outs, W, sorted(inh)


def audit(seed, steps=1040, **kw):
    E, N, outs, W, inh = generate(seed, **kw)
    C = np.array(E, np.int32)
    p = F.mkparams(steps)
    V = F.fsim(C, N, np.asarray(W, np.float32), p)
    sp = {n: F.sp(V, n) for n in range(N)}
    live = [n for n in range(1, N) if sp[n]]
    ocount = [len(sp[o]) for o in outs]
    sub = sum(1 for w in W if abs(w) < W_CRIT)
    # does any neuron fire from ONE arrival?  (a spike with a single contributor over TH)
    return dict(seed=seed, edges=len(E), inh=len(inh), live=len(live), N=N,
                ocount=ocount, subcrit=sub / len(W), outs=outs,
                busiest=max((len(sp[n]) for n in range(1, N)), default=0),
                E=E, W=W, inhset=inh)


if __name__ == "__main__":
    print(f"W_CRIT = {W_CRIT:.1f}   (a weight below this cannot fire a neuron alone)")
    good = []
    for s in range(60):
        a = audit(s)
        # 1040 steps (twice the small cases) and 10 outputs: every output must actually
        # carry constraint, so require them all live and none pathological
        ok = (a["live"] >= 0.6 * a["N"] and all(3 <= c <= 25 for c in a["ocount"])
              and a["busiest"] <= 90)
        if ok:
            good.append(a)
        if ok:
            print(f"  seed{s:>3}: {a['edges']:>3} edges  {a['inh']:>2} inhibitory neurons  "
                  f"{a['live']:>2}/{a['N']-1} live  out spikes {min(a['ocount'])}-{max(a['ocount'])} "
                  f"(total {sum(a['ocount'])})  "
                  f"sub-critical {a['subcrit']*100:.0f}%  busiest {a['busiest']}"
                  + ("   USABLE" if ok else ""))
    print(f"\n{len(good)} usable seeds: {[a['seed'] for a in good]}")
