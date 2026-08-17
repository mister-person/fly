"""What does the current method actually pass to a hidden neuron on cases it SOLVES?

3n D and chain are both 8/8.  3n D's N1 must fire exactly ONCE at t=246, so whatever reaches
it is enough to pin a single spike's time.  Instrument the hidden demand L[n] along a real
trajectory: how much of it is the CREATE (magnitude, rate) channel versus the TIM (signed
earlier/later) channel, where each sits in time relative to the true spike, and how that
changes as the run converges.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G
from _diag import CASES, steps_for

for name, hid in (("3n D", 1), ("chain", 1)):
    E, N, outs, Wl = CASES[name]
    C = np.array(E, np.int32)
    params = G.mkparams(steps_for(name))
    W = np.array(Wl, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    print(f"===== {name}   true hidden N{hid} = {T[hid]}   outputs {[T[o] for o in outs]}")
    w0 = (W * np.random.default_rng(0).uniform(0.5, 1.5, len(Wl))).astype(float)

    snaps = []

    def cb(it, w, upd, g, spall, vsub, L):
        if it in (1, 50, 200, 800, 1600, 3200):
            snaps.append((it, w.copy(), {n: list(spall[n]) for n in range(N)}))

    G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR, cb=cb)

    for it, w, sp in snaps:
        V = G.fsim(C, N, np.asarray(w, np.float32), params)
        s = {n: G.sp(V, n) for n in range(N)}
        # split the hidden demand into its two channels by ablation
        parts = {}
        for tag, over in (("both", {}), ("CREATE only", dict(TIM_GAIN=0.0)),
                          ("TIM only", dict(CREATE=0.0))):
            sv = {k: getattr(G, k) for k in ("TIM_GAIN", "CREATE")}
            for k, v in over.items():
                setattr(G, k, v)
            eps, L, vs, wr = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)
            gg = np.zeros(len(w))
            for n in range(N):
                for si in inc[n]:
                    gg[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
            nz = np.nonzero(L[hid])[0]
            parts[tag] = (L[hid].copy(), gg.copy(), nz)
            for k, v in sv.items():
                setattr(G, k, v)
        Lb, gb, nzb = parts["both"]
        Lc, gc, nzc = parts["CREATE only"]
        Lt, gt, nzt = parts["TIM only"]
        off = (s[hid][0] - T[hid][0]) if (s[hid] and T[hid]) else None
        print(f"  it{it:>5}  N{hid} fires {s[hid][:4]} (true {T[hid][:4]})"
              + (f"  first-spike offset {off:+d}" if off is not None else ""))
        print(f"          L[{hid}] nonzero at {nzb.tolist()[:6]}"
              f"   peak {np.abs(Lb).max():.2e}")
        print(f"          CREATE channel: {len(nzc)} pts peak {np.abs(Lc).max():.2e}"
              f"   |  TIM channel: {len(nzt)} pts peak {np.abs(Lt).max():.2e}"
              f"   (TIM sits AT the neuron's own spikes)")
    print()
