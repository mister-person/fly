"""Does credit assignment self-correct as the DIRECT path converges?

Hypothesis: in 3n D the spurious requests on N1 exist only because w(0->2) is wrong, so
vsub falls short at the N0-driven targets and N1 is blamed for them.  Once w(0->2) is
right, those deficits vanish and the inference should name only the one target N1 truly
owns (293 -> N1 fires at 246).

Test: hold the direct edge at TRUTH and randomise the hidden edges, versus the reverse.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

params = G.mkparams(520)
rng = np.random.default_rng(0)

CASES = {
    # name: (edges, N, outs, true w, index of the DIRECT input->output edge, hidden edges)
    "3n D": ([[0, 1], [0, 2], [1, 2]], 3, [2], [200., 1200., 700.], 1, [0, 2]),
    "4n G": ([[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [250., 500., 1200., 700.], 2, [0, 1, 3]),
}

for name, (E, N, outs, Wl, direct, hidden) in CASES.items():
    C = np.array(E, np.int32); TRUE = np.array(Wl)
    T = {n: G.sp(G.fsim(C, N, TRUE.astype(np.float32), params), n) for n in range(N)}
    hid = [n for n in range(N) if n not in outs and n != 0]
    print(f"=== {name}   true hidden " + "  ".join(f"N{n}={T[n]}" for n in hid))

    def report(tag, mk):
        exact = near = tot = 0
        rows = []
        for k in range(12):
            w = mk()
            V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
            eps, L, vs, wr = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)
            tg = G.infer_hidden_targets(C, N, w, s, params.steps,
                                        {o: T[o] for o in outs}, vs)
            for n in hid:
                got = tg.get(n, []); want = T[n]
                tot += 1
                exact += int(len(got) == len(want))
                near += int(len(got) == len(want) and
                            all(abs(a - b) <= 20 for a, b in zip(sorted(got), sorted(want))))
            if k < 3:
                rows.append(f"      w={np.round(w,0).tolist()}  -> "
                            + "  ".join(f"N{n}={tg.get(n, [])}" for n in hid))
        print(f"   {tag}: right COUNT {exact}/{tot}, right count AND within 20 steps {near}/{tot}")
        for r in rows:
            print(r)

    def mk_direct_true():
        w = TRUE.copy().astype(float)
        for i in hidden:
            w[i] = TRUE[i] * rng.uniform(0.5, 1.5)
        return w

    def mk_direct_wrong():
        w = TRUE.copy().astype(float)
        for i in hidden:
            w[i] = TRUE[i] * rng.uniform(0.5, 1.5)
        w[direct] = TRUE[direct] * rng.uniform(0.5, 1.5)
        return w

    report("direct edge at TRUTH, hidden randomised ", mk_direct_true)
    report("everything randomised                   ", mk_direct_wrong)
    print()
