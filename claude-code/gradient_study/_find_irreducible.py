"""Find a coincidence case that is IRREDUCIBLE: no subset of the inputs can reproduce it.

8n K failed as a coincidence test because its hidden neurons all fire on the same period-100
rhythm (just phase-shifted), so the outputs are period-100 trains and ONE strong edge at the
right phase reproduces them.  The optimiser duly found that: 8 of 12 fan-in weights collapsed
to the clip floor, N4 died, and the output was still 10/11 spikes on target.

Fix: give the hidden neurons DIFFERENT RATES -- some supra-critical (fire every cycle), some
sub-critical accumulators (fire rarely).  Then the coincidence pattern is irregular and no
single-rate source can match it.

TEST OF IRREDUCIBILITY, per output: zero out each subset of its incoming edges and scan the
survivors' weights.  If any subset of size < 4 can reproduce the target train, the case is
reducible and worthless as a coincidence test.
"""
import os, sys, itertools
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

params = G.mkparams(520)
E = [[0, 1], [0, 2], [0, 3], [0, 4]] + [[h, o] for o in (5, 6, 7) for h in (1, 2, 3, 4)]
C = np.array(E, np.int32)


def spikes(w):
    V = G.fsim(C, 8, np.array(w, np.float32), params)
    return {n: G.sp(V, n) for n in range(8)}


def reducible(Wl, T, o, maxk=2, grid=range(20, 3001, 40)):
    """can any subset of <= maxk incoming edges reproduce output o's train?"""
    inc = [si for si in range(len(Wl)) if E[si][1] == o]
    for k in range(1, maxk + 1):
        for keep in itertools.combinations(inc, k):
            base = list(Wl)
            for si in inc:
                if si not in keep:
                    base[si] = 0.0
            if k == 1:
                for x in grid:
                    b = list(base); b[keep[0]] = float(x)
                    if spikes(b)[o] == T[o]:
                        return (keep, x)
            else:
                for x in grid:
                    for y in grid:
                        b = list(base); b[keep[0]] = float(x); b[keep[1]] = float(y)
                        if spikes(b)[o] == T[o]:
                            return (keep, (x, y))
    return None


print("mixed-rate fan-out: two supra-critical, two sub-critical accumulators\n")
for fan in ([900., 500., 250., 200.], [900., 600., 260., 210.], [1200., 500., 250., 190.]):
    for k in (130., 150.):
        Wl = fan + [k, k * 1.1, k * 0.9, k] + [k * 1.2, k, k * 1.1, k * 0.9] \
                 + [k * 0.9, k * 1.2, k, k * 1.1]
        T = spikes(Wl)
        cnt = [len(T[n]) for n in range(1, 5)]
        outs = [len(T[o]) for o in (5, 6, 7)]
        if min(outs) < 2 or len(set(cnt)) < 2:
            continue
        isis = {o: np.diff(T[o]).tolist() for o in (5, 6, 7)}
        reg = all(len(set(isis[o])) <= 1 for o in (5, 6, 7) if len(isis[o]) > 1)
        print(f"fan={[int(x) for x in fan]} k={int(k)}  hidden counts {cnt}  outputs {outs}"
              f"  {'ALL REGULAR (bad)' if reg else 'irregular'}")
        for o in (5, 6, 7):
            print(f"    N{o}={T[o]}  ISIs {isis[o]}")
        if not reg:
            for o in (5, 6, 7):
                r = reducible(Wl, T, o, maxk=1)
                print(f"    N{o} reproducible by ONE edge? {r if r else 'NO'}")
        print()
