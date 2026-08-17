"""Exhaustive search for the three 3n D variants, requiring the hidden path to CREATE
output spikes rather than merely shift them.

A hidden spike only leaves a NEW mark if its arrival (q + DELAY) clears the output's
refractory shadow AND lands in the gap between two output spikes with enough weight to
cross threshold alone.  Requiring extra == (number of hidden spikes) enforces exactly that,
and severing the hidden edge gives the reference train.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

params = G.mkparams(520)


def sim(E, N, w):
    V = G.fsim(np.array(E, np.int32), N, np.array(w, np.float32), params)
    return {n: G.sp(V, n) for n in range(N)}


def report(tag, E, N, out, w, kill, want_counts, want_extra):
    """want_counts = spikes per HIDDEN neuron (in index order); want_extra = how many NEW
    output spikes the hidden path must create.  These are separate things -- conflating
    them into one list is why the first version matched nothing at all."""
    s = sim(E, N, w)
    w2 = list(w); w2[kill] = 0.0
    b = sim(E, N, w2)
    extra = len(s[out]) - len(b[out])
    hid = [n for n in range(N) if n != out and n != 0]
    counts = [len(s[n]) for n in hid]
    if counts != want_counts or extra != want_extra:
        return None
    return (w, {n: s[n] for n in range(N)}, b[out], extra)


# sanity: the ORIGINAL 3n D must pass its own criterion, else the criterion is wrong
_chk = report("D", [[0, 1], [0, 2], [1, 2]], 3, 2, [200., 1200., 700.], 2, [1], 1)
print(f"sanity -- original 3n D passes criterion: {_chk is not None}")
if _chk:
    print(f"   {_chk[1]}  silent {_chk[2]}  extra {_chk[3]}")
print()
print("=== E: 3 neurons, N1 fires TWICE, both marks visible ===")
E_E = [[0, 1], [0, 2], [1, 2]]
hits = []
for w01 in range(200, 445, 5):
    for w12 in range(400, 1301, 50):
        r = report("E", E_E, 3, 2, [float(w01), 1200., float(w12)], 2, [2], 2)
        if r:
            hits.append(r)
for w, s, b, x in hits[:8]:
    print(f"   w={[int(v) for v in w]}  N1={s[1]}  N2={s[2]}   (silent {b})")
print(f"   {len(hits)} candidates\n")

print("=== F: 4 neurons, N0->N1->N2->N3 split chain, ONE mark ===")
E_F = [[0, 1], [1, 2], [0, 3], [2, 3]]
hitsF = []
for w01 in range(150, 300, 5):
    for w12 in range(450, 1501, 50):
        for w23 in (500., 700., 900.):
            r = report("F", E_F, 4, 3, [float(w01), float(w12), 1200., w23], 3, [1, 1], 1)
            if r:
                hitsF.append(r)
for w, s, b, x in hitsF[:8]:
    print(f"   w={[int(v) for v in w]}  N1={s[1]} N2={s[2]} N3={s[3]}   (silent {b})")
print(f"   {len(hitsF)} candidates\n")

print("=== G: 4 neurons, split chain, N1 fires TWICE, both marks ===")
hitsG = []
for w01 in range(200, 445, 5):
    for w12 in range(450, 1501, 50):
        for w23 in (500., 700., 900.):
            r = report("G", E_F, 4, 3, [float(w01), float(w12), 1200., w23], 3, [2, 2], 2)
            if r:
                hitsG.append(r)
for w, s, b, x in hitsG[:8]:
    print(f"   w={[int(v) for v in w]}  N1={s[1]} N2={s[2]} N3={s[3]}   (silent {b})")
print(f"   {len(hitsG)} candidates")
