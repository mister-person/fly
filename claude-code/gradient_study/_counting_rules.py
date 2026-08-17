"""Would smoothing the field fix the one-request-per-run under-count?

    python3 _counting_rules.py ["14n Q"] [seed]

Scores counting rules against the same fields, so the comparison is of the RULES only:

  A  current      one request per positive run
  B  smoothed     Gaussian-convolve the field first (grad_trace's FIELD_SMOOTH), then A
  C  peak-pick    local maxima within a run, separated by at least REFRAC

Two errors, reported separately because they move in opposite directions and netting them
hides both:  UNDER = sum over runs of max(0, true_spikes_inside - requests_placed_there),
OVER = requests placed where no true spike lies within REFRAC.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import field_trace as F
from _diag import CASES, steps_for

name = sys.argv[1] if len(sys.argv) > 1 else "14n Q"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
R = int(F.REFRAC_ITERS)

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
w = np.asarray(F.train(C, N, outs, w0.copy(), T, p, rounds=800, lr=F.LR), float)
V = F.fsim(C, N, np.asarray(w, np.float32), p)
spall = {n: F.sp(V, n) for n in range(N)}
g, Fl, L, ep = F.gradient(C, N, w, spall, p.steps, {o: T[o] for o in outs})
hidden = [n for n in range(1, N) if n not in outs and Fl[n].any()]


def smooth(f, sig):
    if sig <= 0:
        return f
    r = int(max(1, round(3 * sig)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sig) ** 2)
    return np.convolve(f, k / k.sum(), mode="same")


def runs_of(f):
    pos = np.nonzero(f > 0)[0]
    if not len(pos):
        return []
    return [r for r in np.split(pos, np.nonzero(np.diff(pos) > F.GAP)[0] + 1) if len(r)]


def rule_A(f):
    return [int(round(float(np.sum(r * f[r]) / max(float(f[r].sum()), 1e-30)))) for r in runs_of(f)]


def rule_C(f):
    """local maxima inside each run, greedily taken by height, min separation REFRAC"""
    out = []
    for r in runs_of(f):
        v = f[r]
        loc = [i for i in range(len(r)) if (i == 0 or v[i] >= v[i - 1])
               and (i == len(r) - 1 or v[i] >= v[i + 1])]
        loc.sort(key=lambda i: -v[i])
        kept = []
        for i in loc:
            if all(abs(int(r[i]) - q) >= R for q in kept):
                kept.append(int(r[i]))
        out += sorted(kept)
    return out


def score(qs, n):
    tr = list(T[n])
    used, under, over = set(), 0, 0
    for q in qs:
        hit = [i for i, s in enumerate(tr) if i not in used and abs(s - q) < R]
        if hit:
            used.add(min(hit, key=lambda i: abs(tr[i] - q)))
        else:
            over += 1
    under = len(tr) - len(used)
    return under, over


print(f"{name} seed {seed}  (REFRAC={R}; hidden neurons {hidden})\n")
print(f"{'rule':<22} {'requests':>8} {'UNDER':>6} {'OVER':>6}   N9 requests")
rows = [("A current (1/run)", lambda f: rule_A(f))]
for sg in (5, 10, 20, 40):
    rows.append((f"B smoothed sigma={sg}", lambda f, s=sg: rule_A(smooth(f, s))))
rows.append(("C peak-pick >=REFRAC", lambda f: rule_C(f)))
# SMOOTH THEN SPLIT.  The two rules fail in opposite directions -- smoothing can only merge
# runs (so it kills spurious narrow ones and never touches the under-count), peak-picking can
# only add requests (so it splits wide runs and invents extra ones).  Composing them is the
# only combination that can move both halves.
for sg in (5, 10, 20):
    rows.append((f"D smooth {sg} + peak-pick", lambda f, s=sg: rule_C(smooth(f, s))))
tot_true = sum(len(T[n]) for n in hidden)
print(f"(hidden neurons carry {tot_true} true spikes in total)\n")
for lbl, fn in rows:
    tq = tu = to = 0
    n9 = 0
    for n in hidden:
        qs = fn(Fl[n])
        u, o = score(qs, n)
        tq += len(qs); tu += u; to += o
        if n == 9:
            n9 = len(qs)
    print(f"{lbl:<22} {tq:>8} {tu:>6} {to:>6}   {n9} (N9 truly needs {len(T[9])})")
