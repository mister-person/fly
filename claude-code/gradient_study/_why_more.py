"""If the count is already right, why is the field still positive?

    python3 _why_more.py

Claim under test: the positive field on a count-correct neuron is not a request for MORE
spikes.  It is the positive half of a MOVE -- local_demand puts a positive at the bump a spike
is paired with and (for an early spike) a negative at the spike -- and once that positive is
propagated backward through a broad plausibility kernel it is shape-indistinguishable from a
create.  Density mode then reads it as "fire across this whole span".

The code already separates the two channels: Lc / Fc carry the CREATE part alone, propagated
on its own for CREATE_FLOOR.  So the test is direct -- on a state where the counts are RIGHT,
compare the full field against the create-only field.  If the claim holds, F is broadly
positive and Fc is empty.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import field_trace as F
from _diag import CASES, steps_for


def look(name, seed, rounds=800):
    E, N, outs, Wl = CASES[name]
    C = np.array(E, np.int32)
    p = F.mkparams(steps_for(name))
    W = np.array(Wl, np.float32)
    T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = np.asarray(F.train(C, N, outs, w0.copy(), T, p, rounds=rounds, lr=F.LR), float)
    V = F.fsim(C, N, np.asarray(w, np.float32), p)
    spall = {n: F.sp(V, n) for n in range(N)}
    Fl, L, Lc, ep, PR, Fc = F.build(C, N, w, spall, p.steps, {o: list(T[o]) for o in outs})
    print(f"\n{name} seed {seed}   (outputs {outs})")
    print(f"{'neuron':>7} {'fires':>6} {'true':>5} {'count':>6} | {'F>0 samples':>11} "
          f"{'Fc>0 samples':>12} | {'max F':>9} {'max Fc':>9}")
    for n in range(1, N):
        if n in outs:
            continue
        f_, fc_ = np.asarray(Fl[n], float), np.asarray(Fc[n], float)
        ok = "OK" if len(spall[n]) == len(T[n]) else "WRONG"
        print(f"{'N'+str(n):>7} {len(spall[n]):>6} {len(T[n]):>5} {ok:>6} | "
              f"{int((f_ > 0).sum()):>11} {int((fc_ > 0).sum()):>12} | "
              f"{f_.max():>9.2e} {fc_.max():>9.2e}")
    # outputs: is the demand a move or a create?
    for o in outs:
        got, tgt = list(spall[o]), list(T[o])
        kind = ("counts match -> every demand here is a MOVE"
                if len(got) == len(tgt) else "counts differ -> genuine create/delete")
        print(f"   output N{o}: {len(got)} spikes vs {len(tgt)} target — {kind}")


for nm, sd in (("chain", 0), ("4n F", 0), ("14n Q", 0)):
    look(nm, sd)
