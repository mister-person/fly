"""Emit the 14n reproduction cases as literal _diag entries, and verify them.

    python3 _install_mini.py        # prints the CASES lines; also audits and times

Stored literally rather than as a generate() call so a case can never silently change if the
generator is touched -- the same reason the 50n nets are stored that way.
"""
import os, sys, time
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import field_trace as F
import _bignets as B

STEPS = 1040
PICK = [("14n P", 45), ("14n Q", 55)]

for name, seed in PICK:
    E, N, outs, W, inh = B.generate(seed, N=14, n_start=2, n_out=4, fan_in=3,
                                    frac_inh=0.25, w_lo=60.0, w_hi=380.0,
                                    w_start=(650.0, 950.0))
    C = np.array(E, np.int32)
    p = F.mkparams(STEPS)
    Wf = np.asarray(W, np.float32)
    V = F.fsim(C, N, Wf, p)
    T = {n: F.sp(V, n) for n in range(N)}
    # TRUTH MUST BE A FIXED POINT: if the field asks for changes at the true weights, the case
    # is measuring the demand rule's disagreement with itself, not a recovery failure.
    g = F.gradient(C, N, np.asarray(W, float), {n: T[n] for n in range(N)},
                   STEPS, {o: list(T[o]) for o in outs})[0]
    t0 = time.time()
    F.train(C, N, outs, (Wf * 1.1).astype(float), T, p, rounds=50, lr=F.LR)
    dt50 = time.time() - t0
    sub = float(np.mean([abs(x) < B.W_CRIT for x in W]))
    print(f"# {name}: N={N}, {len(E)} edges, {len(outs)} outputs, {sum(len(T[o]) for o in outs)} "
          f"target output spikes, {100*sub:.0f}% sub-critical, {len(inh)} inhibitory neurons")
    print(f"#   max|g| at truth = {np.max(np.abs(g)):.3e}")
    print(f"#   {dt50:.1f}s / 50 rounds  ->  ~{dt50*600/50/60:.1f} min for a 600-round x1 run")
    print(f'    "{name}": ({E}, {N}, {outs}, {[round(float(x),1) for x in W]}),')
    print()
