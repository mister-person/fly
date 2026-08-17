"""Insert 14n P / 14n Q into both case registries (_diag.CASES dict, _suite_mp.CASES list)."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _bignets as B

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = [("14n P", 45), ("14n Q", 55)]

HDR = ("""    # SMALL REPRODUCTION of the 50n hidden-neuron suppression bias ({tag}), 1040 steps:
    # 14 neurons, {ne} edges, {no} outputs, {ns} target output spikes, {sub:.0f}% sub-critical,
    # {ni} inhibitory neurons.  At the perturbed start the OUTPUTS ask correctly (L+/L- > 1)
    # while the 9 hidden neurons -- whose counts are guessed from bumps -- supply ~90% of all
    # suppression mass at L+/L- < 1, and training erodes both weight signs toward zero.
    # Same failure as 50n A/B/C at ~17x the speed.  See _minirepro.py.
""")

diag, smp = open(f"{HERE}/_diag.py").read(), open(f"{HERE}/_suite_mp.py").read()
for name, seed in SPEC:
    E, N, outs, W, inh = B.generate(seed, N=14, n_start=2, n_out=4, fan_in=3,
                                    frac_inh=0.25, w_lo=60.0, w_hi=380.0,
                                    w_start=(650.0, 950.0))
    Wr = [round(float(x), 1) for x in W]
    sub = 100 * float(np.mean([abs(x) < B.W_CRIT for x in W]))
    import field_trace as F
    C = np.array(E, np.int32)
    p = F.mkparams(1040)
    V = F.fsim(C, N, np.asarray(W, np.float32), p)
    ns = sum(len(F.sp(V, o)) for o in outs)
    hdr = HDR.format(tag=f"seed {seed}", ne=len(E), no=len(outs), ns=ns, sub=sub, ni=len(inh))

    anchor = "    # SUB-CRITICAL 50-neuron net (seed 6), 1040 steps:"
    assert anchor in diag, "diag anchor missing"
    diag = diag.replace(anchor, hdr + f'    "{name}": ({E}, {N}, {outs}, {Wr}),\n' + anchor, 1)

    anchor2 = '    ("50n A",'
    assert anchor2 in smp, "suite_mp anchor missing"
    smp = smp.replace(anchor2, f'    ("{name}", {E}, {N}, {outs}, {Wr}),\n' + anchor2, 1)

old = 'CASE_STEPS = {"8n M": 1040, "50n A": 1040, "50n B": 1040, "50n C": 1040}'
assert old in diag, "CASE_STEPS line missing"
diag = diag.replace(old, 'CASE_STEPS = {"8n M": 1040, "50n A": 1040, "50n B": 1040,\n'
                         '               "50n C": 1040, "14n P": 1040, "14n Q": 1040}', 1)

open(f"{HERE}/_diag.py", "w").write(diag)
open(f"{HERE}/_suite_mp.py", "w").write(smp)
print("installed 14n P, 14n Q into _diag.py and _suite_mp.py")
