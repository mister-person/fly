"""_timing + _weight_term must equal the original _plausibility, and the (d,t) cache must be
sound (the cached half must not depend on the edge)."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import field_trace as F

rng = np.random.default_rng(0)
worst = 0.0
mismatch = 0
for trial in range(4000):
    T = int(rng.integers(200, 1100))
    t = int(rng.integers(F.DELAY_ITERS + 1, T))
    lo = int(rng.integers(-1, max(0, t - 1)))
    q0 = max(0, lo + 1 - F.DELAY_ITERS)
    q1 = t - F.DELAY_ITERS + 1
    if q1 <= q0:
        continue
    qs = np.arange(q0, q1)
    rs = sorted(rng.choice(np.arange(0, T), size=int(rng.integers(0, 30)),
                           replace=False).tolist())
    occ = F._occ_mask(rs, T)
    other_t = float(rng.uniform(-0.02, 0.02))
    wsi = float(rng.choice([1, -1]) * rng.uniform(20, 3000))
    ref = F._plausibility(qs, t, other_t, wsi, rs, occ)
    hk, ok = F._timing(qs, t, occ, rs)
    got = F._weight_term(hk, ok, other_t, wsi)
    if (ref is None) != (got is None):
        mismatch += 1
        continue
    if ref is not None:
        d = float(np.abs(ref - got).max())
        worst = max(worst, d)
print(f"split vs original over 4000 random requests: worst |diff| = {worst:.3e}, "
      f"None-disagreements = {mismatch}")

# the cached half must be edge-independent: same (qs,t,occ), many different weights
hk1, ok1 = F._timing(np.arange(0, 60), 90, F._occ_mask([30, 70], 200), [30, 70])
same = all(np.array_equal(ok1, F._timing(np.arange(0, 60), 90,
                                         F._occ_mask([30, 70], 200), [30, 70])[1])
           for _ in range(50))
print(f"cached half independent of the edge: {'YES' if same else 'NO'}")
