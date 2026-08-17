"""Is the field's aggregate demand net-negative on the big nets?

    python3 _demand_sign.py [rounds]

The 50n rasters show late + missing spikes and mean |w| falling to ~0.83 of truth.  Two very
different causes fit that: the DEMAND itself is net-suppressive, or the demand is balanced and
something downstream (trust region, Adam's per-weight normalisation, the magnitude clamp) eats
the create half.  This logs both ends of the pipeline each round and compares.

Sign convention, checked against the update rule: `eps` is the PSP SHAPE and is positive on
every edge, g = dot(L, eps), and upd = +step*Adam(g).  So a POSITIVE update always means MORE
DRIVE onto the postsynaptic neuron -- it raises an excitatory weight and shrinks an inhibitory
one toward zero.  Everything below is therefore reported in drive units, not in |w|.

That distinction matters: mean |w| falling (the earlier finding) is ambiguous on its own, since
weaker inhibition also lowers mean |w| while INCREASING drive.  Excitatory and inhibitory
edges are tracked separately here for exactly that reason.

Small cases where the field works are included as controls -- a net-negative sum is only
diagnostic if the working cases do not show it.
"""
import os, sys, json
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    name, seed, rounds = args
    import numpy as np
    import field_trace as F
    from _diag import CASES, steps_for
    E, N, outs, Wl = CASES[name]
    C = np.array(E, np.int32)
    p = F.mkparams(steps_for(name))
    W = np.array(Wl, np.float32)
    T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
    wt = np.asarray(Wl, float)
    exc, inh = wt > 0, wt < 0
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    rec = []

    def cb(it, w, upd, g, spall, Ff, L):
        Lp = sum(float(np.sum(v[v > 0])) for v in L.values())
        Ln = sum(float(-np.sum(v[v < 0])) for v in L.values())
        w = np.asarray(w, float)
        rec.append(dict(
            it=it,
            L_pos=Lp, L_neg=Ln,                       # demand mass, before any weighting
            g_pos=float(np.sum(g[g > 0])), g_neg=float(-np.sum(g[g < 0])),
            u_mean=float(np.mean(upd)),               # >0 = net more drive
            u_exc=float(np.mean(upd[exc])), u_inh=float(np.mean(upd[inh])),
            mag_exc=float(np.mean(np.abs(w[exc])) / np.mean(np.abs(wt[exc]))),
            mag_inh=(float(np.mean(np.abs(w[inh])) / np.mean(np.abs(wt[inh])))
                     if inh.any() else float("nan")),
        ))

    F.train(C, N, outs, w0.copy(), T, p, rounds=rounds, lr=F.LR, cb=cb)
    return name, seed, rec


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    jobs = ([("50n A", s, rounds) for s in (0, 1, 2)]
            + [("50n B", 0, rounds), ("50n C", 0, rounds)]
            + [(n, s, min(rounds, 800)) for n in ("4n F", "3n D", "3n J") for s in (0, 1)])
    with mp.Pool(min(11, len(jobs))) as pool:
        res = pool.map(run, jobs)
    json.dump([[n, s, r] for n, s, r in res], open(f"{HERE}/demand_sign.json", "w"))

    print(f"{'case':<8} {'seed':>4} | {'L+/L-':>8} {'g+/g-':>8} | "
          f"{'upd exc':>9} {'upd inh':>9} | {'|w|exc':>7} {'|w|inh':>7}")
    print("-" * 78)
    agg = {}
    for name, seed, rec in res:
        h = rec[: max(1, len(rec) // 2)]          # the phase where the shrink happens
        lr_ = np.mean([r["L_pos"] for r in h]) / max(1e-9, np.mean([r["L_neg"] for r in h]))
        gr = np.mean([r["g_pos"] for r in h]) / max(1e-9, np.mean([r["g_neg"] for r in h]))
        ue, ui = np.mean([r["u_exc"] for r in h]), np.mean([r["u_inh"] for r in h])
        me, mi = rec[-1]["mag_exc"], rec[-1]["mag_inh"]
        print(f"{name:<8} {seed:>4} | {lr_:>8.3f} {gr:>8.3f} | {ue:>9.3f} {ui:>9.3f} | "
              f"{me:>7.3f} {mi:>7.3f}")
        agg.setdefault(name, []).append((lr_, gr, ue, ui, me, mi))
    print("\nper case (mean over seeds):")
    for name, v in agg.items():
        a = np.array(v)
        print(f"  {name:<8} L+/L- {a[:,0].mean():.3f}   g+/g- {a[:,1].mean():.3f}   "
              f"upd exc {a[:,2].mean():+.3f}  inh {a[:,3].mean():+.3f}   "
              f"final |w| exc {a[:,4].mean():.3f} inh {a[:,5].mean():.3f}")
