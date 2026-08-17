"""More creation demand -> does the output fire MORE (the intuition), or just EARLIER?"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v,"1")
import multiprocessing as mp
def _job(a):
    cr, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    from _diag import CASES, steps_for
    G.CREATE = cr
    E,N,outs,Wl = CASES["8n M"]; C=np.array(E,np.int32); params=G.mkparams(steps_for("8n M"))
    W=np.array(Wl,np.float32); T={n:G.sp(G.fsim(C,N,W,params),n) for n in range(N)}
    w0=(W*np.random.default_rng(seed).uniform(0.5,1.5,len(Wl))).astype(float)
    w=G.train(C,N,outs,w0.copy(),T,params,rounds=3200,lr=G.LR)
    V=G.fsim(C,N,w,params); F={n:G.sp(V,n) for n in range(N)}
    return cr, seed, float(w[2]), (F[3][0] if F[3] else None), len(F[3]), \
           [len(F[o]) for o in outs], [len(T[o]) for o in outs]
def main():
    import numpy as np
    GAINS=[0.1,0.3,1.0,3.0]
    jobs=[(c,s) for c in GAINS for s in range(4)]
    with mp.get_context("spawn").Pool(16) as p: res=p.map(_job,jobs)
    print("true w(0->3)=250, N3 first spike 173, outputs fire 6/6/6\n")
    print(f"{'CREATE':>7} {'w(0->3)':>9} {'N3 first':>9} {'N3 count':>9} {'out counts':>12}")
    for c in GAINS:
        sub=[r for r in res if r[0]==c]
        w3=np.mean([r[2] for r in sub])
        fs=[r[3] for r in sub if r[3] is not None]
        n3=np.mean([r[4] for r in sub])
        oc=np.mean([sum(r[5]) for r in sub])
        print(f"{c:>7} {w3:>9.0f} {np.mean(fs) if fs else float('nan'):>9.0f} {n3:>9.1f} "
              f"{oc:>7.1f} / 18")
if __name__=="__main__": main()
