"""Crossing demand on a subset, so the slow field path fits in one run."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v,"1")
import multiprocessing as mp
CASES_N = ["chain", "3-cycle", "3n D", "4n G", "5n H"]
def _job(a):
    nm, fx, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    from _diag import CASES, steps_for
    G.FIELD_XING = fx
    E,N,outs,Wl = CASES[nm]; C=np.array(E,np.int32); params=G.mkparams(steps_for(nm))
    W=np.array(Wl,np.float32); T={n:G.sp(G.fsim(C,N,W,params),n) for n in range(N)}
    w0=(W*np.random.default_rng(seed).uniform(0.5,1.5,len(Wl))).astype(float)
    w=G.train(C,N,outs,w0.copy(),T,params,rounds=800,lr=G.LR)
    V=G.fsim(C,N,w,params)
    return nm, fx, seed, all(G.sp(V,o)==T[o] for o in outs)
def main():
    jobs=[(nm,fx,s) for nm in CASES_N for fx in (0.0,1.0) for s in range(6)]
    with mp.get_context("spawn").Pool(16) as p: res=p.map(_job,jobs)
    print(f"{'case':10s} {'FIELD_XING=0':>13} {'FIELD_XING=1':>13}")
    t0=t1=0
    for nm in CASES_N:
        a=sum(r[3] for r in res if r[0]==nm and r[1]==0.0)
        b=sum(r[3] for r in res if r[0]==nm and r[1]==1.0)
        t0+=a; t1+=b
        print(f"{nm:10s} {a:>10}/6 {b:>12}/6" + ("   <- better" if b>a else ("   <- worse" if b<a else "")))
    print(f"{'TOTAL':10s} {t0:>10}/30 {t1:>12}/30")
if __name__=="__main__": main()
