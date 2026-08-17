"""FIELD on 8n M: the signed should-fire-here density, replacing the request path."""
import os, sys, itertools
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v,"1")
import multiprocessing as mp
GRID = {"FIELD": [0, 1], "FIELD_POW": [16.0], "NEW_GAIN": [1.0]}
KEYS = list(GRID)
def _job(a):
    cfg, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    from _diag import CASES, steps_for
    for k, v in cfg.items(): setattr(G, k, v)
    E,N,outs,Wl = CASES["8n M"]; C=np.array(E,np.int32); params=G.mkparams(steps_for("8n M"))
    W=np.array(Wl,np.float32); T={n:G.sp(G.fsim(C,N,W,params),n) for n in range(N)}
    w0=(W*np.random.default_rng(seed).uniform(0.5,1.5,len(Wl))).astype(float)
    w=G.train(C,N,outs,w0.copy(),T,params,rounds=1200,lr=G.LR)
    V=G.fsim(C,N,w,params); F={n:G.sp(V,n) for n in range(N)}
    err=[]
    for o in outs:
        f,t=F[o],T[o]
        err.append(99.0 if len(f)!=len(t) else float(np.mean([abs(a-b) for a,b in zip(f,t)])))
    return tuple(sorted(cfg.items())), seed, all(F[o]==T[o] for o in outs), \
           float(np.mean(err)), sum(1 for n in (3,4) if F[n]), [len(F[o]) for o in outs]
def main():
    import numpy as np
    cfgs=[dict(zip(KEYS,c)) for c in itertools.product(*(GRID[k] for k in KEYS))]
    jobs=[(c,s) for c in cfgs for s in range(3)]
    with mp.get_context("spawn").Pool(16) as p: res=p.map(_job,jobs)
    agg={}
    for k,seed,ok,err,alive,cnt in res: agg.setdefault(k,[]).append((ok,err,alive,cnt))
    print(f"{'meanErr':>8} {'exact':>6} {'N3/N4 alive':>12}  config")
    for k,v in sorted(agg.items(), key=lambda kv: np.mean([x[1] for x in kv[1]])):
        print(f"{np.mean([x[1] for x in v]):8.2f} {sum(x[0] for x in v):>6} "
              f"{sum(x[2] for x in v):>7}/12     {dict(k)}")
if __name__=="__main__": main()
