"""WikiLinks / Flan-T5 (BERT-768d) evaluation runs at the c_LLM selected on tuning seeds 11-20 (c=10000).
Seeds 1-10, experts small/base/large. Same simulator as prerun_study/sim.py (verified to match the cluster
runner action-for-action on WikiLinks). Cold LinUCB does not depend on the expert, so it is computed only in
the 'small' jobs. Every job checkpoints every CKPT rounds and resumes from its checkpoint if restarted."""
import sys, os, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from sim import Data, sm_update
DATA = '/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/wiki_v1_slim.npz'
C_LLM, LAM, DECAY, CKPT = 10000.0, 0.1, 0.9985, 20000
JOBS = [(e, s) for s in range(1, 11) for e in ('small', 'base', 'large')]

def job(D, e, s):
    final = f"{HERE}/out/{e}_s{s}.npz"; ck = f"{HERE}/out/{e}_s{s}.ckpt.npz"; prog = f"{HERE}/logs/{e}_s{s}.progress"
    if os.path.exists(final): return
    n = D.C.shape[0]; T = n; d = D.d; with_cold = (e == 'small')
    np.random.seed(s); order = np.random.permutation(n)
    llm = D.llm[e]; isc = 1.0/np.sqrt(C_LLM)
    if os.path.exists(ck):
        z = np.load(ck); t0 = int(z['t']); Vr, br, Vc, bc = z['Vr'], z['br'], z['Vc'], z['bc']
        ar, rr, ro, ac, rc = z['ar'], z['rr'], z['ro'], z['ac'], z['rc']
    else:
        t0 = 0; Vr = np.eye(d)/LAM; br = np.zeros(d); Vc = np.eye(d)/LAM; bc = np.zeros(d)
        ar = np.zeros(T, np.int32); rr = np.zeros(T, np.int8); ro = np.zeros(T, bool); ac = np.zeros(T, np.int32); rc = np.zeros(T, np.int8)
    start = time.time(); t_start = t0
    for t in range(t0, T):
        j = int(order[t]); cx = D.C[j].astype(np.float64); ys = D.ys[j]; alpha = max(0.1, DECAY**t)
        if with_cold:
            a = D.score(Vc, Vc@bc, cx, alpha); r = 1 if a in ys else 0
            x = cx*D.Z[a]; Vc = sm_update(Vc, x); bc += r*x; ac[t] = a; rc[t] = r
        a0 = D.score(Vr, Vr@br, cx, alpha); x0 = cx*D.Z[a0]
        al = int(llm[j]); xl = cx*D.Z[al]
        use = bool(isc*np.linalg.norm(xl) < np.sqrt(max(0.0, x0@Vr@x0)))
        a, x = (al, xl) if use else (a0, x0); r = 1 if a in ys else 0
        Vr = sm_update(Vr, x); br += r*x; ar[t] = a; rr[t] = r; ro[t] = use
        if (t+1) % CKPT == 0 or t+1 == T:
            tmp = ck + '.tmp.npz'
            np.savez(tmp, t=t+1, Vr=Vr, br=br, Vc=Vc, bc=bc, ar=ar, rr=rr, ro=ro, ac=ac, rc=rc); os.replace(tmp, ck)
            rate = (t+1-t_start)/max(1e-9, time.time()-start)
            with open(prog, 'w') as f:
                json.dump(dict(expert=e, seed=s, t=t+1, T=T, router_acc=float(rr[:t+1].mean()),
                               cold_acc=float(rc[:t+1].mean()) if with_cold else None, routed=int(ro[:t+1].sum()),
                               rounds_per_sec=rate, eta_hours=(T-t-1)/max(rate,1e-9)/3600, updated=time.strftime('%Y-%m-%d %H:%M:%S')), f)
    llm_acc = float(np.mean([1 if llm[j] in D.ys[j] else 0 for j in range(n)]))
    out = dict(a_router=ar, r_router=rr, routed=ro, llm_acc=llm_acc, c_llm=C_LLM, seed=s)
    if with_cold: out.update(a_cold=ac, r_cold=rc)
    tmp = final + '.tmp.npz'; np.savez(tmp, **out); os.replace(tmp, final); os.remove(ck)
    print(f"[{e} s{s}] DONE router {rr.mean()*100:.2f}%" + (f" cold {rc.mean()*100:.2f}%" if with_cold else "") + f" llm {llm_acc*100:.2f}% routed {ro.mean()*100:.2f}%", flush=True)

if __name__ == '__main__':
    W, I = int(sys.argv[1]), int(sys.argv[2]); D = Data(DATA)
    for e, s in JOBS[I::W]: job(D, e, s)
    print(f"worker {I} DONE", flush=True)
