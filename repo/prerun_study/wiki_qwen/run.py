"""WikiLinks / Qwen2.5 (BGE-1024d): the c_LLM sweep on tuning seeds 11-20, then the evaluation on
seeds 1-10 at the selected c. Same simulator as prerun_study/sim.py, checked against the cluster runner
on seed 11 (Router, LLM, cold accuracy and routed share equal to 4 decimals at step 5,000).

Jobs: ('cold', s) runs cold LinUCB once per seed (it does not depend on the expert or c);
('router', e, c, s) runs the gated Router. Phase 1 = all sweep jobs; phase 2 = eval jobs at the c with
the highest mean Router accuracy over experts and seeds 11-20 (ties -> smaller c), written to
out/chosen_c.json. Workers claim jobs through lock files, checkpoint every CKPT rounds, and resume."""
import sys, os, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from sim import Data, sm_update
DATA = '/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/wiki_v3_slim.npz'
LAM, DECAY, CKPT = 0.1, 0.9985, 20000
EXPERTS = ('small', 'base', 'large'); GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
TUNE, EVAL = range(11, 21), range(1, 11)
NICE = {'small': 'Qwen-0.5B', 'base': 'Qwen-3B', 'large': 'Qwen-7B'}

def name(j): return f"cold_s{j[1]}" if j[0] == 'cold' else f"{j[1]}_c{j[2]:g}_s{j[3]}"
def sweep_jobs():
    J = []
    for s in TUNE: J += [('cold', s)] + [('router', e, c, s) for c in GRID for e in EXPERTS]
    return J
def eval_jobs(c):
    J = []
    for s in EVAL: J += [('cold', s)] + [('router', e, c, s) for e in EXPERTS]
    return J
def done(j): return os.path.exists(f"{HERE}/out/{name(j)}.npz")

def choose_c():
    p = f"{HERE}/out/chosen_c.json"
    if os.path.exists(p): return json.load(open(p))['c']
    means = {c: np.mean([np.load(f"{HERE}/out/{e}_c{c:g}_s{s}.npz")['r_router'].mean() for e in EXPERTS for s in TUNE])
             for c in GRID}
    best = max(GRID, key=lambda c: (round(means[c], 10), -c))
    tmp = p + '.tmp'; json.dump(dict(c=best, mean_router_acc={f"{c:g}": means[c] for c in GRID}), open(tmp, 'w')); os.replace(tmp, p)
    return best

def claim(j):
    """Atomic claim; a claim whose process is gone is taken over (the checkpoint makes that safe)."""
    p = f"{HERE}/logs/{name(j)}.claim"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        try: pid = int(open(p).read().strip() or 0)
        except Exception: pid = 0
        try: os.kill(pid, 0); return False            # owner still alive
        except (ProcessLookupError, PermissionError, ValueError, OSError): pass
        open(p, 'w').write(str(os.getpid())); return True

def job(D, j):
    final = f"{HERE}/out/{name(j)}.npz"; ck = f"{HERE}/out/{name(j)}.ckpt.npz"; prog = f"{HERE}/logs/{name(j)}.progress"
    if os.path.exists(final): return
    cold = j[0] == 'cold'; s = j[1] if cold else j[3]
    n = D.C.shape[0]; T = n; d = D.d
    np.random.seed(s); order = np.random.permutation(n)
    if not cold: e, c = j[1], j[2]; llm = D.llm[e]; isc = 1.0/np.sqrt(c)
    if os.path.exists(ck):
        z = np.load(ck); t0 = int(z['t']); V, b, A, R, RO = z['V'], z['b'], z['A'], z['R'], z['RO']
    else:
        t0 = 0; V = np.eye(d)/LAM; b = np.zeros(d); A = np.zeros(T, np.int32); R = np.zeros(T, np.int8); RO = np.zeros(T, bool)
    start = time.time()
    for t in range(t0, T):
        jj = int(order[t]); cx = D.C[jj].astype(np.float64); ys = D.ys[jj]; alpha = max(0.1, DECAY**t)
        a0 = D.score(V, V@b, cx, alpha); x0 = cx*D.Z[a0]; use = False
        if not cold:
            al = int(llm[jj]); xl = cx*D.Z[al]
            use = bool(isc*np.linalg.norm(xl) < np.sqrt(max(0.0, x0@V@x0)))
        a, x = (al, xl) if use else (a0, x0); r = 1 if a in ys else 0
        V = sm_update(V, x); b += r*x; A[t] = a; R[t] = r; RO[t] = use
        if (t+1) % CKPT == 0 or t+1 == T:
            tmp = ck + '.tmp.npz'; np.savez(tmp, t=t+1, V=V, b=b, A=A, R=R, RO=RO); os.replace(tmp, ck)
            rate = (t+1-t0)/max(1e-9, time.time()-start)
            with open(prog + '.tmp', 'w') as f:
                json.dump(dict(job=name(j), t=t+1, T=T, acc=float(R[:t+1].mean()), routed=float(RO[:t+1].mean()),
                               rounds_per_sec=rate, eta_hours=(T-t-1)/max(rate, 1e-9)/3600,
                               updated=time.strftime('%Y-%m-%d %H:%M:%S')), f)
            os.replace(prog + '.tmp', prog)
    out = dict(seed=s, r=R, a=A)
    if cold: out.update(a_cold=A, r_cold=R)
    else: out.update(a_router=A, r_router=R, routed=RO, c_llm=c, llm_acc=float(np.mean([llm[k] in D.ys[k] for k in range(n)])))
    tmp = final + '.tmp.npz'; np.savez(tmp, **out); os.replace(tmp, final); os.remove(ck)
    print(f"[{time.strftime('%m-%d %H:%M')}] {name(j)} DONE acc {R.mean()*100:.2f}% routed {RO.mean()*100:.1f}%", flush=True)

def work(D, jobs):
    for j in jobs:
        if not done(j) and claim(j): job(D, j)

if __name__ == '__main__':
    D = Data(DATA)
    work(D, sweep_jobs())
    while not all(done(j) for j in sweep_jobs()): time.sleep(60)   # other workers finishing the sweep
    c = choose_c(); print(f"selected c_LLM = {c:g}", flush=True)
    work(D, eval_jobs(c))
    print("worker DONE", flush=True)
