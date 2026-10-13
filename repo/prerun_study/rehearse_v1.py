"""Pre-run rehearsal worker. Runs the verified simulator (sim.py) on a labeled
held-out stream. Each job: one track (cold, or the gated Router with one expert),
one seed. Worker i of W takes jobs i, i+W, ... Output written atomically."""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sim import Data, run
W, I = int(sys.argv[1]), int(sys.argv[2])
DATA = {'A': (HERE+'/reh_v1.npz', 100.0, 10000, range(1,21)),
        'B': (HERE+'/fresh_v1.npz', 100.0, 50000, range(1,11)),
        'C': (HERE+'/reh_v3.npz', 0.1, 10000, range(1,21)),
        'D': (HERE+'/reh_v1_int.npz', 100.0, 10000, range(1,21))}
EXPERTS = {'D': ('large_no1471', 'small_plus1471')}
jobs = []
for ph in sys.argv[3].split(','):
    path, c, T, seeds = DATA[ph]
    for s in seeds:
        if ph not in EXPERTS: jobs.append((ph, 'cold', None, s))
        for e in EXPERTS.get(ph, ('small','base','large')): jobs.append((ph, 'router', e, s))
mine = jobs[I::W]; loaded = {}
for ph, mode, e, s in mine:
    out = f"{HERE}/reh_out/{ph}_{mode}{'_'+e if e else ''}_s{s}.npz"
    if os.path.exists(out): continue
    path, c, T, _ = DATA[ph]
    if path not in loaded: loaded = {path: Data(path)}
    D = loaded[path]; n = D.C.shape[0]
    np.random.seed(s); order = np.random.permutation(n)       # same convention as the runner
    t0 = time.time()
    if mode == 'cold': o = run(D, order, T, cold=True, router=False)
    else:              o = run(D, order, T, llm=D.llm[e], c=c, cold=False, router=True)
    tmp = out + '.tmp.npz'; np.savez(tmp, **o); os.replace(tmp, out)
    k = 'r_cold' if mode == 'cold' else 'r_router'
    print(f"[w{I}] {ph} {mode} {e or ''} s{s}: acc {o[k].mean()*100:.2f}%  {time.time()-t0:.0f}s", flush=True)
print(f"[w{I}] DONE", flush=True)
