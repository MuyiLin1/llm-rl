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
        'D': (HERE+'/reh_v1_int.npz', 100.0, 10000, range(1,21)),
        # E: exact gated Router + deployed alpha schedule on the OLD dry run's streams
        #    (10k held-out contexts resampled with replacement to 50k; default_rng(seed)).
        'E': (HERE+'/reh_v1.npz', 100.0, 50000, range(1,11)),
        # F: old dry-run structure (forced 1600-round acquisition from default_rng(seed+10000),
        #    then LinUCB on stream[1600:]) but with the deployed alpha schedule (t offset 1600).
        'F': (HERE+'/reh_v1.npz', None, 50000, range(1,11)),
        # P: accuracy-matched placebos for small_plus1471 (build_placebo.py)
        'P': (HERE+'/reh_v1_plc.npz', 100.0, 10000, range(1,21))}
TAU_F = 1600
EXPERTS = {'D': ('large_no1471', 'small_plus1471'), 'P': ('small_plusrand', 'small_plus7961')}
NOCOLD = {'D', 'F', 'P'}
jobs = []
for ph in sys.argv[3].split(','):
    path, c, T, seeds = DATA[ph]
    for s in seeds:
        if ph not in NOCOLD: jobs.append((ph, 'cold', None, s))
        for e in EXPERTS.get(ph, ('small','base','large')): jobs.append((ph, 'router', e, s))
mine = jobs[I::W]; loaded = {}
for ph, mode, e, s in mine:
    out = f"{HERE}/reh_out/{ph}_{mode}{'_'+e if e else ''}_s{s}.npz"
    if os.path.exists(out): continue
    path, c, T, _ = DATA[ph]
    if path not in loaded: loaded = {path: Data(path)}
    D = loaded[path]; n = D.C.shape[0]
    t0 = time.time()
    if ph in ('E', 'F'):
        pool = np.arange(n)
        order = np.random.default_rng(s).choice(pool, size=T, replace=True)   # = dry_run_task.streams
        acq = np.random.default_rng(s + 10_000).choice(pool, size=5000, replace=True)
    else:
        np.random.seed(s); order = np.random.permutation(n)   # same convention as the runner
    if ph == 'F':
        warm = [(int(j), int(D.llm[e][int(j)])) for j in acq[:TAU_F]]
        o = run(D, order[TAU_F:], T - TAU_F, cold=False, router=True, warm=warm, t0=TAU_F)
        o['acq_correct'] = np.array(sum(1.0 for j, a in warm if a in D.ys[j]))
        o['acq_dom_correct'] = np.array(sum(1 for j, a in warm if a == 1471 and a in D.ys[j]))
    elif mode == 'cold': o = run(D, order, T, cold=True, router=False)
    else:              o = run(D, order, T, llm=D.llm[e], c=c, cold=False, router=True)
    tmp = out + '.tmp.npz'; np.savez(tmp, **o); os.replace(tmp, out)
    k = 'r_cold' if mode == 'cold' else 'r_router'
    print(f"[w{I}] {ph} {mode} {e or ''} s{s}: acc {o[k].mean()*100:.2f}%  {time.time()-t0:.0f}s", flush=True)
print(f"[w{I}] DONE", flush=True)
