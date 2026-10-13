"""Phase B driver: is the AmazonCat multiplier an artifact of a weak cold baseline?
Four chains run on one pool of GPU workers; each chain starts its next stage as soon as the current one
is done, using the selections of the earlier stages (tuning seeds 11-20, evaluation seeds 1-10).

  prod chain (paper features, exact float32 scoring), for BERT and for BGE:
    S1 cold LinUCB, 16 exploration settings (schedule x floor), lambda=0.1, tuning seeds   [Tier 1]
    S2 cold LinUCB, lambda in {0.01,1,10} at the best S1 setting, tuning seeds             [Tier 1]
    S3 best cold setting on evaluation seeds (+ default for BGE)                           [Tier 1]
    S4 Router at the best cold setting, c in {c0/10,c0,10c0} x 3 experts, tuning seeds    [Tier 3]
    S5 Router at best setting and chosen c on evaluation seeds (+ default Router for BGE)  [Tier 3]
  arm chain (features [x*z_a, z_a], TF32 scoring), for BERT and for BGE; starts after S1:
    A1 cold, top-3 S1 settings + default, tuning seeds                                     [Tier 2]
    A2 best (and default) arm cold setting on evaluation seeds                             [Tier 2]
    A3 Router, c in {c0/10,c0,10c0} x 3 experts at the best arm setting, tuning seeds      [Tier 2]
    A4 Router at best c on evaluation seeds                                                [Tier 2]
BGE at the default setting doubles as a check of this simulator against the paper's BGE rows; BERT
defaults are already available from prerun_study/replay (identical simulator)."""
import os, sys, json, time, hashlib, traceback, numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('PB_OUT', f'{HERE}/out'); LOG = os.environ.get('PB_LOG', f'{HERE}/logs')
T = int(os.environ.get('PB_T', 50000)); NW = int(os.environ.get('PB_WORKERS', 12))
FAMS = {'bert': dict(slim='amazon_v1_slim.npz', c0=100.0), 'bge': dict(slim='amazon_v3_slim.npz', c0=0.1)}
TUNE = list(range(11, 21)); EVAL = list(range(1, 11)); EXPERTS = ['small', 'base', 'large']
DEFAULT = ('fast', 0.1, 0.1)                                                   # (schedule, floor, lambda)
GRID = [(s, a) for s in ('fast', 'slow', 'const') for a in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
        if not (a >= 1.0 and s != 'const')]                                    # floor 1.0 is the same for every schedule
LAMS = [0.01, 1.0, 10.0]
KMAX = {'prod': 3, 'prod+arm': 2}

def spec(kind, sched, amin, lam, seed, expert=None, c=None):
    return dict(kind=kind, expert=expert, c=c, sched=sched, amin=amin, lam=lam, seed=seed)
def key(fam, feat, s):
    return (f"{fam}_{feat.replace('+', '')}_{s['kind']}_{s['expert'] or '-'}_c{(s['c'] or 0):g}_"
            f"{s['sched']}_a{s['amin']:g}_l{s['lam']:g}_s{s['seed']}")
def path(fam, feat, s): return f"{OUT}/{key(fam, feat, s)}.npz"
def done(fam, feat, s): return os.path.exists(path(fam, feat, s))
def acc(fam, feat, s): return float(np.load(path(fam, feat, s))['R'].mean())
def macc(fam, feat, specs): return float(np.mean([acc(fam, feat, s) for s in specs]))
SELF = f'{OUT}/selections.json'
def sel_get():
    try: return json.load(open(SELF))
    except Exception: return {}
def sel_put(k, v):
    d = sel_get(); d[k] = v; tmp = SELF + '.tmp'; json.dump(d, open(tmp, 'w'), indent=1); os.replace(tmp, SELF)

# ---------------- chains (generators yielding stages; 'WAIT' = dependency not ready) ----------------
def prod_chain(fam):
    c0 = FAMS[fam]['c0']
    yield ('S1 cold, exploration grid, tuning seeds', 'prod', [spec('cold', g[0], g[1], 0.1, s) for g in GRID for s in TUNE])
    rank = sorted(((g[0], g[1], macc(fam, 'prod', [spec('cold', g[0], g[1], 0.1, s) for s in TUNE])) for g in GRID), key=lambda r: -r[2])
    sel_put(f'{fam}_S1_rank', rank); b1 = rank[0][:2]
    yield ('S2 cold, lambda, tuning seeds', 'prod', [spec('cold', b1[0], b1[1], l, s) for l in LAMS for s in TUNE])
    cands = [(b1[0], b1[1], l) for l in [0.1] + LAMS]
    scores = {str(g): macc(fam, 'prod', [spec('cold', *g, s) for s in TUNE]) for g in cands}
    best = max(cands, key=lambda g: scores[str(g)]); sel_put(f'{fam}_S2_lambda', scores); sel_put(f'{fam}_cold_best', list(best))
    ev = [spec('cold', *best, s) for s in EVAL] if (best != DEFAULT or fam == 'bge') else []
    if fam == 'bge' and best != DEFAULT: ev += [spec('cold', *DEFAULT, s) for s in EVAL]
    yield ('S3 cold, best setting, evaluation seeds', 'prod', ev)
    c = c0
    if best != DEFAULT:
        cs = [c0/10, c0, c0*10]
        yield ('S4 Router at best cold setting, c grid, tuning seeds', 'prod',
               [spec('router', *best, s, expert=e, c=cc) for cc in cs for e in EXPERTS for s in TUNE])
        cscore = {f'{cc:g}': macc(fam, 'prod', [spec('router', *best, s, expert=e, c=cc) for e in EXPERTS for s in TUNE]) for cc in cs}
        c = max(cs, key=lambda cc: cscore[f'{cc:g}']); sel_put(f'{fam}_S4_c', cscore)
    sel_put(f'{fam}_router_c_at_best', c)
    ev = [spec('router', *best, s, expert=e, c=c) for e in EXPERTS for s in EVAL] if best != DEFAULT else []
    if fam == 'bge': ev += [spec('router', *DEFAULT, s, expert=e, c=c0) for e in EXPERTS for s in EVAL]
    yield ('S5 Router, evaluation seeds', 'prod', ev)

def arm_chain(fam):
    c0 = FAMS[fam]['c0']; F = 'prod+arm'
    while f'{fam}_S1_rank' not in sel_get(): yield 'WAIT'
    rank = sel_get()[f'{fam}_S1_rank']; sets = []
    for r in rank[:3] + [list(DEFAULT[:2])]:
        g = (r[0], r[1])
        if g not in sets: sets.append(g)
    yield ('A1 cold, category-term features, tuning seeds', F, [spec('cold', g[0], g[1], 0.1, s) for g in sets for s in TUNE])
    scores = {f'{g[0]}/{g[1]:g}': macc(fam, F, [spec('cold', g[0], g[1], 0.1, s) for s in TUNE]) for g in sets}
    best = max(sets, key=lambda g: scores[f'{g[0]}/{g[1]:g}']); sel_put(f'{fam}_A1', scores); sel_put(f'{fam}_arm_best', list(best))
    ev = [spec('cold', best[0], best[1], 0.1, s) for s in EVAL]
    if best != DEFAULT[:2]: ev += [spec('cold', *DEFAULT, s) for s in EVAL]
    yield ('A2 cold, category-term features, evaluation seeds', F, ev)
    cs = [c0/10, c0, c0*10]
    yield ('A3 Router, category-term features, c grid, tuning seeds', F,
           [spec('router', best[0], best[1], 0.1, s, expert=e, c=cc) for cc in cs for e in EXPERTS for s in TUNE])
    cscore = {f'{cc:g}': macc(fam, F, [spec('router', best[0], best[1], 0.1, s, expert=e, c=cc) for e in EXPERTS for s in TUNE]) for cc in cs}
    c = max(cs, key=lambda cc: cscore[f'{cc:g}']); sel_put(f'{fam}_A3_c', cscore)
    yield ('A4 Router, category-term features, evaluation seeds', F,
           [spec('router', best[0], best[1], 0.1, s, expert=e, c=c) for e in EXPERTS for s in EVAL])

# ---------------- worker ----------------
_D = {}
def init_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch; torch.set_num_threads(1)
def bkey(fam, feat, seed, specs):
    h = hashlib.md5('|'.join(key(fam, feat, s) for s in specs).encode()).hexdigest()[:10]
    return f"{fam}_{feat.replace('+', '')}_s{seed}_{h}"
def work(fam, feat, seed, specs):
    import torch; sys.path.insert(0, HERE); import pb_batch
    tf32 = feat == 'prod+arm'
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    todo = [s for s in specs if not done(fam, feat, s)]
    if not todo: return 0
    if fam not in _D: _D[fam] = pb_batch.Data(f"{HERE}/{FAMS[fam]['slim']}")
    bk = bkey(fam, feat, seed, todo); ck = f'{OUT}/ckpt_{bk}.npz'; pg = f'{LOG}/{bk}.prog'
    with open(f'{LOG}/{bk}.specs', 'w') as f: json.dump([key(fam, feat, s) for s in todo], f)
    A, R, RO = pb_batch.run_batch(_D[fam], feat, seed, todo, T=T, ckpt=ck, prog=pg)
    for k, s in enumerate(todo):
        p = path(fam, feat, s); tmp = p + '.tmp.npz'
        np.savez(tmp, A=A[k], R=R[k], RO=RO[k], spec=json.dumps(s), tf32=tf32); os.replace(tmp, p)
    for f in (ck, pg, f'{LOG}/{bk}.specs'):
        if os.path.exists(f): os.remove(f)
    return len(todo)

# ---------------- scheduler ----------------
class Chain:
    def __init__(self, name, prio, fam, gen):
        self.name, self.prio, self.fam, self.gen = name, prio, fam, gen
        self.stage = None; self.feat = None; self.specs = []; self.pending = []; self.running = 0
        self.finished = False; self.waiting = False
    def advance(self):
        while True:
            try: st = next(self.gen)
            except StopIteration: self.finished = True; self.stage = 'finished'; return
            if st == 'WAIT': self.waiting = True; self.stage = 'waiting for S1'; return
            self.waiting = False; self.stage, self.feat, self.specs = st
            todo = [s for s in self.specs if not done(self.fam, self.feat, s)]
            by = {}
            for s in todo: by.setdefault(s['seed'], []).append(s)
            k = KMAX[self.feat]
            self.pending = [(self.fam, self.feat, sd, L[i:i+k]) for sd, L in sorted(by.items()) for i in range(0, len(L), k)]
            if self.pending: return
            log(f'{self.name}: {self.stage} already complete')

def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)

def status(chains, running):
    now = time.strftime('%Y-%m-%d %H:%M:%S'); ch = []
    for c in chains:
        n = len(c.specs); d = sum(done(c.fam, c.feat, s) for s in c.specs) if c.specs and c.feat else 0
        ch.append(dict(name=c.name, stage=c.stage, specs_done=d, specs_total=n, batches_pending=len(c.pending), batches_running=c.running))
    json.dump(dict(updated=now, workers=NW, chains=ch, selections=sel_get()), open(f'{LOG}/status.json.tmp', 'w'), indent=1)
    os.replace(f'{LOG}/status.json.tmp', f'{LOG}/status.json')

def main(extra=(), skip=()):
    os.makedirs(OUT, exist_ok=True); os.makedirs(LOG, exist_ok=True)
    chains = [Chain('BERT Tier 1+3', 0, 'bert', prod_chain('bert')), Chain('BGE Tier 1+3', 1, 'bge', prod_chain('bge')),
              Chain('BERT Tier 2', 2, 'bert', arm_chain('bert')), Chain('BGE Tier 2', 3, 'bge', arm_chain('bge'))] + list(extra)
    chains = [c for c in chains if c.name not in skip]
    log(f'driver start, T={T}, workers={NW}, out={OUT}')
    fails = {}
    with ProcessPoolExecutor(NW, mp_context=mp.get_context('spawn'), initializer=init_worker) as ex:
        running = {}
        while True:
            for c in chains:
                if not c.finished and (c.waiting or (not c.pending and c.running == 0)):
                    before = c.stage; c.advance()
                    if c.stage != before: log(f'{c.name}: now at "{c.stage}"')
            if all(c.finished for c in chains) and not running: break
            free = NW - len(running)
            for c in sorted(chains, key=lambda c: c.prio):
                while free > 0 and c.pending:
                    b = c.pending.pop(0); f = ex.submit(work, *b); running[f] = (c, b); c.running += 1; free -= 1
            status(chains, running)
            if running:
                fin, _ = wait(list(running), timeout=60, return_when=FIRST_COMPLETED)
                for f in fin:
                    c, b = running.pop(f); c.running -= 1
                    try: f.result()
                    except Exception:
                        k = bkey(*b); fails[k] = fails.get(k, 0) + 1
                        log(f'{c.name}: batch {k} failed ({fails[k]}):\n{traceback.format_exc()}')
                        if fails[k] <= 3: c.pending.append(b)
            else: time.sleep(20)
    status(chains, {}); log('ALL CHAINS FINISHED')

if __name__ == '__main__': main()
