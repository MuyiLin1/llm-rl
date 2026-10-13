"""Phase B on the Mac: the same four chains, stages and selections as phaseb.py (the cluster driver), with each
run computed by pb_sim.run -- float64 Sherman-Morrison state on the host, float32 scoring on the Apple GPU,
the split of prerun_study/sim.py. Several worker processes share the GPU; each run checkpoints every 2,000
rounds and resumes. Start/stop/status: start_local.sh, stop_local.sh, status_local.sh. Outputs keep the cluster's file names and keys (A, R, RO, spec)."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('PB_OUT', f'{HERE}/out_local'); os.environ.setdefault('PB_LOG', f'{HERE}/logs_local')
os.environ.setdefault('PB_WORKERS', '3')
import numpy as np
import phaseb, skew
for _n in skew.CAPS: phaseb.FAMS[_n] = dict(slim='amazon_v1_slim.npz', c0=100.0, skew=True)

_D = {}
def work(fam, feat, seed, specs):
    import pb_sim
    n = 0
    F = phaseb.FAMS[fam]; slim = f"{HERE}/{F['slim']}"
    for s in specs:
        if phaseb.done(fam, feat, s): continue
        if (slim, feat) not in _D: _D[(slim, feat)] = pb_sim.Data(slim, feat)
        T, order = (skew.TS, skew.order(fam, slim, s['seed'])) if F.get('skew') else (phaseb.T, None)
        k = phaseb.key(fam, feat, s); ck = f'{phaseb.OUT}/ckpt_{k}.npz'; pg = f'{phaseb.LOG}/{k}.prog'
        start = time.time()
        def prog(t, T, acc, rate):
            with open(pg + '.tmp', 'w') as f:
                json.dump(dict(t=t, T=T, K=1, acc=[acc], rounds_per_sec=rate, eta_min=(T-t)/max(rate, 1e-9)/60,
                               updated=time.strftime('%m-%d %H:%M:%S')), f)
            os.replace(pg + '.tmp', pg)
        A, R, RO = pb_sim.run(_D[(slim, feat)], s, T=T, ckpt=ck, every=2000, progress=prog, order=order)
        p = phaseb.path(fam, feat, s); tmp = p + '.tmp.npz'
        np.savez(tmp, A=A, R=R, RO=RO, spec=json.dumps(s), tf32=False); os.replace(tmp, p)
        for f in (ck, pg):
            if os.path.exists(f): os.remove(f)
        phaseb.log(f'{k} done: acc {R.mean()*100:.2f}% routed {RO.mean()*100:.2f}% ({(time.time()-start)/60:.0f} min)')
        n += 1
    return n

phaseb.work = work
phaseb.KMAX = {'prod': 1, 'prod+arm': 1}           # one run per task: finer scheduling across chains

XFLOORS = [0.1, 0.3, 1.0, 3.0, 10.0]
def explore_chain(fam='bert'):
    """Exploration sweep: cold LinUCB and the Router (paper's c) at constant exploration levels, evaluation seeds.
    Remark 6 predicts the Router's multiplier falls toward the coverage-only gain as exploration grows."""
    c0 = phaseb.FAMS[fam]['c0']
    yield ('X1 exploration sweep, cold + Router, evaluation seeds', 'prod',
           [phaseb.spec('cold', 'const', a, 0.1, s) for a in XFLOORS for s in phaseb.EVAL] +
           [phaseb.spec('router', 'const', a, 0.1, s, expert=e, c=c0) for a in XFLOORS for e in phaseb.EXPERTS for s in phaseb.EVAL])

def skew_chain(name):
    """One skew level: cold LinUCB and the Router at the paper's settings on the capped stream, evaluation seeds."""
    yield (f'K1 skew level {name[-2:]}%, cold + Router, evaluation seeds', 'prod',
           [phaseb.spec('cold', *phaseb.DEFAULT, s) for s in phaseb.EVAL] +
           [phaseb.spec('router', *phaseb.DEFAULT, s, expert=e, c=100.0) for e in phaseb.EXPERTS for s in phaseb.EVAL])

if __name__ == '__main__':
    os.makedirs(phaseb.LOG, exist_ok=True)
    with open(f'{phaseb.LOG}/driver.pid', 'w') as f: f.write(str(os.getpid()))
    sk = f'{phaseb.LOG}/skip.txt'   # one chain name per line; delete the file to run them again
    skip = [l.strip() for l in open(sk) if l.strip()] if os.path.exists(sk) else []
    if skip: phaseb.log(f'skipping chains: {skip}')
    phaseb.main(extra=[phaseb.Chain('BERT exploration sweep', 0.5, 'bert', explore_chain('bert'))] +
                      [phaseb.Chain(f'Skew {n[-2:]}%', 0.6, n, skew_chain(n)) for n in skew.CAPS], skip=skip)
