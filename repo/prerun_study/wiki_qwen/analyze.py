"""Evaluation-seed summaries for WikiLinks: Flan-T5 (BERT-768d) at c=1e4 and Qwen2.5 (BGE-1024d) at the selected c.
Prints the rows of the paper's WikiLinks tables: LLM, cold, Router (mean+-std over seeds 1-10), routed share,
accuracy multiplier, regret ratio, paired t vs cold, and the decomposition A/I/B in correct answers."""
import os, json, numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/Users/linmuyi/code/DMwithLLM/reproducibility_package/data'
SEEDS = range(1, 11)

def llm_correct(path):
    z = np.load(path); fl, of = z['y_flat'], z['y_offsets']
    ys = [set(fl[of[i]:of[i+1]].tolist()) for i in range(len(of)-1)]
    out = {}
    for k in z.files:
        if k.startswith('llm_'):
            a = z[k]; out[k[4:]] = np.array([int(a[j]) in ys[j] for j in range(len(ys))], np.int8)
    return out

def tstat(d):
    d = np.asarray(d, float); return d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else float('nan')

def report(title, names, load_router, load_cold, lc):
    print(f'\n== {title} ==')
    cold = {s: load_cold(s) for s in SEEDS}
    ca = np.array([cold[s].mean() for s in SEEDS])
    print(f'cold {ca.mean()*100:.2f} +- {ca.std(ddof=1)*100:.2f}')
    for e, nice in names:
        rows = []
        for s in SEEDS:
            r, ro = load_router(e, s); order = np.random.RandomState(s).permutation(len(r))
            l = lc[e][order]; c = cold[s]
            A = int((l[ro] - c[ro]).sum()); I = int((r[~ro] - c[~ro]).sum()); B = int((r[~ro] - l[~ro]).sum())
            rows.append((r.mean(), ro.mean(), A, I, B, r.mean() - c.mean()))
        R = np.array(rows); llm = lc[e].mean(); base = max(llm, ca.mean())
        print(f'{nice:14} LLM {llm*100:5.2f}  Router {R[:,0].mean()*100:5.2f} +- {R[:,0].std(ddof=1)*100:.2f}  '
              f'routed {R[:,1].mean()*100:5.2f}%  lift {R[:,5].mean()*100:+.2f}pp  mult {R[:,0].mean()/base:.3f}x  '
              f'regret ratio {(1-base)/(1-R[:,0].mean()):.3f}  t vs cold {tstat(R[:,5]):.2f}  '
              f'A {R[:,2].mean():.0f}  I {R[:,3].mean():.0f} (I>0 on {int((R[:,3]>0).sum())}/10)  B {R[:,4].mean():.0f}')

def _flan_router(e, s):
    z = np.load(f'{P}/wiki_c10000/out/{e}_s{s}.npz'); return z['r_router'].astype(np.int8), z['routed'].astype(bool)
report('WikiLinks Flan-T5 (BERT-768d), c=1e4, seeds 1-10',
       [('small', 'Flan-T5-small'), ('base', 'Flan-T5-base'), ('large', 'Flan-T5-large')],
       _flan_router, lambda s: np.load(f'{P}/wiki_c10000/out/small_s{s}.npz')['r_cold'].astype(np.int8),
       llm_correct(f'{DATA}/wiki_v1_slim.npz'))

c = json.load(open(f'{P}/wiki_qwen/out/chosen_c.json'))['c']
def _qwen_router(e, s):
    z = np.load(f'{P}/wiki_qwen/out/{e}_c{c:g}_s{s}.npz'); return z['r_router'].astype(np.int8), z['routed'].astype(bool)
report(f'WikiLinks Qwen2.5 (BGE-1024d), c={c:g}, seeds 1-10',
       [('small', 'Qwen-0.5B'), ('base', 'Qwen-3B'), ('large', 'Qwen-7B')],
       _qwen_router, lambda s: np.load(f'{P}/wiki_qwen/out/cold_s{s}.npz')['r_cold'].astype(np.int8),
       llm_correct(f'{DATA}/wiki_v3_slim.npz'))

print('\n== Qwen sweep, tuning seeds 11-20: routed share and Router accuracy by c ==')
for cc in (0.01, 0.1, 1.0, 10.0, 100.0):
    cells = []
    for e in ('small', 'base', 'large'):
        zs = [np.load(f'{P}/wiki_qwen/out/{e}_c{cc:g}_s{s}.npz') for s in range(11, 21)]
        cells.append((np.mean([z['routed'].mean() for z in zs])*100, np.mean([z['r_router'].mean() for z in zs])*100))
    cold = np.mean([np.load(f'{P}/wiki_qwen/out/cold_s{s}.npz')['r_cold'].mean() for s in range(11, 21)])*100
    print(f'c={cc:g}: routed ' + ' / '.join(f'{x[0]:.2f}' for x in cells) + '   Router ' + ' / '.join(f'{x[1]:.2f}' for x in cells) + f'   cold {cold:.2f}')
