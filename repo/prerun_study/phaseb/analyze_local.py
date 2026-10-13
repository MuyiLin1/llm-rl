"""Summaries of the finished local Phase B experiments: the exploration sweep and the skew experiment.
Multiplier = mean Router accuracy / max(mean LLM accuracy, mean cold accuracy) on the same seeds and stream;
regret ratio = (1 - max(LLM, cold)) / (1 - Router); t = paired t of Router - cold over seeds 1-10."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import skew
OUT = f'{HERE}/out_local'; SLIM = f'{HERE}/amazon_v1_slim.npz'; SEEDS = range(1, 11); EXPERTS = ('small', 'base', 'large')
z = np.load(SLIM); fl, of = z['y_flat'], z['y_offsets']
YS = [set(fl[of[i]:of[i+1]].tolist()) for i in range(len(of)-1)]
LLM = {e: z[f'llm_{e}'] for e in EXPERTS}

def R(name): return np.load(f'{OUT}/{name}.npz')['R'].astype(float)
def A(name): return np.load(f'{OUT}/{name}.npz')['A']
def settle(a, cat=1471): late = a[-len(a)//5:]; return float((late == cat).mean() > 0.2)
def tstat(d): d = np.asarray(d); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))

def row(label, cold_names, router_names, orders):
    c = np.array([R(n).mean() for n in cold_names]); out = [f'{label}: cold {c.mean()*100:5.2f}% (settles on 1471 in {int(sum(settle(A(n)) for n in cold_names))}/10)']
    for e in EXPERTS:
        r = np.array([R(n.format(e=e)).mean() for n in router_names])
        llm = np.mean([np.mean([int(LLM[e][j]) in YS[j] for j in o]) for o in orders])
        base = max(llm, c.mean())
        st = int(sum(settle(A(n.format(e=e))) for n in router_names))
        out.append(f'    {e:5}: Router {r.mean()*100:5.2f}%  LLM {llm*100:4.2f}%  mult {r.mean()/base:4.2f}x  '
                   f'regret ratio {(1-base)/(1-r.mean()):.3f}  t {tstat(r-c):6.2f}  settles {st}/10')
    print('\n'.join(out))

stream = lambda s: np.random.RandomState(s).permutation(50000)
print('== Exploration sweep (BERT, constant exploration, lambda 0.1, c=100, T=50,000, seeds 1-10) ==')
for a in (0.1, 0.3, 1.0, 3.0, 10.0):
    row(f'floor {a:g}', [f'bert_prod_cold_-_c0_const_a{a:g}_l0.1_s{s}' for s in SEEDS],
        [f'bert_prod_router_{{e}}_c100_const_a{a:g}_l0.1_s{s}' for s in SEEDS], [stream(s) for s in SEEDS])

print(f'\n== Skew (BERT, paper settings, c=100, T={skew.TS:,}, seeds 1-10) ==')
for name in skew.CAPS:
    P = skew.pool(name, SLIM); k = np.zeros(len(YS), bool); k[P] = True
    cnt = np.bincount(fl, weights=np.repeat(k, np.diff(of))); top = cnt.max()/len(P)
    row(f'{name} (top category on {top*100:.1f}% of contexts)',
        [f'{name}_prod_cold_-_c0_fast_a0.1_l0.1_s{s}' for s in SEEDS],
        [f'{name}_prod_router_{{e}}_c100_fast_a0.1_l0.1_s{s}' for s in SEEDS],
        [skew.order(name, SLIM, s) for s in SEEDS])
