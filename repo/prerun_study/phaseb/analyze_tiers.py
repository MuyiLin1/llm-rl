"""Summaries of the finished Phase B tuned-baseline tiers on the evaluation seeds 1-10: BERT tuned (Tier 1+3),
BERT category-term features (Tier 2) and BGE tuned (Tier 1+3, with the default Router reproduced).
Multiplier = mean Router accuracy / max(mean LLM accuracy, mean cold accuracy); regret ratio =
(1 - max(LLM, cold)) / (1 - Router); t = paired t of Router - cold over seeds 1-10; settle = dominant category
predicted on more than 20% of the last fifth of rounds."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out_local'
SEEDS = range(1, 11); EXPERTS = ('small', 'base', 'large')

_C = {}
def load(slim):
    if slim in _C: return _C[slim]
    _C[slim] = _load(slim); return _C[slim]

def _load(slim):
    z = np.load(f'{HERE}/{slim}'); fl, of = z['y_flat'], z['y_offsets']
    ys = [set(fl[of[i]:of[i+1]].tolist()) for i in range(len(of)-1)]
    top = int(np.bincount(fl).argmax())
    L = {e: z[f'llm_{e}'] for e in EXPERTS}
    llm = {e: np.mean([np.mean([int(L[e][j]) in ys[j] for j in np.random.RandomState(s).permutation(50000)]) for s in SEEDS]) for e in EXPERTS}
    return top, np.bincount(fl).max() / len(ys), llm

def R(n): return np.load(f'{OUT}/{n}.npz')['R'].astype(float)
def settle(n, cat): a = np.load(f'{OUT}/{n}.npz')['A']; late = a[-len(a)//5:]; return (late == cat).mean() > 0.2
def tstat(d): d = np.asarray(d); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))

def block(title, slim, cold, router):
    top, frac, llm = load(slim)
    c = np.array([R(cold.format(s=s)).mean() for s in SEEDS])
    print(f'{title}\n  cold {c.mean()*100:5.2f} +- {c.std(ddof=1)*100:4.2f}%  settles {sum(settle(cold.format(s=s), top) for s in SEEDS)}/10'
          f'   (top category {top} on {frac*100:.1f}% of contexts)')
    for e in EXPERTS:
        r = np.array([R(router.format(e=e, s=s)).mean() for s in SEEDS]); base = max(llm[e], c.mean())
        ro = np.mean([np.load(f'{OUT}/{router.format(e=e, s=s)}.npz')['RO'].mean() for s in SEEDS])
        print(f'  {e:5}: Router {r.mean()*100:5.2f} +- {r.std(ddof=1)*100:4.2f}%  LLM {llm[e]*100:5.2f}%  mult {r.mean()/base:4.2f}x  '
              f'regret ratio {(1-base)/(1-r.mean()):.3f}  t {tstat(r-c):6.2f}  routed {ro*100:5.2f}%  settles {sum(settle(router.format(e=e, s=s), top) for s in SEEDS)}/10')

block('BERT default', 'amazon_v1_slim.npz', 'bert_prod_cold_-_c0_fast_a0.1_l0.1_s{s}', 'bert_prod_router_{e}_c100_fast_a0.1_l0.1_s{s}') if os.path.exists(f'{OUT}/bert_prod_router_small_c100_fast_a0.1_l0.1_s1.npz') else None
block('BERT tuned (fast/0.05, lambda 0.1, c 1000)', 'amazon_v1_slim.npz', 'bert_prod_cold_-_c0_fast_a0.05_l0.1_s{s}', 'bert_prod_router_{e}_c1000_fast_a0.05_l0.1_s{s}')
block('BERT category-term features (fast/0.05, lambda 0.1, c 100)', 'amazon_v1_slim.npz', 'bert_prodarm_cold_-_c0_fast_a0.05_l0.1_s{s}', 'bert_prodarm_router_{e}_c100_fast_a0.05_l0.1_s{s}')
block('BERT category-term features at default schedule (fast/0.1), Router at fast/0.05 c 100 [cold only differs]', 'amazon_v1_slim.npz', 'bert_prodarm_cold_-_c0_fast_a0.1_l0.1_s{s}', 'bert_prodarm_router_{e}_c100_fast_a0.05_l0.1_s{s}')
block('BGE default (fast/0.1, lambda 0.1, c 0.1)', 'amazon_v3_slim.npz', 'bge_prod_cold_-_c0_fast_a0.1_l0.1_s{s}', 'bge_prod_router_{e}_c0.1_fast_a0.1_l0.1_s{s}')
block('BGE tuned (const/0.2, lambda 0.01, c 0.1)', 'amazon_v3_slim.npz', 'bge_prod_cold_-_c0_const_a0.2_l0.01_s{s}', 'bge_prod_router_{e}_c0.1_const_a0.2_l0.01_s{s}')
