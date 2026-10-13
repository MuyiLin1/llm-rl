import numpy as np, collections
D='/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/'
def stats(path, idx_eval, idx_prev, label):
    z=np.load(path); fl,of=z['y_flat'],z['y_offsets']
    Y=lambda i: fl[of[i]:of[i+1]]
    prev=collections.Counter()
    for i in idx_prev: prev.update(Y(i).tolist())
    P=np.zeros(z['Z'].shape[0]); 
    for a,c in prev.items(): P[a]=c/len(idx_prev)
    print(f"[{label}] max prevalence {P.max()*100:.2f}%")
    for e in ('small','base','large'):
        lab=z[f'llm_{e}']; hit=np.array([lab[i] in set(Y(i).tolist()) for i in idx_eval])
        pv=np.array([P[lab[i]] for i in idx_eval])
        print(f"   {e:6} accuracy {hit.mean()*100:6.2f}%   dominant-hit q_E {np.mean(hit&(lab[idx_eval]==P.argmax()))*1e3:7.2f}/1000   "
              f"prevalence-weighted hit H_E {np.mean(hit*pv)*1e3:7.3f}/1000")
# prevalences always estimated on a block DISJOINT from the evaluation block
stats(D+'amazon_v1_slim.npz', np.arange(50000,60000), np.arange(0,50000), 'AmazonCat BERT, held-out 50-60k')
stats(HERE:='/private/tmp/claude-502/-Users-linmuyi-code-DMwithLLM/68dc0ee2-93ec-4e14-b569-9bbcd9293426/scratchpad/prerun/fresh_v1.npz', np.arange(0,50000), np.arange(0,50000), 'AmazonCat BERT, fresh 60-110k')
stats(D+'amazon_v3_slim.npz', np.arange(40000,50000), np.arange(0,40000), 'AmazonCat BGE, 40-50k')
stats(D+'wiki_v1_slim.npz', np.arange(0,62200), np.arange(62200,622000), 'WikiLinks Flan-T5 (10% sample)')
stats(D+'wiki_v3_slim.npz', np.arange(0,62200), np.arange(62200,622000), 'WikiLinks Qwen (10% sample)')
