import json, numpy as np, collections
K=json.load(open('seeding_online.json')); B=json.load(open('basins_online.json'))
bas={(x['fam'],x['tag'],x['e'],x['c'],x['s']):x['router'] for x in B}
def basin(t):
    if t['share']<=0.2: return 'none'
    return {1:'r1',2:'r2',3:'r3'}.get(t['rank'],'rX')
for fam in ('v1','v3'):
    X=[x for x in K if x['fam']==fam and (fam,x['tag'],x['e'],x['c'],x['s']) in bas]
    top=X[0]['top']; print(f"\n=== {fam}: top labels {top}; n={len(X)} gated runs (routing<50%)")
    print("  mean routed-correct counts by expert (c=100/0.1 main configs):")
    main=100.0 if fam=='v1' else 0.1
    for e in ('small','base','large'):
        L=[x for x in X if x['e']==e and x['c']==main and x['tag'].startswith('amazon')]
        print(f"    {e:6} n={len(L)} " + "  ".join(f"{t}:{np.mean([x['k'][str(t)] for x in L]):.2f}" for t in top[:4]))
    # basin outcome vs k1 (dominant) and k_r3/k_r2
    print("  P(basin | k_dominant) across all gated runs:")
    by=collections.defaultdict(list)
    for x in X: by[min(x['k'][str(top[0])],5)].append(basin(bas[(fam,x['tag'],x['e'],x['c'],x['s'])]))
    for k in sorted(by): c=collections.Counter(by[k]); n=len(by[k]); print(f"    k1={k}{'+' if k==5 else ' '} n={n:3}  " + "  ".join(f"{b}:{c[b]/n*100:4.0f}%" for b in ('r1','r2','r3','rX','none')))
