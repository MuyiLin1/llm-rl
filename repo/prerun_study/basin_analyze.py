import json, numpy as np, collections
B=json.load(open('basins_online.json'))
print(collections.Counter((x['fam'],x['tag']) for x in B))
def basin(t): return ('r'+str(t['rank'])) if t['share']>0.2 else 'none'
for fam in ('v1','v3'):
    print(f"\n==== {fam} ====")
    # cold: dedupe by seed (cold is expert-independent)
    cold={}
    for x in B:
        if x['fam']==fam: cold.setdefault(x['s'],x['cold'])
    cb=collections.Counter(basin(t) for t in cold.values())
    print(f"cold (unique seeds n={len(cold)}): basins {dict(cb)}  acc {np.mean([t['acc'] for t in cold.values()])*100:.2f}%")
    for k,g in sorted(collections.groupby if False else {}.items()): pass
    grp=collections.defaultdict(list)
    for x in B:
        if x['fam']==fam: grp[(x['tag'],x['c'],x['e'])].append(x)
    for k in sorted(grp):
        L=grp[k]; c=collections.Counter(basin(x['router']) for x in L)
        print(f"  {str(k):48} n={len(L):2} {dict(sorted(c.items()))}  acc {np.mean([x['router']['acc'] for x in L])*100:5.2f}%")
    # all router + cold tracks: late acc vs prevalence of locked label
    T=[x['router'] for x in B if x['fam']==fam]+list(cold.values())
    print(" basin -> late acc (all tracks):")
    for b in sorted(set(basin(t) for t in T), key=lambda s:(s=='none', int(s[1:]) if s!='none' else 0)):
        L=[t for t in T if basin(t)==b]; print(f"    {b:5} n={len(L):3} prev {np.mean([t['prev'] for t in L])*100:5.1f}%  share {np.mean([t['share'] for t in L])*100:4.0f}%  late acc {np.mean([t['late'] for t in L])*100:5.2f}%  full acc {np.mean([t['acc'] for t in L])*100:5.2f}%")
