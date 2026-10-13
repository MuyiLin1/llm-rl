import numpy as np, sys, collections
fam=sys.argv[1]
z=np.load(f'/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_{fam}_slim.npz')
flat,offs=z['y_flat'],z['y_offsets']; N=len(offs)-1
ys=[flat[offs[i]:offs[i+1]] for i in range(N)]
held=np.arange(50000,60000); online=np.arange(0,50000)
# label frequencies on the ONLINE stream (what the learner will face) and held-out
def freq(idx):
    c=collections.Counter()
    for i in idx: c.update(ys[i].tolist())
    return c
f_on=freq(online); f_ho=freq(held)
K=z['Z'].shape[0]
top=f_ho.most_common(10)
print(f"[{fam}] labels/context {np.mean([len(ys[i]) for i in held]):.2f}; top-10 held-out label frequencies (share of contexts):")
print("   ", [(a, round(c/len(held),3)) for a,c in top])
for e in ('small','base','large'):
    lab=z[f'llm_{e}']
    corr=[i for i in held if lab[i] in set(ys[i].tolist())]
    acc=len(corr)/len(held)
    corr_labels=collections.Counter(int(lab[i]) for i in corr)
    pop=np.array([f_ho[int(lab[i])]/len(held) for i in corr])
    distinct=len(corr_labels)
    # "reach": share of ONLINE contexts containing at least one of the expert's correct-pick labels (weighted by how often it is right on it)
    S_=set(corr_labels); reach=np.mean([len(S_.intersection(ys[i].tolist()))>0 for i in online[:20000]])
    print(f"  {e:6} held-out acc {acc*100:5.2f}%  correct picks {len(corr):4d}  distinct labels {distinct:4d}  "
          f"mean popularity of a correct label {pop.mean()*100:5.2f}%  median {np.median(pop)*100:5.2f}%  reach {reach*100:5.1f}%")
    print(f"         top correct labels: {[(a,c, round(f_ho[a]/len(held),3)) for a,c in corr_labels.most_common(5)]}")
