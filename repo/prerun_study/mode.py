import numpy as np, glob, collections
r=np.load('reh_v1.npz'); fl,of=r['y_flat'],r['y_offsets']; n=len(of)-1
prev=collections.Counter(l for i in range(n) for l in set(fl[of[i]:of[i+1]].tolist()))
prev={k:v/n for k,v in prev.items()}
rank={k:i+1 for i,(k,_) in enumerate(sorted(prev.items(), key=lambda x:-x[1]))}
rows=[]
for p in sorted(glob.glob('reh_out/[ADP]_*_s*.npz')):
    if p.endswith('.tmp.npz'): continue
    o=np.load(p); key='a_router' if 'router' in p else 'a_cold'; a=o[key]; rr=o['r_router' if 'router' in p else 'r_cold']
    late=a[8000:10000]; top,cnt=collections.Counter(late.tolist()).most_common(1)[0]
    rows.append(dict(p=p.split('/')[-1][:-4], top=top, share=cnt/len(late), prev=prev.get(top,0), rank=rank.get(top,9999), late_acc=rr[8000:10000].mean(), acc=rr[:10000].mean()))
S_=np.array([x['share'] for x in rows]); print("late top-label share histogram:", np.histogram(S_,bins=[0,.05,.1,.2,.3,.4,.5,.7,1])[0])
mode=[x for x in rows if x['share']>0.2]; non=[x for x in rows if x['share']<=0.2]
print(f"runs {len(rows)}: mode-locked (top share>20%) {len(mode)}, not {len(non)}")
print("locked-label rank counts:", collections.Counter(x['rank'] for x in mode).most_common(10))
for rk in sorted(set(x['rank'] for x in mode)):
    L=[x for x in mode if x['rank']==rk]; print(f"  rank {rk} (prev {L[0]['prev']*100:.1f}%): n={len(L)} share {np.mean([x['share'] for x in L])*100:.0f}%  late acc {np.mean([x['late_acc'] for x in L])*100:.1f}%")
print(f"  not locked: n={len(non)} late acc {np.mean([x['late_acc'] for x in non])*100:.1f}%")
# does prevalence of locked label predict late accuracy?
x=np.array([m['prev'] for m in mode]); y=np.array([m['late_acc'] for m in mode])
b=np.polyfit(x,y,1); print(f"late_acc ~ {b[0]:.2f}*prev + {b[1]*100:.1f}pp ; corr {np.corrcoef(x,y)[0,1]:.3f}")
print("\nby run type: basin counts (rank1=1471, rank3=7892, other-locked, none)")
g=collections.defaultdict(list)
for x in rows: g[x['p'].rsplit('_s',1)[0]].append(x)
for k in sorted(g):
    L=g[k]; c=collections.Counter(('r'+str(x['rank'])) if x['share']>0.2 else 'none' for x in L)
    print(f"  {k:28} n={len(L):2}  {dict(c)}   acc@10k {np.mean([x['acc'] for x in L])*100:5.2f}%")
