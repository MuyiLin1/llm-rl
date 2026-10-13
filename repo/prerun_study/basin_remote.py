import numpy as np, glob, json, collections, re, sys
out=[]
for fam in ('v1','v3'):
    z=np.load(f'amazon_{fam}_slim.npz'); fl,of=z['y_flat'],z['y_offsets']; n=min(50000,len(of)-1)
    prev=collections.Counter(l for i in range(n) for l in set(fl[of[i]:of[i+1]].tolist())); prev={k:v/n for k,v in prev.items()}
    rank={k:i+1 for i,(k,_) in enumerate(sorted(prev.items(), key=lambda x:-x[1]))}
    for p in sorted(glob.glob(f'perround_algo1_*_{fam}_*.npz')):
        m=re.match(r'perround_algo1_(.+?)_'+fam+r'_(small|base|large)_c([0-9.e+]+)_pfull(?:_(\w+?))?_s(\d+)\.npz',p)
        if not m: continue
        tag,e,c,rt,s=m.groups()
        try: o=np.load(p)
        except Exception as ex: continue
        if int(o['step'])<49999 and len(o['r_router'])<50000: continue
        rec=dict(tag=tag,fam=fam,e=e,c=float(c),router=rt or 'gate',s=int(s))
        for tr in ('router','cold'):
            a=o['a_'+tr][40000:50000]; r=o['r_'+tr]
            top,cnt=collections.Counter(a.tolist()).most_common(1)[0]
            rec[tr]=dict(top=int(top),share=cnt/len(a),rank=rank.get(int(top),9999),prev=prev.get(int(top),0),late=float(r[40000:50000].mean()),acc=float(r.mean()))
        rec['routed']=int(o['routed'].sum())
        out.append(rec)
json.dump(out,open('basins_online.json','w')); print(len(out))
