import numpy as np, time, json
rng=np.random.default_rng(0)
z=np.load('/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_v1_slim.npz')
C=z['contexts'].astype(np.float32); Z=z['Z'].astype(np.float32)
flat,offs=z['y_flat'],z['y_offsets']; ys=[set(flat[offs[i]:offs[i+1]].tolist()) for i in range(len(offs)-1)]
acq=np.arange(50000,55000); ev=rng.choice(np.arange(55000,60000),2000,replace=False)
Ce=C[ev]; d=C.shape[1]; lam=0.1; DOM=1471
res={}
for e in ('small','base','large'):
    lab=z[f'llm_{e}']
    for tau in (24,600,1500,7000):
        rows=[]
        for rep in range(60):
            S_=rng.choice(acq if tau<=5000 else np.arange(50000,60000),tau,replace=tau>5000)
            X=C[S_].astype(np.float64)*Z[lab[S_]].astype(np.float64)
            r=np.array([1.0 if lab[i] in ys[i] else 0.0 for i in S_])
            k=int(sum(1 for i in S_ if lab[i]==DOM and lab[i] in ys[i]))
            th=np.linalg.solve(lam*np.eye(d)+X.T@X, X.T@r).astype(np.float32)
            sc=(Ce*th)@Z.T; am=sc.argmax(1)
            p1=np.mean([am[j] in ys[ev[j]] for j in range(len(ev))]); dom=np.mean(am==DOM)
            rows.append((k,p1,dom))
        a=np.array(rows)
        res[f'{e}_{tau}']=a.tolist()
        print(f"{e:6} tau={tau:5d}: mean correct-1471 {a[:,0].mean():5.2f} | static greedy P@1 {a[:,1].mean()*100:5.2f}% | "
              f"dominant-top share {a[:,2].mean()*100:5.1f}% | P(share>25%) {np.mean(a[:,2]>0.25)*100:5.1f}%",flush=True)
json.dump(res,open(__file__.replace('.py','.json'),'w'))
