import numpy as np, os, math
D=os.path.expanduser('~/DMwithLLM/amazoncat13k/study')
SEEDS=list(range(1,11)); EXP=['small','base','large']
PAT={'gate':'perround_algo1_amazonv1_v1_{e}_c100.0_pfull_s{s}.npz',
     'corral':'perround_algo1_corral_v1_{e}_c100.0_pfull_corral_s{s}.npz',
     'off1600':'perround_algo1_off1600_v1_{e}_c100.0_pfull_offline_s{s}.npz',
     'off5000':'perround_algo1_off5000_v1_{e}_c100.0_pfull_offline_s{s}.npz'}
def mean(v): return sum(v)/len(v)
def sd(v):
    m=mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
print(f"{'expert':7}{'arm':>9}{'routed':>8}{'A':>12}{'V':>14}{'B':>10}{'V>0':>6}")
for e in EXP:
    for a in PAT:
        A=[];V=[];B=[];R=[]
        for s in SEEDS:
            p=os.path.join(D,PAT[a].format(e=e,s=s))
            if not os.path.exists(p): continue
            L=np.load(p); n=int(L['step'])
            ro=L['routed'][:n].astype(bool)
            rr=L['r_router'][:n]; rc=L['r_cold'][:n]; rl=L['r_llm'][:n]
            A.append(float((rr[ro]-rc[ro]).sum())); V.append(float((rr[~ro]-rc[~ro]).sum()))
            B.append(float((rr[~ro]-rl[~ro]).sum())); R.append(int(ro.sum()))
        if not A: print(f"{e:7}{a:>9}  (no per-round logs)"); continue
        print(f"{e:7}{a:>9}{mean(R):>8.0f}{mean(A):>12.0f}{mean(V):>9.0f}+-{sd(V):<4.0f}{mean(B):>10.0f}{sum(1 for x in V if x>0):>4}/{len(V)}")
