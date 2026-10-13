import json, os, math
D = os.path.expanduser('~/DMwithLLM/amazoncat13k/study')
SEEDS=list(range(1,11)); EXP=['small','base','large']
ARMS={'gate':'results_algo1_amazonv1_v1_{e}_c100.0_pfull_s{s}.json',
      'corral':'results_algo1_corral_v1_{e}_c100.0_pfull_corral_s{s}.json',
      'off1600':'results_algo1_off1600_v1_{e}_c100.0_pfull_offline_s{s}.json',
      'off5000':'results_algo1_off5000_v1_{e}_c100.0_pfull_offline_s{s}.json'}
def get(arm,e,s):
    p=os.path.join(D,ARMS[arm].format(e=e,s=s))
    if not os.path.exists(p): return None
    d=json.load(open(p))
    if d['step'][-1]!=50000: return ('SHORT',d['step'][-1])
    return d
def mean(v): return sum(v)/len(v)
def sd(v):
    m=mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1)) if len(v)>1 else 0.0

# 1. cold consistency check across arms, per seed
print("=== cold LinUCB accuracy per seed (must be identical across arms) ===")
print(f"{'seed':>4}"+"".join(f"{a:>10}" for a in ARMS))
bad=0
for s in SEEDS:
    row=[]
    for a in ARMS:
        d=get(a,'base',s); row.append(d['linucb_cold'][-1] if isinstance(d,dict) else float('nan'))
    if max(row)-min(row) > 1e-9: bad+=1
    print(f"{s:>4}"+"".join(f"{x:>10.5f}" for x in row))
print("seeds where cold differs across arms:",bad)

print("\n=== router accuracy, mean +/- sd over 10 seeds, and multiplier vs cold ===")
print(f"{'expert':7}{'arm':>9}{'router%':>16}{'cold%':>9}{'mult':>8}{'LLM%':>8}")
res={}
for e in EXP:
    for a in ARMS:
        ds=[get(a,e,s) for s in SEEDS]
        short=[d for d in ds if isinstance(d,tuple)]
        ds=[d for d in ds if isinstance(d,dict)]
        r=[d['hybrid'][-1]*100 for d in ds]; c=[d['linucb_cold'][-1]*100 for d in ds]
        lf=[d['llm_fraction'][-1]*100 for d in ds]
        res[(e,a)]=dict(r=r,c=c)
        note=f"  !! {len(short)} short" if short else ""
        print(f"{e:7}{a:>9}{mean(r):>10.2f}+-{sd(r):<5.2f}{mean(c):>9.2f}{mean(r)/mean(c):>8.2f}{mean(lf):>8.2f}{note}")
json.dump({f"{e}|{a}":v for (e,a),v in res.items()}, open('/tmp/baseline_res.json','w'))
