import json,os,statistics as st
D=os.path.expanduser('~/DMwithLLM/amazoncat13k/study')
m={}
for e in ('small','base','large'):
    rr=[]
    for s in range(1,11):
        h=json.load(open(f"{D}/results_algo1_amazonv1_v1_{e}_c100.0_pfull_s{s}.json"))
        rr.append((1-h['linucb_cold'][-1])/(1-h['hybrid'][-1]))
    m[e]=st.mean(rr)
print({k:round(v,3) for k,v in m.items()}, "span of per-seed-mean regret ratios:", round(max(m.values())-min(m.values()),3))
for tag,suf in (('linear','_pfull_linear'),('lin0.01','_pfull_linear'),('lin0.001','_pfull_linear')):
    p=f"{D}/linear_regret_{tag}.json"
    if os.path.exists(p):
        o=json.load(open(p)); mm={e:st.mean(v['mult'] for k,v in o.items() if k.startswith(e+'_')) for e in ('small','base','large')}
        print(tag,{k:round(v,3) for k,v in mm.items()},"span",round(max(mm.values())-min(mm.values()),3))
    else: print(tag,"no json")
