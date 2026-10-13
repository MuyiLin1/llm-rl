import json, os, math, statistics as st
D=os.path.expanduser('~/DMwithLLM/amazoncat13k/study')
C={'full':'results_algo1_amazonv1_v1_{m}_c100.0_pfull_s{s}.json',
   'nom':'results_algo1_nom_v1_{m}_c100.0_pfull_nom_s{s}.json',
   'passive':'results_algo1_passiveonly_v1_{m}_c100.0_s{s}.json',
   'corral':'results_algo1_corral_v1_{m}_c100.0_pfull_corral_s{s}.json',
   'off1600':'results_algo1_off1600_v1_{m}_c100.0_pfull_offline_s{s}.json',
   'off5000':'results_algo1_off5000_v1_{m}_c100.0_pfull_offline_s{s}.json'}
def pt(a,b):
    d=[x-y for x,y in zip(a,b)]; s=st.stdev(d); return st.mean(d)/(s/math.sqrt(len(d)))
print(f"{'cond':8}{'llm':7}{'acc ratio':>10}{'regret ratio':>13}{'regret saved %':>15}{'per-seed RR spread':>19}")
per={}
for c,pat in C.items():
    for m in ('small','base','large'):
        R=[];K=[]
        for s in range(1,11):
            p=os.path.join(D,pat.format(m=m,s=s))
            if not os.path.exists(p): continue
            h=json.load(open(p)); R.append(h['hybrid'][-1]); K.append(h['linucb_cold'][-1])
        if not R: continue
        rr=[(1-k)/(1-r) for r,k in zip(R,K)]; per[(c,m)]=rr
        acc=st.mean(R)/st.mean(K); reg=(1-st.mean(K))/(1-st.mean(R))
        print(f"{c:8}{m:7}{acc:>10.2f}{reg:>13.3f}{100*(1-1/reg):>14.1f}%{max(rr)-min(rr):>19.3f}")
print("\npaired t on per-seed REGRET ratio (t*=2.26):")
for c in ('full','nom','passive'):
    print(f"  {c:8} large-small {pt(per[(c,'large')],per[(c,'small')]):+.2f}   base-small {pt(per[(c,'base')],per[(c,'small')]):+.2f}   large-base {pt(per[(c,'large')],per[(c,'base')]):+.2f}")
