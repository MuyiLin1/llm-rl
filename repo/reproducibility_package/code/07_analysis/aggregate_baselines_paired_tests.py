import json, os, math
D = os.path.expanduser('~/DMwithLLM/amazoncat13k/study')
SEEDS=list(range(1,11)); EXP=['small','base','large']
ARMS={'gate':'results_algo1_amazonv1_v1_{e}_c100.0_pfull_s{s}.json',
      'corral':'results_algo1_corral_v1_{e}_c100.0_pfull_corral_s{s}.json',
      'off1600':'results_algo1_off1600_v1_{e}_c100.0_pfull_offline_s{s}.json',
      'off5000':'results_algo1_off5000_v1_{e}_c100.0_pfull_offline_s{s}.json'}
def load(arm,e,s):
    return json.load(open(os.path.join(D,ARMS[arm].format(e=e,s=s))))
def mean(v): return sum(v)/len(v)
def sd(v):
    m=mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
def paired_t(a,b):
    d=[x-y for x,y in zip(a,b)]; n=len(d)
    s=sd(d)
    return mean(d), s/math.sqrt(n), (mean(d)/(s/math.sqrt(n)) if s>0 else float('inf'))

acc={}; qry={}
for e in EXP:
    for a in ARMS:
        acc[(e,a)]=[load(a,e,s)['hybrid'][-1]*100 for s in SEEDS]
        qry[(e,a)]=[load(a,e,s)['llm_fraction'][-1]*100 for s in SEEDS]
cold=[load('gate','base',s)['linucb_cold'][-1]*100 for s in SEEDS]

print("t* = 2.26 at n=10 (two-sided, 5%)\n")
print(f"{'expert':7}{'comparison':>18}{'mean diff':>11}{'se':>8}{'t':>8}{'sig':>6}")
for e in EXP:
    for b in ['corral','off1600','off5000']:
        md,se,t=paired_t(acc[(e,'gate')],acc[(e,b)])
        print(f"{e:7}{'gate - '+b:>18}{md:>11.2f}{se:>8.2f}{t:>8.2f}{('YES' if abs(t)>2.26 else 'ns'):>6}")
    print()

print("=== expert queries actually issued (mean count over 50,000 rounds) ===")
print(f"{'expert':7}{'gate':>10}{'corral':>10}{'off1600':>10}{'off5000':>10}")
for e in EXP:
    row=[mean(qry[(e,a)])/100*50000 for a in ARMS]
    print(f"{e:7}"+"".join(f"{x:>10.0f}" for x in row))

print("\n=== per-seed multiplier (mean of ratios) vs ratio of means ===")
print(f"{'expert':7}{'arm':>9}{'ratio-of-means':>16}{'mean-of-ratios':>16}")
for e in EXP:
    for a in ARMS:
        rom=mean(acc[(e,a)])/mean(cold)
        mor=mean([x/c for x,c in zip(acc[(e,a)],cold)])
        print(f"{e:7}{a:>9}{rom:>16.2f}{mor:>16.2f}")
