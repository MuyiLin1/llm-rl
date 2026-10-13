import json, numpy as np, collections
from scipy.optimize import minimize_scalar
out=json.load(open(__file__.rsplit('/',1)[0]+'/cause.json'))
ok=[o for o in out if not (o['tag']=='amazonv1' and o['c']==10000)]
gate=[o for o in ok if o['tag']=='amazonv1']; off=[o for o in ok if o['tag'].startswith('off')]; cor=[o for o in ok if o['tag']=='corral']
# ---- Test A: two-state model with OBSERVED lock rate explains config accuracy
aL=np.mean([o['acc'] for o in gate if o['lock']]); aN=np.mean([o['acc'] for o in gate if not o['lock']])
print(f"gate runs: a_lock {aL*100:.2f}%  a_nolock {aN*100:.2f}%")
def cfgs(runs):
    g=collections.defaultdict(list)
    for o in runs: g[(o['tag'],o['c'],o['e'])].append(o)
    return g
pred=[];obs=[]
for k,v in sorted(cfgs(ok).items()):
    p=np.mean([o['lock'] for o in v]); pr=p*aL+(1-p)*aN; ob=np.mean([o['acc'] for o in v])
    pred.append(pr); obs.append(ob)
pred=np.array(pred); obs=np.array(obs)
r2=1-np.sum((obs-pred)**2)/np.sum((obs-obs.mean())**2)
print(f"Test A (observed lock rate -> accuracy), {len(obs)} configs: R^2 {r2:.3f}, mean |err| {np.mean(abs(obs-pred))*100:.2f} pp")
# ---- Test B: seeding model fitted on GATE runs only, applied OUT OF SAMPLE to offline arms
k_g=np.array([o['n_dom_correct'] for o in gate]); l_g=np.array([o['lock'] for o in gate],float)
p0=np.mean(l_g[k_g==0])
def nll(rho):
    p=1-(1-p0)*(1-rho)**k_g; p=np.clip(p,1e-6,1-1e-6)
    return -np.sum(l_g*np.log(p)+(1-l_g)*np.log(1-p))
rho=minimize_scalar(nll,bounds=(1e-4,0.999),method='bounded').x
print(f"\nseeding model fitted on gate runs only: P(lock|k) = 1-(1-{p0:.3f})(1-{rho:.3f})^k")
for k in range(0,7): print(f"   k={k}: model {100*(1-(1-p0)*(1-rho)**k):5.1f}%")
# held-out (pre-run) dominant-hit rates q_E from the offline arms' own pool
q={'small':0.20e-3,'base':1.90e-3,'large':1.50e-3}
print("\nTest B -- offline warm starts, predicted from held-out q_E and tau alone (Poisson k), vs observed:")
print(f"{'arm':9}{'expert':7}{'kappa=tau*q':>12}{'P_lock pred':>12}{'obs':>6}{'acc pred':>10}{'acc obs':>9}")
for k,v in sorted(cfgs(off).items()):
    tag,c,e=k; tau=int(tag[3:]); kap=tau*q[e]
    ks=np.arange(0,60); from scipy.stats import poisson; pk=poisson.pmf(ks,kap)
    pl=np.sum(pk*(1-(1-p0)*(1-rho)**ks))
    ob=np.mean([o['acc'] for o in v]); pr=pl*aL+(1-pl)*aN
    print(f"{tag:9}{e:7}{kap:>12.2f}{pl*100:>11.0f}%{np.mean([o['lock'] for o in v])*100:>5.0f}%{pr*100:>9.2f}%{ob*100:>8.2f}%")
