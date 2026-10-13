"""Full-horizon pre-run rehearsal (E) and ablation (F) vs the online runs and the old dry run.
All on the ACCURACY-multiplier scale: mean Router acc / mean cold acc (the paper's headline
definition, Table amazon-results); regret ratio printed alongside."""
import numpy as np, glob, json
T=50000
def ld(ph,mode,e=None):
    R={}
    for p in glob.glob(f"reh_out/{ph}_{mode}{'_'+e if e else ''}_s*.npz"):
        if p.endswith('.tmp.npz'): continue
        s=int(p.rsplit('_s',1)[1].split('.')[0]); o=np.load(p)
        if mode=='cold': R[s]=dict(acc=o['r_cold'].mean(), lock=np.mean(o['a_cold'][int(.8*T):]==1471)>0.25)
        elif ph=='F':
            tail=o['r_router']; R[s]=dict(acc=(float(o['acq_correct'])+tail.sum())/T, lock=np.mean(o['a_router'][-int(.2*T):]==1471)>0.25,
                                        k=int(o['acq_dom_correct']))
        else:
            a,r,rt=o['a_router'],o['r_router'],o['routed']
            R[s]=dict(acc=r.mean(), lock=np.mean(a[int(.8*T):]==1471)>0.25, k=int(((a==1471)&rt&(r>0)).sum()), routed=int(rt.sum()))
    return R
on=json.load(open('online10k.json'))
cold=ld('E','cold'); 
if not cold: raise SystemExit("no E cold yet")
ca=np.mean([v['acc'] for v in cold.values()])
oc=json.load(open('olddry/cold_s1.json'))
od_cold=np.mean([json.load(open(f'olddry/cold_s{s}.json'))['acc'] for s in range(1,11)])
print(f"cold acc: rehearsal E {ca*100:.2f}% (n={len(cold)}, lock {np.mean([v['lock'] for v in cold.values()])*100:.0f}%) | old dry run (beta=1) {od_cold*100:.2f}% | online s1-10 8.39%")
print(f"{'expert':7}{'n':>3} {'E mult':>7} {'E Plock':>8} {'E k':>5} | {'F mult':>7} {'F Plock':>8} | {'old dry':>8} | {'online s1-10':>12} {'online 20s':>10} {'P_lock50':>9}")
for e in ('small','base','large'):
    E=ld('E','router',e); F=ld('F','router',e)
    W=[json.load(open(f'olddry/warm_{e}_tau1600_s{s}.json')) for s in range(1,11)]
    od=np.mean([((w['tau']-w['C_E'])+w['rounds']*w['warm_tail_acc'])/50000 for w in W])/od_cold
    O=on['amazonv1_'+e]; O10=[x for x in O if x['s']<=10]
    m=lambda L: np.mean([x['acc50'] for x in L])/np.mean([x['cold50'] for x in L])
    em = np.mean([v['acc'] for v in E.values()])/ca if E else np.nan
    fm = np.mean([v['acc'] for v in F.values()])/ca if F else np.nan
    print(f"{e:7}{len(E):>3} {em:>6.2f}x {np.mean([v['lock'] for v in E.values()])*100 if E else np.nan:>7.0f}% {np.mean([v['k'] for v in E.values()]) if E else np.nan:>5.1f} |"
          f" {fm:>6.2f}x {np.mean([v['lock'] for v in F.values()])*100 if F else np.nan:>7.0f}% | {od:>7.2f}x | {m(O10):>11.2f}x {m(O):>9.2f}x {np.mean([x['lock50'] for x in O])*100:>8.0f}%")
