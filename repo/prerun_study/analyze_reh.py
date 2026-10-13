"""Compare pre-run rehearsals with the online runs at a matched horizon.
lock-on := share of the dominant label (1471) among the Router's plays in the
last 20% of the horizon exceeds 25% (same definition for rehearsal and online)."""
import numpy as np, json, os, glob, sys
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__))
on=json.load(open(HERE+'/online10k.json'))
def reh(ph, mode, e, T):
    rows=[]
    for p in sorted(glob.glob(f"{HERE}/reh_out/{ph}_{mode}{'_'+e if e else ''}_s*.npz")):
        o=np.load(p); a=o['a_router'] if mode=='router' else o['a_cold']; r=o['r_router'] if mode=='router' else o['r_cold']
        w=a[int(0.8*T):T]
        rows.append(dict(lock=np.mean(w==1471)>0.25, acc=r[:T].mean(), routed=int(o['routed'][:T].sum()) if mode=='router' else 0))
    return rows
def summary(ph, T, fam):
    cold=reh(ph,'cold',None,T)
    if not cold: print(f"phase {ph}: no cold runs yet"); return
    ca=np.mean([x['acc'] for x in cold])
    print(f"\n=== phase {ph} (horizon {T}) : rehearsal cold acc {ca*100:.2f}% over {len(cold)} seeds, cold lock {np.mean([x['lock'] for x in cold])*100:.0f}%")
    print(f"{'expert':7}{'n':>3}{'routed':>8}{'P_lock reh':>11}{'P_lock online':>14}{'acc reh':>9}{'acc online':>11}{'mult reh':>10}{'mult online':>12}")
    for e in ('small','base','large'):
        R=reh(ph,'router',e,T)
        if not R: continue
        key=f"{fam}_{e}"; ON=on.get(key,[])
        if T==10000: lo=np.mean([x['lock10'] for x in ON]); ao=np.mean([x['acc10'] for x in ON]); co=np.mean([x['cold10'] for x in ON])
        else:        lo=np.mean([x['lock50'] for x in ON]); ao=np.mean([x['acc50'] for x in ON]); co=np.mean([x['cold50'] for x in ON])
        ar=np.mean([x['acc'] for x in R])
        print(f"{e:7}{len(R):>3}{np.mean([x['routed'] for x in R]):>8.0f}{np.mean([x['lock'] for x in R])*100:>10.0f}%{lo*100:>13.0f}%"
              f"{ar*100:>8.2f}%{ao*100:>10.2f}%{ar/ca:>9.2f}x{ao/co:>11.2f}x")
summary('A',10000,'amazonv1'); summary('C',10000,'amazonv3'); summary('B',50000,'amazonv1')
