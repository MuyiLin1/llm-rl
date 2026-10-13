"""Test 4 read-out: the same AmazonCat+BERT experiment under three conditions.
  full        routed rounds add (V, b) with the observed binary reward  [the Router]
  nom         routed rounds add V, and b gets <theta_or, phi> -- the expert-independent
              part M theta_or only; the transfer m is removed (Test 4)
  passive     routed rounds add V only; b untouched (the earlier covariance-only test)
Prediction (Prop transfer): 'nom' collapses the three experts together, like
'passive', because everything expert-dependent in the response sum is m.
"""
import json, os, statistics as st, numpy as np
L = 'study'
conds = [('full', 'amazonv1', '_pfull'), ('nom', 'nom', '_pfull_nom'), ('passive', 'passiveonly', '')]
print(f"{'expert':7} {'condition':10} {'n':>2} {'Router':>14} {'Cold':>8} {'mult':>7} {'V_tau':>15}")
for m in ('small', 'base', 'large'):
    for cname, tag, suf in conds:
        R = []; C = []; V = []
        for s in range(1, 11):
            p = f"{L}/results_algo1_{tag}_v1_{m}_c100.0{suf}_s{s}.json"
            q = f"{L}/perround_algo1_{tag}_v1_{m}_c100.0{suf}_s{s}.npz"
            if not (os.path.exists(p) and os.path.exists(q)): continue
            h = json.load(open(p)); R.append(h['hybrid'][-1]); C.append(h['linucb_cold'][-1])
            z = np.load(q); n = int(z['step']); rt = z['routed'][:n].astype(bool)
            V.append(float((z['r_router'][:n][~rt] - z['r_cold'][:n][~rt]).sum()))
        if not R: print(f"{m:7} {cname:10}  (no runs yet)"); continue
        print(f"{m:7} {cname:10} {len(R):>2} {st.mean(R)*100:>7.2f}+-{(st.stdev(R)*100 if len(R)>1 else 0):<5.2f} "
              f"{st.mean(C)*100:>7.2f}% {st.mean(R)/st.mean(C):>6.2f}x {st.mean(V):>8.0f}+-{(st.stdev(V) if len(V)>1 else 0):<5.0f}")
