"""Recompute the Appendix I basin-mixture R^2 with the old own-routing 'nom' ablation replaced by the
replay's fixed-routed-rounds 'transfer removed' runs (prerun_study/replay/out). Same records and same
leave-config-out fit as basin_remote.py / basin_fit.py. Writes basins_online_replay.json (a new file)."""
import json, collections, numpy as np, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PS = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), 'out')
z = np.load('/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_v1_slim.npz')
fl, of = z['y_flat'], z['y_offsets']; n = 50000
prev = collections.Counter(l for i in range(n) for l in set(fl[of[i]:of[i+1]].tolist())); prev = {k: v/n for k, v in prev.items()}
rank = {k: i+1 for i, (k, _) in enumerate(sorted(prev.items(), key=lambda x: -x[1]))}
def rec(a, r):
    a = a[40000:50000]; top, cnt = collections.Counter(a.tolist()).most_common(1)[0]
    return dict(top=int(top), share=cnt/len(a), rank=rank.get(int(top), 9999), prev=prev.get(int(top), 0),
                late=float(r[40000:50000].mean()), acc=float(r.mean()))
cold = {s: rec(np.load(f'{OUT}/small_s{s}.npz')['a_cold'], np.load(f'{OUT}/small_s{s}.npz')['r_cold']) for s in range(1, 11)}
B = [x for x in json.load(open(os.path.join(PS, 'basins_online.json'))) if x['tag'] != 'nom']
for e in ('small', 'base', 'large'):
    for s in range(1, 11):
        o = np.load(f'{OUT}/{e}_s{s}.npz')
        B.append(dict(tag='nom', fam='v1', e=e, c=100.0, router=rec(o['a_nom'], o['r_nom']), s=s, cold=cold[s],
                      routed=int(o['routed'].sum())))
json.dump(B, open(os.path.join(PS, 'basins_online_replay.json'), 'w')); print('records', len(B))
