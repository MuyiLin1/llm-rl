"""Show progress of the WikiLinks c=10000 runs.  Usage:  python3 status.py"""
import os, json, glob, subprocess, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
jobs = [(e, s) for s in range(1, 11) for e in ('small', 'base', 'large')]
ps = subprocess.run(['ps', '-eo', 'command'], capture_output=True, text=True).stdout.splitlines()
alive = {l.split()[-1] for l in ps if 'wiki_c10000/run.py 3 ' in l and 'caffeinate' not in l}
print(f"workers running: {len(alive)}   (0 means stopped: restart with  bash {H}/start.sh)\n")
done = 0
for e, s in jobs:
    f = f"{H}/out/{e}_s{s}.npz"; p = f"{H}/logs/{e}_s{s}.progress"
    if os.path.exists(f):
        z = np.load(f); done += 1
        print(f"{e:5s} seed {s:2d}: DONE     router {z['r_router'].mean()*100:5.2f}%" + (f"  cold {z['r_cold'].mean()*100:5.2f}%" if 'r_cold' in z.files else ''))
    elif os.path.exists(p):
        d = json.load(open(p))
        print(f"{e:5s} seed {s:2d}: {100*d['t']/d['T']:5.1f}%   router so far {d['router_acc']*100:5.2f}%   ~{d['eta_hours']:.1f} h left   (updated {d['updated']})")
    else:
        print(f"{e:5s} seed {s:2d}: waiting")
print(f"\n{done}/30 runs finished")
