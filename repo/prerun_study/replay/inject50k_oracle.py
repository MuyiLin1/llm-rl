"""Part C of teammate-overlap-analysis.md: one more condition for tab:inject50k.
  oracle : same 75 changed answers as 'seen'/'hidden' (inject50k.build); on a routed changed context the
           learner's response sum gets <theta_or, phi(x_t, a_t)> instead of the observed reward (1) or 0 (hidden).
theta_or is theta_or_v1.npy, the oracle of the transfer-removed curve (replay.py 'nom').
Everything else is inject50k.py / hide.run_hide verbatim. Output out/J50k_oracle_s{s}.npz."""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from sim import Data, sm_update
from inject50k import build, SLIM, N_ON
T, C_CONF, LAM, DECAY = 50000, 100.0, 0.1, 0.9985

def run_oracle(D, order, llm, chg, th):
    d = D.d; Vr = np.eye(d)/LAM; br = np.zeros(d); isc = 1.0/np.sqrt(C_CONF)
    ar = np.zeros(T, np.int64); rr = np.zeros(T); ro = np.zeros(T, bool); orc = np.zeros(T, bool)
    for t in range(T):
        j = int(order[t]); cx = D.C[j].astype(np.float64); ys = D.ys[j]; alpha = max(0.1, DECAY**t)
        a0 = D.score(Vr, Vr@br, cx, alpha); x0 = cx*D.Z[a0]
        al = int(llm[j]); xl = cx*D.Z[al]
        use = bool(isc*np.linalg.norm(xl) < np.sqrt(max(0.0, x0@Vr@x0)))
        a, x = (al, xl) if use else (a0, x0)
        r = 1.0 if a in ys else 0.0
        o = use and j in chg; orc[t] = o
        rb = float(x @ th) if o else r
        Vr = sm_update(Vr, x); br += rb*x; ar[t] = a; rr[t] = r; ro[t] = use
    return dict(a_router=ar, r_router=rr, routed=ro, oracle=orc)

if __name__ == '__main__':
    W, I = int(sys.argv[1]), int(sys.argv[2]); seeds = range(int(sys.argv[3]), int(sys.argv[4]) + 1)
    D = Data(SLIM); seen, rand, chg = build(D); th = np.load(os.path.join(HERE, 'theta_or_v1.npy'))
    for s in list(seeds)[I::W]:
        out = f"{HERE}/out/J50k_oracle_s{s}.npz"
        if os.path.exists(out): continue
        np.random.seed(s); order = np.random.permutation(N_ON)
        t0 = time.time(); o = run_oracle(D, order, seen, chg, th)
        o['injected_routed'] = np.array(int(o['oracle'].sum()))
        tmp = out + '.tmp.npz'; np.savez(tmp, **o); os.replace(tmp, out)
        print(f"[o{I}] oracle s{s}: acc {o['r_router'].mean()*100:.2f}%  injected routed {int(o['injected_routed'])}  "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"[o{I}] DONE", flush=True)
