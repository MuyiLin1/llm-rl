"""4d control: the +15-answers-on-1471 intervention with the reward HIDDEN on the injected answers.
Same data (reh_v1_int.npz), expert (small_plus1471), stream convention, T and seeds as phase D of
rehearse_v1.py. When the Router routes one of the 15 injected contexts, the learner records the
covariance of the injected answer but its response sum gets 0 instead of the (correct) reward, so
the injection adds coverage along 1471 but no transfer. Accuracy is still scored on the true reward."""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from sim import Data, sm_update
T, C_CONF, LAM, DECAY = 10000, 100.0, 0.1, 0.9985

def run_hide(D, order, llm, hide):
    d = D.d; Vr = np.eye(d)/LAM; br = np.zeros(d); isc = 1.0/np.sqrt(C_CONF)
    ar = np.zeros(T, np.int64); rr = np.zeros(T); ro = np.zeros(T, bool); hid = np.zeros(T, bool)
    for t in range(T):
        j = int(order[t]); cx = D.C[j].astype(np.float64); ys = D.ys[j]; alpha = max(0.1, DECAY**t)
        a0 = D.score(Vr, Vr@br, cx, alpha); x0 = cx*D.Z[a0]
        al = int(llm[j]); xl = cx*D.Z[al]
        use = bool(isc*np.linalg.norm(xl) < np.sqrt(max(0.0, x0@Vr@x0)))
        a, x = (al, xl) if use else (a0, x0)
        r = 1.0 if a in ys else 0.0
        rb = 0.0 if (use and j in hide) else r; hid[t] = use and j in hide
        Vr = sm_update(Vr, x); br += rb*x; ar[t] = a; rr[t] = r; ro[t] = use
    return dict(a_router=ar, r_router=rr, routed=ro, hidden=hid)

if __name__ == '__main__':
    W, I = int(sys.argv[1]), int(sys.argv[2])
    D = Data(os.path.dirname(HERE)+'/reh_v1_int.npz')
    base = np.load(os.path.dirname(HERE)+'/reh_v1.npz')['llm_small']
    llm = D.llm['small_plus1471']; hide = set(np.where(base != llm)[0].tolist())
    assert len(hide) == 15, len(hide)
    n = D.C.shape[0]
    for s in list(range(1, 21))[I::W]:
        out = f"{HERE}/out/H_small_plus1471_hide_s{s}.npz"
        if os.path.exists(out): continue
        np.random.seed(s); order = np.random.permutation(n)   # same convention as rehearse_v1.py
        t0 = time.time(); o = run_hide(D, order, llm, hide)
        tmp = out + '.tmp.npz'; np.savez(tmp, **o); os.replace(tmp, out)
        print(f"[h{I}] s{s}: acc {o['r_router'].mean()*100:.2f}%  hidden rounds {int(o['hidden'].sum())}  "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"[h{I}] DONE", flush=True)
