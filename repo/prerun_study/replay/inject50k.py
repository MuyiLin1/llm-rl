"""Injection test on the 50,000-round evaluation stream (seeds as in the main results).
Same construction as the 10k held-out test (build_placebo.py / reh_v1_int.npz), scaled to keep the
same share of the stream: 75 changed answers out of 50,000 (15 out of 10,000 before), i.e. +0.15 points
of expert accuracy. Conditions (Flan-T5-small, c=100):
  seen   : 75 wrong answers -> 1471 on contexts that contain 1471
  hidden : same answers; on a routed changed context the learner gets covariance, reward 0 in the response sum
  rand   : 75 correct answers on a random true label other than 1471 (same contexts where possible)
Baseline and cold come from the replay runs (same simulator, same seeds)."""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import hide as H
from sim import Data
SLIM = '/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_v1_slim.npz'
N_ON, N_INJ = 50000, 75
def build(D):
    sm = D.llm['small']; rng = np.random.default_rng(12345)
    wrong = [j for j in range(N_ON) if sm[j] not in D.ys[j]]
    c1471 = [j for j in wrong if 1471 in D.ys[j]]
    ch = sorted(rng.choice(c1471, N_INJ, replace=False).tolist())
    seen = sm.copy(); seen[ch] = 1471
    rand = sm.copy(); used = []
    for j in ch:
        other = sorted(D.ys[j] - {1471})
        if other: rand[j] = rng.choice(other); used.append(j)
    cand = [j for j in wrong if j not in set(ch) and len(D.ys[j] - {1471}) > 0]
    extra = rng.choice(cand, N_INJ - len(used), replace=False)
    for j in extra: rand[j] = rng.choice(sorted(D.ys[j] - {1471}))
    return seen, rand, set(ch)
if __name__ == '__main__':
    W, I = int(sys.argv[1]), int(sys.argv[2]); seeds = range(int(sys.argv[3]), int(sys.argv[4]) + 1)
    D = Data(SLIM); seen, rand, chg = build(D); H.T = 50000
    acc = lambda p: np.mean([p[j] in D.ys[j] for j in range(N_ON)]) * 100
    if I == 0: print(f"expert acc on stream: small {acc(D.llm['small']):.2f}%  seen {acc(seen):.2f}%  rand {acc(rand):.2f}%", flush=True)
    jobs = [(c, s) for s in seeds for c in ('seen', 'hidden', 'rand')][I::W]
    for c, s in jobs:
        out = f"{HERE}/out/J50k_{c}_s{s}.npz"
        if os.path.exists(out): continue
        np.random.seed(s); order = np.random.permutation(N_ON)
        llm = rand if c == 'rand' else seen; hs = chg if c == 'hidden' else set()
        t0 = time.time(); o = H.run_hide(D, order, llm, hs)
        o['injected_routed'] = np.array(int(sum(1 for t in range(H.T) if o['routed'][t] and int(order[t]) in chg)))
        tmp = out + '.tmp.npz'; np.savez(tmp, **o); os.replace(tmp, out)
        print(f"[j{I}] {c} s{s}: acc {o['r_router'].mean()*100:.2f}%  injected routed {int(o['injected_routed'])}  {time.time()-t0:.0f}s", flush=True)
    print(f"[j{I}] DONE", flush=True)
