# identity check: batched runner vs pb_sim.run (== sim.py on the default setting), on the cluster GPU
import sys, time, numpy as np, pb_sim, pb_batch
p = 'amazon_v1_slim.npz'; D1 = pb_sim.Data(p); DB = pb_batch.Data(p); T = 1500
specs = [dict(kind='cold', sched='fast', amin=0.1, lam=0.1, seed=3),
         dict(kind='router', expert='large', c=100.0, sched='fast', amin=0.1, lam=0.1, seed=3),
         dict(kind='cold', sched='const', amin=0.5, lam=1.0, seed=3),
         dict(kind='router', expert='base', c=10.0, sched='slow', amin=0.05, lam=0.1, seed=3)]
t0 = time.time(); A, R, RO = pb_batch.run_batch(DB, 'prod', 3, specs, T=T); tb = time.time() - t0
ok = True
for k, s in enumerate(specs):
    a, r, ro = pb_sim.run(D1, s, T=T); same = np.array_equal(a, A[k]); ok &= same
    print(k, s['kind'], s['sched'], s['amin'], 'identical' if same else 'DIFFERENT', int(ro.sum()), int(RO[k].sum()), flush=True)
print(f'batched K=4: {4*T/tb:.0f} learner-rounds/s', flush=True)
D2 = pb_sim.Data(p, 'prod+arm'); T = 500
s = dict(kind='router', expert='large', c=100.0, sched='fast', amin=0.1, lam=0.1, seed=3)
A, R, RO = pb_batch.run_batch(DB, 'prod+arm', 3, [s], T=T); a, r, ro = pb_sim.run(D2, s, T=T)
print('prod+arm', 'identical' if np.array_equal(a, A[0]) else 'DIFFERENT', flush=True)
