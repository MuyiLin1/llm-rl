import sys, os, time, numpy as np, torch, pb_batch
torch.backends.cuda.matmul.allow_tf32 = os.environ.get("TF32") == "1"
fam, K = sys.argv[1], int(sys.argv[2]); D = pb_batch.Data(f'amazon_{fam}_slim.npz'); T = 600
specs = [dict(kind='cold', sched='fast', amin=a, lam=0.1, seed=11) for a in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0][:K]]
t0 = time.time(); pb_batch.run_batch(D, 'prod', 11, specs, T=T); print(f'{K*T/(time.time()-t0):.0f}', flush=True)
