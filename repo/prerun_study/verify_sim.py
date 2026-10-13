import numpy as np, time, sys
sys.path.insert(0, __file__.rsplit('/',1)[0]); from sim import Data, run, DEV
D = Data('/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_v1_slim.npz')
L = np.load(__file__.rsplit('/',1)[0]+'/online_large_s1.npz')
np.random.seed(1); perm = np.random.permutation(50000)
T = 3000; t0 = time.time()
o = run(D, perm, T, llm=D.llm['large'], c=100.0)
dt = time.time()-t0
print(f"device {DEV}: {T} rounds in {dt:.1f}s ({dt/T*1000:.1f} ms/round, both tracks)")
for k in ('a_cold','a_router','routed'):
    ref = L[k][:T]; agree = np.mean(o[k]==ref)
    first = np.argmax(o[k]!=ref) if agree<1 else T
    print(f"  {k:9} agreement with the online run: {agree*100:6.2f}%   first divergence at round {first}")
print(f"  cold acc sim {o['r_cold'].mean()*100:.2f}% vs online {L['r_cold'][:T].mean()*100:.2f}%; "
      f"router acc sim {o['r_router'].mean()*100:.2f}% vs online {L['r_router'][:T].mean()*100:.2f}%; "
      f"routed sim {o['routed'].sum()} vs online {int(L['routed'][:T].sum())}")
