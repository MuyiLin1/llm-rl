import numpy as np
z=np.load('/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_v1_slim.npz')
fl,of=z['y_flat'],z['y_offsets']; K=z['Z'].shape[0]
ys=[fl[of[i]:of[i+1]] for i in range(50000)]
# indicator matrix row lookup: for context i, reward of arm a = a in ys[i]
print(f"always-play-1471 on the online stream: {np.mean([1471 in set(y.tolist()) for y in ys])*100:.2f}%")
def ucb1(seed, T=50000):
    np.random.seed(seed); perm=np.random.permutation(50000)
    n=np.zeros(K); s=np.zeros(K); acc=0.0
    for t in range(T):
        y=set(ys[perm[t]].tolist())
        if t<K: a=t          # one pull per arm first (standard UCB1 initialisation)
        else: a=int(np.argmax(s/n+np.sqrt(2*np.log(t+1)/n)))
        r=1.0 if a in y else 0.0; n[a]+=1; s[a]+=r; acc+=r
    return acc/T
def greedy_counts(seed,T=50000):   # pure empirical-mean greedy with optimistic prior 1/1
    np.random.seed(seed); perm=np.random.permutation(50000)
    n=np.ones(K); s=np.ones(K); acc=0.0
    for t in range(T):
        y=set(ys[perm[t]].tolist()); a=int(np.argmax(s/n)); r=1.0 if a in y else 0.0
        n[a]+=1; s[a]+=r; acc+=r
    return acc/T
u=[ucb1(s) for s in (1,2,3)]; g=[greedy_counts(s) for s in (1,2,3)]
print(f"context-free UCB1 (13,330 arms), seeds 1-3: {[round(x*100,2) for x in u]}  mean {np.mean(u)*100:.2f}%")
print(f"context-free optimistic greedy,  seeds 1-3: {[round(x*100,2) for x in g]}  mean {np.mean(g)*100:.2f}%")
