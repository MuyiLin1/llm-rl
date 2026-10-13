"""Placebo controls for the small_plus1471 intervention: give Flan-T5-small the same
number (15) of extra correct answers, but NOT on the dominant label 1471.
  plusrand : same 15 contexts where possible; the new answer is a uniformly random
             true label other than 1471 (2 contexts have no other label -> replaced by
             random contexts where small is wrong and a non-1471 label exists)
  plus7961 : 15 correct answers on the SECOND most common label (7961, 16.1%), on random
             contexts that contain 7961 and where small is wrong."""
import numpy as np
r=np.load('reh_v1.npz'); I=np.load('reh_v1_int.npz')
fl,of=r['y_flat'],r['y_offsets']; ys=[set(fl[of[i]:of[i+1]].tolist()) for i in range(len(of)-1)]
sm=r['llm_small']; ch=np.where(sm!=I['llm_small_plus1471'])[0]
rng=np.random.default_rng(12345)
wrong=[j for j in range(len(ys)) if sm[j] not in ys[j]]
A=sm.copy(); used=[]
for j in ch:
    other=sorted(ys[j]-{1471})
    if other: A[j]=rng.choice(other); used.append(j)
cand=[j for j in wrong if j not in set(ch) and len(ys[j]-{1471})>0]
for j in rng.choice(cand, 15-len(used), replace=False):
    A[j]=rng.choice(sorted(ys[j]-{1471})); used.append(int(j))
B=sm.copy(); c79=[j for j in wrong if 7961 in ys[j] and j not in set(ch)]
for j in rng.choice(c79, 15, replace=False): B[j]=7961
acc=lambda p: np.mean([p[j] in ys[j] for j in range(len(ys))])*100
print(f"small {acc(sm):.2f}%  plusrand {acc(A):.2f}%  plus7961 {acc(B):.2f}%  plus1471 {acc(I['llm_small_plus1471']):.2f}%")
print("plusrand new labels:", sorted(int(A[j]) for j in used)); print("changed:", (A!=sm).sum(), (B!=sm).sum())
print("any 1471 introduced?", ((A==1471)&(sm!=1471)).sum(), ((B==1471)&(sm!=1471)).sum())
np.savez('reh_v1_plc.npz', contexts=r['contexts'], Z=r['Z'], y_flat=fl, y_offsets=of, n_keep=r['n_keep'],
         llm_small_plusrand=A, llm_small_plus7961=B)
