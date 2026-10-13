"""Fresh rehearsal stream for AmazonCat BERT: contexts 60,000-109,999, which no
online run, sweep, baseline or held-out analysis has ever used."""
import gzip, pickle, sys, numpy as np, torch
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,*a,**k): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self,i): return i
sys.modules['__main__'].MyDataset=MyDataset
LO,HI=60000,110000
src='/Users/linmuyi/code/DMwithLLM/amazoncat13k/'
with gzip.open(src+'mydataset.0.pickle.gz','rb') as f: ds=pickle.load(f)
zd={}
for k in ds.labelfeats:
    i,e=ds.labelfeats[k]; zd[i]=np.asarray(e,dtype=np.float32)
Z=np.nan_to_num(np.stack([zd[i] for i in sorted(zd)]),nan=0.0,posinf=0.0,neginf=0.0)
Xs=ds.Xs.numpy() if isinstance(ds.Xs,torch.Tensor) else ds.Xs
d=Xs.shape[1]//2; print('total contexts',Xs.shape[0],flush=True)
C=np.nan_to_num(Xs[LO:HI,:d].astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
ys=ds.ys[LO:HI]; ys=ys.numpy() if hasattr(ys,'numpy') else ys
flat=np.concatenate([np.asarray(y,dtype=np.int32) for y in ys]); offs=np.cumsum([0]+[len(y) for y in ys]).astype(np.int64)
# sanity: Z must equal the slim file's Z so the feature space is identical
Zs=np.load('/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_v1_slim.npz')['Z']
assert np.array_equal(Z,Zs), 'Z mismatch'
out=dict(contexts=C,Z=Z,y_flat=flat,y_offsets=offs,n_keep=np.int64(HI-LO))
for e in ('small','base','large'):
    with gzip.open(src+f'llm_cache_amazon_v1_{e}.pkl.gz','rb') as f: lab=np.asarray(pickle.load(f)['labels'])
    assert len(lab)>=HI, (e,len(lab)); out[f'llm_{e}']=lab[LO:HI].astype(np.int64)
    acc=np.mean([out[f'llm_{e}'][i] in set(flat[offs[i]:offs[i+1]].tolist()) for i in range(HI-LO)])
    dom=np.mean([(out[f'llm_{e}'][i]==1471) and (1471 in set(flat[offs[i]:offs[i+1]].tolist())) for i in range(HI-LO)])
    print(f'{e}: cache len {len(lab)}, fresh-block acc {acc*100:.2f}%, dominant-hit rate {dom*1e3:.2f} per 1000',flush=True)
np.savez(sys.argv[1],**out); print('saved',sys.argv[1],C.shape,flush=True)
