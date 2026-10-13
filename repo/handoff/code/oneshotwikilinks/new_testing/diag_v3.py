"""Quick diagnostic for v3 numerical stability."""
import gzip, pickle, torch, numpy as np
from experiment_algorithm1_v3 import MyDataset

with gzip.open('mydataset.2000.v3.pickle.gz', 'rb') as f:
    ds = pickle.load(f)

Xs = ds.Xs
print(f'Xs shape: {Xs.shape}, dtype: {Xs.dtype}')
print(f'Xs has NaN: {torch.isnan(Xs).any().item()}')
print(f'Xs has Inf: {torch.isinf(Xs).any().item()}')
print(f'Xs min: {Xs.min().item():.6f}, max: {Xs.max().item():.6f}')
norms = torch.norm(Xs, dim=1)
print(f'Xs norm per sample: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}')

# Half norms (pre-text part)
half = Xs.shape[1] // 2
pre_norms = torch.norm(Xs[:, :half], dim=1)
print(f'Pre-text half norms: min={pre_norms.min():.4f}, max={pre_norms.max():.4f}, mean={pre_norms.mean():.4f}')

# Entity embeddings
d = {}
for k in ds.labelfeats:
    id_, embd = ds.labelfeats[k]
    d[id_] = torch.FloatTensor(embd)
entity_embd = torch.stack([d[k] for k in sorted(d)])
print(f'\nEntity embd shape: {entity_embd.shape}, dtype: {entity_embd.dtype}')
print(f'Entity has NaN: {torch.isnan(entity_embd).any().item()}')
print(f'Entity has Inf: {torch.isinf(entity_embd).any().item()}')
ent_norms = torch.norm(entity_embd, dim=1)
print(f'Entity norms: min={ent_norms.min():.4f}, max={ent_norms.max():.4f}')

# Feature products
action_features = entity_embd.numpy()
cx = Xs[0].numpy()[:half]
joint = action_features * cx
feat_norms = np.linalg.norm(joint, axis=1)
print(f'\nFeature product (sample 0), dtype={joint.dtype}:')
print(f'  norms: min={feat_norms.min():.6f}, max={feat_norms.max():.6f}, mean={feat_norms.mean():.6f}')

# SM stability test: 128 updates
lam = 0.1
dd = half
V_inv = (1.0/lam) * np.eye(dd)
print(f'\nV_inv init: dtype={V_inv.dtype}, diag={V_inv[0,0]}')

for i in range(128):
    cx_i = Xs[i].numpy()[:half]
    joint_i = action_features * cx_i
    x_i = joint_i[0]  # always action 0
    Vx_i = V_inv @ x_i
    denom_i = 1.0 + x_i @ Vx_i
    V_inv = V_inv - np.outer(Vx_i, Vx_i) / denom_i
    if np.isnan(V_inv).any() or np.isinf(V_inv).any():
        print(f'  BLEW UP at step {i}! denom={denom_i}')
        print(f'  x_i dtype={x_i.dtype}, V_inv dtype={V_inv.dtype}')
        print(f'  ||x_i||={np.linalg.norm(x_i):.6f}')
        break
    if i % 20 == 0:
        print(f'  step {i}: V_inv range=[{V_inv.min():.4e}, {V_inv.max():.4e}], denom={denom_i:.6f}')
else:
    print(f'128 updates OK. V_inv range=[{V_inv.min():.4e}, {V_inv.max():.4e}]')
