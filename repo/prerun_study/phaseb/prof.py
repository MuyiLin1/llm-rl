import time, numpy as np, torch
d = 768; V = np.eye(d)/0.1; b = np.random.randn(d); x = np.random.randn(d)*0.05; cx = np.random.randn(d)
n = 300
t = time.time()
for _ in range(n):
    th = V @ b; q = x @ V @ x; Vx = V @ x; den = 1 + x @ Vx; V = V - np.outer(Vx, Vx)/den
print(f'host float64 per learner-round: {(time.time()-t)/n*1e3:.2f} ms')
dev = torch.device('cuda'); Zt = torch.randn(13330, d, device=dev); Vg = torch.eye(d, dtype=torch.float64, device=dev); cxg = torch.as_tensor(cx, device=dev)
for _ in range(5): (Zt @ ((Vg*cxg[:,None])*cxg[None,:]).float()).sum().item()
torch.cuda.synchronize(); t = time.time()
for _ in range(n):
    Cx = ((Vg*cxg[:,None])*cxg[None,:]).float(); var = ((Zt @ Cx) * Zt).sum(1); i = torch.argmax(var)
torch.cuda.synchronize(); print(f'GPU scoring per learner-round (no per-round sync): {(time.time()-t)/n*1e3:.2f} ms')
t = time.time()
for _ in range(n):
    Cx = ((Vg*cxg[:,None])*cxg[None,:]).float(); i = torch.argmax(((Zt @ Cx) * Zt).sum(1)).item()
print(f'GPU scoring with sync each: {(time.time()-t)/n*1e3:.2f} ms')
t = time.time()
for _ in range(n):
    Cx = ((Vg*cxg[:,None])*cxg[None,:]).float()
torch.cuda.synchronize(); print(f'  of which fp64 Cx build: {(time.time()-t)/n*1e3:.2f} ms')
t = time.time()
for _ in range(n):
    Vxg = torch.as_tensor(np.random.randn(d)); Vg = Vg - torch.outer(Vxg.to(dev), Vxg.to(dev))/2.0
torch.cuda.synchronize(); print(f'  fp64 mirror update: {(time.time()-t)/n*1e3:.2f} ms')
