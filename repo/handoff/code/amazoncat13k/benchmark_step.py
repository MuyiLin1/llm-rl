"""Quick benchmark: how fast is one LinUCB step with 13K actions x 768d?"""
import time, numpy as np
np.random.seed(0)

K, d = 13330, 768
Z = np.random.randn(K, d)
V_inv = (1.0/0.1) * np.eye(d, dtype=np.float64)
b = np.zeros(d, dtype=np.float64)
x = np.random.randn(d).astype(np.float64)

# Warm up
Cx = (V_inv * x[:, None]) * x[None, :]
_ = Z @ Cx

t0 = time.time()
for i in range(5):
    theta = V_inv @ b
    weighted = x * theta
    mu = Z @ weighted                           # O(K*d)
    Cx = (V_inv * x[:, None]) * x[None, :]      # O(d^2)
    ZCx = Z @ Cx                                 # O(K*d^2) — bottleneck
    var = np.sum(ZCx * Z, axis=1)               # O(K*d)
    ucb = mu + 1.0 * np.sqrt(np.maximum(var, 0))
    choice = int(np.argmax(ucb))
t1 = time.time()

per_step = (t1 - t0) / 5
print(f"Per-step (1 LinUCB pass, K={K}, d={d}): {per_step:.3f}s")
print(f"For cold+warm (x2): {per_step*2:.3f}s")
print(f"50K samples: {per_step*2*50000/3600:.1f} hours")
print(f"200K samples: {per_step*2*200000/3600:.1f} hours")
print(f"1.18M samples: {per_step*2*1186239/3600:.1f} hours")
