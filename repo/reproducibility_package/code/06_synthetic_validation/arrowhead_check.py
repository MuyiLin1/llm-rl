"""Validate Lemma [Arrowhead inverse] against dense numpy inverse.

V = lam*I + S e1 e1^T + sum_j (M_j e_j e_j^T + n_j q_j q_j^T),  q_j = alpha e1 + c e_j.
Claimed entries:
  D_j = lam + M_j + n_j c^2,  u_j = alpha c n_j,
  Q = lam + S + alpha^2 sum_j n_j (lam + M_j)/D_j
  (V^-1)_11 = 1/Q, (V^-1)_1j = -u_j/(D_j Q),
  (V^-1)_jj = 1/D_j + u_j^2/(D_j^2 Q), (V^-1)_jj' = u_j u_j'/(D_j D_j' Q).
"""
import numpy as np

rng = np.random.default_rng(0)
K = 6
d = K + 1
alpha, c, lam = 0.93, 0.7, 3.5

for trial in range(200):
    S = rng.uniform(0, 50)
    M = rng.uniform(0, 40, size=K) * rng.integers(0, 2, size=K)  # some zero
    n = rng.integers(0, 20, size=K).astype(float)
    V = lam * np.eye(d)
    V[0, 0] += S
    for j in range(K):
        V[j + 1, j + 1] += M[j]
        q = np.zeros(d); q[0] = alpha; q[j + 1] = c
        V += n[j] * np.outer(q, q)
    Vinv = np.linalg.inv(V)
    D = lam + M + n * c**2
    u = alpha * c * n
    Q = lam + S + (alpha**2) * np.sum(n * (lam + M) / D)
    # claimed inverse
    W = np.zeros((d, d))
    W[0, 0] = 1 / Q
    for j in range(K):
        W[0, j + 1] = W[j + 1, 0] = -u[j] / (D[j] * Q)
        for jp in range(K):
            if j == jp:
                W[j + 1, j + 1] = 1 / D[j] + u[j]**2 / (D[j]**2 * Q)
            else:
                W[j + 1, jp + 1] = u[j] * u[jp] / (D[j] * D[jp] * Q)
    err = np.max(np.abs(W - Vinv))
    assert err < 1e-10, (trial, err)
print("Arrowhead inverse lemma: PASSED 200 random trials (max entrywise err < 1e-10)")
