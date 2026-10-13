"""Validate Prop [three-way] and Prop [one-pass] on the canonical instance.

Instance: theta*=e1, A_t = {g=e1, b_t = L e_{t+1}}, horizon n_h, d = n_h+1.
Everything is diagonal, so LinUCB reduces to scalar bookkeeping:
  state: s = pulls of g, sumXg = sum of rewards on g, and per-direction pulls of b (0 or more).
  Since bad arms never repeat, each b_t is fresh with width L/sqrt(lam) and estimate 0.
Cold: V diag; estimate on e1 from g pulls only; on e_{t+1} always 0 before round t.
Warm: V0 = lam I + m L^2 sum_{j>=2} e_j e_j^T.
"""
import numpy as np

def run_canonical(n_h, L, lam, beta, sigma, m_warm, expert_mode, tau, seed):
    """expert_mode: 'none' (pure LinUCB from round 1, cold or warm via m_warm),
    'bad' expert plays b_t for t<=tau, 'good' expert plays g for t<=tau.
    Returns pseudo-regret trajectory total."""
    rng = np.random.default_rng(seed)
    s = 0            # pulls of g
    sumXg = 0.0
    regret = 0.0
    for t in range(1, n_h + 1):
        # widths
        wg2 = 1.0 / (lam + s)
        wb2 = L**2 / (lam + m_warm * L**2)   # fresh direction: warm mass m_warm*L^2, online mass 0
        est_g = sumXg / (lam + s)
        est_b = 0.0
        if t <= tau and expert_mode == 'bad':
            a = 'b'
        elif t <= tau and expert_mode == 'good':
            a = 'g'
        else:
            Ug = est_g + beta * np.sqrt(wg2)
            Ub = est_b + beta * np.sqrt(wb2)
            a = 'b' if Ub > Ug else 'g'
        if a == 'g':
            s += 1
            sumXg += 1.0 + sigma * rng.standard_normal()
        else:
            regret += 1.0
    return regret

n_h, L, lam, sigma = 2000, 2.0, 1.0, 0.3
beta = sigma * np.sqrt(2 * np.log(2 * n_h / 0.05)) + np.sqrt(lam)  # standard-ish radius

# (2) cold LinUCB regret = n exactly
r_cold = run_canonical(n_h, L, lam, beta, sigma, m_warm=0, expert_mode='none', tau=0, seed=1)
print(f"Cold LinUCB regret: {r_cold:.0f}  (claim: exactly {n_h})")

# (3) warm-started with m per eq (m-condition): try m = ceil(lambda(L^2-1)/L^2)+ slack, and larger
for m in [1, 2, 5, 20]:
    rs = [run_canonical(n_h, L, lam, beta, sigma, m_warm=m, expert_mode='none', tau=0, seed=s) for s in range(10)]
    print(f"Warm m={m:3d}: regret over 10 seeds min/med/max = {np.min(rs):.0f}/{np.median(rs):.0f}/{np.max(rs):.0f}")

# one-pass: any expert + handoff at tau -> hybrid plays b for all t>tau
print("\n[one-pass] hybrid tail regret (claim: = n_h - tau for every expert/tau):")
for expert_mode in ['bad', 'good']:
    for tau in [0, 100, 1000]:
        rs = []
        for s in range(5):
            total = run_canonical(n_h, L, lam, beta, sigma, 0, expert_mode, tau, seed=100+s)
            head = tau if expert_mode == 'bad' else 0.0
            rs.append(total - head)   # tail regret
        print(f"  expert={expert_mode:4s} tau={tau:5d}: tail regret = {sorted(set(round(x) for x in rs))} (claim {n_h - tau})")
