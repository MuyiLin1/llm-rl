"""Two catalog claims the shipped suite does not simulate.
Row 4  (prop:cat-mimic): expert plays b_t always, no probes.
  (i) tau <= K*m0 -> Hyb and Cold identical, A=V=0.
  (ii) tau > K*m0 -> R(Hyb) >= tau, R(Cold) <= K(4beta^2+1), ratio <= (16lam+1)/m0 <= 961.
Row 8 (cor:cat-dear): eps >= c^2/L^2 -> R(Cold)/R(Hyb) <= 2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from template_sim import LinUCBState, simulate

lam = 60.0; beta = 2*np.sqrt(lam); L, c, sigma = 3.0, 1.0, 0.5
m0 = int(lam//30)

def run_row4(K, T, tau, seed):
    rng = np.random.default_rng(seed)
    st = LinUCBState(K, lam, L, c, 1.0, beta)
    reg = 0.0; actions = []
    for t in range(1, tau+1):
        j = (t-1) % K
        st.play('b', j, 0.0 + sigma*rng.standard_normal()); reg += 1.0; actions.append('b')
    for t in range(tau+1, T+1):
        j = (t-1) % K
        Ug, Ub, _ = st.indices(j)
        arm = 'b' if Ub > Ug else 'g'
        actions.append(arm)
        if arm == 'g': st.play('g', j, 1.0 + sigma*rng.standard_normal())
        else: st.play('b', j, 0.0 + sigma*rng.standard_normal()); reg += 1.0
    return reg, actions

K, T = 32, 200_000
cold, acts_cold = run_row4(K, T, 0, seed=9)
cap = K*(4*beta**2+1)
print(f"Row 4: m0={m0}, K*m0={K*m0}, cap K(4b^2+1)={cap:.0f}, bound (16lam+1)/m0={(16*lam+1)/m0:.0f} (paper: <=961)")
print(f"  Cold regret {cold:.0f}  <= cap? {cold <= cap}")
for tau in [K*m0, K*m0+1, 5*K*m0, 20000]:
    r, acts = run_row4(K, T, tau, seed=9)
    same = acts[:len(acts_cold)] == acts_cold
    ratio = cold/r if r > 0 else float('nan')
    tag = "(i) tau<=K*m0" if tau <= K*m0 else "(ii) tau>K*m0"
    print(f"  tau={tau:6d} {tag:14s} R(Hyb)={r:8.0f} >=tau? {r>=tau}  "
          f"identical-to-cold? {same}  ratio={ratio:.3f} <=961? {ratio<=961}")

print(f"\nRow 8 (cor:cat-dear): threshold eps >= c^2/L^2 = {c**2/L**2:.4f}")
K8, n8, Ttail = 32, 19440, 100_000; T8 = K8*n8 + Ttail
for eps in [0.05, 0.111112, 0.15, 0.3]:
    cd = simulate('cold', K8, n8, T8, L, c, eps, lam, beta, sigma, seed=11, m0=m0)
    hb = simulate('hybrid', K8, n8, T8, L, c, eps, lam, beta, sigma, seed=11)
    ratio = cd['regret']/hb['regret']; dear = eps >= c**2/L**2
    print(f"  eps={eps:<9} ratio={ratio:5.3f}  dear? {dear}  claim(ratio<=2 when dear): "
          f"{'OK' if (not dear or ratio <= 2) else 'VIOLATED'}")
