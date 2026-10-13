"""(1) eps sweep at K=32: is ratio-1 proportional to 1/eps (price ratio scaling)?
(2) one run at certified K=324 (where 2*beta*L*||e_j|| < 1 provably at handoff).
(3) row-7 nuisance-probe variant: hybrid should be trapped after handoff.
"""
import numpy as np
from template_sim import LinUCBState, simulate

lam = 60.0; beta = 2*np.sqrt(lam); L, c, sigma = 3.0, 1.0, 0.5

print("=== (1) eps sweep, K=32, n=19440, tail 100k ===")
K, nprobe, Ttail = 32, 19440, 100_000
tau0 = K*nprobe; T = tau0 + Ttail; m0 = int(lam//30)
print(f"{'eps':>8} {'R(Cold)':>9} {'R(Hyb)':>9} {'ratio':>7} {'(ratio-1)*eps':>13} {'bracket_hi':>10} {'priceratio':>10} {'tailCold':>8} {'hybG?':>5}")
for eps in [0.0005, 0.002, 0.008, 0.032]:
    cold = simulate('cold', K, nprobe, T, L, c, eps, lam, beta, sigma, seed=11, m0=m0)
    hyb  = simulate('hybrid', K, nprobe, T, L, c, eps, lam, beta, sigma, seed=11)
    ratio = cold['regret']/hyb['regret']
    print(f"{eps:>8} {cold['regret']:>9.0f} {hyb['regret']:>9.0f} {ratio:>7.2f} "
          f"{(ratio-1)*eps:>13.5f} {1+(8*beta**2+1)/(eps*nprobe):>10.1f} "
          f"{c**2/(eps*L**2):>10.1f} {cold['tail']:>8.1f} {hyb['tail_g_violations']==0!s:>5}")

print("\n=== (2) certified K=324 (K >= 36 L^2), eps=0.002, tail 50k ===")
K2, Ttail2 = 324, 50_000
tau2 = K2*nprobe; T2 = tau2 + Ttail2
a = 1-0.002; D = lam + nprobe*c**2
Q = lam + a**2*K2*nprobe*lam/D
vjj = 1/D + (a*c*nprobe)**2/(D**2*Q)
print(f"handoff cert: 2*beta*L*||e_j|| = {2*beta*L*np.sqrt(vjj):.3f} (must be < 1)")
cold = simulate('cold', K2, nprobe, T2, L, c, 0.002, lam, beta, sigma, seed=3, m0=m0)
hyb  = simulate('hybrid', K2, nprobe, T2, L, c, 0.002, lam, beta, sigma, seed=3)
print(f"Cold regret {cold['regret']:.0f} (head {cold['head']:.0f}, tail {cold['tail']:.1f}); "
      f"Hyb regret {hyb['regret']:.0f} (claim eps*K*n = {0.002*tau2:.0f}); tail non-g: {hyb['tail_g_violations']}")
print(f"ratio {cold['regret']/hyb['regret']:.2f}; V_tau0 = {cold['tail']-hyb['tail']:.1f}; "
      f"A_tau0 = {cold['head']-0.002*tau2:.0f}; K*beta^2 = {K2*beta**2:.0f}")

print("\n=== (3) row 7: nuisance probes (probe couples e1 with direction K+j) ===")
# reuse LinUCBState with 2K directions; bad dirs 0..K-1 (b only), nuisance K..2K-1 (p only)
def simulate_row7(policy, K, nprobe, T, L, c, eps, lam, beta, sigma, seed):
    rng = np.random.default_rng(seed)
    a = 1.0 - eps
    tau0 = K*nprobe
    st = LinUCBState(2*K, lam, L, c, a, beta)
    reg_head = reg_tail = 0.0; bad_tail = np.zeros(K); viol = 0
    if policy == 'hybrid':
        for j in range(K, 2*K):
            cSQ, cW = st._contribs(j)
            st.n[j] = nprobe
            st.Rp[j] = nprobe*(1-eps) + sigma*np.sqrt(nprobe)*rng.standard_normal()
            st.sumRp += st.Rp[j]
            nSQ, nW = st._contribs(j)
            st.SQ += nSQ - cSQ; st.W += nW - cW
        reg_head = eps*tau0; t0 = tau0+1
    else:
        t0 = 1
    for t in range(t0, T+1):
        jb = (t-1) % K; jp = K + jb
        probe_offered = (t <= tau0)
        # indices: Ug and Ub at direction jb; Up at direction jp
        Ug, Ub, _ = st.indices(jb)
        if probe_offered:
            _, _, Up = st.indices(jp)
            best = max(Ug, Ub, Up)
            arm, jdir = ('g', jb) if Ug == best else (('b', jb) if Ub == best else ('p', jp))
        else:
            arm, jdir = ('b', jb) if Ub > Ug else ('g', jb)
        if arm == 'g': r = 1.0 + sigma*rng.standard_normal(); dr = 0.0
        elif arm == 'b': r = sigma*rng.standard_normal(); dr = 1.0
        else: r = (1-eps) + sigma*rng.standard_normal(); dr = eps
        if t <= tau0: reg_head += dr
        else:
            reg_tail += dr
            if arm == 'b': bad_tail[jb] += 1
        st.play(arm, jdir, r)
    return dict(regret=reg_head+reg_tail, head=reg_head, tail=reg_tail, bad_tail=bad_tail)

K7, n7, Ttail7 = 32, 19440, 100_000
tau7 = K7*n7; T7 = tau7 + Ttail7
cold7 = simulate_row7('cold', K7, n7, T7, L, c, 0.002, lam, beta, sigma, seed=5)
hyb7  = simulate_row7('hybrid', K7, n7, T7, L, c, 0.002, lam, beta, sigma, seed=5)
print(f"Cold regret {cold7['regret']:.0f} (tail {cold7['tail']:.1f}); "
      f"Hybrid regret {hyb7['regret']:.0f} (head {hyb7['head']:.0f}, tail {hyb7['tail']:.0f})")
print(f"Hybrid tail bad pulls/direction: mean {hyb7['bad_tail'].mean():.1f} "
      f"(claim: trapped, >= floor(lam/2)={int(lam//2)} forced per direction at paper constants)")
print(f"ratio Cold/Hyb = {cold7['regret']/hyb7['regret']:.2f}  (paper bound for row 7: <= 65)")
