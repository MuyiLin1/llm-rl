"""Evaluate the paper's pre-runtime net-value rule
   N(tau) = (T-tau)[Phi(lam I) - Phi(lam I + (tau/n_held) M_E)] - C_E(tau)
on the template instance, where the true multiplier is known from theory+simulation.

Phi(V) = E_context max_a min{Delta_a, 2 beta ||x_a||_{V^{-1}}}   (exact on template)
Row 5 expert: plays probes  -> M_E covers e_j via q_j = a e1 + c e_j
Row 4 expert: plays bad arms -> M_E = L^2 e_j e_j^T mass
Row 7 expert: nuisance probes -> M_E orthogonal to bad directions
"""
import numpy as np

lam = 60.0; beta = 2*np.sqrt(lam); L, c = 3.0, 1.0

def phi_cold():
    # competitors: b (gap 1, width L/sqrt(lam)), p (gap eps) -> max of mins
    return min(1.0, 2*beta*L/np.sqrt(lam))

def width_ej_row5(K, n_per_dir, eps):
    a = 1-eps
    D = lam + n_per_dir*c**2
    Q = lam + a**2*K*n_per_dir*lam/D
    return np.sqrt(1/D + (a*c*n_per_dir)**2/(D**2*Q))

def width_ej_row4(m_per_dir):
    return np.sqrt(1.0/(lam + m_per_dir*L**2))

for K, tailT, label in [(32, 100_000, "K=32 (multiplies 3.2x in sim)"),
                        (324, 50_000, "K=324 (certified; multiplies 3.4x in sim)")]:
    n = 19440; eps = 0.002; tau = K*n
    print(f"--- {label}: n={n} tau={tau} tail={tailT} ---")
    # Row 5: expert Gram = probes, tau/n_held * M_E == true handoff Gram
    w5 = width_ej_row5(K, n, eps)
    phi5 = max(min(1.0, 2*beta*L*w5), eps)   # b clipped by min(gap, width term); probe contributes eps while offered
    dphi = phi_cold() - phi5
    C_E = eps*tau
    N = tailT*dphi - C_E
    print(f"row5: 2bL||e_j||={2*beta*L*w5:.3f}  Phi_cold={phi_cold():.3f} Phi_warm={phi5:.3f} "
          f"dPhi={dphi:.3f}  C_E={C_E:.0f}  N={N:.0f}  N/(tail*Phi_cold)={N/(tailT*phi_cold()):.3f}")
    # Row 4 expert (plays b): match acquisition tau4 s.t. same coverage mass on e_j: m=tau4/K pulls
    for tau4 in [K*80, tau]:
        m = tau4/K
        w4 = width_ej_row4(m)
        phi4 = max(min(1.0, 2*beta*L*w4), eps)
        N4 = tailT*(phi_cold()-phi4) - 1.0*tau4      # expert error rate 1 -> C = tau
        print(f"row4 tau={tau4}: dPhi={phi_cold()-phi4:.3f} C_E={tau4} N={N4:.0f}")
    # Row 7: nuisance -> widths on bad dirs unchanged
    print(f"row7: dPhi=0.000 C_E={eps*tau:.0f} N={-eps*tau:.0f}")
    print()

print("Ground truth: row5 multiplies (3.2-3.4x), rows 4 and 7 do not.")
print("Paper's rule ('multiplier only if N large positive fraction, N>0 necessary'):")
print("  -> row5 at K=32 has dPhi=0, N<0, yet multiplies: the 'necessary' claim FAILS.")
print("  -> row5 at K=324 has N<0 for tail 50k, marginally >0 only for very long tails.")
