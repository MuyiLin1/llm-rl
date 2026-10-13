"""Exact simulation of the paper's template instance I(K, L, c, eps, tau0=K*n)
using the arrowhead closed form (validated separately), so LinUCB costs O(1)/round.

Checks Theorem [thm:price]:
  (i)   expert regret = eps*K*n + (T-K*n)
  (ii)  cold plays b whenever m_{j(t)} < m0 within probe window (forced pulls >= K*m0)
  (iii) cold total regret <= K(8 beta^2 + 1) + eps K n
  (iv)  hybrid plays g at every t > tau0  ->  R(Hyb) = eps K n
  ratio R(Cold)/R(Hyb) vs price ratio c^2/(eps L^2)
Tests the conjecture: R_{[Kn+1,T]}(Cold) =? Theta(K beta^2)   (V_tau0 magnitude)
"""
import numpy as np
from dataclasses import dataclass, field

@dataclass
class LinUCBState:
    K: int; lam: float; L: float; c: float; alpha: float; beta: float
    s: int = 0
    Rg: float = 0.0
    m: np.ndarray = None
    n: np.ndarray = None
    Rb: np.ndarray = None
    Rp: np.ndarray = None
    SQ: float = 0.0      # sum_j n_j (lam + m_j L^2)/D_j
    W: float = 0.0       # sum_j u_j b_j / D_j
    sumRp: float = 0.0

    def __post_init__(self):
        Z = np.zeros(self.K)
        self.m, self.n, self.Rb, self.Rp = Z.copy(), Z.copy(), Z.copy(), Z.copy()

    def Dj(self, j):
        return self.lam + self.m[j] * self.L**2 + self.n[j] * self.c**2

    def uj(self, j):
        return self.alpha * self.c * self.n[j]

    def bj(self, j):
        return self.L * self.Rb[j] + self.c * self.Rp[j]

    def _contribs(self, j):
        D = self.Dj(j)
        return (self.n[j] * (self.lam + self.m[j] * self.L**2) / D,
                self.uj(j) * self.bj(j) / D)

    def indices(self, j):
        """Return (Ug, Ub, Up) for direction j at current state."""
        lam, L, c, a, beta = self.lam, self.L, self.c, self.alpha, self.beta
        D = self.Dj(j); u = self.uj(j); bj = self.bj(j)
        Q = lam + self.s + a**2 * self.SQ
        b1 = self.Rg + a * self.sumRp
        th1 = (b1 - self.W) / Q
        thj = bj / D - (u / (D * Q)) * (b1 - self.W)
        wg = 1.0 / np.sqrt(Q)
        vjj = 1.0 / D + u**2 / (D**2 * Q)
        wb = L * np.sqrt(vjj)
        wp2 = a**2 / Q - 2 * a * c * u / (D * Q) + c**2 * vjj
        wp = np.sqrt(max(wp2, 0.0))
        Ug = th1 + beta * wg
        Ub = L * thj + beta * wb
        Up = a * th1 + c * thj + beta * wp
        return Ug, Ub, Up

    def play(self, arm, j, reward):
        cSQ, cW = self._contribs(j)
        if arm == 'g':
            self.s += 1; self.Rg += reward
        elif arm == 'b':
            self.m[j] += 1; self.Rb[j] += reward
        else:
            self.n[j] += 1; self.Rp[j] += reward; self.sumRp += reward
        nSQ, nW = self._contribs(j)
        self.SQ += nSQ - cSQ
        self.W += nW - cW


def simulate(policy, K, nprobe, T, L, c, eps, lam, beta, sigma, seed,
             m0=None, track=False):
    """policy in {'cold','hybrid','expert'}; tau0 = K*nprobe.
    Returns dict of results."""
    rng = np.random.default_rng(seed)
    a = 1.0 - eps
    tau0 = K * nprobe
    st = LinUCBState(K, lam, L, c, a, beta)
    reg_head = 0.0; reg_tail = 0.0
    bad_head = np.zeros(K); bad_tail = np.zeros(K); probe_cold = 0
    forced_ok = True   # (ii): whenever m_j < m0 in probe window, cold plays b
    tail_g_violations = 0

    if policy in ('hybrid', 'expert') :
        # expert phase in closed form: probes each direction nprobe times
        for j in range(K):
            cSQ, cW = st._contribs(j)
            st.n[j] = nprobe
            noise = sigma * np.sqrt(nprobe) * rng.standard_normal()
            st.Rp[j] = nprobe * (1 - eps) + noise
            st.sumRp += st.Rp[j]
            nSQ, nW = st._contribs(j)
            st.SQ += nSQ - cSQ; st.W += nW - cW
        reg_head = eps * tau0
        if policy == 'expert':
            return dict(regret=reg_head + (T - tau0), head=reg_head, tail=T - tau0)
        t0 = tau0 + 1
    else:
        t0 = 1

    for t in range(t0, T + 1):
        j = (t - 1) % K
        probe_offered = (t <= tau0)
        Ug, Ub, Up = st.indices(j)
        if probe_offered:
            best = max(Ug, Ub, Up)
            arm = 'g' if Ug == best else ('b' if Ub == best else 'p')
        else:
            arm = 'b' if Ub > Ug else 'g'
        if policy == 'cold' and probe_offered and m0 is not None:
            if st.m[j] < m0 and arm != 'b':
                forced_ok = False
        if arm == 'g':
            r = 1.0 + sigma * rng.standard_normal(); dr = 0.0
        elif arm == 'b':
            r = 0.0 + sigma * rng.standard_normal(); dr = 1.0
            if t <= tau0: bad_head[j] += 1
            else: bad_tail[j] += 1
        else:
            r = (1 - eps) + sigma * rng.standard_normal(); dr = eps
            probe_cold += 1
        if t <= tau0: reg_head += dr
        else:
            reg_tail += dr
            if policy == 'hybrid' and arm != 'g':
                tail_g_violations += 1
        st.play(arm, j, r)

    return dict(regret=reg_head + reg_tail, head=reg_head, tail=reg_tail,
                bad_head=bad_head, bad_tail=bad_tail, probe_pulls=probe_cold,
                forced_ok=forced_ok, tail_g_violations=tail_g_violations)


if __name__ == '__main__':
    lam = 60.0; beta = 2 * np.sqrt(lam)          # paper's convention
    L, c, sigma = 3.0, 1.0, 0.5
    K, nprobe = 32, 19440                        # nc^2 large; actual (not bound) width cert holds
    tau0 = K * nprobe
    Ttail = 200_000
    T = tau0 + Ttail
    m0 = int(lam // 30)
    print(f"lam={lam} beta={beta:.2f} L={L} c={c} K={K} n={nprobe} tau0={tau0} T={T} m0={m0}")
    print(f"price ratio c^2/(eps L^2) at eps=0.002: {c**2/(0.002*L**2):.1f}")

    # handoff width certificate (actual, from arrowhead closed form)
    a = 1 - 0.002
    D = lam + nprobe * c**2
    Q = lam + a**2 * K * nprobe * lam / D
    vjj = 1 / D + (a * c * nprobe)**2 / (D**2 * Q)
    print(f"handoff: 2*beta*L*||e_j|| = {2*beta*L*np.sqrt(vjj):.3f}  (<1 -> hybrid certified to play g)")

    for eps in [0.002]:
        cold = simulate('cold', K, nprobe, T, L, c, eps, lam, beta, sigma, seed=7, m0=m0)
        hyb  = simulate('hybrid', K, nprobe, T, L, c, eps, lam, beta, sigma, seed=7)
        exp  = simulate('expert', K, nprobe, T, L, c, eps, lam, beta, sigma, seed=7)
        print(f"\n=== eps={eps} ===")
        print(f"Expert regret {exp['regret']:.0f} (claim (i): {eps*tau0 + (T-tau0):.0f})")
        print(f"Cold  regret {cold['regret']:.1f}  head {cold['head']:.1f} tail {cold['tail']:.1f}; "
              f"upper bound (iii) K(8b^2+1)+eps K n = {K*(8*beta**2+1)+eps*K*nprobe:.0f}")
        print(f"Cold forced-pull claim (ii) held: {cold['forced_ok']}; "
              f"bad pulls/direction head: mean {cold['bad_head'].mean():.1f} "
              f"min {cold['bad_head'].min():.0f} max {cold['bad_head'].max():.0f}; "
              f"tail: mean {cold['bad_tail'].mean():.2f}; probe pulls {cold['probe_pulls']}")
        print(f"Hybrid regret {hyb['regret']:.1f} (claim (iv): eps K n = {eps*tau0:.0f}); "
              f"tail non-g plays: {hyb['tail_g_violations']}")
        print(f"Ratio Cold/Hyb = {cold['regret']/hyb['regret']:.2f}; "
              f"proven bracket [{m0/(eps*nprobe):.3f}, {1+(8*beta**2+1)/(eps*nprobe):.1f}]; "
              f"price ratio {c**2/(eps*L**2):.1f}")
        print(f"Conjecture check: cold TAIL regret {cold['tail']:.1f} vs K*beta^2 = {K*beta**2:.0f} "
              f"-> V_tau0 = tail(Cold) - tail(Hyb) = {cold['tail']-hyb['tail'] if 'tail' in hyb else float('nan'):.1f}")
        print(f"A_tau0 = head(Cold) - head(Exp) = {cold['head'] - eps*tau0:.1f}; "
              f"B_tau0 = tail(Exp) - tail(Hyb) = {(T-tau0) - hyb['tail']:.1f}")
