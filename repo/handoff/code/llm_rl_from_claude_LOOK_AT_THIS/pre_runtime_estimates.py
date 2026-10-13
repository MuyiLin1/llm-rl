"""Pre-runtime estimates for the paper (fills the TODOs in Table `diagnostics`).

Inputs: an .npz with
  X          : float array [n_held, A, d]  joint features phi(x_s, a) for every candidate a
               (for WikiLinks A=311; for AmazonCat use the top-A candidate set actually scored)
  y          : int   array [n_held]        index of the correct action in axis 1 of X
  expert     : int   array [n_held]        index of the expert's (LLM's) proposal
Optional:
  mask       : bool  array [n_held, A]     valid-candidate mask (default: all valid)

Outputs, for each requested tau:
  Phi_cold          = mean_s max_a min{Delta_hat_{s,a}, 2 beta ||x_{s,a}||_{(lam I)^{-1}}}
  Phi_warm(tau)     = same with V = lam I + (tau/n) M_E,  M_E = sum_s x_{s,expert_s} x^T
  C_E(tau)          = tau * held-out expert error rate          (binary reward)
  N(tau)            = (T - tau) [Phi_cold - Phi_warm] - C_E(tau)          (paper's rule)
  dry-run           : simulate cold LinUCB and warm-started LinUCB on contexts resampled
                      from the held-out set with rewards = labels, horizon T_dry, and report
                      R_cold, R_warm_tail, and the implied multiplier estimate
                      R_cold / (C_E(tau_scaled) + R_warm_tail).   This covers BOTH channels
                      (A_tau and V_tau) of the decomposition, which N(tau) does not.

Usage: python pre_runtime_estimates.py data.npz --T 50000 --taus 5000 --lam 0.1 --beta 2.0
"""
import argparse
import numpy as np


def widths_sq(X, Vinv):
    # X: [n, A, d] -> ||x||^2_{Vinv} = sum over d of (X @ Vinv) * X
    return np.einsum('nad,de,nae->na', X, Vinv, X)


def phi_hat(X, y, beta, Vinv, mask=None):
    n, A, d = X.shape
    Delta = np.ones((n, A))
    Delta[np.arange(n), y] = 0.0                      # binary reward: gap 1 off the label
    w = 2 * beta * np.sqrt(np.maximum(widths_sq(X, Vinv), 0))
    term = np.minimum(Delta, w)
    if mask is not None:
        term = np.where(mask, term, -np.inf)
    return term.max(axis=1).mean()


def dry_run(X, y, lam, beta_fn, T, V0=None, rng=None, batch=None):
    """Simulate LinUCB on contexts resampled iid from the held-out set,
    rewards = 1{a == y}.  Returns cumulative pseudo-regret and final V."""
    n, A, d = X.shape
    rng = rng or np.random.default_rng(0)
    V = (lam * np.eye(d)) if V0 is None else V0.copy()
    Vinv = np.linalg.inv(V)
    b = np.zeros(d)
    reg = 0.0
    order = rng.integers(0, n, size=T)
    for t, s in enumerate(order, start=1):
        Xs = X[s]                                     # [A, d]
        theta = Vinv @ b
        wid = np.sqrt(np.maximum(np.einsum('ad,de,ae->a', Xs, Vinv, Xs), 0))
        a = int(np.argmax(Xs @ theta + beta_fn(t) * wid))
        r = 1.0 if a == y[s] else 0.0
        reg += 1.0 - r
        x = Xs[a]
        Vx = Vinv @ x
        Vinv -= np.outer(Vx, Vx) / (1.0 + x @ Vx)     # Sherman--Morrison
        b += r * x
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--T', type=int, required=True, help='online horizon')
    ap.add_argument('--taus', type=int, nargs='+', required=True)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=2.0)
    ap.add_argument('--T_dry', type=int, default=20000,
                    help='dry-run horizon (scale results by T/T_dry to compare)')
    args = ap.parse_args()

    Z = np.load(args.npz)
    X, y, ex = Z['X'], Z['y'], Z['expert']
    mask = Z['mask'] if 'mask' in Z else None
    n, A, d = X.shape
    err = float(np.mean(ex != y))
    ME = np.einsum('nd,ne->de', X[np.arange(n), ex], X[np.arange(n), ex])

    Vinv_cold = np.eye(d) / args.lam
    phi_c = phi_hat(X, y, args.beta, Vinv_cold, mask)
    print(f"n_held={n}  A={A}  d={d}  expert err={err:.3f}  Phi_cold={phi_c:.4f}")

    for tau in args.taus:
        Vw = args.lam * np.eye(d) + (tau / n) * ME
        phi_w = phi_hat(X, y, args.beta, np.linalg.inv(Vw), mask)
        CE = tau * err
        N = (args.T - tau) * (phi_c - phi_w) - CE
        print(f"tau={tau}: Phi_warm={phi_w:.4f}  dPhi={phi_c-phi_w:.4f}  "
              f"C_E={CE:.0f}  N={N:.0f}  N/((T-tau)Phi_cold)={N/((args.T-tau)*phi_c):.3f}")

        # dry run (covers the A_tau channel too): scale tau and T down together
        scale = args.T_dry / args.T
        tau_s = max(1, int(tau * scale))
        rng = np.random.default_rng(1)
        R_cold = dry_run(X, y, args.lam, lambda t: args.beta, args.T_dry, rng=rng)
        R_warm_tail = dry_run(X, y, args.lam, lambda t: args.beta, args.T_dry - tau_s,
                              V0=args.lam * np.eye(d) + (tau_s / n) * ME,
                              rng=np.random.default_rng(2))
        mult = R_cold / (tau_s * err + R_warm_tail)
        print(f"  dry-run (T'={args.T_dry}): R_cold={R_cold:.0f}  "
              f"C_E'={tau_s*err:.0f}  R_warm_tail={R_warm_tail:.0f}  "
              f"implied regret multiplier ~ {mult:.2f}")


if __name__ == '__main__':
    main()
