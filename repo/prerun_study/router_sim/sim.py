"""Router (confidence-gated meta-algorithm) on the instance of Theorem thm:transfer-separation (v8.tex).

Instance: d = T+2, actions D = e1, R = e2, fresh arm b_t = L e_{t+2} at round t. Hidden type
y_t in {D, R, F, 0}; reward 1 iff the action matches the type (F -> fresh arm). LinUCB is diagonal,
so its state is the scalar counts (n_D, s_D, n_R, s_R) and every fresh arm has index
omega = beta*L/sqrt(lam) (ties favour the fresh arm), exactly as in transfer_thm/stronger.py.

Policies (all share one type sequence per seed, i.e. common random numbers):
  cold                        LinUCB only
  hybrid_{A,B}                expert on rounds t < tau = 4k, LinUCB after (H_tau)
  hybrid_{A,B}_noreward       same, but routed rounds update n (covariance) and not s (response sum)
  router_{A,B}                Router: LinUCB proposes a0 with width w0 = ||x_a0||_{V^-1}; expert
                              proposes ae with width ||x_ae||_2/sqrt(c); route iff expert width < w0
  router_{A,B}_noreward       Router whose routed rounds update n but not s
Every round (routed or not) updates LinUCB with the played action and reward.

Designs:
  scripted   first 4k types = theorem window (k D, k R, 2k 0, shuffled), i.i.d. afterwards; routers are
             forced to the expert for t < tau and gated afterwards; hybrids use the window.
  adaptive   i.i.d. types from round 1; routers gated from round 1; hybrids still hand off at tau=4k.

The vectorised simulator `simulate` steps over t with state arrays of shape [n_configs, n_seeds].
`scalar_run` is a direct per-seed loop in the style of stronger.py; `python3 sim.py --check`
asserts that both give identical action sequences and identical per-run metrics.
"""
import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

D, R, F, N = 0, 1, 2, 3            # action / type codes (N = type '0': nothing pays)
NAMES = 'DRF0'
# EXPERT[e, y] = action of expert e (0=A, 1=B) on type y
EXPERT = np.array([[D, D, D, R],    # A: D on D/R/F, R on 0
                   [R, R, F, D]],   # B: R on D/R, fresh on F, D on 0
                  dtype=np.int8)
TIE = 1e-12                         # same tie tolerance as stronger.py


@dataclass(frozen=True)
class Config:
    policy: str          # full name, e.g. 'router_A_noreward'
    kind: str            # 'cold' | 'hybrid' | 'router'
    expert: int          # -1 (cold), 0 (A), 1 (B)
    noreward: bool
    c: float             # router expert-confidence parameter (nan for cold/hybrid)


def make_configs(c_values, noreward=True):
    cfgs = [Config('cold', 'cold', -1, False, float('nan'))]
    for e, en in ((0, 'A'), (1, 'B')):
        cfgs.append(Config(f'hybrid_{en}', 'hybrid', e, False, float('nan')))
        if noreward:
            cfgs.append(Config(f'hybrid_{en}_noreward', 'hybrid', e, True, float('nan')))
    for c in c_values:
        for e, en in ((0, 'A'), (1, 'B')):
            cfgs.append(Config(f'router_{en}', 'router', e, False, float(c)))
            if noreward:
                cfgs.append(Config(f'router_{en}_noreward', 'router', e, True, float(c)))
    return cfgs


def gen_types(rng, T, k, S, p, q, f, design):
    """int8 array [T, S] of hidden types."""
    u = rng.random((T, S))
    ty = np.where(u < p, D, np.where(u < p + q, R, np.where(u < p + q + f, F, N))).astype(np.int8)
    if design == 'scripted':
        win = np.array([D] * k + [R] * k + [N] * (2 * k), dtype=np.int8)
        ty[:4 * k] = rng.permuted(np.tile(win[:, None], (1, S)), axis=0)
    elif design != 'adaptive':
        raise ValueError(design)
    return np.ascontiguousarray(ty)


def _consts(p, q, f, omega, lam, L):
    beta = omega * np.sqrt(lam) / L
    w = beta * L / np.sqrt(lam)       # recomputed exactly as stronger.py does
    return beta, w


METRICS = ['regret', 'reward', 'routed', 'last_routed', 'routed_after_tau', 'settled_half',
           'settled_tau', 'fracD_half', 'nD', 'nR', 'routed_right', 'routed_right_D',
           'routed_nD', 'routed_nR', 'routed_nF', 'routed_regret']


def simulate(types, configs, p, q, f, omega, lam, L, k, design, record_actions=False):
    """Vectorised over configs x seeds. types: int8 [T, S]. Returns dict metric -> array [C, S]
    (plus 'actions' int8 [T, C, S] if record_actions)."""
    T, S = types.shape
    C = len(configs)
    tau = 4 * k
    beta, w = _consts(p, q, f, omega, lam, L)
    n = np.arange(T + 1)
    bt = beta / np.sqrt(lam + n)      # exploration bonus by count (same float ops as scalar code)
    wt = 1 / np.sqrt(lam + n)         # LinUCB width of D/R by count
    wF = L / np.sqrt(lam)             # LinUCB width of an untried fresh arm

    kind = np.array([c.kind for c in configs])
    is_router = (kind == 'router')[:, None]
    is_hybrid = (kind == 'hybrid')[:, None]
    exp_idx = np.array([max(c.expert, 0) for c in configs])[:, None]
    noreward = np.array([c.noreward for c in configs])[:, None]
    cc = np.array([c.c if c.kind == 'router' else 1.0 for c in configs])
    ewD = (1 / np.sqrt(cc))[:, None]
    ewF = (L / np.sqrt(cc))[:, None]
    scripted = design == 'scripted'
    # per-phase routing masks [C,1]
    forced_win = is_hybrid | (is_router & scripted)          # t < tau
    gate_win = is_router & (not scripted)                    # t < tau
    gate_late = is_router                                    # t >= tau
    no_force = np.zeros_like(forced_win)

    z = lambda dt=np.int32: np.zeros((C, S), dtype=dt)
    nD, nR = z(), z()
    sD, sR = z(np.float64), z(np.float64)
    reward, routed, r_after, dtau, dhalf = z(), z(), z(), z(), z()
    last = np.full((C, S), -1, dtype=np.int32)
    rr, rrD, rnD, rnR = z(), z(), z(), z()
    acts = np.empty((T, C, S), dtype=np.int8) if record_actions else None
    half0 = T // 2

    for t in range(T):
        y = types[t]                                         # [S]
        bD, bR = bt[nD], bt[nR]
        uD = sD / (lam + nD) + bD
        uR = sR / (lam + nR) + bR
        isF = w >= np.maximum(uD, uR) - TIE
        DgeR = uD >= uR
        a0 = np.where(isF, F, np.where(DgeR, D, R)).astype(np.int8)
        if t < tau:
            forced, gate = forced_win, gate_win
        else:
            forced, gate = no_force, gate_late
        if gate.any():
            w0 = np.where(isF, wF, np.where(DgeR, wt[nD], wt[nR]))
            ae = EXPERT[exp_idx, y[None, :]]
            ew = np.where(ae == F, ewF, ewD)
            route = forced | (gate & (ew < w0))
        else:
            ae = EXPERT[exp_idx, y[None, :]]
            route = np.broadcast_to(forced, (C, S))
        a = np.where(route, ae, a0)
        if record_actions:
            acts[t] = a
        r = a == y
        isD = a == D
        isR = a == R
        nD += isD
        nR += isR
        upd = r & ~(route & noreward)
        sD += upd & isD
        sR += upd & isR
        reward += r
        routed += route
        last[route] = t
        rr += route & r
        rrD += route & r & isD
        rnD += route & isD
        rnR += route & isR
        if t >= tau:
            r_after += route
            dtau += isD
        if t >= half0:
            dhalf += isD

    nF = T - nD - nR
    rnF = routed - rnD - rnR
    out = dict(
        regret=nR * (p - q) + nF * (p - f),
        reward=reward, routed=routed, last_routed=last, routed_after_tau=r_after,
        settled_half=(dhalf == T - half0), settled_tau=(dtau == T - tau),
        fracD_half=dhalf / (T - half0), nD=nD, nR=nR,
        routed_right=rr, routed_right_D=rrD, routed_nD=rnD, routed_nR=rnR, routed_nF=rnF,
        routed_regret=rnR * (p - q) + rnF * (p - f))
    if record_actions:
        out['actions'] = acts
    return out


# ----------------------------------------------------------------------------------------------
# direct scalar reference (mirrors transfer_thm/stronger.py and the router probe line by line)
EXP_STR = {'A': {'D': 'D', 'R': 'D', 'F': 'D', '0': 'R'}, 'B': {'D': 'R', 'R': 'R', 'F': 'F', '0': 'D'}}


def scalar_run(types_str, cfg, p, q, f, omega, lam, L, k, design):
    beta = omega * np.sqrt(lam) / L
    w = beta * L / np.sqrt(lam)
    tau = 4 * k
    T = len(types_str)
    nD = nR = 0
    sD = sR = 0.0
    m = dict(regret=0.0, reward=0, routed=0, last_routed=-1, routed_after_tau=0, routed_right=0,
             routed_right_D=0, routed_nD=0, routed_nR=0, routed_nF=0, routed_regret=0.0)
    dtau = dhalf = 0
    acts = []
    en = 'AB'[cfg.expert] if cfg.expert >= 0 else None
    for t in range(T):
        y = types_str[t]
        uD = sD / (lam + nD) + beta / np.sqrt(lam + nD)
        uR = sR / (lam + nR) + beta / np.sqrt(lam + nR)
        mx = max(uD, uR, w)
        a0 = 'F' if w >= mx - 1e-12 else ('D' if uD >= uR else 'R')
        wid0 = {'F': L / np.sqrt(lam), 'D': 1 / np.sqrt(lam + nD), 'R': 1 / np.sqrt(lam + nR)}[a0]
        if cfg.kind == 'cold':
            route = False
        elif cfg.kind == 'hybrid':
            route = t < tau
        else:
            ae = EXP_STR[en][y]
            wide = (L if ae == 'F' else 1) / np.sqrt(cfg.c)
            route = (design == 'scripted' and t < tau) or wide < wid0
        a = EXP_STR[en][y] if route else a0
        r = 1.0 if a == y else 0.0
        rr = 0.0 if (route and cfg.noreward) else r
        if a == 'D':
            nD += 1; sD += rr
        elif a == 'R':
            nR += 1; sR += rr
        g = {'D': 0.0, 'R': p - q, 'F': p - f}[a]
        m['regret'] += g
        m['reward'] += int(r)
        if route:
            m['routed'] += 1; m['last_routed'] = t; m['routed_right'] += int(r)
            m['routed_right_D'] += int(r and a == 'D'); m['routed_regret'] += g
            m['routed_n' + a] += 1
            if t >= tau:
                m['routed_after_tau'] += 1
        if t >= tau and a == 'D':
            dtau += 1
        if t >= T // 2 and a == 'D':
            dhalf += 1
        acts.append(a)
    m.update(settled_half=dhalf == T - T // 2, settled_tau=dtau == T - tau,
             fracD_half=dhalf / (T - T // 2), nD=nD, nR=nR)
    return acts, m


def check(n_seeds=4, T=300, verbose=True):
    """Assert vectorised == scalar (actions and metrics) on small runs over all settings/designs."""
    settings = [(0.3, 0.05, 0.0, 0.135, 0.1, 2.0), (0.5, 0.1, 0.0, 0.225, 0.1, 2.0),
                (0.2, 0.0, 0.0, 0.09, 0.1, 2.0), (0.3, 0.1, 0.25, 0.135, 1.0, 2.0),
                (0.3, 0.05, 0.0, 2.0, 0.1, 2.0), (0.3, 0.1, 0.25, 2.0, 1.0, 2.0)]
    n_runs = 0
    t0 = time.time()
    for si, (p, q, f, om, lam, L) in enumerate(settings):
        cvals = [lam / L ** 2 * 0.5, 0.05, 1.0, 10.0, 100.0, 1000.0]
        cfgs = make_configs(cvals)
        for design in ('scripted', 'adaptive'):
            for k in (1, 2, 4):
                rng = np.random.default_rng([7, si, k, design == 'scripted'])
                ty = gen_types(rng, T, k, n_seeds, p, q, f, design)
                out = simulate(ty, cfgs, p, q, f, om, lam, L, k, design, record_actions=True)
                for ci, cfg in enumerate(cfgs):
                    for s in range(n_seeds):
                        tstr = [NAMES[v] for v in ty[:, s]]
                        acts, m = scalar_run(tstr, cfg, p, q, f, om, lam, L, k, design)
                        va = ''.join(NAMES[v] for v in out['actions'][:, ci, s])
                        sa = ''.join(acts)
                        if va != sa:
                            i = next(j for j in range(T) if va[j] != sa[j])
                            raise AssertionError(f'action mismatch {cfg} setting {si} {design} k={k} seed {s} at t={i}: '
                                                 f'vec {va[i:i+10]} scalar {sa[i:i+10]}')
                        for key in METRICS:
                            a, b = float(out[key][ci, s]), float(m[key])
                            if abs(a - b) > 1e-9 * max(1.0, abs(b)):
                                raise AssertionError(f'metric {key} mismatch {cfg} setting {si} {design} k={k} seed {s}: {a} vs {b}')
                        n_runs += 1
    if verbose:
        print(f'CHECK OK: {n_runs} runs (T={T}), vectorised == scalar on every action and metric '
              f'[{time.time() - t0:.1f}s]')
    return n_runs


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='vectorised vs scalar equality check')
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--T', type=int, default=300)
    ap.add_argument('--bench', action='store_true', help='time one cell')
    a = ap.parse_args()
    if a.check:
        check(a.seeds, a.T)
    if a.bench:
        cfgs = make_configs([0.025, 0.05, 1, 10, 100, 1000])
        for T, S in ((2000, 500), (20000, 200)):
            ty = gen_types(np.random.default_rng(0), T, 2, S, 0.3, 0.05, 0.0, 'scripted')
            t0 = time.time()
            simulate(ty, cfgs, 0.3, 0.05, 0.0, 0.135, 0.1, 2.0, 2, 'scripted')
            print(f'T={T} seeds={S} configs={len(cfgs)}: {time.time() - t0:.2f}s')
    if not (a.check or a.bench):
        ap.print_help()
        sys.exit(1)
