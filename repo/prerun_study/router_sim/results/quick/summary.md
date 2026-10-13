# Router on the transfer-separation instance

Source: `/Users/linmuyi/code/DMwithLLM/prerun_study/router_sim/results/quick/results.csv` (348 rows). Headline c = 100. Settings:

- s1: p=0.3 q=0.05 f=0 omega=0.135 lam=0.1 L=2
- fgt: p=0.3 q=0.1 f=0.25 omega=0.135 lam=1 L=2
- s1_bigR: p=0.3 q=0.05 f=0 omega=2 lam=0.1 L=2

Definitions: regret = pseudo-regret vs theta* (gaps D 0, R p-q, fresh p-f); reward = number of reward-1 rounds; "settle (half)" = plays D on every round of the second half; "settle (tau)" = plays D on every round after tau = 4k (the theorem's event for H_tau). Router widths: LinUCB 1/sqrt(lam+n) for D/R, L/sqrt(lam) for an untried fresh arm; expert 1/sqrt(c) (L/sqrt(c) for fresh). c=never is lam/(2L^2).

## 0. Sanity: c = never-route

Adaptive design: Router(never) must equal cold on every seed; scripted design: it must equal H_tau (same expert, same noreward flag). Passed 48 / 48 (cell, policy) pairs.

Theorem (ii)/(iii) on scripted cells: max over cells of mean LinUCB-chosen D plays (nD - routed D) for hybrid_B, hybrid_A_noreward, hybrid_B_noreward = 0 (theorem: 0).

Cold regret = (p-f)T in 8/8 small-radius cells.

## 1. Does Router-A settle at least as often as H_tau-A? (scripted)

T = 500. Router settle = all-D second half. "(<H)" marks Router-A settling less often than H_A (second-half definition) by more than 2 binomial s.e.

| setting | k | bound | H_A settle(tau) | H_A settle(half) | R_A c=never | R_A c=0.05 | R_A c=1 | R_A c=10 | R_A c=100 | R_A c=1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 1 | 0.612 | 0.900 | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 | 0.020 (<H) | 0.000 (<H) |
| s1 | 2 | 0.852 | 0.960 | 0.960 | 0.960 | 1.000 | 1.000 | 1.000 | 0.120 (<H) | 0.000 (<H) |
| fgt | 1 | 0.544 | 0.860 | 0.860 | 0.860 | 0.860 | 1.000 | 1.000 | 1.000 | 0.000 (<H) |
| fgt | 2 | 0.826 | 0.960 | 0.960 | 0.960 | 0.960 | 1.000 | 1.000 | 1.000 | 0.000 (<H) |

Router-A routed rounds (mean; after tau in brackets), scripted:

T = 500

| setting | k | c=never | c=0.05 | c=1 | c=10 | c=100 | c=1000 |
|---|---|---|---|---|---|---|---|
| s1 | 1 | 4 (0) | 4 (0) | 4 (0) | 27 (23) | 284 (280) | 500 (496) |
| s1 | 2 | 8 (0) | 8 (0) | 8 (0) | 26 (18) | 278 (270) | 500 (492) |
| fgt | 1 | 4 (0) | 4 (0) | 4 (0) | 15 (11) | 151 (147) | 500 (496) |
| fgt | 2 | 8 (0) | 8 (0) | 8 (0) | 15 (7) | 153 (145) | 500 (492) |

## 2. Does Router-B become the expert? (routed fraction, scripted)

T = 500. Entry = routed fraction / fraction of second-half rounds on D.

| setting | k | c=never | c=0.05 | c=1 | c=10 | c=100 | c=1000 |
|---|---|---|---|---|---|---|---|
| s1 | 1 | 0.008 / D2=0.00 | 0.872 / D2=0.60 | 0.872 / D2=0.60 | 0.872 / D2=0.60 | 0.887 / D2=0.50 | 1.000 / D2=0.65 |
| s1 | 2 | 0.016 / D2=0.00 | 0.865 / D2=0.60 | 0.865 / D2=0.60 | 0.865 / D2=0.60 | 0.879 / D2=0.49 | 1.000 / D2=0.65 |
| fgt | 1 | 0.008 / D2=0.00 | 0.008 / D2=0.00 | 0.314 / D2=0.16 | 0.419 / D2=0.16 | 0.464 / D2=0.12 | 1.000 / D2=0.36 |
| fgt | 2 | 0.016 / D2=0.00 | 0.016 / D2=0.00 | 0.282 / D2=0.13 | 0.374 / D2=0.13 | 0.424 / D2=0.09 | 1.000 / D2=0.34 |

## 3. Multipliers cold / Router-A vs cold / Router-B (c = 100)

Design scripted.

| setting | T | k | cold | Router-A | Router-B | H_A | H_B | cold/RA | cold/RB | cold/H_A | cold/H_B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 500 | 1 | 150.0 | 46.0±0.6 | 53.0±1.1 | 15.2 | 148.8 | 3.26 | 2.83 | 9.90 | 1.01 |
| s1 | 500 | 2 | 150.0 | 44.5±0.9 | 54.3±1.1 | 6.5 | 147.8 | 3.37 | 2.76 | 23.09 | 1.02 |
| fgt | 500 | 1 | 25.0 | 10.5±0.2 | 76.2±1.0 | 3.8 | 29.7 | 2.38 | 0.33 | 6.58 | 0.84 |
| fgt | 500 | 2 | 25.0 | 10.8±0.2 | 78.7±0.8 | 1.8 | 31.7 | 2.31 | 0.32 | 14.20 | 0.79 |
| s1_bigR | 500 | 1 | 150.0 | 80.8±0.3 | 44.2±0.3 | 149.3 | 149.3 | 1.86 | 3.40 | 1.00 | 1.00 |
| s1_bigR | 500 | 2 | 150.0 | 81.0±0.4 | 44.0±0.4 | 148.6 | 148.6 | 1.85 | 3.41 | 1.01 | 1.01 |

Design adaptive.

| setting | T | k | cold | Router-A | Router-B | H_A | H_B | cold/RA | cold/RB | cold/H_A | cold/H_B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 500 | 1 | 150.0 | 46.4±0.8 | 50.8±0.9 | 45.1 | 149.1 | 3.23 | 2.95 | 3.33 | 1.01 |
| s1 | 500 | 2 | 150.0 | 46.0±0.9 | 51.0±1.1 | 13.0 | 148.1 | 3.26 | 2.94 | 11.53 | 1.01 |
| fgt | 500 | 1 | 25.0 | 10.6±0.3 | 76.1±1.0 | 8.7 | 26.4 | 2.36 | 0.33 | 2.87 | 0.95 |
| fgt | 500 | 2 | 25.0 | 10.8±0.3 | 76.7±1.0 | 2.9 | 26.9 | 2.32 | 0.33 | 8.58 | 0.93 |
| s1_bigR | 500 | 1 | 150.0 | 80.3±0.3 | 44.7±0.3 | 149.4 | 149.2 | 1.87 | 3.36 | 1.00 | 1.01 |
| s1_bigR | 500 | 2 | 150.0 | 81.2±0.3 | 43.8±0.3 | 148.9 | 148.3 | 1.85 | 3.42 | 1.01 | 1.01 |

Ratios at every c (scripted, largest T, k=2 if present):

| setting | c=never (A/B) | c=0.05 (A/B) | c=1 (A/B) | c=10 (A/B) | c=100 (A/B) | c=1000 (A/B) |
|---|---|---|---|---|---|---|
| s1 | 23.09 / 1.02 | 150.00 / 2.70 | 150.00 / 2.70 | 37.17 / 2.71 | 3.37 / 2.76 | 1.86 / 3.37 |
| fgt | 14.20 / 0.79 | 14.20 / 0.79 | 31.09 / 0.31 | 20.56 / 0.31 | 2.31 / 0.32 | 0.72 / 0.53 |
| s1_bigR | 1.01 / 1.01 | 1.85 / 3.41 | 1.85 / 3.41 | 1.85 / 3.41 | 1.85 / 3.41 | 1.85 / 3.41 |

## 4. f > p - q: pseudo-regret vs realized reward (does the ranking flip?)

Expert accuracies after tau: A = p, B = q + f. FLIP = the two measures rank the pair differently.

| setting | design | T | k | pair | regret A | regret B | reward A | reward B | reward cold | better by regret | better by reward |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fgt | scripted | 500 | 1 | H | 3.8 | 29.7 | 146.5 | 120.5 | 123.8 | A | A |  |
| fgt | scripted | 500 | 1 | R c=0.05 | 3.8 | 29.7 | 146.5 | 120.5 | 123.8 | A | A |  |
| fgt | scripted | 500 | 1 | R c=1 | 0.4 | 77.7 | 149.6 | 100.8 | 123.8 | A | A |  |
| fgt | scripted | 500 | 1 | R c=10 | 1.2 | 77.6 | 149.6 | 99.1 | 123.8 | A | A |  |
| fgt | scripted | 500 | 1 | R c=100 | 10.5 | 76.2 | 149.6 | 85.3 | 123.8 | A | A |  |
| fgt | scripted | 500 | 1 | R c=1000 | 35.3 | 46.1 | 149.6 | 173.7 | 123.8 | A | B | FLIP |
| fgt | scripted | 500 | 2 | H | 1.8 | 31.7 | 149.8 | 118.7 | 123.2 | A | A |  |
| fgt | scripted | 500 | 2 | R c=0.05 | 1.8 | 31.7 | 149.8 | 118.7 | 123.2 | A | A |  |
| fgt | scripted | 500 | 2 | R c=1 | 0.8 | 80.4 | 150.7 | 96.4 | 123.2 | A | A |  |
| fgt | scripted | 500 | 2 | R c=10 | 1.2 | 80.3 | 150.7 | 95.4 | 123.2 | A | A |  |
| fgt | scripted | 500 | 2 | R c=100 | 10.8 | 78.7 | 150.7 | 82.2 | 123.2 | A | A |  |
| fgt | scripted | 500 | 2 | R c=1000 | 34.8 | 46.7 | 150.7 | 175.5 | 123.2 | A | B | FLIP |
| fgt | adaptive | 500 | 1 | H | 8.7 | 26.4 | 140.2 | 123.8 | 124.7 | A | A |  |
| fgt | adaptive | 500 | 1 | R c=0.05 | 25.0 | 25.0 | 124.7 | 124.7 | 124.7 | B | B |  |
| fgt | adaptive | 500 | 1 | R c=1 | 0.4 | 77.2 | 149.3 | 101.6 | 124.7 | A | A |  |
| fgt | adaptive | 500 | 1 | R c=10 | 1.0 | 77.2 | 149.3 | 100.5 | 124.7 | A | A |  |
| fgt | adaptive | 500 | 1 | R c=100 | 10.6 | 76.1 | 149.3 | 86.5 | 124.7 | A | A |  |
| fgt | adaptive | 500 | 1 | R c=1000 | 35.2 | 46.0 | 149.3 | 174.5 | 124.7 | A | B | FLIP |
| fgt | adaptive | 500 | 2 | H | 2.9 | 26.9 | 146.0 | 125.6 | 126.7 | A | A |  |
| fgt | adaptive | 500 | 2 | R c=0.05 | 25.0 | 25.0 | 126.7 | 126.7 | 126.7 | B | B |  |
| fgt | adaptive | 500 | 2 | R c=1 | 0.2 | 78.1 | 148.4 | 103.7 | 126.7 | A | A |  |
| fgt | adaptive | 500 | 2 | R c=10 | 0.9 | 78.0 | 148.4 | 102.8 | 126.7 | A | A |  |
| fgt | adaptive | 500 | 2 | R c=100 | 10.8 | 76.7 | 148.4 | 87.5 | 126.7 | A | A |  |
| fgt | adaptive | 500 | 2 | R c=1000 | 34.9 | 46.1 | 148.4 | 177.2 | 126.7 | A | B | FLIP |

## 5. Scripted vs adaptive window (c = 100)

Each entry: scripted / adaptive.

| setting | T | k | RA regret | RA settle | RA routed | RB regret | H_A regret | H_A settle(tau) |
|---|---|---|---|---|---|---|---|---|
| s1 | 500 | 1 | 46.0 / 46.4 | 0.020 / 0.040 | 284 / 286 | 53.0 / 50.8 | 15.2 / 45.1 | 0.900 / 0.700 |
| s1 | 500 | 2 | 44.5 / 46.0 | 0.120 / 0.080 | 278 / 284 | 54.3 / 51.0 | 6.5 / 13.0 | 0.960 / 0.920 |
| fgt | 500 | 1 | 10.5 / 10.6 | 1.000 / 1.000 | 151 / 152 | 76.2 / 76.1 | 3.8 / 8.7 | 0.860 / 0.660 |
| fgt | 500 | 2 | 10.8 / 10.8 | 1.000 / 1.000 | 153 / 153 | 78.7 / 76.7 | 1.8 / 2.9 | 0.960 / 0.900 |
| s1_bigR | 500 | 1 | 80.8 / 80.3 | 0.000 / 0.000 | 500 / 500 | 44.2 / 44.7 | 149.3 / 149.4 | 0.000 / 0.000 |
| s1_bigR | 500 | 2 | 81.0 / 81.2 | 0.000 / 0.000 | 500 / 500 | 44.0 / 43.8 | 148.6 / 148.9 | 0.000 / 0.000 |

## 6. Coverage only (routed rewards withheld), scripted, c = 100

| setting | T | k | H_A(nr) regret | H_A(nr) LinUCB D plays | R_A(nr) regret | R_A(nr) settle | R_B(nr) regret | cold |
|---|---|---|---|---|---|---|---|---|
| s1 | 500 | 1 | 149.3 | 0.00 | 81.2 | 0.000 | 43.8 | 150.0 |
| s1 | 500 | 2 | 148.6 | 0.00 | 80.5 | 0.000 | 44.5 | 150.0 |
| fgt | 500 | 1 | 25.2 | 0.00 | 35.3 | 0.000 | 46.1 | 25.0 |
| fgt | 500 | 2 | 25.4 | 0.00 | 34.8 | 0.000 | 46.7 | 25.0 |

## 7. Large radius (omega = 2.00 > p; valid-radius regime)

Entries: cold/Router-A / cold/Router-B (routed fraction A/B).

| setting | design | T | k | cold | cold/H_A | cold/H_B | c=never | c=0.05 | c=1 | c=10 | c=100 | c=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_bigR | scripted | 500 | 1 | 150.0 | 1.00 | 1.00 | 1.00/1.00 (rt 0.01/0.01) | 1.86/3.40 (rt 1.00/1.00) | 1.86/3.40 (rt 1.00/1.00) | 1.86/3.40 (rt 1.00/1.00) | 1.86/3.40 (rt 1.00/1.00) | 1.86/3.40 (rt 1.00/1.00) |
| s1_bigR | scripted | 500 | 2 | 150.0 | 1.01 | 1.01 | 1.01/1.01 (rt 0.02/0.02) | 1.85/3.41 (rt 1.00/1.00) | 1.85/3.41 (rt 1.00/1.00) | 1.85/3.41 (rt 1.00/1.00) | 1.85/3.41 (rt 1.00/1.00) | 1.85/3.41 (rt 1.00/1.00) |
| s1_bigR | adaptive | 500 | 1 | 150.0 | 1.00 | 1.01 | 1.00/1.00 (rt 0.00/0.00) | 1.87/3.36 (rt 1.00/1.00) | 1.87/3.36 (rt 1.00/1.00) | 1.87/3.36 (rt 1.00/1.00) | 1.87/3.36 (rt 1.00/1.00) | 1.87/3.36 (rt 1.00/1.00) |
| s1_bigR | adaptive | 500 | 2 | 150.0 | 1.01 | 1.01 | 1.00/1.00 (rt 0.00/0.00) | 1.85/3.42 (rt 1.00/1.00) | 1.85/3.42 (rt 1.00/1.00) | 1.85/3.42 (rt 1.00/1.00) | 1.85/3.42 (rt 1.00/1.00) | 1.85/3.42 (rt 1.00/1.00) |

