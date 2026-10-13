import numpy as np
rng = np.random.default_rng(1)
lam, sigma = 60.0, 0.01
L, Lp, c, eps = 13.0, 12.0, 1.0, 0.05      # eps < c/L = 0.077
K, n = 4, 300; tau0 = K*n
Kp = tau0                                   # capacity lam/Lp^2 = 0.42 < 1 -> fresh distractor every round
tail = 400; T = tau0 + tail
d = 1 + K + Kp
beta = 2*np.sqrt(lam); m0 = int(lam//30)
theta = np.zeros(d); theta[0] = 1.0
J = list(range(1, K+1)); Jp = list(range(K+1, K+1+Kp))
def g(): v=np.zeros(d); v[0]=1; return v
def p(j): v=np.zeros(d); v[0]=1-eps; v[j]=c; return v
def b(j): v=np.zeros(d); v[j]=L; return v
def bp(jp): v=np.zeros(d); v[0]=1; v[jp]=Lp; return v

Vi = np.eye(d)/lam; bvec = np.zeros(d)
maxdev = 0.0; window_probes = 0; window_reg = 0.0
tail_pulls = {j:0 for j in J}; tail_reg = 0.0; played_g_in_window = 0
for t in range(T):
    th = Vi@bvec
    j = J[t % K]
    if t < tau0:
        A = [('g',g()), ('p',p(j)), ('bp',bp(Jp[t % Kp]))]
    else:
        A = [('g',g()), ('b',b(j))]
    idx = [x@th + beta*np.sqrt(x@Vi@x) for _,x in A]
    name, x = A[int(np.argmax(idx))]
    mu = x@theta; best = max(v@theta for _,v in A)
    r = mu + rng.normal(scale=sigma)
    # check sharpened confidence event: ||th-theta||_V <= 1.25 sqrt(lam) ; V = inv(Vi)
    if t % 50 == 0:
        V = np.linalg.inv(Vi); dev = np.sqrt((th-theta)@V@(th-theta)); maxdev = max(maxdev, dev)
    if t < tau0:
        window_reg += best-mu
        if name=='p': window_probes += 1
        if name=='g': played_g_in_window += 1
    else:
        tail_reg += best-mu
        if name=='b': tail_pulls[j] += 1
    # Sherman-Morrison update
    Vx = Vi@x; Vi -= np.outer(Vx,Vx)/(1+x@Vx); bvec += r*x

print(f"d={d}, lam={lam}, beta={beta:.2f}, m0={m0}, 16*sigma^2*d*l0 ~ {16*sigma**2*d*(np.log(1+T*L**2/lam)+2*np.log(20)):.2f} <= lam OK")
print(f"sharpened conf event max dev / (1.25 sqrt lam) = {maxdev/(1.25*np.sqrt(lam)):.3f}  (needs <= 1)")
print(f"window: probes played = {window_probes} (need 0); g played = {played_g_in_window}; window regret = {window_reg:.3f} (need 0)")
print(f"tail: bad-arm pulls per direction = {tail_pulls} (need each >= m0={m0} and <= 4beta^2+1={4*beta**2+1:.0f})")
print(f"tail regret = {tail_reg:.1f} (need >= K*m0 = {K*m0})")