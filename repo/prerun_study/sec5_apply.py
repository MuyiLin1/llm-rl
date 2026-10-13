P = '/Users/linmuyi/code/DMwithLLM/v7.tex'
s = open(P).read()
def one(x):
    n = s.count(x); assert n == 1, (n, x[:90])
def rep(old, new):
    global s; one(old); s = s.replace(old, new)
def cut(a, b):
    """return text from a (inclusive) to b (exclusive) and remove it"""
    global s; one(a); one(b); i = s.index(a); j = s.index(b); assert i < j
    blk = s[i:j]; s = s[:i] + s[j:]; return blk, i

# ---------- grab the old proof pieces before replacing Section 5 ----------
pa = s[s.index(r"\emph{(a)} Fix $x\in \mathcal S$"):s.index(r"\emph{(b)} Fix $x\in U$")]
pb = s[s.index(r"\emph{(b)} Fix $x\in U$"):s.index(r"\emph{(c)} Split the sum")]
pc = s[s.index(r"\emph{(c)} Split the sum"):s.index(r"""hybrid plays the expert's actions on $[1,\tau]$.
\end{proof}""")] + "hybrid plays the expert's actions on $[1,\\tau]$.\n"
pa = pa.replace(r"\emph{(a)} ", "").replace("Summing over $x\\in \\mathcal S$ gives (a).", "Summing over $x\\in \\mathcal S$ gives the lemma.").rstrip() + "\n"
pb = pb.replace(r"\emph{(b)} Fix $x\in U$", r"Fix $x\in U$").rstrip() + "\n"
pc = (pc.replace(r"\emph{(c)} Split the sum in (a) into", r"Split the sum of Lemma~\ref{lem:cold-price} into")
        .replace("Summing over $U$ and applying (b) bounds", "Summing over $U$ and applying Lemma~\\ref{lem:cert-cost} bounds"))
assert '(a)' not in pc and '(b)' not in pc

# ---------- new Section 5 core: setting + theorem + sketch ----------
old_start = r"""\paragraph{Setting for the theorem.}"""
old_end = r"""The proof is in Appendix~\ref{app:proof-diverge}. Two relaxations of"""
a = s.index(old_start); b = s.index(r"""\paragraph{The two parts of the bound.}""")
new_core = r"""\paragraph{Setting for the theorem.}
In words: we fix a finite set of arms in which the best arm is always
offered, compare cold LinUCB with an arbitrary policy $\pi$ up to a cutoff
$t_0$, and measure how much coverage $\pi$ bought, at what price, and which
arms that coverage lets the confidence bound rule out. Let
$\mathcal X\subset\R^d$ be a finite set of arm vectors with $\|x\|_2\le L$,
let $\mathcal A_t\subseteq\mathcal X$ for every $t$, and suppose an arm
$x^\star$ with $\mu(x^\star)=\max_{x\in\mathcal X}\mu(x)$ belongs to every
$\mathcal A_t$. The gap of an arm is then the same at every round,
$\Delta(x):=\mu(x^\star)-\mu(x)$; write $\mathcal S:=\{x\in\mathcal X:\Delta(x)>0\}$
and fix $\beta\ge\beta_t$ for every $t\le T$ (one exists because
$\|x\|_2\le L$ bounds $\det V_t$).

\emph{What cold LinUCB pays.} For $x\in\mathcal S$ put $u_x:=x/\|x\|_2$ and
\[
\kappa_x:=\frac{\Delta(x)}{\|x\|_2^2},\qquad
C^\star_x:=\frac{4\beta^2\|x\|_2^2}{\Delta(x)^2},\qquad
n^\star_x:=\Bigl(\frac{C^\star_x-\lambda}{\|x\|_2^2}+1\Bigr)\ind\{C^\star_x\ge\lambda\},\qquad
N_x:=\#\{t\in[T]:x\in\mathcal A_t\}.
\]
Here $\kappa_x$ is the price of covering $u_x$ by pulling $x$ itself,
$C^\star_x-\lambda$ is the least coverage of $u_x$ at which the confidence
bound can exclude $x$, and $n^\star_x$ bounds how often cold LinUCB pulls $x$
before its own pulls exclude it.

\emph{What $\pi$ pays.} Write $x^\pi_t$ for the arm $\pi$ plays at round $t$
and, for a set $U\subseteq\mathcal S$ of arms,
$\mathrm{cov}_U(x):=\sum_{y\in U}\langle x,u_y\rangle^2$. The price $\pi$ pays
for coverage of $U$ up to $t_0$, and the coverage of $U$ it obtains for free
by playing optimal arms, are
\[
\kappa^\pi_U(t_0):=\min\Bigl\{\frac{\Delta(x^\pi_t)}{\mathrm{cov}_U(x^\pi_t)}:
\ t\le t_0,\ \Delta(x^\pi_t)>0,\ \mathrm{cov}_U(x^\pi_t)>0\Bigr\},
\qquad
C^{\mathrm{free}}_U(t_0):=\sum_{t\le t_0:\,\Delta(x^\pi_t)=0}\mathrm{cov}_U(x^\pi_t),
\]
with the minimum over an empty set equal to $+\infty$. Finally
$U_{\mathrm{cert}}(t_0)$ is the set of arms $x\in\mathcal S$ that $\pi$'s
first $t_0$ rounds certify: $2\beta\|x\|_{(V^\pi_{t_0+1})^{-1}}<\Delta(x)$, with
$V^\pi_{t_0+1}=\lambda I+\sum_{t\le t_0}x^\pi_t (x^\pi_t)^\top$. For the
handoff hybrid with $t_0=\tau$, $\kappa^\pi_U(\tau)$ is the expert's price and
$R_{[1,t_0]}(\pi)$ its acquisition regret, but nothing below uses this.

\begin{theorem}[Price bound for the online multiplier]
\label{thm:general-price}
In the setting above, on the confidence event of cold LinUCB, let $\pi$ be
any policy, $t_0\in\{0,\dots,T\}$ any cutoff and $U\subseteq
U_{\mathrm{cert}}(t_0)$ any set of certified arms, and write
$\kappa^{\Cold}_U:=\max_{x\in U}\kappa_x$ ($:=0$ if $U=\emptyset$) and
$P_U:=\kappa^{\Cold}_U/\kappa^\pi_U(t_0)$ for the price ratio. Then
\[
R_T(\Cold)\ \le\ P_U\,R_{[1,t_0]}(\pi)\ +\ \kappa^{\Cold}_U\,C^{\mathrm{free}}_U(t_0)\ +\ \sum_{x\in U}\Delta(x)
\ +\sum_{x\in \mathcal S\setminus U}\Delta(x)\min\{n^\star_x,N_x\},
\]
and consequently, whenever $R_T(\pi)>0$,
\begin{equation}
\label{eq:general-price}
\frac{R_T(\Cold)}{R_T(\pi)}\ \le\ P_U\ +\
\frac{\kappa^{\Cold}_U\,C^{\mathrm{free}}_U(t_0)+\sum_{x\in U}\Delta(x)+\sum_{x\in \mathcal S\setminus U}\Delta(x)\min\{n^\star_x,N_x\}}{R_T(\pi)} .
\end{equation}
For $\pi=\Hyb_\tau$ and $t_0=\tau$ this bounds the multiplier of the handoff
hybrid by the price ratio between cold exploration and the expert on the
arms the warm start certifies, plus a residual.
\end{theorem}

\emph{Proof sketch.} The proof (Appendix~\ref{app:proof-general-price}) has
two steps. First, cold LinUCB pays its own price, once per arm: on its
confidence event it plays a suboptimal arm $x$ only while
$2\beta\|x\|_{V_t^{-1}}\ge\Delta(x)$, and its own pulls of $x$ shrink that
width, so it pulls $x$ at most $\min\{n^\star_x,N_x\}$ times
(Lemma~\ref{lem:cold-price}). Second, certification costs at least the price:
by Cauchy--Schwarz, $\pi$ certifies $x$ only once its coverage of $u_x$
exceeds $C^\star_x-\lambda$, and it buys that coverage at a price of at least
$\kappa^\pi_U(t_0)$ per unit (Lemma~\ref{lem:cert-cost}). Splitting cold
LinUCB's bill into certified and uncertified arms and paying for the
certified ones at the rate $\kappa^{\Cold}_U$ gives the first display; since
$R_T(\pi)\ge R_{[1,t_0]}(\pi)$, dividing by $R_T(\pi)$ gives the second.

Only the confidence event of cold LinUCB is used, and
$\pi$ is arbitrary, so the bound covers the fixed-$\tau$ hybrid, the adaptive
router (take $t_0$ to be its last routed round), a CORRAL-style master, and
cold LinUCB itself. It bounds the ratio of the two policies' regrets on the
same instance, not one upper bound against another: two policies with the
same asymptotic rate can still differ by a diverging factor, so a statement
about the multiplier must bound one policy's regret in terms of the other's,
as the theorem does. Appendix~\ref{app:price-template} evaluates the bound on the
instance of Theorem~\ref{thm:price}.

"""
s = s[:a] + new_core + s[b:]

# ---------- appendix: full proof as two lemmas + assembly ----------
rep(r"""\subsection{The Price Bound on the Template}
\label{app:price-template}

The proof of Theorem~\ref{thm:general-price} is in Section~\ref{sec:price}.
Here we check the bound on the template instance.""",
r"""\subsection{Proof of Theorem~\ref{thm:general-price}}
\label{app:proof-general-price}

We work in the setting of Section~\ref{sec:price}, on the confidence event
$\mathcal E_\delta$ of cold LinUCB, and prove the two steps of the proof
sketch as lemmas.

\begin{lemma}[Cold LinUCB pays its own price, once per arm]
\label{lem:cold-price}
$R_T(\Cold)\le\sum_{x\in\mathcal S}\Delta(x)\min\{n^\star_x,N_x\}$.
\end{lemma}

\begin{proof}
""" + pa + r"""\end{proof}

\begin{lemma}[Certification costs at least the price]
\label{lem:cert-cost}
For every policy $\pi$, every cutoff $t_0\in\{0,\dots,T\}$ and every set of
arms $U\subseteq U_{\mathrm{cert}}(t_0)$,
\[
\sum_{x\in U}(C^\star_x-\lambda)_+\ \le\ \sum_{x\in U}\sum_{t\le t_0}\langle x^\pi_t,u_x\rangle^2
\ \le\ \frac{R_{[1,t_0]}(\pi)}{\kappa^\pi_U(t_0)}+C^{\mathrm{free}}_U(t_0),
\]
with $1/\kappa^\pi_U(t_0):=0$ when $\kappa^\pi_U(t_0)=+\infty$.
\end{lemma}

\begin{proof}
""" + pb + r"""\end{proof}

\begin{proof}[Proof of Theorem~\ref{thm:general-price}]
""" + pc + r"""\end{proof}

\subsection{The Price Bound on the Template}
\label{app:price-template}

Here we check the bound of Theorem~\ref{thm:general-price} on the template
instance.""")
rep(r"""values over rounds at which $x$ is offered. Part (a) holds with $C^\star_x$ computed
from $\Delta^-(x)$ and each pull charged $\Delta^+(x)$, and (b) with
$\Delta^-(x)$;""", r"""values over rounds at which $x$ is offered. Lemma~\ref{lem:cold-price} holds
with $C^\star_x$ computed from $\Delta^-(x)$ and each pull charged $\Delta^+(x)$,
and Lemma~\ref{lem:cert-cost} with $\Delta^-(x)$;""")
rep(r"""Theorem~\ref{thm:general-price}(b) written out for the template.""", r"""Lemma~\ref{lem:cert-cost} written out for the template.""")
rep(r"""Sections~\ref{sec:price} and~\ref{sec:two-channels} other than
Theorem~\ref{thm:general-price}, whose proof is in the main text.""",
r"""Sections~\ref{sec:price} and~\ref{sec:two-channels}, including
Theorem~\ref{thm:general-price}.""")
rep(r"""Section~\ref{sec:experiments} the experiments. The proof of
Theorem~\ref{thm:general-price} is in Section~\ref{sec:price}; the catalog
and the other proofs are in the appendix.""", r"""Section~\ref{sec:experiments} the experiments. The catalog and all proofs
are in the appendix.""")

# ---------- fix 6 ----------
rep(r"""Five quantities recur.""", r"""Six quantities recur.""")

# ---------- fix 5: move the price-ratio-on-real-data paragraph ----------
rep(r"""\textbf{The price ratio on real data.} For the AmazonCat BERT configurations
we also report an empirical analogue of the price ratio $P_U$ of
Theorem~\ref{thm:general-price}. It is measured after the run from per-round
logs, taking for $U$ the leading directions of the features cold LinUCB
played on rounds it lost after routing stopped, in place of certified arms,
on streams outside the theorem's setting: it is an explanatory statistic, not
a bound the theorem certifies, and by construction it is blind to the
transfer.

""", "")
rep(r"""same prices, so their post-hoc price ratios ($1.73$, $1.59$, $1.58$) are close""",
r"""same prices, so their price ratios, measured after the run as an explanatory
statistic (Appendix~\ref{app:amazon-transfer}), are close ($1.73$, $1.59$, $1.58$),
close""")
rep(r"""ratio of Theorem~\ref{thm:general-price}, measured post hoc as described in
Section~\ref{sec:exp-setup}, is $1.73$, $1.59$ and $1.58$ for the
three---close to one another, and ordered opposite to the observed
multipliers.""", r"""ratio of Theorem~\ref{thm:general-price}, measured after the run, is $1.73$,
$1.59$ and $1.58$ for the three---close to one another, and ordered opposite
to the observed multipliers. We compute it from per-round logs, taking for $U$
the leading directions of the features cold LinUCB played on rounds it lost
after routing stopped, in place of certified arms. The streams are outside
the theorem's setting, so this is an explanatory statistic rather than a
bound the theorem certifies, and by construction it is blind to the
transfer.""")

# ---------- fix 4: conclusion leads with the transfer ----------
a = s.index(r"""Reward multiplication in LLM--bandit hybrids has an exact accounting, and the""")
b = s.index(r"""\paragraph{Limitations.}""")
s = s[:a] + r"""Reward multiplication in LLM--bandit hybrids has an exact accounting, and the
data the expert leaves behind reaches the learner through coverage and
through the transfer. The transfer is what decides the gain. When
exploration is below the valid radius, as in practice, it can separate
experts of identical coverage, cost and number of routed right answers by a
factor of order the horizon, even when the losing expert is the more
accurate. On AmazonCat-13K it carries every significant difference between
experts and most of the regret the router saves, through a handful of right
answers on the most frequent category: with their rewards hidden from the
learner, the same answers buy nothing. Coverage cannot play this role. With a
valid radius the multiplier it can deliver is capped by a ratio of prices,
the cap is attained on explicit instances, and the information value it
controls vanishes on the canonical one; under linear rewards with Gaussian
noise, for an expert whose choices ignore the noise, coverage is all there
is; and on AmazonCat it saves about $2$ to $3.5\%$ of cold LinUCB's regret
for every expert. A warm-start expert cannot be chosen by its accuracy or by
its coverage: what decides its value is which of its answers are right.
Information value is not policy value.

""" + s[b:]
open(P, 'w').write(s); print('applied')
