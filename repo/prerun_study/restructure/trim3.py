p='restructure/v7_new.tex'; s=open(p).read()
def swap(a,b,new):
    global s
    assert s.count(a)==1,a[:60]; i=s.index(a); j=s.index(b,i); s=s[:i]+new+s[j:]
swap('\\textbf{LLMs and bandits.}','\\textbf{Warm starts and model selection.}', r"""\textbf{LLMs and bandits.} \cite{chen2024llmdecision} combine an LLM policy
with a contextual bandit through an online model-selection master and report
reward multiplication; \cite{huangcontextual} calibrate LLM-derived priors to
set exploration bonuses; \cite{jumpstart2026} warm-start a linear bandit from
LLM-generated preferences through a single prior-error term; and
\cite{calibgated2026} inject LLM pseudo-observations into LinUCB with a
calibration-dependent weight. We separate the design matrix from the response
sum, bound what the first can deliver by a price ratio, and give the
expert-dependent remainder in closed form; our decomposition and bound hold
for any routed set, so they apply to these rules as well.
""")
swap('\nThe gap and the width sit in one expression.','\\section{The Price Bound}', r"""
A competitor with a large gap but a larger width is as dangerous as its gap,
one with a tiny gap is cheap however wide it is, and the only way to lower the
regret LinUCB can incur on a round is to shrink the widths of competitors
whose widths exceed their gaps. Widths shrink through a warm start and through
LinUCB's own observations, both by exact linear algebra
(Lemmas~\ref{lem:loewner} and~\ref{lem:transfer},
Appendix~\ref{app:extra-theory}). One consequence is that the
determinant-ratio bound behind $\widetilde O(\sqrt{rT})$-type guarantees
cannot certify a multiplier: any multiplier must come from competitors whose
widths the warm start pushes \emph{below their gaps}
(Proposition~\ref{thm:handoff}).

""")
swap('\nThe condition on $\\beta$ is the optimism condition','\\paragraph{The price ratio is attained.}', r"""
No online router realizes the offline separation on this instance: a bad
expert's data has $V_\tau=0$, and a good expert has $B_\tau\le0$ because it is
already optimal (Corollary~\ref{cor:cat-oracle},
Appendix~\ref{app:catalog-statements}). Online multiplication needs an expert
whose routed actions are cheap \emph{and} cover directions that recur.

""")
open(p,'w').write(s); print('ok')
