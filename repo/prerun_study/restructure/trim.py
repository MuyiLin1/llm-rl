import sys, re
sys.path.insert(0,'restructure')
from part1_new_text import ABSTRACT, INTRO
from part3_new_text import CONCLUSION, APP_MORE_EXP_HEAD, APP_CAT_HEAD
p='restructure/v7_new.tex'; s=open(p).read()
def rep(old,new):
    global s
    assert s.count(old)==1,(s.count(old),old[:90]); s=s.replace(old,new)
def between(a,b):
    assert s.count(a)==1,a[:60]; i=s.index(a); j=s.index(b,i); return s[i:j]

rep(ABSTRACT, r"""\begin{abstract}
Routing the early rounds of a linear contextual bandit to a frozen LLM and
then handing control to LinUCB can beat both, sometimes by a large factor,
and the LLM's accuracy does not predict when. We explain why. The expert's
routed rounds help the learner through \emph{coverage}, the directions its
actions explore, and through the \emph{transfer}, what its observed rewards
say beyond that coverage. For coverage we prove a price bound: on any
instance with a fixed finite arm set in which the best arm is always
available, the multiplier over cold LinUCB is at most the ratio of the prices
cold exploration and the expert pay for the directions the warm start
certifies, plus cold LinUCB's regret on the directions left uncertified. The
bound is attained, and on a canonical instance on which an offline warm start
multiplies by a factor of the horizon, no online handoff gains anything.
Under linear rewards, for an expert that sees only what the learner sees,
experts of equal coverage are interchangeable. Otherwise the estimate at
handoff carries a transfer term, which we give in closed form, and we
construct two experts of identical coverage, price and accuracy whose
multipliers differ by a factor that grows with the horizon, decided by a
handful of right answers. On AmazonCat-13K, three LLM experts of similar, low
accuracy cover the same directions yet multiply the router's accuracy over
cold LinUCB by $1.71$, $2.72$ and $3.29$. Coverage alone saves about $5\%$ of
cold LinUCB's regret for each; the transfer, which steers the learner to the
most frequent category, carries the rest and every significant difference
between them. Information value is not policy value.
\end{abstract}
""")

old_data = between('\\paragraph{What the data shows.}', '\\paragraph{Contributions.}')
rep(old_data, r"""\paragraph{What the data shows.}
On AmazonCat-13K the three experts above cover the same directions to within
$5\%$, and coverage alone saves about $5\%$ of cold LinUCB's regret for each;
every significant difference between them disappears when the transfer is
removed. The transfer acts as Theorem~\ref{thm:transfer-separation}
describes: one category is present on $30\%$ of contexts, a learner that
settles on it ends near $28\%$ accuracy against cold LinUCB's $8\%$, and
whether it settles is decided by a few of the expert's routed answers; adding
$15$ correct answers on that category to the weakest expert doubles its
multiplier. On OneShotWikiLinks, where no entity covers more than $0.36\%$ of
contexts, the Flan-T5 experts gain $1\%$.

""")

conc = between('\\section{Conclusion}', '\\paragraph{Broader impact.}')
rep(conc, r"""\section{Conclusion}
\label{sec:conclusion}

Reward multiplication in LLM--bandit hybrids has an exact accounting, and the
data the expert leaves behind enters through coverage and through the
transfer. Coverage has a price: the multiplier it can deliver is at most a
ratio of prices, attained on explicit instances and zero on the canonical
one, and under linear rewards, for an expert that sees only what the learner
sees, coverage is all there is. Real rewards are not linear and an LLM reads
text the features only summarize, so the estimate at handoff acquires a
transfer, which can separate experts of identical coverage and accuracy by a
factor of order the horizon. On AmazonCat-13K coverage saves about $5\%$ of
cold LinUCB's regret for every expert and the full router up to $21\%$; the
rest, and every significant difference between experts, is the transfer,
carried by a handful of right answers on the most frequent category. A
warm-start expert cannot be chosen by its accuracy or by its coverage: what
decides its value is which of its answers are right. Information value is
not policy value.

\paragraph{Limitations.}
The price bound assumes linear rewards and a fixed arm set with the best arm
always available; real streams satisfy these only approximately, and the
price ratio we report on them is a post-hoc explanatory statistic. Its
residual, cold LinUCB's regret on arms the warm start does not certify, is
not governed by a price, and we have no lower bound showing that it can be
large without a large price ratio, so the theorem characterizes the
certified saving rather than the multiplier outright. The constructions use a
fixed radius and unoptimized constants. Theorem~\ref{thm:transfer-separation}
needs exploration below a valid confidence radius, as in our experiments, and
compares two experts that both see the reward; how an informed expert
compares with one that sees only the features is shown only numerically
(Appendix~\ref{app:proof-transfer-sep}). We have no estimator that predicts
the size of the multiplier before the expert is run. The large AmazonCat
multipliers depend on a dominant category that the learner can be steered
to, and we have not tested other datasets with a skewed label distribution.
Only two real datasets are used, multi-expert behaviour is not stress-tested,
reward models suited to binary feedback may change the split between
coverage and the transfer, and Lemma~\ref{lem:width-limited} would need an
analogue for Thompson sampling.

""")

two = between('\\textbf{Two scales for one effect.}', '\\subsection{Setup}')
rep(two, r"""\textbf{Two scales.} The literature reports \emph{reward multiplication},
which with binary Precision@1 is a ratio of accuracies; we call it the
\emph{accuracy multiplier}. The theorems bound a ratio of regrets, and since
every context has a correct action, $R_T(\pi)=T(1-\mathrm{acc}(\pi))$; we
therefore also report the \emph{regret ratio}
$(1-\max\{\mathrm{acc}_{\mathrm{LLM}},\mathrm{acc}_{\mathrm{Cold}}\})/(1-\mathrm{acc}_{\mathrm{Router}})$.
At low accuracies the two differ sharply: on AmazonCat an accuracy multiplier
of $3.29$ from a base of $8.4\%$ is a regret ratio of $1.27$, or $21\%$ less
regret. $A_\tau$, $V_\tau$ and $B_\tau$ are counted in correct answers and are
the same on either scale.

""")

pol = between('\\textbf{Policies.} Each experiment tracks three policies', '\\textbf{The price ratio on real data.}')
choice_c = r"""\paragraph{Choice of $c$.} On AmazonCat the tuning seeds $11$--$20$ select
$c=100$ for BERT features (mean Router accuracy $17.2$, $19.5$, $22.2$, $21.4$
and $5.4\%$ at $c=1,10,10^2,10^3,10^4$) and $c=0.1$ for BGE features
(Table~\ref{tab:bge-csweep}). On WikiLinks the Flan-T5 grid is
$c\in\{1,10,10^2,10^3,10^4\}$ (Table~\ref{tab:wiki-csweep}) and the Qwen grid
$c\in\{0.01,0.1,1,10,100\}$ (Table~\ref{tab:bge-csweep}).
\todo{WikiLinks sweep}

"""
assert '17.2' in pol
rep(pol, r"""\textbf{Policies.} Each run tracks three policies on the same stream: the
\textbf{Router} (Algorithm~\ref{alg:confidence-gated},
Appendix~\ref{app:meta-algorithm}), which routes a round to the LLM when the
LLM's width $\|\phi(x_t,a_t^{(1)})\|_2/\sqrt c$ is below LinUCB's width
$\|\phi(x_t,a_t^{(0)})\|_{V_t^{-1}}$ and updates $(V_t,b_t)$ on every round
with the played action and observed reward, so that its LinUCB component is
the online-warm-started learner of Section~\ref{sec:setup}; the \textbf{LLM}
alone, which proposes the action whose embedding is closest to its textual
output \cite{chen2024llmdecision}; and \textbf{cold LinUCB}. We use
$\lambda=0.1$ and the exploration scale $\alpha_t=\max(0.1,0.9985^t)$, which
reaches its floor at $t\approx1{,}500$. This is smaller than the confidence
radius the theory requires, so the confidence event of
Section~\ref{sec:width} is not guaranteed in our runs: we use the theory to
explain the experiments, not to certify them. A seed fixes the order of the
stream and is shared by all policies, so every comparison is paired. The
confidence parameter $c$ of each configuration is chosen on seeds $11$--$20$
by mean Router accuracy and evaluated on seeds $1$--$10$, so tuning and
evaluation never share a seed; this selects $c=100$ for AmazonCat/BERT and
$c=0.1$ for AmazonCat/BGE (Appendix~\ref{app:more-exp}).
\todo{WikiLinks sweep}

""")
rep(APP_MORE_EXP_HEAD.split('\\subsection{AmazonCat-13K with BERT Features}')[1], APP_MORE_EXP_HEAD.split('\\subsection{AmazonCat-13K with BERT Features}')[1] + choice_c)

pr = between('\\textbf{The price ratio on real data.}', '\\subsection{AmazonCat-13K}')
rep(pr, r"""\textbf{The price ratio on real data.} For the AmazonCat BERT configurations
we also report the price ratio $P_U$ of Theorem~\ref{thm:general-price}. It is
measured after the run from per-round logs, taking for $U$ the leading
directions of the features cold LinUCB played on rounds it lost after routing
stopped, on streams outside the theorem's fixed-arm setting: it is an
explanatory statistic, not a prediction, and by construction it is blind to
the transfer.

""")

c11 = between('The catalog has eleven rows rather than', '\\begin{table}[t]')
rep(c11, '')
rep(APP_CAT_HEAD, APP_CAT_HEAD + c11.strip('\n') + '\n\n')
open(p,'w').write(s); print('ok')
