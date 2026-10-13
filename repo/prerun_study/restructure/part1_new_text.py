# New prose for the restructured paper (part 1: abstract, intro, related work, setup)
ABSTRACT = r"""\begin{abstract}
Routing the early rounds of a linear contextual bandit to a frozen LLM and
then handing control to LinUCB can beat both the LLM and cold-started
LinUCB, sometimes by a large factor. The LLM's accuracy does not predict when
this happens, and we explain why. The expert's routed rounds help the learner
in two ways: through \emph{coverage}, the directions its actions explore, and
through the \emph{transfer}, what its observed rewards say beyond that
coverage. For coverage we prove a price bound: on any instance with a fixed
finite arm set in which the best arm is always available, the multiplier over
cold LinUCB is at most the ratio of the price cold exploration pays for the
directions the warm start certifies to the price the expert pays for them,
plus cold LinUCB's regret on the directions left uncertified. The bound is
attained, and on a canonical instance on which an offline warm start
multiplies by a factor of the horizon, no online handoff gains anything.
Under linear rewards, and for an expert that sees only what the learner
sees, coverage is the whole story: experts of equal coverage are
interchangeable. Otherwise the estimate at handoff carries a transfer term,
which we give in closed form, and we construct two experts of identical
coverage, price and accuracy whose multipliers differ by a factor that grows
with the horizon, decided by a handful of right answers. On AmazonCat-13K
three LLM experts of similar, low accuracy ($3.6$--$5.3\%$) cover the same
directions yet raise the router's accuracy over cold LinUCB by factors of
$1.71$, $2.72$ and $3.29$. Coverage alone saves about $5\%$ of cold LinUCB's
regret, the same for all three; the full router saves up to $21\%$, and the
difference, including every significant difference between the experts,
comes from the transfer, which steers the learner to the most frequent
category. Information value is not policy value: a warm-start expert cannot
be chosen by its accuracy.
\end{abstract}
"""

INTRO = r"""\section{Introduction}
\label{sec:intro}

Large pretrained language models carry prior knowledge that can guide the
early decisions of a sequential learner
\cite{brown2020language,ouyang2022training,achiam2023gpt,alamdari2024jumpstarting,abbasiyadkori2024believe},
in recommendation \cite{li2010contextual,agarwal2016making}, healthcare
\cite{tewari2017ads,svensson2023sequential} and dialogue \cite{li2016deep}.
A striking empirical finding is that combining a frozen LLM with an adaptive
linear bandit can strictly dominate both, sometimes by a large factor
\cite{chen2024llmdecision,huangcontextual}: routing the early rounds to the
LLM and then handing control to LinUCB beats the LLM alone and LinUCB started
from scratch. The literature on warm starts and model selection
\cite{bastani2021mostly,zhang2019warm,agarwal2017corralling} reports nothing
comparable, and the LLM's accuracy is a poor guide: on AmazonCat-13K, three
LLM experts that are right on $3.6$--$5.3\%$ of contexts raise the accuracy of
the same learner by factors from $1.71$ to $3.29$, and the least accurate is
not the least useful. This paper explains where the gain comes from, bounds
it, and shows why accuracy cannot predict it.

\paragraph{Policy value and information value.}
The \emph{policy value} of an expert is the reward its own actions earn; its
\emph{information value} is how much its actions teach a learner that keeps
their data. The two can be wildly different. On the instance of
Proposition~\ref{prop:three-way}, a constant policy that is wrong on every
round produces data under which LinUCB's regret falls from linear to
constant: no policy value, enormous information value. That data, however, is
collected before the evaluated horizon and not paid for. When the expert's
rounds count toward the regret, as in a handoff, the same instance yields no
gain at all, for every expert (Proposition~\ref{prop:one-pass}). The question
is when an \emph{online} handoff can multiply, and by how much.

\paragraph{The accounting.}
Write $\Cold$ for cold LinUCB, $\Exp$ for the frozen expert, and $\Hyb_\tau$
for the hybrid that follows $\Exp$ for $\tau$ rounds, feeding every
observation to LinUCB, and then hands over. Then
\[
R_T(\Cold)-R_T(\Hyb_\tau)=A_\tau+V_\tau,\qquad R_T(\Exp)-R_T(\Hyb_\tau)=B_\tau
\]
(Proposition~\ref{prop:decomp}), where $A_\tau$ is the expert's advantage on
the routed rounds, $V_\tau$ the regret LinUCB saves afterwards because of the
data it inherited, and $B_\tau$ the learner's advantage over the expert on
the tail. The inherited data enters LinUCB through two objects: the design
matrix, which sets every confidence width, and the response sum, which sets
the estimate. The first is \emph{coverage}; beyond coverage, the second
contributes a single expert-dependent term, the \emph{transfer}.

\paragraph{Coverage has a price (main result).}
Data helps LinUCB only by shrinking the confidence widths of actions it would
otherwise mistake for the best, and widths shrink only with coverage.
Coverage is never free: the \emph{price} of a direction is the least regret
any action pays per unit of coverage of it, and cold LinUCB buys the same
coverage on its own, at the price of the arms its optimism forces it to pull.
Our main theorem (Theorem~\ref{thm:general-price}) bounds the multiplier of
any policy over cold LinUCB, on any instance with a fixed finite arm set in
which the best arm is always available, by the ratio of cold LinUCB's price
to the expert's price on the directions the warm start certifies, plus cold
LinUCB's regret on the directions it does not certify. The confidence radius
cancels. A diverging multiplier therefore requires a diverging price ratio:
an action of vanishing gap that covers a direction cold exploration can buy
only dearly (Corollary~\ref{cor:diverge}). The bound is attained
(Theorem~\ref{thm:price}), whether the saving is collected during the routed
rounds or after them, and a catalog of eleven instances
(Table~\ref{tab:catalog}) shows that removing any one of recurrence,
cheapness, correct targeting, or the expert's suboptimality on the tail can
remove the gain.

\paragraph{Beyond coverage: the transfer.}
Under linear rewards, and for an expert that sees only what the learner
sees, the response sum is the Gram matrix applied to the true parameter plus
noise whose law depends only on that matrix
(Proposition~\ref{prop:response-sum}). Coverage is then the whole of
information value, and experts that cover the same directions are
interchangeable (Corollary~\ref{cor:coverage-complete}). This is a
prediction, and it fails in practice for two reasons: rewards are not linear
in the features, and an LLM reads text that the features only summarize, so
it can prefer an action on exactly the rounds where that action pays. Either
way the estimate at handoff acquires a transfer term, which we give in closed
form (Proposition~\ref{prop:transfer}). The transfer can decide everything:
we construct two experts with identical coverage, price and accuracy whose
multipliers over cold LinUCB differ by a factor that grows with the horizon
(Theorem~\ref{thm:transfer-separation}). What separates them is where their
right answers fall. A handful of right answers on an action that pays often
settles the learner on it, with a failure probability that decays
exponentially in the number of such answers and does not depend on the
horizon.

\paragraph{What the data shows.}
On AmazonCat-13K the three experts above cover the same directions to within
$5\%$, so the price bound treats them alike, and coverage alone saves about
$5\%$ of cold LinUCB's regret for each. Their multipliers differ, and every
significant difference disappears when the transfer is removed. The transfer
acts as Theorem~\ref{thm:transfer-separation} describes: one category is
present on $30\%$ of contexts, a learner that settles on it ends near $28\%$
accuracy against cold LinUCB's $8\%$, and whether it settles is decided by a
few of the expert's routed answers; adding $15$ correct answers on that
category to the weakest expert, $0.15$ points of accuracy, doubles its
multiplier. On OneShotWikiLinks, where no entity covers more than $0.36\%$ of
contexts, there is nothing to steer the learner toward, and the Flan-T5
experts gain $1\%$.

\paragraph{Contributions.}
\begin{itemize}[itemsep=1pt, topsep=1pt, leftmargin=*]
\item \textbf{A price bound for the online multiplier}
(Theorem~\ref{thm:general-price}, Corollary~\ref{cor:diverge}), on any
instance with a fixed finite arm set in which the best arm is always
available, stated for any policy and any cutoff, so that it covers fixed and
adaptive handoffs and randomized masters alike.
\item \textbf{Instances attaining it, and a catalog}
(Theorem~\ref{thm:price}, Table~\ref{tab:catalog}): the bound is attained
with the saving collected during or after the routed rounds, and eleven
instances with complete proofs isolate which ingredient each negative case
lacks. On the canonical instance an offline warm start multiplies by a factor
of the horizon while no online handoff gains anything
(Propositions~\ref{prop:three-way} and~\ref{prop:one-pass}).
\item \textbf{Coverage and the transfer} (Section~\ref{sec:two-channels}):
coverage is a complete account of information value under linear rewards,
for an expert that sees only the features; beyond that, the transfer is the
only expert-dependent part of the warm start, in closed form; and it can
separate experts of identical coverage, price and accuracy by a factor of
order the horizon (Theorem~\ref{thm:transfer-separation}).
\item \textbf{Experiments that test the prediction}
(Section~\ref{sec:experiments}): on two datasets and twelve configurations,
with every tuning parameter chosen on seeds disjoint from the evaluation
seeds, coverage gives a small gain that is the same for every expert, and the
transfer carries most of the gain and every significant difference between
experts.
\end{itemize}
Section~\ref{sec:related-main} discusses related work,
Section~\ref{sec:setup} fixes notation, Section~\ref{sec:decomp} gives the
accounting, Sections~\ref{sec:price}--\ref{sec:two-channels} the theory, and
Section~\ref{sec:experiments} the experiments. Proofs are in the appendix.

"""

RELATED = r"""\section{Related Work}
\label{sec:related-main}

\textbf{LLMs and bandits.} \cite{chen2024llmdecision} combine an LLM policy
with a contextual bandit through an online model-selection master and report
reward multiplication; \cite{huangcontextual} calibrate LLM-derived priors to
set exploration bonuses; \cite{jumpstart2026} warm-start a linear bandit from
LLM-generated preferences and bound the estimator's error through a single
prior-error term; \cite{calibgated2026} inject LLM pseudo-observations into
LinUCB with a calibration-dependent weight. These analyses treat the design
matrix and the response vector together or study a bias--variance trade-off;
we separate the two, bound what the first can deliver by a price ratio, and
give the expert-dependent remainder in closed form. Our decomposition and
price bound hold for any routed set, so they apply to these rules as well.
\textbf{Warm starts and model selection.} Warm-start analyses
\cite{bastani2021mostly,zhang2019warm} bound the regret of a learner started
from logged data through the quality of that data; they are offline in our
sense, and our contribution relative to them is the online accounting, in
which acquiring the data is charged and can cost more than it is worth.
Corralling and model selection
\cite{agarwal2017corralling,pacchiano2020model,marinov2021pareto} compete
with the best of several base learners; we ask when a combination beats both
components, and our bound applies to a CORRAL master as to any other policy.
\textbf{Geometry of exploration and misspecification.} The price of
Section~\ref{sec:price} is the single-direction shadow of the allocation
programs behind instance-dependent lower bounds
\cite{graves1997asymptotically,lattimore2017end}; our use is comparative,
with two policies paying different prices for the same coverage. Unlike a
hint about $\theta^\star$ \cite{cutkosky2022leveraging}, whose value grows
with its accuracy, a warm start is data in the design matrix, and its value
is not a function of accuracy. Work on misspecified linear bandits treats the
part of the reward the linear model misses as a cost to be controlled
\todo{cite misspecified linear bandits, e.g.\ Ghosh et al.\ 2017; Lattimore
et al.\ 2020; Foster et al.\ 2020}; here that part, carried by the transfer,
is the main source of the gain. Appendix~\ref{sec:related} gives an extended
discussion.

"""

KEYTERMS = r"""
\paragraph{Key terms.}
Five quantities recur. The \emph{confidence width} of an action with features
$x$ under a design matrix $V$ is $\|x\|_{V^{-1}}$; LinUCB plays a suboptimal
action only while $2\beta_t$ times its width is at least its gap
(Lemma~\ref{lem:width-limited}). The \emph{coverage} of a direction $u$ by a
sequence of played actions is $\sum_s\langle x_s,u\rangle^2$, the growth of the
design matrix along $u$; widths along $u$ fall only as its coverage grows
(Lemma~\ref{lem:loewner}). The \emph{price} of $u$ is the least regret any
available action pays per unit of coverage of $u$ \eqref{eq:price}. A design
matrix \emph{certifies} an action once $2\beta$ times its width is below its
gap, so that the confidence bound alone rules the action out. The
\emph{acquisition regret} of a warm start is the regret of the routed rounds
that produced it. Finally, the \emph{multiplier} of a policy $\pi$ over cold
LinUCB is $R_T(\Cold)/R_T(\pi)$; the experiments also report the ratio of
accuracies used in the literature (Section~\ref{sec:experiments}).
"""
