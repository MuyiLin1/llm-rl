# New prose, part 3: experiments, conclusion, appendix glue
EXP_HEAD = r"""\section{Experiments}
\label{sec:experiments}

Corollary~\ref{cor:coverage-complete} predicts that experts of equal coverage
leave warm starts of equal value unless the transfer intervenes, and
Theorem~\ref{thm:transfer-separation} predicts how it can intervene. We test
both on real data, running the router, the LLM alone, and cold LinUCB on the
same stream; the setup mirrors \cite{chen2024llmdecision}.

"""

AMAZON_HEAD = r"""\subsection{AmazonCat-13K}
\label{sec:amazon}

AmazonCat-13K has $13{,}330$ categories and $T=50{,}000$ rounds. We pair
BERT-768d features with Flan-T5 experts and BGE-1024d features with Qwen2.5
experts (Table~\ref{tab:amazon-results}).

"""

AMAZON_RESULTS_CAPTION = r"""\caption{AmazonCat-13K, $T=50{,}000$, mean $\pm$ standard deviation over
seeds $1$--$10$. Acc.\ mult.\ is the seed-averaged Router accuracy over the
seed-averaged $\max\{$LLM, cold$\}$; the regret ratio is
$(1-\max\{\text{LLM},\text{cold}\})/(1-\text{Router})$, the quantity
Theorem~\ref{thm:general-price} bounds. Cold LinUCB does not depend on the
expert. BERT rows are at $c=100$ and BGE rows at $c=0.1$, each selected on
the disjoint seeds $11$--$20$ (Appendix~\ref{app:more-exp}). Held-out
accuracy orders the BERT experts differently from the multiplier; on BGE the
two larger experts gain significantly (paired $t=3.92$ and $2.59$ against
cold) and the smallest does not ($t=0.01$).}
"""

EQUAL_COVERAGE = r"""\paragraph{Equal coverage, unequal value.}
The three Flan-T5 experts have held-out accuracies $5.20\%$, $3.61\%$ and
$5.27\%$ and deliver accuracy multipliers of $1.71\times$, $2.72\times$ and
$3.29\times$ (regret ratios $1.07$, $1.19$ and $1.27$): accuracy orders them
differently from the multiplier. The decomposition locates the difference
(Table~\ref{tab:amazon-decomp}, Appendix~\ref{app:more-exp}): $A_\tau$ is
$47$, $27$ and $53$ correct answers, while $V_\tau$ is $2{,}936$, $7{,}178$ and
$9{,}571$, and $96$--$100\%$ of each Router's advantage over cold LinUCB
accrues after its last routed round. The three experts cover the directions
cold LinUCB is trapped on to within $5\%$ ($1.40$, $1.46$ and
$1.47\times10^5$), at nearly the same prices, so the price ratio, measured
post hoc as described in Section~\ref{sec:exp-setup}, is $1.73$, $1.59$ and
$1.58$: equal within measurement error, ordered opposite to the multipliers,
and above every observed regret ratio. By
Corollary~\ref{cor:coverage-complete} the separation must come from the
transfer, and Figure~\ref{fig:two-channel} shows that it does. Feeding the
learner $\langle\theta_{\mathrm{or}},\phi\rangle$ instead of the observed
reward on routed rounds only, which deletes $m$ of
Proposition~\ref{prop:transfer} with $\theta^\star=\theta_{\mathrm{or}}$ and
changes nothing else, collapses the spread of per-seed multipliers from
$1.74$ to $0.56$ and turns every significant pairwise difference into a null
result (large over small: paired $t=5.19$ becomes $0.62$). Dropping the
response sum altogether gives the same collapse, and replacing the reward by
a linear one makes the experts interchangeable, as
Proposition~\ref{prop:response-sum} predicts. On the regret scale, coverage
alone saves about $5\%$ of cold LinUCB's regret for every expert, while the
full router saves $6.5\%$, $16\%$ and $21\%$: most of the saving, and all of
the difference between experts, is the transfer
(Appendix~\ref{app:more-exp}).

"""

CARRIES_EXTRA = r""" This is the pattern
Theorem~\ref{thm:transfer-separation} predicts: a handful of right answers on
a frequent action, settling that is all or nothing, and a probability of
settling that rises steeply with the number of such answers."""

BGE_SHORT = r"""\paragraph{BGE features.}
With BGE-1024d features and Qwen2.5 experts cold LinUCB is much stronger
($32.59\%$) and settles on the dominant category by itself in $7$ of $20$
seeds, so there is less to add. At $c=0.1$, selected on the tuning seeds, the
Router reaches $40.20\%$ with Qwen-3B and $38.71\%$ with Qwen-7B (accuracy
multipliers $1.23\times$ and $1.19\times$, paired $t=3.92$ and $2.59$) while
routing $0.05\%$ of rounds; the saving is again almost all $V_\tau$
($3{,}801$ and $3{,}056$ correct answers, against $A_\tau$ of $5$ and $6$).
Qwen-0.5B, consulted just as often, does not move the Router ($1.00\times$).
At $c=100$ the gate never engages and the Router equals the LLM; the
confidence sweep is in Appendix~\ref{app:more-exp}.

"""

WIKI_SHORT = r"""\subsection{OneShotWikiLinks-311}
\label{sec:wiki}

On OneShotWikiLinks-311 ($622{,}000$ rounds, $311$ entities) the most
frequent entity covers $0.36\%$ of contexts, so there is no category for a
warm start to steer the learner toward. With Flan-T5 experts the Router
gains $0.13$--$0.17$ points over cold LinUCB ($1.01\times$), although the
routed data does reach useful directions ($V_\tau$ is $748$, $832$ and $886$
correct answers and positive on every seed). With Qwen2.5 experts the expert
is better than the learner ($29$--$39\%$ against $19\%$), so $B_\tau<0$ and no
handoff can beat the expert (row~3 of Table~\ref{tab:catalog}).
\todo{WikiLinks sweep} Full results are in Appendix~\ref{app:more-exp}.

\subsection{Other Routers}
\label{sec:other-routers}

The decomposition and the price bound hold for any routed set, so they apply
to the randomized CORRAL master \cite{agarwal2017corralling} used by
\cite{chen2024llmdecision} as to the gate. Table~\ref{tab:baselines}
(Appendix~\ref{app:more-exp}) compares the gate with CORRAL and with offline
warm starts of $1{,}600$ and $5{,}000$ free expert answers on AmazonCat/BERT.
No alternative is significantly better on any expert. The gate beats CORRAL
on Flan-T5-small ($t=2.64$) with $16\times$ fewer expert queries, and where an
alternative comes out ahead in the mean the margin is small, not
significant, and paid for with $3$--$5\times$ as many expert answers. In
every row $V_\tau$ carries the value and $A_\tau$ is where the choice of
router shows up.

"""

CONCLUSION = r"""\section{Conclusion}
\label{sec:conclusion}

Reward multiplication in LLM--bandit hybrids has an exact accounting: the
hybrid beats cold LinUCB by the expert's direct advantage plus the value of
the data it leaves behind, and beats the expert by the learner's advantage on
the tail. The data enters through coverage and through the transfer.
Coverage has a price, and the multiplier it can deliver is bounded by a ratio
of prices, attained on explicit instances and zero on the canonical one.
Under linear rewards, and for an expert that sees only what the learner
sees, coverage is all there is. Real rewards are not linear, and an LLM reads
text that the features only summarize; either way the estimate at handoff
acquires a transfer, which we give in closed form and which can separate
experts of identical coverage and accuracy by a factor of order the horizon.
On AmazonCat-13K this is what happens: coverage saves about $5\%$ of cold
LinUCB's regret for every expert and the full router up to $21\%$, and every
significant difference between three experts of similar accuracy disappears
when the transfer is removed, because a handful of the experts' right answers
on the most frequent category decide where the learner settles. A warm-start
expert cannot be chosen by its accuracy, nor by the coverage its actions
provide: what decides its value is which of its answers are right.
Information value is not policy value.

\paragraph{Limitations.}
The price bound assumes linear rewards and a fixed arm set with the best arm
always available. Real streams satisfy these only approximately, and the
price ratio we report on them is measured after the run, as an explanatory
statistic rather than a prediction. The bound has a residual---cold LinUCB's
regret on arms the warm start does not certify---that is not governed by a
price; we have no lower bound showing that it can be large without a large
price ratio, and no lower bound on $R_T(\Hyb_\tau)$ beyond the acquisition
regret, so the theorem characterizes the certified saving rather than the
multiplier outright. The positive instances use a fixed confidence radius
and a regularization scaled to the dimension, and the catalog's constants are
not optimized. Theorem~\ref{thm:transfer-separation} needs exploration below
a valid confidence radius, as in our experiments; with a valid radius
coverage governs. It compares two experts that both see the reward; how an
informed expert compares with one that sees only the features is shown only
numerically (Appendix~\ref{app:proof-transfer-sep}). We have no estimator
that predicts the size of the multiplier before the expert is run. The large
AmazonCat multipliers depend on a dominant category that the learner can be
steered to; where no category dominates, as on WikiLinks, we see no
multiplication, and we have not tested other datasets with a skewed label
distribution. Only two real datasets are used; reward models suited to binary
feedback, such as generalized linear bandits, may change the split between
coverage and the transfer; and Lemma~\ref{lem:width-limited} uses the
optimistic selection rule, so extending the analysis to Thompson sampling
needs an analogue of it.

"""

APP_EXTRA_HEAD = r"""\section{Further Theory}
\label{app:extra-theory}

This appendix collects results used in the main text but stated there only in
passing: the two width identities, the regret of the handoff hybrid, the
width-limited bound with the realized estimation error, what the transfer can
buy, two relaxations of Theorem~\ref{thm:general-price}, and a discussion of
when cheap informative actions exist.

\subsection{How Widths Shrink}
\label{app:widths}

"""

APP_PROP4_HEAD = r"""
\subsection{Width-Limited Regret with the Realized Estimation Error}
\label{app:two-objects}

Lemma~\ref{lem:width-limited} uses only the radius. With the realized
estimation error in its place it reads as follows, which shows how the two
objects of a warm start act on it.

"""

APP_PROP7_HEAD = r"""
\subsection{What the Transfer Can Buy}
\label{app:transfer-regret}

Under the model of Section~\ref{sec:two-channels}, the transfer's effect on a
single round is bounded as follows.

"""

APP_PROP7_AFTER = r"""
Proposition~\ref{prop:transfer-regret} is a statement about what the transfer
\emph{can} deliver, not a prediction of what a given expert will deliver: $m$
depends on the residuals at the actions the expert chooses, which are not
known before it is run. Theorem~\ref{thm:transfer-separation} shows how large
the difference between two experts can be on a constructed instance, and
Section~\ref{sec:amazon} measures $m$ on real data, finds that it accounts for
essentially the whole estimate at handoff, and shows that removing it
collapses the differences between experts of similar accuracy.

"""

APP_RELAX_HEAD = r"""
\subsection{Two Relaxations of the Price Bound}
\label{app:relaxations}

"""

APP_PROBES_HEAD = r"""
\subsection{When Do Cheap Probes Exist?}
\label{app:cheap-probes}

"""

APP_CAT_HEAD = r"""\section{The Catalog: Full Statements}
\label{app:catalog-statements}

This appendix states the rows of Table~\ref{tab:catalog} that the main text
only summarizes. The canonical instance (rows 1--2) and the template are
defined in Section~\ref{sec:catalog}; proofs are in Appendix~\ref{app:catalog}.

"""

APP_THM2_DISC_HEAD = r"""
\paragraph{Discussion of Theorem~\ref{thm:price}.}
"""

APP_MORE_EXP_HEAD = r"""\section{Additional Experimental Results}
\label{app:more-exp}

This appendix gives the full results summarized in Section~\ref{sec:experiments}:
the decomposition and learning dynamics on AmazonCat/BERT, the complete
coverage-versus-transfer analysis, the BGE confidence sweep, the WikiLinks
results, a summary across configurations, and the comparison with other
routers.

\subsection{AmazonCat-13K with BERT Features}
\label{app:amazon-bert}

"""
