p='restructure/v7_new.tex'; s=open(p).read()
def between(a,b):
    assert s.count(a)==1,a[:60]; i=s.index(a); j=s.index(b,i); return s[i:j]
def swap(a,b,new):
    global s
    old=between(a,b); assert s.count(old)==1; s=s.replace(old,new)

swap('\\paragraph{Coverage has a price (main result).}','\\paragraph{Beyond coverage: the transfer.}', r"""\paragraph{Coverage has a price (main result).}
Data helps LinUCB only by shrinking the confidence widths of actions it would
otherwise mistake for the best, and widths shrink only with coverage.
Coverage is never free: the \emph{price} of a direction is the least regret
any action pays per unit of coverage of it, and cold LinUCB buys the same
coverage on its own, at the price of the arms its optimism forces it to pull.
Our main theorem (Theorem~\ref{thm:general-price}) bounds the multiplier of
any policy over cold LinUCB, on any instance with a fixed finite arm set in
which the best arm is always available, by the ratio of cold LinUCB's price
to the expert's price on the directions the warm start certifies, plus cold
LinUCB's regret on the directions it does not certify. A diverging multiplier
therefore requires an action of vanishing gap that covers a direction cold
exploration can buy only dearly (Corollary~\ref{cor:diverge}). The bound is
attained (Theorem~\ref{thm:price}), and a catalog of eleven instances
(Table~\ref{tab:catalog}) shows that removing any one of recurrence,
cheapness, correct targeting, or the expert's suboptimality on the tail can
remove the gain.

""")
swap('\\paragraph{Beyond coverage: the transfer.}','\\paragraph{What the data shows.}', r"""\paragraph{Beyond coverage: the transfer.}
Under linear rewards, and for an expert that sees only what the learner
sees, the response sum is determined in law by the Gram matrix
(Proposition~\ref{prop:response-sum}), so experts that cover the same
directions are interchangeable (Corollary~\ref{cor:coverage-complete}). This
prediction fails in practice for two reasons: rewards are not linear in the
features, and an LLM reads text the features only summarize, so it can prefer
an action on exactly the rounds where that action pays. Either way the
estimate at handoff acquires a transfer term, in closed form
(Proposition~\ref{prop:transfer}), and the transfer can decide everything:
two experts with identical coverage, price and accuracy can have multipliers
over cold LinUCB that differ by a factor growing with the horizon, according
to where their right answers fall (Theorem~\ref{thm:transfer-separation}). A
handful of right answers on an action that pays often settles the learner on
it, with a failure probability that decays exponentially in their number.

""")
swap('\\item \\textbf{A price bound for the online multiplier}','\\item \\textbf{Coverage and the transfer}', r"""\item \textbf{A price bound for the online multiplier}
(Theorem~\ref{thm:general-price}, Corollary~\ref{cor:diverge}), on any
instance with a fixed finite arm set in which the best arm is always
available, for any policy and any cutoff, so that it covers fixed and
adaptive handoffs and randomized masters alike.
\item \textbf{Instances attaining it, and a catalog}
(Theorem~\ref{thm:price}, Table~\ref{tab:catalog}): the bound is attained
with the saving collected during or after the routed rounds, eleven instances
with complete proofs isolate which ingredient each negative case lacks, and
on the canonical instance an offline warm start multiplies by a factor of the
horizon while no online handoff gains anything
(Propositions~\ref{prop:three-way} and~\ref{prop:one-pass}).
""")
swap('The proof is in Appendix~\\ref{app:proof-general-price}; part (a)','\\paragraph{The two parts of the bound.}', r"""The proof is in Appendix~\ref{app:proof-general-price}: part (a) is
Lemma~\ref{lem:width-limited} combined with the Sherman--Morrison identity,
(b) is Lemma~\ref{lem:no-free-coverage} summed over the certified directions,
and (c) is arithmetic. Only the confidence event of cold LinUCB is used, and
$\pi$ is arbitrary, so the bound covers the fixed-$\tau$ hybrid, the adaptive
router (take $t_0$ to be its last routed round), a CORRAL-style master, and
cold LinUCB itself. It bounds the ratio of the two policies' regrets on the
same instance, not one upper bound against another: two policies with the
same asymptotic rate can still differ by a diverging factor, so a statement
about the multiplier must bound one policy's regret in terms of the other's,
as (c) does.

""")
swap('Two experts that buy exactly the same coverage','\\begin{remark}[Where the theorem sits]', r"""Two experts that buy exactly the same coverage, at exactly the same price,
with exactly as many right answers, give warm starts whose multipliers over
cold LinUCB differ by a factor of order $T/k$. Nothing that depends on
coverage alone---the design matrix, the widths, the prices, the bound of
Theorem~\ref{thm:general-price}---can tell them apart, and neither can their
accuracy. What separates them is where their right answers fall:
$\Exp_A$'s are on an action right on a fraction $p$ of later rounds,
$\Exp_B$'s on one right on a fraction $q$. The number of right answers needed
does not grow with the horizon, and the learner settles all or nothing: once
$\Hyb^A_\tau$ leaves $D$ it never returns. Making $\Exp_B$ strictly more
accurate than $\Exp_A$ changes none of this
(Remark~\ref{rem:transfer-extensions}, Appendix~\ref{app:proof-transfer-sep}).

""")
swap('The transfer can help or hurt. Proposition~\\ref{prop:transfer-regret}','\\paragraph{How far the transfer can go.}', r"""The transfer can help or hurt: Proposition~\ref{prop:transfer-regret}
(Appendix~\ref{app:extra-theory}) shows that it helps exactly when it moves the
estimate toward a reference whose residual is smaller on the best action than
on the actions LinUCB would otherwise play.

""")
swap('\\paragraph{Equal coverage, unequal value.}\nThe three Flan-T5 experts','\\paragraph{What the transfer carries.}', r"""\paragraph{Equal coverage, unequal value.}
The three Flan-T5 experts have held-out accuracies $5.20\%$, $3.61\%$ and
$5.27\%$ and deliver accuracy multipliers of $1.71\times$, $2.72\times$ and
$3.29\times$ (regret ratios $1.07$, $1.19$ and $1.27$): accuracy orders them
differently from the multiplier. The decomposition locates the difference:
$A_\tau$ is $47$, $27$ and $53$ correct answers and $V_\tau$ is $2{,}936$,
$7{,}178$ and $9{,}571$ (Table~\ref{tab:amazon-decomp}), and $96$--$100\%$ of
each Router's advantage accrues after its last routed round. The experts
cover the directions cold LinUCB is trapped on to within $5\%$, at nearly the
same prices, so their post-hoc price ratios ($1.73$, $1.59$, $1.58$) are equal
within measurement error, ordered opposite to the multipliers, and above
every observed regret ratio. By Corollary~\ref{cor:coverage-complete} the
separation must come from the transfer, and Figure~\ref{fig:two-channel}
shows that it does: feeding the learner $\langle\theta_{\mathrm{or}},\phi\rangle$
instead of the observed reward on routed rounds only, which deletes $m$ of
Proposition~\ref{prop:transfer} with $\theta^\star=\theta_{\mathrm{or}}$, turns
every significant pairwise difference into a null result (large over small:
paired $t=5.19$ becomes $0.62$); dropping the response sum gives the same
collapse; and a linear reward makes the experts interchangeable, as
Proposition~\ref{prop:response-sum} predicts. On the regret scale coverage
alone saves about $5\%$ of cold LinUCB's regret for every expert, while the
full router saves $6.5\%$, $16\%$ and $21\%$ (Appendix~\ref{app:more-exp}).

""")
swap('\nThe bracket \\eqref{eq:price-ratio} is the price ratio. Cold exploration buys\ncoverage $L^2$ of a bad direction per unit of regret, so its price is\n$1/L^2$; the probe','\\paragraph{The catalog.}', r"""
The bracket \eqref{eq:price-ratio} is the price ratio: cold exploration buys
coverage $L^2$ of a bad direction per unit of regret (price $1/L^2$), the
probe buys $c^2$ per $\eps$ (price $\eps/c^2$), and the multiplier over cold
LinUCB is their ratio $c^2/(\eps L^2)$ up to constants. The multiplier over
the expert, $(T-\tau_0)/(\eps\tau_0)$, diverges as the tail lengthens. Here
the saving is collected during the routed rounds, as $A_{\tau_0}$; keeping
cold LinUCB busy elsewhere during the window moves it after them, into
$V_{\tau_0}$, with the same price ratio (Proposition~\ref{prop:cat-tail-saving},
row~$5'$, Appendix~\ref{app:catalog-statements}).

""")
swap('The decomposition and the price bound hold for any routed set, so they apply\nto the randomized CORRAL master','\\section{Conclusion}', r"""The decomposition and the price bound hold for any routed set, so they apply
to the randomized CORRAL master \cite{agarwal2017corralling} used by
\cite{chen2024llmdecision} as to the gate. Against CORRAL and offline warm
starts of $1{,}600$ and $5{,}000$ free expert answers on AmazonCat/BERT
(Table~\ref{tab:baselines}, Appendix~\ref{app:more-exp}), no alternative is
significantly better on any expert; the gate beats CORRAL on Flan-T5-small
($t=2.64$) with $16\times$ fewer expert queries, and where an alternative comes
out ahead in the mean, the margin is not significant and costs $3$--$5\times$
as many expert answers.

""")
open(p,'w').write(s); print('ok')
