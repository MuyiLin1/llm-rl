import re
P = '/Users/linmuyi/code/DMwithLLM/v7.tex'
s = open(P).read()
def one(x):
    n = s.count(x); assert n == 1, (n, x[:90])
def rep(old, new):
    global s; one(old); s = s.replace(old, new)
def between(a, b, new, keep_b=True):
    """replace text from marker a (inclusive) up to marker b (b kept if keep_b)"""
    global s; one(a); one(b); i = s.index(a); j = s.index(b)
    assert i < j, a[:40]; s = s[:i] + new + (s[j:] if keep_b else s[j+len(b):])

# ---------- 1. abstract ----------
between(r"""\begin{abstract}""", r"""\end{abstract}""", r"""\begin{abstract}
Routing the early rounds of a linear contextual bandit to a frozen LLM and
then handing control to LinUCB can beat both, sometimes by a large factor,
and the LLM's accuracy does not predict when. We explain why. The expert's
routed rounds reach the learner in two ways: through \emph{coverage}, the
directions its actions explore, and through the \emph{transfer}, what its
observed rewards say beyond that coverage. Coverage alone cannot produce
large gains when LinUCB explores as the theory requires: on any instance with
a fixed finite arm set in which the best arm is always available, the saving
it certifies over cold LinUCB is capped by the ratio of the prices the two
pay for coverage, and the cap is attained. Under linear rewards with Gaussian
noise, experts of equal coverage are interchangeable. The transfer is not
capped. When LinUCB explores less than the theory requires, as it does in
practice, we construct two experts with identical coverage, cost and number
of right answers, the losing one possibly the more accurate, whose
multipliers over cold LinUCB differ in most runs by a factor that grows with
the horizon, and in expectation once the expert gives about $\log T$ right
answers. On AmazonCat-13K, three LLM experts of similar, low accuracy
multiply the router's accuracy over cold LinUCB by $1.71$, $2.72$ and $3.29$,
cutting its regret by $6.5$ to $21\%$. Coverage alone saves about $2$ to
$3.5\%$ of cold LinUCB's regret for each; the transfer, which steers the
learner to the most frequent category, carries the rest and every
significant difference between experts, and hiding the rewards of a handful
of right answers on that category, while keeping their coverage, removes
their gain. Information value is not policy value.
""")

# ---------- 2. intro ----------
rep(r"""to $21\%$), and the least accurate is not the least useful. This paper explains where the gain comes from, bounds
it, and shows why accuracy cannot predict it.""",
r"""to $21\%$), and the least accurate is not the least useful. This paper
explains where the gain comes from and why accuracy cannot predict it.""")
between(r"""\paragraph{Coverage has a price (main result).}""", r"""\paragraph{What the data shows.}""", r"""\paragraph{Coverage has a price.}
With a valid confidence radius, LinUCB's regret is limited by the confidence
widths of actions it would otherwise mistake for the best, and widths shrink
only with coverage. Coverage is never free: the \emph{price} of a direction is
the least regret any action pays per unit of coverage of it, and cold LinUCB
buys the same coverage on its own, at the price of the arms its optimism
forces it to pull. Theorem~\ref{thm:general-price} holds for LinUCB with a
valid confidence radius, on any instance with a fixed finite arm set in which
the best arm is always available: the multiplier of any policy over cold
LinUCB is at most the ratio of cold LinUCB's price to the expert's price on
the arms the warm start certifies, plus cold LinUCB's regret on the arms it
does not certify. It thus caps the certified saving rather than the
multiplier outright, and a large certified saving requires an action of small
gap that covers a direction cold exploration can buy only dearly
(Corollary~\ref{cor:diverge}). The cap is attained (Theorem~\ref{thm:price}),
and a catalog of eleven instances (Appendix~\ref{app:catalog-statements})
shows that removing any one of recurrence, cheapness, correct targeting, or
the expert's suboptimality on the tail can remove the gain. Under valid
exploration, then, coverage multiplies only through cheap informative
actions, and it cannot explain experts of similar accuracy and similar
coverage whose multipliers differ severalfold.

\paragraph{The transfer decides (main result).}
Under linear rewards with Gaussian noise, and for an expert whose choices do
not depend on the noise, the response sum is determined in law by the Gram
matrix (Proposition~\ref{prop:response-sum}), so experts that cover the same
directions are interchangeable (Corollary~\ref{cor:coverage-complete}). This
fails in practice for two reasons: rewards are not linear in the features,
and an LLM reads text the features only summarize, so it can prefer an action
on exactly the rounds where that action pays. Either way the estimate at
handoff acquires a transfer term (Definition~\ref{prop:transfer}). Our main
result (Theorem~\ref{thm:transfer-separation}) shows that when LinUCB explores
less than the theory requires, as it does in practice, the transfer can
decide everything: two experts with identical coverage, cost and number of
right answers on the routed rounds can have multipliers over cold LinUCB that
differ, in most runs, by a factor growing with the horizon, according to
where their right answers fall, and the losing expert can be the more
accurate one. A handful of right answers on an action that pays often settles
the learner on it, with a failure probability that decays exponentially in
their number; about $\log T$ of them make the separation hold in expectation.
With a valid radius the effect disappears. The two theorems therefore
describe two regimes: when exploration is valid, coverage governs and its
value is capped by prices; when exploration is reduced, as in practice, the
transfer can exceed that cap.

""")
between(r"""\paragraph{Contributions.}""", r"""\section{Related Work}
\label{sec:related-main}""", r"""\paragraph{Contributions.}
\begin{itemize}[itemsep=1pt, topsep=1pt, leftmargin=*]
\item \textbf{The transfer can decide everything}
(Theorem~\ref{thm:transfer-separation}): with reduced exploration, two
experts equal in coverage, cost and number of routed right answers can differ
by a factor of order the horizon, and the more accurate one can lose.
\item \textbf{Coverage is capped by prices}
(Theorems~\ref{thm:general-price} and~\ref{thm:price},
Corollary~\ref{cor:coverage-complete}): with a valid radius the certified
saving of any policy at any cutoff is bounded by a price ratio, the bound is
attained, and under linear rewards with Gaussian noise coverage is all there
is; on the canonical instance an offline warm start multiplies by a factor
of the horizon while an online handoff inherits nothing
(Propositions~\ref{prop:three-way} and~\ref{prop:one-pass}).
\item \textbf{Experiments that isolate the transfer}
(Section~\ref{sec:experiments}): on two datasets and twelve configurations,
with every tuning parameter chosen on seeds disjoint from the evaluation
seeds, coverage gives a small gain that is the same for every expert, the
transfer carries most of the gain and every significant difference between
experts, and right answers on the most frequent category double the
multiplier only when their rewards reach the learner.
\end{itemize}
Section~\ref{sec:related-main} discusses related work,
Section~\ref{sec:setup} fixes notation, Section~\ref{sec:decomp} gives the
accounting, Sections~\ref{sec:price}--\ref{sec:two-channels} the theory, and
Section~\ref{sec:experiments} the experiments. The proof of
Theorem~\ref{thm:general-price} is in Section~\ref{sec:price}; the catalog
and the other proofs are in the appendix.

""")

# ---------- 3. related work citations ----------
rep(r"""\todo{cite Song et al.\ 2023, Hybrid RL; Wagenmaker \& Pacchiano 2023,
Leveraging offline data in online RL}""", r"""\cite{song2022hybrid,wagenmaker2023leveraging}""")
rep(r"""\todo{cite Hao et al.\ 2023, Leveraging demonstrations
to improve online learning: quality matters}""", r"""\cite{hao2023leveraging}""")
rep(r"""\todo{cite Wu et al.\ 2016,
Conservative bandits}""", r"""\cite{wu2016conservative}""")

# ---------- 4. section 4 / 5 wording ----------
rep(r"""Everything in this paper is organized by one identity.""", r"""The catalog and the experiments are organized by one identity.""")
rep(r"""That exchange rate is the price. The main theorem says that no""", r"""That exchange rate is the price. The theorem below says that no""")
rep(r"""The lemma is one half of the main theorem; the other is that cold LinUCB""", r"""The lemma is one half of the theorem; the other is that cold LinUCB""")
rep(r"""\paragraph{Setting for the theorem.}
""", r"""\paragraph{Setting for the theorem.}
In words: we fix a finite set of arms in which the best arm is always
offered, compare cold LinUCB with an arbitrary policy $\pi$ up to a cutoff
$t_0$, and measure how much coverage $\pi$ bought, at what price, and which
arms that coverage lets the confidence bound rule out.
""")
rep(r"""(zero when the optimal arm is orthogonal to the certified directions, as on
every instance of Section~\ref{sec:catalog}),""", r"""(zero when the optimal arm is orthogonal to the certified directions, as on
every instance of the catalog),""")
# S -> \mathcal S for the set of suboptimal arms (S also bounds ||theta*||)
rep(r"""$S:=\{x\in\mathcal X:\Delta(x)>0\}$""", r"""$\mathcal S:=\{x\in\mathcal X:\Delta(x)>0\}$""")
a = s.index(r"""\paragraph{Setting for the theorem.}"""); b = s.index(r"""\begin{corollary}[Diverging online multipliers""")
blk = s[a:b]
blk2 = re.sub(r'\\in S(?=[\\$ }])', r'\\in \\mathcal S', blk).replace(r'\subseteq S$', r'\subseteq \mathcal S$')
assert blk2.count(r'\mathcal S') == 11, blk2.count(r'\mathcal S')
s = s[:a] + blk2 + s[b:]
rep(r"""\kappa^{\Cold}_U\le\max_{x\in S}\kappa_x""", r"""\kappa^{\Cold}_U\le\max_{x\in\mathcal S}\kappa_x""")

# ---------- 5. move Theorem 2 + catalog to the appendix ----------
i = s.index(r"""\paragraph{The price ratio is attained.}
All instances below"""); j = s.index(r"""\section{Coverage and the Transfer}""")
moved = s[i:j]
s = s[:i] + r"""\paragraph{The price ratio is attained.}
Theorem~\ref{thm:general-price} is tight up to constants. We modify the
canonical instance so that directions recur and, for a while, a cheap
informative action is available: a \emph{probe} of gap $\eps$ that covers a
bad direction at rate $\gamma^2$ per round. The hybrid that routes the probes
and then hands off beats cold LinUCB by a factor between
$\gamma^2/(2220\,\eps L^2)$ and $1+\gamma^2/(\eps L^2)$
(Theorem~\ref{thm:price}, Appendix~\ref{app:template-thm}), the ratio of cold
exploration's price $1/L^2$ to the probe's price $\eps/\gamma^2$. A catalog of
eleven instances built the same way (Table~\ref{tab:catalog},
Appendix~\ref{app:catalog-statements}) shows that removing any one of
recurrence, cheapness, correct targeting, or the expert's suboptimality on
the tail can remove the gain: across these linear instances a gain over cold
LinUCB appears only when the expert buys the directions cold LinUCB would
otherwise pay for, more cheaply and early enough for them to recur.

""" + s[j:]
moved = moved.replace(r"""\paragraph{The price ratio is attained.}
All instances below""", r"""\subsection{The Template, and the Price Ratio Attained}
\label{app:template-thm}

All instances below""")
moved = moved.replace(r"""We work under the standing conventions \eqref{eq:conventions} of
Appendix~\ref{app:catalog-statements}:""", r"""We work under the standing conventions \eqref{eq:conventions} of
Appendix~\ref{sec:template}:""")
moved = moved.replace(r"""(Proposition~\ref{prop:cat-tail-saving},
row~$5'$, Appendix~\ref{app:catalog-statements}).""", r"""(Proposition~\ref{prop:cat-tail-saving},
row~$5'$, below).""")
assert r'\label{app:template-thm}' in moved and r'Appendix~\ref{sec:template}' in moved
k = s.index(r"""\paragraph{Row 3: an oracle expert on the canonical instance.}""")
s = s[:k] + moved + s[k:]
rep(r"""\section{The Catalog: Full Statements}
\label{app:catalog-statements}""", r"""\section{The Price Ratio Attained, and the Catalog}
\label{app:catalog-statements}""")
rep(r"""only summarizes. The canonical instance (rows 1--2) and the template are
defined in Section~\ref{sec:catalog}; proofs are in Appendix~\ref{app:catalog}.""",
r"""only summarizes. The canonical instance (rows 1--2) is defined in
Section~\ref{sec:catalog} and the template in Appendix~\ref{app:template-thm}
below; proofs are in Appendix~\ref{app:catalog}.""")
rep(r"""Every claim in Section~\ref{sec:catalog} is a statement about an explicitly""",
r"""Every claim of the catalog (Section~\ref{sec:catalog} and
Appendix~\ref{app:catalog-statements}) is a statement about an explicitly""")

# ---------- 6. section 7 additions ----------
rep(r"""constant radius $\beta>0$; we write $\omega:=\beta L/\sqrt\lambda$ for the
index of an untried fresh arm, and break ties in favour of the fresh arm.""",
r"""constant radius $\beta>0$; we write $\omega:=\beta L/\sqrt\lambda$ for the
index of an untried fresh arm, and break ties in favour of the fresh arm.
The two experts are built so that everything coverage can see is the same:
each plays $D$ and $R$ exactly $2k$ times in the window, pays the same
regret, and is right exactly $k$ times. Their wrong answers are padding that
makes these totals equal, so the only difference left is which action their
right answers fall on.""")
rep(r"""answers make that fraction small enough for the separation to hold in
expectation as well, at order $T/\log T$.""", r"""answers make that fraction small enough for the separation to hold in
expectation as well, at order $T/\log T$.

That a learner which explores too little can lock onto whatever paid first
is known: greedy and near-greedy linear bandits can succeed or fail
depending on the data \cite{bastani2021mostly}, and the value of
demonstrations to an online learner depends on the demonstrator's quality
\cite{hao2023leveraging}. What Theorem~\ref{thm:transfer-separation} adds is
that, for an online warm start, lock-in makes the value of an expert's data
depend on where its right answers fall rather than on its coverage, its cost
or its accuracy: two experts equal in all three differ by a factor of order
the horizon.""")
rep(r"""arms serve only to make cold LinUCB fail and to make leaving $D$ permanent;
$d=T+2$ is what they cost.
\end{remark}""", r"""arms serve only to make cold LinUCB fail and to make leaving $D$ permanent;
$d=T+2$ is what they cost.
\end{remark}

\begin{remark}[The instance as a cartoon of AmazonCat]
\label{rem:transfer-amazon}
Each piece of the instance has a counterpart in Section~\ref{sec:amazon}.
The action $D$ plays the role of the category present on $29.8\%$ of
AmazonCat contexts, and $R$ that of a rarer category. The fresh arms stand for
the thousands of categories cold LinUCB keeps trying because their widths
are large, which is why it settles on the dominant category in only $1$ of
$20$ seeds. The number $k$ of right answers on $D$ corresponds to the number
of rewarded routed answers on the dominant category: about $0.2$ per run for
Flan-T5-small and $3.7$ and $4.0$ for Flan-T5-base and Flan-T5-large. The
all-or-nothing settling of part~(iv) and its steep dependence on $k$ appear
in Tables~\ref{tab:settle} and~\ref{tab:dose}. The instance is not fitted to
the data; it isolates the mechanism the data display.
\end{remark}""")

# ---------- 7. section 8: split paragraph, soften ----------
between(r"""\paragraph{What the transfer carries.} On AmazonCat one category is present on""",
        r"""\paragraph{BGE features and other routers.}""", r"""\paragraph{The dominant category.} On AmazonCat one category is present on
$29.8\%$ of contexts, so a policy that knew this in advance and always
predicted it would score $29.8\%$. An online learner does not know it, and
finding it is not easy: context-free bandits that try categories without
using the context reach $0.04\%$ (UCB1) and $2.31\%$ (optimistic greedy) over
the same $50{,}000$ rounds and $13{,}330$ categories, seeds $1$--$10$. The
Routers that multiply are the ones that discover this category. LinUCB
settles on one frequent category and predicts it on a large share of rounds,
and its accuracy then tracks that category's frequency: runs that settle on
the dominant category finish near $28\%$, while cold LinUCB settles on a
category present on $11\%$ of contexts, or on none, in $18$ of $20$ seeds.
Whether a Router settles on the dominant category is decided by a handful of
the expert's routed answers. Across $177$ Router runs on BERT features (every
$c_{\mathrm{LLM}}$ and CORRAL, excluding configurations that route most rounds), a run whose
expert gave no correct routed answer on this category reached it in $3$ of
$45$ runs, and a run that received five or more reached it in all $54$.

\paragraph{A controlled test.} Adding $15$ correct answers on this category
to Flan-T5-small ($+0.15$ points of held-out accuracy) doubled its multiplier
($1.33\times$ to $2.62\times$, paired $p=0.002$), while $15$ correct answers on
random categories gave no significant gain ($1.65\times$, $p=0.15$), and the
same $15$ answers with their rewards hidden from the learner, so that they add
coverage along the category but no transfer, gave none either ($1.48\times$,
$p=0.47$; against the same answers with their rewards, $p=0.003$). Only about
two of the $15$ changed answers reach the learner in a typical run, because
the Router consults the expert on few rounds, almost all of them early: two
rewarded answers on the dominant category are enough to double the
multiplier, and the same two answers with their rewards hidden do nothing.

\paragraph{On the evaluation stream.} The same test on the $50{,}000$-round
stream and seeds of Table~\ref{tab:amazon-results} gives the same result:
$75$ added answers on this category raise the multiplier from $1.71\times$ to
$2.56\times$ ($p=0.007$), and with their rewards hidden only to $1.84\times$
($p=0.61$; Table~\ref{tab:inject50k}). This is the pattern of
Theorem~\ref{thm:transfer-separation}. As in its part~(ii), the routed
covariance without the routed rewards buys nothing; every correct answer
carries a residual near $1$, but only a handful on a frequent action push the
estimate in a direction that keeps paying, settling is all or nothing, and
its probability rises steeply with the number of such answers.
Appendix~\ref{app:dominant} gives the details.

""")
rep(r"""With Qwen2.5 experts the expert
is better than the learner ($29$--$39\%$ against $19\%$), so $B_\tau<0$ and no
handoff can beat the expert (row~3 of Table~\ref{tab:catalog}).""",
r"""With Qwen2.5 experts the expert
is already better than the learner ($29$--$39\%$ against $19\%$), so
$B_\tau<0$ and the Router has little to add over the expert, as in row~3 of
Table~\ref{tab:catalog}.""")
rep(r"""On WikiLinks the expert is
genuinely better than the learner---row 3 of Table~\ref{tab:catalog}, where no
combination can beat the better component.}""", r"""On WikiLinks the expert is
already better than the learner, as in row~3 of Table~\ref{tab:catalog}, and the
Router has little to add.}""")
rep(r"""$18.88\%$), which is row~3 of Table~\ref{tab:catalog}: an expert at or above the
learner's ceiling, where $B_\tau\le0$ and no combination can beat the better
component.""", r"""$18.88\%$), as in row~3 of Table~\ref{tab:catalog}: with the expert above the
learner, $B_\tau\le0$ and the Router has little to add over the expert.""")
rep(r"""learner ($29$--$39\%$ against $19\%$), so $B_\tau<0$ for every $\tau$ and no
handoff can beat the expert.""", r"""learner ($29$--$39\%$ against $19\%$), so $B_\tau<0$ and the Router has
little to add over the expert.""")
rep(r"""explain the experiments, not to certify them. A seed fixes the order of the""",
r"""explain the experiments, not to certify them.
\todo{Phase B: add the tuned cold-LinUCB baselines (exploration schedule and
floor, $\lambda$, category-term features) and the Router re-tuned against them.}
A seed fixes the order of the""")

# ---------- 8. limitations ----------
rep(r"""certified saving rather than the multiplier outright. The constructions use a""",
r"""certified saving rather than the multiplier outright. It also counts cold
LinUCB's regret arm by arm, so with many arms that share directions, as on
AmazonCat, the residual is large and the bound is loose: it explains why
coverage cannot produce large gains under valid exploration rather than
predicting the size of a gain. The constructions use a""")

# ---------- 9. small errors ----------
rep(r"""$\widehat\theta_t=V_t^{-1}b_t$ lies in $\mathrm{span}\{e_1,\dots,e_t\}$.""", r"""$\widehat\theta_t=V_t^{-1}c_t$ lies in $\mathrm{span}\{e_1,\dots,e_t\}$.""")
rep(r"""$V_t$ is diagonal with entry $\lambda+s_t$ on $e_1$ and the rewards on bad
arms enter $b_t$ only in directions orthogonal to $e_1$,""", r"""$V_t$ is diagonal with entry $\lambda+s_t$ on $e_1$ and the rewards on bad
arms enter $c_t$ only in directions orthogonal to $e_1$,""")
rep(r"""The hybrid's LinUCB phase uses $(V_{\tau+1},b_{\tau+1})$ built from the""", r"""The hybrid's LinUCB phase uses $(V_{\tau+1},c_{\tau+1})$ built from the""")
rep(r"""response vector $b_t$ is $L\sum r$ over the pulls of $b$ in direction $e_j$""", r"""response vector $c_t$ is $L\sum r$ over the pulls of $b$ in direction $e_j$""")
rep(r"""$(V_{\tau+1},b_{\tau+1})$ coincide. From round $\tau+1$ on both run LinUCB""", r"""$(V_{\tau+1},c_{\tau+1})$ coincide. From round $\tau+1$ on both run LinUCB""")
rep(r"""round-level update protocol for $(V_t,b_t)$.""", r"""round-level update protocol for $(V_t,c_t)$.""")
rep(r"""$R_{[1,\tau]}(\Exp)>\pi(e_j)\,(4\beta^2L^2-\lambda)$""", r"""$R_{[1,\tau]}(\Exp)>\kappa(e_j)\,(4\beta^2L^2-\lambda)$""")
rep(r"""($m=1$), the LLM, with $V^{(1)}=cI_d$, and do not use the permanent handover:""", r"""($m=1$), the LLM, with $V^{(1)}=c_{\mathrm{LLM}}I_d$, and do not use the permanent handover:""")

open(P, 'w').write(s); print('Phase A edits applied')
