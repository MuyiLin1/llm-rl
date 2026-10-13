# New prose, part 2: Sections 4-7 connective text
DECOMP_HEAD = r"""\section{Where the Gain Comes From}
\label{sec:decomp}

Everything in this paper is organized by one identity.

"""

DECOMP_REMARK = r"""
\begin{remark}
\label{rem:decomp}
The identities hold pathwise in the reward-table model, in which the $k$-th
pull of an action always receives the same noise draw, so that two policies
can be run on the same noise \cite[\S4.6]{lattimore2020bandit}, and
therefore also in expectation. They also hold when the routed rounds are not
an initial segment: for any set $\mathcal T\subseteq[T]$ of routed rounds,
fixed in advance or chosen adaptively (as by the router of
Appendix~\ref{app:meta-algorithm}), replacing $[1,\tau]$ by $\mathcal T$
gives the same identities, because each side is a difference of sums over a
partition of $[T]$. If the warm start $(M,c)$ is built from data outside the
horizon (an \emph{offline} warm start $\Hyb^{\mathrm{off}}$, LinUCB started
from $(\lambda I+M,c)$), the routed rounds are not charged and the identities
read $R_T(\Cold)-R_T(\Hyb^{\mathrm{off}})=:V^{\mathrm{off}}$ and
$R_T(\Exp)-R_T(\Hyb^{\mathrm{off}})=:B^{\mathrm{off}}$. A multiplier over both
base policies means that $A_\tau+V_\tau$ is almost all of $R_T(\Cold)$ and
$B_\tau$ almost all of $R_T(\Exp)$; every diverging ratio in this paper is
read along a sequence of instances.
\end{remark}

\paragraph{What a warm start can change.}
\label{sec:width}
"""

WIDTH_AFTER = r"""
The gap and the width sit in one expression. A competitor with a large gap
but an even larger width is as dangerous as its gap; one with a tiny gap is
cheap however wide it is; and the only way to lower the regret LinUCB can
incur on a round is to shrink the widths of competitors whose widths exceed
their gaps. Widths shrink in exactly two ways, through a warm start and
through LinUCB's own observations, and both are exact linear algebra
(Lemmas~\ref{lem:loewner} and~\ref{lem:transfer},
Appendix~\ref{app:extra-theory}). One consequence is that the
determinant-ratio bound behind every $\widetilde O(\sqrt{rT})$-type guarantee
cannot certify a multiplier: with $\tau=o(T)$ the warm and cold bounds agree
to first order, so any multiplier must come from competitors whose widths the
warm start pushes \emph{below their gaps} (Proposition~\ref{thm:handoff},
Appendix~\ref{app:extra-theory}).

"""

PRICE_HEAD = r"""\section{The Price Bound}
\label{sec:price}

\paragraph{Why a price.}
By Lemma~\ref{lem:width-limited}, the routed rounds can lower LinUCB's later
regret only by shrinking widths, and widths shrink only where the routed
actions add coverage. Coverage is never free: covering a direction means
playing actions that point along it, and those actions have gaps. Cold
LinUCB buys the same coverage on its own---optimism makes it pull each arm
whose width exceeds its gap until that width falls below the gap---so the
expert and cold LinUCB buy the same thing, and the question is what each pays
per unit. That exchange rate is the price. The main theorem says that no
policy beats cold LinUCB by more than the ratio of the two prices on the
directions it certifies, plus what cold LinUCB pays on the directions nobody
certified.

\paragraph{Prices.}
"""

LEFTOVER = r"""\paragraph{The two parts of the bound.}
The bound has two parts. The price ratio covers the directions the policy
paid to certify. The residual is what nobody paid for, and it has three
pieces: the coverage the policy obtains for free by playing optimal arms
(zero when the optimal arm is orthogonal to the certified directions, as on
every instance of Section~\ref{sec:catalog}), one pull of each certified arm,
and cold LinUCB's regret on the arms whose widths the warm start does not push
below their gaps. Only the last can be large. On those arms the hybrid faces
the same confidence bound as cold LinUCB, so no coverage argument can say
which of the two does better there, and the term cannot be dropped: on the
canonical instance of Section~\ref{sec:catalog} no online expert certifies a
direction that recurs, and all of cold LinUCB's regret is residual
(Corollary~\ref{cor:diverge}). When the directions cold LinUCB would
otherwise pay for are certified, the residual is small and the price ratio is
the bound, and Theorem~\ref{thm:price} shows that it is then attained.
Whether the residual can be large without a large price ratio is open. The
bound holds whether the saving is collected during the routed rounds
($A_\tau$) or after them ($V_\tau$), and the confidence radius cancels in
$P_U$ because it scales $c_x$, the coverage both policies must reach,
identically.

"""

PRICE_TAIL = r"""
The proof is in Appendix~\ref{app:proof-diverge}. Two relaxations of
Theorem~\ref{thm:general-price} and a discussion of when cheap informative
actions exist are in Appendix~\ref{app:extra-theory}.

"""

CATALOG_HEAD = r"""\section{The Bound Is Attained, and Can Be Zero}
\label{sec:catalog}

Theorem~\ref{thm:general-price} is sharp in both directions. On the
canonical separating instance an offline warm start multiplies by a factor of
the horizon while no online handoff gains anything; on a modification of it
the price ratio is attained.

\paragraph{The canonical instance: offline against online.}
\label{sec:canonical}
"""

CANON_MID = r"""
Cold LinUCB fails because at each round the fresh arm $b_t$ has width
$L/\sqrt\lambda$ whatever was played before, while $g$ has width
$1/\sqrt\lambda$ and estimated value $0$, so optimism picks $b_t$ every time.
The offline design matrix $M=mL^2\sum_{j\ge2}e_je_j^\top$ does not teach
LinUCB that $g$ is good; it teaches it that the \emph{alternatives} are not
uncertain, since $\|b_t\|^2_{(\lambda I+M)^{-1}}=L^2/(\lambda+mL^2)$ falls
below the gap once $m$ is moderate. Acquiring that data is not charged.
Online it would be, and the instance is hostile to online acquisition: every
round introduces a direction that no earlier observation touches.

"""

CANON_AFTER = r"""
The condition on $\beta$ is the optimism condition that makes cold LinUCB
fail in the first place. No online router realizes the offline separation on
this instance: a bad expert's data has $V_\tau=0$, and a good expert has
$B_\tau\le0$ because it is already optimal (Corollary~\ref{cor:cat-oracle},
Appendix~\ref{app:catalog-statements}). Online multiplication needs an
expert whose routed actions are cheap \emph{and} cover directions that
recur.

\paragraph{The price ratio is attained.}
"""

CONVENTIONS_SHORT = r"""We work under the standing conventions \eqref{eq:conventions} of
Appendix~\ref{app:catalog-statements}: $L\ge8$, $c\in(0,1]$, $0<\eps<c/L$,
$\lambda\ge\max\{\sigma^2d\,\ell_0,8\sigma^2\ell_1,60\}$ with $\ell_0,\ell_1$
logarithmic in $T/\delta$, and the constant radius $\beta=2\sqrt\lambda$, in
the reward-table noise model of Remark~\ref{rem:decomp}; we write
$m_0:=\lfloor\lambda/30\rfloor$ for the number of pulls of a fresh bad arm
that optimism forces.
"""

THM2_DISC = r"""
The bracket \eqref{eq:price-ratio} is the price ratio. Cold exploration buys
coverage $L^2$ of a bad direction per unit of regret, so its price is
$1/L^2$; the probe buys $c^2$ per $\eps$, price $\eps/c^2$; the multiplier
over cold LinUCB is the ratio $c^2/(\eps L^2)$ up to constants, and is $O(1)$
when the ratio is. The multiplier over the expert, $(T-\tau_0)/(\eps\tau_0)$,
diverges as the tail lengthens. On this instance the saving is collected
during the routed rounds, as $A_{\tau_0}$, because cold LinUCB completes its
forced exploration inside the window. Keeping cold LinUCB busy elsewhere
during the window moves the same saving after it, into $V_{\tau_0}$, with the
same price ratio (Proposition~\ref{prop:cat-tail-saving}, row~$5'$);
Appendix~\ref{app:catalog-statements} states it and discusses both
instances.

\paragraph{The catalog.}
Theorem~\ref{thm:price} is one row of a catalog of eleven instances, each
obtained from the canonical one by changing a single ingredient
(Table~\ref{tab:catalog}). Exactly two rows multiply over both base policies,
rows~5 and~$5'$, and they realize the same price ratio, one during the routed
rounds and one after them.
"""

CATALOG_AFTER = r"""
Across these linear instances, a gain over cold LinUCB appears only when the
expert buys the directions cold LinUCB would otherwise pay for, more cheaply
and early enough for them to recur, and a gain over the expert only when, in
addition, the expert is worse than the warm-started learner on the tail.

"""

CATALOG_CAPTION = r"""\caption{The catalog: what the three terms of Proposition~\ref{prop:decomp}
can be (statements in Appendix~\ref{app:catalog-statements}, proofs in
Appendix~\ref{app:catalog}). Row 1 is the offline identity, in which $A$ is
not charged; rows 2--10 are online handoffs. Rows 3 and 6 kill $B_\tau$ from
the two sides. Rows 4, 7 and 8 kill the multiplier over cold LinUCB: the
expert buys what cold buys at the same price, buys directions nobody needs,
or buys the right directions too dearly. Row 9 is the case in which cold
LinUCB's bill is already negligible, and row 10 shows that the gap rescales
both prices. Rows 5 and $5'$ realize the same price ratio, collected as
$A_\tau$ and as $V_\tau$. Blank cells are not claimed; constants are not
optimized.}
"""

TRANSFER_HEAD = r"""\section{Coverage and the Transfer}
\label{sec:two-channels}

A warm start $(M,c)$ enters LinUCB through two objects: the design matrix
$V_1=\lambda I+M$, which fixes every confidence width, and the response sum
$b_1=c$, which fixes the estimate $\widehat\theta_1=V_1^{-1}c$.
Theorem~\ref{thm:general-price} bounds the value of a warm start through the
first. We now show that in the model of Section~\ref{sec:setup}, with an
expert that sees only what the learner sees, the first is the whole story,
and that the second matters only when the logged rewards depart from that
model, in which case it has an explicit form.

"""

PREDICTION = r"""
Corollary~\ref{cor:coverage-complete} is a prediction that can be checked on
data: experts whose routed rounds cover the same directions should leave
warm starts of the same value. Section~\ref{sec:experiments} checks it.

"""

WHEN_TRANSFER_EXTRA = r""" A second departure concerns the expert
rather than the reward. Proposition~\ref{prop:response-sum} assumes that the
noise is independent of which actions were logged. An LLM reads the raw
context, which the features only summarize, so it can prefer an action on
exactly the rounds where that action pays; its logged rewards are then
selected, and the response sum is no longer determined by the Gram matrix,
even when $\varepsilon\equiv0$.
"""

PROP7_SENTENCE = r"""
The transfer can help or hurt. Proposition~\ref{prop:transfer-regret}
(Appendix~\ref{app:extra-theory}) bounds LinUCB's regret against any
reference parameter and shows that the transfer helps exactly when it moves
the estimate toward a reference whose residual is smaller on the best action
than on the actions LinUCB would otherwise play. How much it can help is the
subject of the next result.

"""

TRANSFER_CLOSE = r"""
Because the transfer is not a function of coverage, estimates of the
multiplier built from coverage before the run cannot rank the experts; the
two we tried are reported in Appendix~\ref{app:diagnostics}.

"""
