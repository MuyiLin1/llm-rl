# Theorem 3 pieces (from theorem_draft.tex, with fixes) and Appendix F text
DRAFT = open('/Users/linmuyi/code/DMwithLLM/prerun_study/transfer_thm/theorem_draft.tex').read()
def dcut(a, b, inc=True):
    assert DRAFT.count(a)==1, a[:60]
    i=DRAFT.index(a); j=DRAFT.index(b,i)
    return DRAFT[i:j+len(b)] if inc else DRAFT[i:j]

PROP6_NEW = dcut('\\begin{proposition}[The transfer]', '\\end{proposition}')
INSTANCE = dcut('\\emph{The instance.}', '\\begin{theorem}[Equal coverage, unequal value]', inc=False)
THM3 = dcut('\\begin{theorem}[Equal coverage, unequal value]', '\\end{theorem}')
MEANING = dcut('Two experts that buy exactly the same coverage', 'decays exponentially in $k$ alone.')
REGIME = dcut('\\begin{remark}[Where the theorem sits]', '\\end{remark}')
EXTENSIONS = dcut('\\begin{remark}[Settling, and a more accurate but useless expert]', '\\end{remark}')
PROOF3 = dcut('\\begin{proof}[Proof of Theorem~\\ref{thm:transfer-separation}]', '\\end{proof}')

old_end = REGIME[REGIME.index('for every warm start, so no'):REGIME.index('\\end{remark}')]
REGIME = REGIME.replace(old_end, r"""for every warm start,
and likewise for $R$, so after round $\tau$ both hybrids play the fresh arm
at every round, as cold LinUCB does, and the transfer buys nothing after the
handoff. With a valid radius coverage governs; with the annealed exploration
used in practice (Section~\ref{sec:exp-setup}) the transfer can decide.
""")
assert 'neither expert helps' not in REGIME

MEANING = MEANING + r""" The learner also
settles all or nothing: once $\Hyb^A_\tau$ leaves $D$, the statistics of $D$
stop changing and every later fresh arm has index $c$, so it never returns.
Making $\Exp_B$ strictly more accurate than $\Exp_A$ changes none of this
(Remark~\ref{rem:transfer-extensions}, Appendix~\ref{app:proof-transfer-sep})."""

THM3_LEAD = r"""\paragraph{How far the transfer can go.}
Proposition~\ref{prop:response-sum} says that two experts of equal coverage
are interchangeable when the logged noise is independent of the logged
actions. The next result shows how far this fails once the expert sees more
than the learner. It modifies the canonical instance of
Proposition~\ref{prop:three-way}: the good action is right only on a fraction
$p$ of the rounds, which rounds is visible to the expert but not to the
learner, and a second action, right on a smaller fraction $q$, is added.

"""

APP_F = r"""\section{Proof of Theorem~\ref{thm:transfer-separation} and Numerical Checks}
\label{app:proof-transfer-sep}

""" + PROOF3 + "\n\n" + EXTENSIONS + r"""

\paragraph{An expert that sees only the features.}
Theorem~\ref{thm:transfer-separation} compares two experts that both see the
type. A third expert plays $D$ and $R$ equally often, $2k$ times each, but
chooses which window rounds get $D$ without seeing the type, so that its right
answers come at the natural rate. It has the same coverage and price as
$\Exp_A$ and $\Exp_B$ and, on average, the same number of right answers. We
have no theorem for it. Numerically, with the parameters of the first row of
Table~\ref{tab:transfer-check} and $500$ seeds, its hybrid spends the whole
second half of the horizon on $D$ in $62\%$, $72\%$ and $84\%$ of runs at
$k=2,4,8$, and its mean regret is $576$, $412$ and $238$, against $28$, $2$
and $4$ for $\Exp_A$, $1{,}498$, $1{,}495$ and $1{,}491$ for $\Exp_B$, and
$1{,}500$ for cold LinUCB. The two informed experts sit on either side of it:
selection toward the frequent action helps far more than unselected data, and
selection away from it removes the gain.

\paragraph{Numerical checks.}
The instance is simple enough to simulate exactly. We ran LinUCB with the full
$d\times d$ design matrix and, for long horizons, the scalar recursion used in
the proof; the two chose the same action at every round of $36$ runs with
$T\le300$. With $\lambda=0.1$, $L=2$ and $T=5{,}000$, across $k\in\{1,\dots,32\}$
and the three settings of $(p,q,c)$ in Table~\ref{tab:transfer-check}: cold
LinUCB's regret equalled $pT$ in each of the $600$ runs checked;
$\Hyb^B_\tau$ never played $D$ after round $\tau$ in $12{,}000$ runs; a hybrid
that left $D$ never returned to it; when either hybrid received the routed
covariance without the routed rewards, it never played $D$ after $\tau$
($300$ runs each); with $\Exp_B$ made two to three times as accurate as
$\Exp_A$ (Remark~\ref{rem:transfer-extensions}(b)), $\Hyb^B_\tau$ stayed off
$D$ in all $500$ runs of each setting while $\Hyb^A_\tau$ settled on $D$ in
$496$--$500$; and with $\beta(L-1)/\sqrt\lambda\ge1$ neither hybrid played $D$
after $\tau$ ($200$ runs), as Remark~\ref{rem:transfer-regime} states.
Table~\ref{tab:transfer-check} compares the frequency with which
$\Hyb^A_\tau$ settles on $D$ with the bound of
Theorem~\ref{thm:transfer-separation}(iv).

\begin{table}[h]
\centering
\caption{Frequency with which $\Hyb^A_\tau$ settles on $D$ after the routed
window ($2{,}000$ seeds per cell, $\lambda=0.1$, $L=2$, $T=5{,}000$) against the
lower bound $1-e^{-8(p-c)h}$ of Theorem~\ref{thm:transfer-separation}(iv),
with $h=k(1-2c)-c\lambda$. The bound holds in every cell.}
\label{tab:transfer-check}
\small
\begin{tabular}{@{}llccccc@{}}
\toprule
$(p,q,c)$ & & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=6$ \\
\midrule
\multirow{2}{*}{$(0.3,\,0.05,\,0.135)$} & observed & $0.862$ & $0.979$ & $0.996$ & $1.000$ & $1.000$ \\
 & bound & $0.612$ & $0.852$ & $0.943$ & $0.978$ & $0.997$ \\
\midrule
\multirow{2}{*}{$(0.5,\,0.1,\,0.225)$} & observed & $0.865$ & $0.983$ & $0.996$ & $0.999$ & $1.000$ \\
 & bound & $0.687$ & $0.907$ & $0.972$ & $0.992$ & $0.999$ \\
\midrule
\multirow{2}{*}{$(0.2,\,0,\,0.09)$} & observed & $0.870$ & $0.979$ & $0.997$ & $1.000$ & $1.000$ \\
 & bound & $0.510$ & $0.762$ & $0.884$ & $0.944$ & $0.987$ \\
\bottomrule
\end{tabular}
\end{table}

"""

PROP6_PROOF_NEW = r"""\begin{proof}[Proof of Proposition~\ref{prop:transfer}]
$c=\sum_sr_sx_s=\sum_s\langle\theta^\star,x_s\rangle x_s+\sum_s(r_s-\langle\theta^\star,x_s\rangle)x_s
=M\theta^\star+V_1m$, and
$V_1^{-1}M\theta^\star=V_1^{-1}(V_1-\lambda I)\theta^\star=\theta^\star-\lambda V_1^{-1}\theta^\star$.
Under the linear model $r_s-\langle\theta^\star,x_s\rangle=\eta_s$, so
$V_1m=\xi$, whose law given $M$ is that of
Proposition~\ref{prop:response-sum} when the noise is independent of the
logged actions. Otherwise $r_s-\langle\theta^\star,x_s\rangle=\varepsilon(x_s)+\eta_s$,
and when the logged actions depend on the noise, $\E[\eta_s\mid x_s]$ need
not vanish on the routed rounds.
\end{proof}"""
