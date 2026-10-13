import sys
P = '/Users/linmuyi/code/DMwithLLM/v7.tex'
s = open(P).read()
R = []
def rep(old, new):
    assert s.count(old) == 1, (s.count(old), old[:80]); R.append((old, new))

# ---- abstract
rep(r"""regret by $6.5$ to $21\%$. Coverage alone saves about $5\%$ of
cold LinUCB's regret for each;""",
r"""regret by $6.5$ to $21\%$. Coverage alone saves about $2$ to $3.5\%$ of
cold LinUCB's regret for each;""")
# ---- intro
rep(r"""$5\%$, and coverage alone saves about $5\%$ of cold LinUCB's regret for each;
every significant difference between them disappears when the transfer is
removed.""",
r"""$5\%$, and coverage alone saves about $2$ to $3.5\%$ of cold LinUCB's regret
for each; every significant difference between them disappears when the
transfer is removed.""")
# ---- conclusion
rep(r"""is the more accurate. On AmazonCat-13K coverage saves about $5\%$ of
cold LinUCB's regret for every expert""",
r"""is the more accurate. On AmazonCat-13K coverage saves about $2$ to $3.5\%$ of
cold LinUCB's regret for every expert""")
# ---- Figure 1 caption
rep(r"""\caption{Coverage versus the transfer on AmazonCat/BERT-768d, ten seeds, standard
errors over seeds. The three experts cover the same directions to within
$5\%$. With the transfer present (dark bars) they span $1.71\times$ to
$3.29\times$; with it removed they collapse onto a common multiplier near
$1.5\times$ and no pairwise difference survives. \emph{Transfer removed} keeps
the expert-independent part $M\theta_{\mathrm{or}}$ of the response sum and
\emph{coverage only} discards the response sum entirely, yet the two agree on
every expert, as they must if the transfer is the only thing both lack.}""",
r"""\caption{Coverage versus the transfer on AmazonCat/BERT-768d, ten seeds, standard
errors over seeds. All three versions of the learner route exactly the same
rounds, the ones the full Router chose, and play the expert's answer there;
they differ only in what a routed round adds to the response sum. The three
experts cover the same directions to within $5\%$. With the transfer present
(dark bars) they span $1.71\times$ to $3.29\times$; with it removed they fall
to $1.16\times$--$1.50\times$ and no pairwise difference survives.
\emph{Transfer removed} keeps the expert-independent part
$M\theta_{\mathrm{or}}$ of the response sum and \emph{no response sum}
discards it entirely; the two do not differ significantly on any expert.}""")
# ---- Section 8 paragraph
rep(r"""the transfer carries the separation, and Figure~\ref{fig:two-channel} shows
that it does: feeding the learner $\langle\theta_{\mathrm{or}},\phi\rangle$
instead of the observed reward on routed rounds only, which deletes $m$ of
Definition~\ref{prop:transfer} with $\theta^\star=\theta_{\mathrm{or}}$, turns
every significant pairwise difference into a null result (large over small:
paired $t=5.19$ becomes $0.62$); dropping the response sum gives the same
collapse; and a linear reward makes the experts interchangeable, as
Proposition~\ref{prop:response-sum} predicts. On the regret scale coverage
alone saves about $5\%$ of cold LinUCB's regret for every expert, while the
full router saves $6.5\%$, $16\%$ and $21\%$ (Appendix~\ref{app:more-exp}).""",
r"""the transfer carries the separation, and Figure~\ref{fig:two-channel} shows
that it does. We replay each run with the routed rounds held fixed: all three
versions of the learner route exactly the rounds the full Router chose and
play the expert's answer there, so they have the same routed data and the same
coverage, and differ only in what those rounds add to the response sum.
Feeding the learner $\langle\theta_{\mathrm{or}},\phi\rangle$ instead of the
observed reward on the routed rounds, which deletes $m$ of
Definition~\ref{prop:transfer} with $\theta^\star=\theta_{\mathrm{or}}$, turns
every significant pairwise difference into a null result (large over small:
paired $t=5.18$ becomes $1.33$); dropping the response sum on the routed
rounds gives the same collapse; and a linear reward makes the experts
interchangeable, as Proposition~\ref{prop:response-sum} predicts. On the
regret scale coverage alone saves about $2$ to $3.5\%$ of cold LinUCB's
regret for every expert, while the full router saves $6.5\%$, $16\%$ and
$21\%$ (Appendix~\ref{app:more-exp}).""")
# ---- Section 8: 50k sentence, after the hidden-reward result
rep(r"""$p=0.47$; against the same answers with their rewards, $p=0.003$). Only about""",
r"""$p=0.47$; against the same answers with their rewards, $p=0.003$). The same
holds on the evaluation stream: $75$ added answers on this category over the
$50{,}000$ rounds raise the multiplier from $1.71\times$ to $2.56\times$
($p=0.007$), and with their rewards hidden to $1.84\times$ ($p=0.61$)
(Table~\ref{tab:inject50k}). Only about""")
# ---- Appendix: coverage & transfer paragraph
rep(r"""Removing $m$ while leaving the binary problem otherwise intact collapses the
spread of per-seed multipliers from $1.74$ to $0.56$ and turns every
significant pairwise difference into a null result: large over small falls
from $t=5.19$ to $t=0.62$. Suppressing the response sum entirely gives the
same collapse, to a spread of $0.11$; and the two transfer-free conditions
agree with each other on every expert (paired $t=-1.18$, $+0.26$, $-0.60$),
as they must if $m$ is the only thing both lack. Removing the transfer costs
each expert in proportion to what it gained, $-2.22$ for large ($t=5.12$)
against $-0.63$ for small ($t=2.47$).""",
r"""The comparison is a replay in which the routed rounds are held fixed. For
each expert and seed, three versions of the learner run in lockstep on the
same stream: the full Router, which decides which rounds are routed, and two
ablated learners that route exactly the same rounds and play the expert's
answer there, whether or not their own confidence would have routed them. On
every other round each learner plays its own LinUCB choice and sees the true
reward. The three versions therefore have the same routed rounds, the same
routed actions and the same routed covariance, and differ only in what the
routed rounds add to the response sum; the full version reproduces the
Router of Table~\ref{tab:amazon-results} exactly. Removing $m$ collapses the
spread of the experts' mean per-seed multipliers from $1.74$ to $0.50$ and
turns every significant pairwise difference into a null result: large over
small falls from $t=5.18$ to $t=1.33$. Suppressing the response sum on the
routed rounds gives the same collapse, to a spread of $0.01$; and the two
transfer-free conditions do not differ from each other on any expert (paired
$t=-0.66$, $+0.64$, $+0.46$). Removing the transfer costs each expert in
proportion to what it gained: the per-seed multiplier falls by $2.04$ for
large ($t=4.57$), $1.27$ for base ($t=3.94$) and $0.70$ for small
($t=2.51$). Whether a run settles on the dominant category
(Appendix~\ref{app:dominant}) follows the same pattern: $2$, $8$ and $9$ of
ten runs with the transfer, and at most $2$ of ten for any expert without
it.""")
# ---- Where the regret goes
rep(r"""have very different sizes. With the response sum suppressed, coverage alone gives
regret ratios $1.051$, $1.060$ and $1.051$: about $5\%$ less regret than cold
LinUCB, indistinguishable across experts, and below the price ratio for each,
consistent with Theorem~\ref{thm:general-price} although the stream is outside
its setting. The full Router gives $1.070$, $1.187$ and $1.266$. For
Flan-T5-base and Flan-T5-large, therefore, most of the regret saved is saved
by the transfer, roughly $10$ of $16$ and $16$ of $21$ percentage points. The
separation between experts survives the change of scale intact: on per-seed
regret ratios, large beats small with paired $t=+5.66$ and base beats small
with $t=+3.33$ when the transfer is present, and neither difference is
significant with the transfer removed ($t=+0.92$, $+1.74$) or the response sum
suppressed ($t=+0.15$, $+0.33$);""",
r"""have very different sizes. With the response sum suppressed on the routed
rounds, coverage alone gives regret ratios $1.035$, $1.029$ and $1.019$:
about $2$ to $3.5\%$ less regret than cold LinUCB, indistinguishable across
experts, and below the price ratio for each, consistent with
Theorem~\ref{thm:general-price} although the stream is outside its setting.
The full Router gives $1.070$, $1.187$ and $1.266$. For Flan-T5-base and
Flan-T5-large, therefore, almost all of the regret saved is saved by the
transfer, roughly $13$ of $16$ and $19$ of $21$ percentage points, and for
Flan-T5-small about half, $3$ of $6.5$. The separation between experts
survives the change of scale intact: on per-seed regret ratios, large beats
small with paired $t=+5.65$ and base beats small with $t=+3.32$ when the
transfer is present, and neither difference is significant with the transfer
removed ($t=+1.25$, $+1.24$) or the response sum suppressed ($t=-0.47$,
$-0.13$);""")
# ---- Table caption
rep(r"""\caption{Separating coverage from the transfer on AmazonCat/BERT-768d, ten seeds. In
\emph{full} a routed round contributes both the covariance and the response
sum, as the Router does. In \emph{transfer removed} the round contributes the
covariance and the expert-independent part $M\theta_{\mathrm{or}}$ of the
response sum---the learner is fed
$\langle\theta_{\mathrm{or}},\phi\rangle$ instead of the observed reward on
routed rounds only, which deletes $m$ of Definition~\ref{prop:transfer} and
changes nothing else. In \emph{no response sum} the round contributes the
covariance alone. Multipliers are accuracy multipliers, Router over cold
LinUCB, with the regret ratio in parentheses; $t$-statistics are paired over
seeds against small, on each scale, $t^\star=2.26$. With the transfer present the
experts separate; without it, by either mechanism, no pairwise difference
survives.}""",
r"""\caption{Separating coverage from the transfer on AmazonCat/BERT-768d, ten seeds,
with the routed rounds held fixed: all three versions route exactly the rounds
the full Router chose and play the expert's answer there, so their routed
covariance is identical. In \emph{full} a routed round also adds its observed
reward to the response sum, as the Router does. In \emph{transfer removed} it
adds the expert-independent part: the learner is fed
$\langle\theta_{\mathrm{or}},\phi\rangle$ instead of the observed reward on
the routed rounds, which deletes $m$ of Definition~\ref{prop:transfer}. In
\emph{no response sum} it adds nothing to the response sum. On every other
round each version plays its own LinUCB choice. Multipliers are accuracy
multipliers, Router over cold LinUCB ($8.40\%$), with the regret ratio in
parentheses. The spread is the largest minus the smallest of the three
experts' mean per-seed multipliers; $t$-statistics are paired over seeds, on
each scale, $t^\star=2.26$. With the transfer present the experts separate;
without it, by either mechanism, no pairwise difference survives.}""")
# ---- Table body
rep(r"""\small
\begin{tabular}{@{}lccccc@{}}
\toprule
& \multicolumn{2}{c}{Router accuracy} & \multicolumn{3}{c}{Multiplier over cold} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}
LLM & full & transfer removed & full & transfer removed & no response sum \\
\midrule
Flan-T5-small & $14.36\pm6.12\%$ & $9.71\pm2.08\%$  & $1.71\times$ ($1.07$) & $1.16\times$ ($1.02$) & $1.53\times$ ($1.05$) \\
Flan-T5-base  & $22.80\pm9.00\%$ & $15.21\pm9.13\%$ & $2.72\times$ ($1.19$) & $1.81\times$ ($1.08$) & $1.61\times$ ($1.06$) \\
Flan-T5-large & $27.64\pm5.56\%$ & $11.62\pm5.97\%$ & $3.29\times$ ($1.27$) & $1.38\times$ ($1.04$) & $1.53\times$ ($1.05$) \\
\midrule
\multicolumn{3}{@{}l}{spread of per-seed multipliers} & $1.74$ & $0.56$ & $0.11$ \\
\multicolumn{3}{@{}l}{large $-$ small, paired $t$} & $\mathbf{+5.19}$ & $+0.62$ & $+0.24$ \\
\multicolumn{3}{@{}l}{base $-$ small, paired $t$} & $\mathbf{+3.03}$ & $+1.25$ & $+0.23$ \\
\multicolumn{3}{@{}l}{large $-$ small, paired $t$, regret ratio} & $\mathbf{+5.66}$ & $+0.92$ & $+0.15$ \\
\multicolumn{3}{@{}l}{base $-$ small, paired $t$, regret ratio} & $\mathbf{+3.33}$ & $+1.74$ & $+0.33$ \\
\bottomrule
\end{tabular}""",
r"""\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
& \multicolumn{3}{c}{Router accuracy} & \multicolumn{3}{c}{Multiplier over cold} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
LLM & full & transfer removed & no response sum & full & transfer removed & no response sum \\
\midrule
Flan-T5-small & $14.37\pm6.12\%$ & $9.71\pm4.18\%$  & $11.53\pm6.61\%$ & $1.71\times$ ($1.07$) & $1.16\times$ ($1.01$) & $1.37\times$ ($1.04$) \\
Flan-T5-base  & $22.80\pm9.00\%$ & $12.56\pm5.66\%$ & $11.02\pm6.86\%$ & $2.72\times$ ($1.19$) & $1.50\times$ ($1.05$) & $1.31\times$ ($1.03$) \\
Flan-T5-large & $27.64\pm5.56\%$ & $12.49\pm6.35\%$ & $10.10\pm5.08\%$ & $3.29\times$ ($1.27$) & $1.49\times$ ($1.05$) & $1.20\times$ ($1.02$) \\
\midrule
\multicolumn{4}{@{}l}{runs settling on $1471$ (small / base / large)} & $2/8/9$ & $1/2/2$ & $1/2/1$ \\
\multicolumn{4}{@{}l}{spread of mean per-seed multipliers} & $1.74$ & $0.50$ & $0.01$ \\
\multicolumn{4}{@{}l}{large $-$ small, paired $t$} & $\mathbf{+5.18}$ & $+1.33$ & $+0.01$ \\
\multicolumn{4}{@{}l}{base $-$ small, paired $t$} & $\mathbf{+3.03}$ & $+1.36$ & $+0.03$ \\
\multicolumn{4}{@{}l}{large $-$ small, paired $t$, regret ratio} & $\mathbf{+5.65}$ & $+1.25$ & $-0.47$ \\
\multicolumn{4}{@{}l}{base $-$ small, paired $t$, regret ratio} & $\mathbf{+3.32}$ & $+1.24$ & $-0.13$ \\
\bottomrule
\end{tabular}""")
# ---- R^2 paragraph: the ablation it pooled was the earlier (own-routing) version
rep(r"""Table~\ref{tab:baselines} and the transfer-removed ablation of
Table~\ref{tab:two-channel}, for each of the three experts, plus cold""",
r"""Table~\ref{tab:baselines} and a transfer-removed ablation in which the
ablated learner chose its own routed rounds (an earlier version of the
ablation of Table~\ref{tab:two-channel}), for each of the three experts, plus cold""")
for o, n in R: s = s.replace(o, n)
open(P, 'w').write(s); print('applied', len(R))
