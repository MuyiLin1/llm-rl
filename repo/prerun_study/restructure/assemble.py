import re, sys
sys.path.insert(0, '/Users/linmuyi/code/DMwithLLM/prerun_study/restructure')
from part1_new_text import *
from part2_new_text import *
from part3_new_text import *
from part4_transfer import *

SRC = '/Users/linmuyi/code/DMwithLLM/prerun_study/v7.before_restructure.tex'
OUT = '/Users/linmuyi/code/DMwithLLM/prerun_study/restructure/v7_new.tex'
s = open(SRC).read()

def uniq(a):
    n = s.count(a); assert n == 1, (n, a[:80]); return s.index(a)
def between(a, b, inc_b=False):
    i = uniq(a); j = s.index(b, i + len(a))
    return s[i:j + len(b)] if inc_b else s[i:j]
def env(begin):
    i = uniq(begin); name = re.match(r'\\begin\{(\w+)\}', begin).group(1)
    e = '\\end{%s}' % name; j = s.index(e, i) + len(e); return s[i:j]
def flt(label, kind):
    li = uniq('\\label{%s}' % label); a = s.rindex('\\begin{%s}' % kind, 0, li)
    e = '\\end{%s}' % kind; b = s.index(e, li) + len(e); return s[a:b]
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80]); return text.replace(old, new)

# ---------------- extract ----------------
PRE = s[:uniq('\\begin{abstract}')]
prelim = between('We work in a standard linear contextual bandit', '\\section{Main Results}')
prelim = rep(prelim, "right object for understanding why a multiplier appears, and we comment on\nadaptive $\\tau$ after Proposition~\\ref{thm:handoff}.",
             "right object for understanding why a multiplier appears; Appendix~\\ref{sec:handoff}\ncomments on adaptive $\\tau$.")
decomp_prop = env('\\begin{proposition}[Decomposition of the hybrid')
decomp_expl = between("$A_\\tau$ is the expert's \\emph{direct} advantage", "of the data in $M_\\tau$.", True)
catalog11 = between('The catalog has eleven rows rather than', "and on the tail (row~$5'$).", True)
cat_table = flt('tab:catalog', 'table')
cap_i = cat_table.index('\\caption{The catalog:'); cap_j = cat_table.index('\\label{tab:catalog}')
cat_table = cat_table[:cap_i] + CATALOG_CAPTION + cat_table[cap_j:]
canon_def = between('The canonical instance, which we construct to model the separation reported', 'arms $b_t=Le_{t+1}$ with $L>1$.', True)
prop_threeway = env('\\begin{proposition}[Offline three-way separation]')
prop_onepass = env('\\begin{proposition}[No online gain on the canonical instance]')
row3 = between('\\paragraph{Row 3: an oracle expert on the canonical instance.}', '\\end{corollary}', True)
price_def = between('For a direction $u\\in\\R^d$ and an\naction set', '\\paragraph{Setting for the theorem.}')
setting_thm1 = between('\\paragraph{Setting for the theorem.}', '\\end{theorem}', True)
thm1_proofpara = between('The proof is in Appendix~\\ref{app:proof-general-price}; part (a)', "must bound one policy's regret in terms of the other's, as (c) does.", True)
remark_relax = env('\\begin{remark}[Two relaxations]')
cor_diverge = env('\\begin{corollary}[Diverging online multipliers require vanishing prices]')
i = uniq('\\begin{corollary}[Diverging online multipliers'); pj = s.index('\\begin{proof}', i); pe = s.index('\\end{proof}', pj) + len('\\end{proof}')
cor_diverge_proof = s[pj:pe]
cheap = between('\\paragraph{When do cheap probes exist?}', '\\subsection{Coverage and the Transfer}').replace('\\paragraph{When do cheap probes exist?}\n', '', 1)
cheap = rep(cheap, "of the rows below is the exchange rate", "of rows~5 and~$5'$ is the exchange rate")
prop4 = env('\\begin{proposition}[Width-limited regret with estimation error]')
prop5 = env('\\begin{proposition}[The response sum is determined by the design matrix]')
prop5 = rep(prop5, "$\\mathcal N(0,\\sigma^2)$. Then", "$\\mathcal N(0,\\sigma^2)$, independent of which actions were logged (as when the\nexpert sees only what the learner sees). Then")
prop5_after = between('For $\\sigma$-sub-Gaussian noise the distributional identity', 'another of equal coverage.', True)
prop5_after = rep(prop5_after, "every bound in Sections~\\ref{sec:width}--\\ref{sec:handoff} is\ntherefore unchanged when one expert is exchanged for another of equal coverage.",
                  "every bound in Sections~\\ref{sec:decomp}--\\ref{sec:catalog} is\ntherefore unchanged when one expert is exchanged for another of equal coverage,\nprovided both see only what the learner sees.")
i = uniq('\\begin{corollary}[Coverage is complete under linear rewards]'); j = s.index('\\end{proof}', i) + len('\\end{proof}')
cor_complete = s[i:j]
cor_complete = rep(cor_complete, "Under the model of Section~\\ref{sec:preliminaries}, two experts whose routed\nrounds leave the same Gram matrix have the same $V_\\tau$ in law. Any\ndifference in $V_\\tau$ between experts of equal coverage is a departure from\nthe linear model.",
                   "Under the model of Section~\\ref{sec:setup}, with noise independent of the\nlogged actions, two experts whose routed rounds leave the same Gram matrix\nhave the same $V_\\tau$ in law. Any difference in $V_\\tau$ between experts of\nequal coverage is a departure from the linear model or comes from an expert\nwhose choices depend on the noise.")
misspec = between('\\paragraph{Misspecification creates the transfer.}', '$\\Delta_{t,a}=\\bar r(x_{t,a^\\star_t})-\\bar r(x_{t,a})$.', True)
misspec = misspec.replace('\\paragraph{Misspecification creates the transfer.}', '\\paragraph{When the transfer appears.}') + WHEN_TRANSFER_EXTRA
prop7 = env('\\begin{proposition}[What the transfer can buy]')
template = between('\\paragraph{The template.}', '\\paragraph{Standing conventions.}').replace('\\paragraph{The template.}\n', '', 1)
conventions = between('\\paragraph{Standing conventions.}', '\\paragraph{Row 4: an expert that mimics cold exploration.}').replace('\\paragraph{Standing conventions.}\n', '', 1)
row4 = between('\\paragraph{Row 4: an expert that mimics cold exploration.}', '\\paragraph{Row 5: cheap recurring probes, the positive case.}')
row5intro = between('\\paragraph{Row 5: cheap recurring probes, the positive case.}', '\\begin{theorem}[Online multiplication').replace('\\paragraph{Row 5: cheap recurring probes, the positive case.}\n', '', 1)
row5intro = rep(row5intro, 'when the probes disappear therefore tests every clause of the thesis at once.', 'when the probes disappear therefore isolates the price ratio.')
thm2 = env('\\begin{theorem}[Online multiplication, and the price ratio as the multiplier]')
thm2_disc = between('The bracket \\eqref{eq:price-ratio} is the price ratio.', '\\begin{remark}[What certification costs on the template]')
remark_cert = env('\\begin{remark}[What certification costs on the template]')
row5p = between("\\paragraph{Row $5'$: the same price ratio, collected on the tail.}", '\\paragraph{Row 6: the same expert, optimal on the tail.}')
rows6_10 = between('\\paragraph{Row 6: the same expert, optimal on the tail.}', '\\paragraph{Where the experiments sit.}')
where = between('\\paragraph{Where the experiments sit.}', '\\subsection{What Governs the Regret of LinUCB at a Single Round}')
width_A = between('Fix the confidence event $\\mathcal E_\\delta$ on which', '\\end{lemma}', True)
width_B = between('The lemma is the familiar fact that an optimistic algorithm', '\\subsection{Regret of the Handoff Hybrid}')
width_B = rep(width_B, 'The lemma is the familiar fact', 'Lemma~\\ref{lem:width-limited} is the familiar fact')
handoff = between('\\subsection{Regret of the Handoff Hybrid}', '\\subsection{Pre-Runtime Estimates}')
twoscales = between('\\textbf{Two scales for one effect.}', '\\subsection{Setup}')
exp_setup = between('\\subsection{Setup}\n\\label{sec:exp-setup}', '\\subsection{OneShotWikiLinks-311}')
wiki_full = between('\\subsection{OneShotWikiLinks-311}\n\\label{sec:wiki}', '\\subsection{AmazonCat-13K}')
wiki_full = wiki_full.replace('\\label{sec:wiki}', '\\label{app:wiki}', 1)
tab_res = flt('tab:amazon-results', 'table')
ci = tab_res.index('\\caption{'); cj = tab_res.index('\\label{tab:amazon-results}')
tab_res = tab_res[:ci] + AMAZON_RESULTS_CAPTION + tab_res[cj:]
tab_decomp = flt('tab:amazon-decomp', 'table')
fig_two = flt('fig:two-channel', 'figure')
tab_bge = flt('tab:bge-csweep', 'table')
fig_dyn = flt('fig:amazon-dynamics', 'figure')
p_mult = between('\\textbf{The multiplying configuration.}', '\\textbf{Similar accuracy, unequal information')
p_s34 = between('Section~\\ref{sec:two-channels} says where that difference can live.', 'Removing $m$ while leaving')
p_remove = between('Removing $m$ while leaving the binary problem', '\\textbf{Where the regret goes.}')
p_regret = between('\\textbf{Where the regret goes.}', 'Finally, the transfer is not a small correction')
p_final = between('Finally, the transfer is not a small correction', '\\textbf{What the transfer carries.}')
p_carries = between('\\textbf{What the transfer carries.}', '\\textbf{BGE-1024d.}')
p_carries = rep(p_carries, 'why accuracy does not order the experts. Appendix~\\ref{app:dominant} gives the\ndetails.',
                'why accuracy does not order the experts.' + CARRIES_EXTRA + ' Appendix~\\ref{app:dominant} gives\nthe details.')
p_carries = p_carries.replace('\\textbf{What the transfer carries.}', '\\paragraph{What the transfer carries.}', 1)
p_bge = between('\\textbf{BGE-1024d.}', '\\paragraph{Summary across configurations.}')
p_summary = between('\\paragraph{Summary across configurations.}', 'Across the three Flan-T5 experts coverage is').replace('\\paragraph{Summary across configurations.}\n', '', 1)
p_across = between('Across the three Flan-T5 experts coverage is', '\\begin{table}[h]\n\\centering\n\\caption{Alternative routing mechanisms')
tab_base = flt('tab:baselines', 'table')
p_three = between('Three things follow.', '\\section{Conclusion}')
broader = between('\\paragraph{Broader impact.}', '\\bibliography{references}')
broader = rep(broader, 'The work is foundational theory and pre-runtime estimation for contextual\nbandits.', 'The work is foundational theory for combining frozen models with contextual\nbandits.')
APPX = s[uniq('\\bibliography{references}'):]
def acut(a, b):
    i = APPX.index(a); j = APPX.index(b, i); return APPX[i:j]
app_head = APPX[:APPX.index('\\section{Related Work}\\label{sec:related}')]
app_related = acut('\\section{Related Work}\\label{sec:related}', '\\section{Proofs}\n\\label{app:proofs}')
app_related = rep(app_related, 'This appendix expands on the related-work discussion deferred from the\nintroduction.', 'This appendix expands on Section~\\ref{sec:related-main}.')
app_proofs = acut('\\section{Proofs}\n\\label{app:proofs}', '\\section{Proofs for the Catalog of Section')
app_proofs = rep(app_proofs, "Sections~\\ref{app:proof-width}--\\ref{app:proof-transfer} prove the three\nlemmas of Section~\\ref{sec:width}, and Section~\\ref{app:proof-handoff}\nproves Proposition~\\ref{thm:handoff}.",
                 "Sections~\\ref{app:proof-width}--\\ref{app:proof-transfer} prove\nLemma~\\ref{lem:width-limited} and the two width identities of\nAppendix~\\ref{app:widths}, Section~\\ref{app:proof-handoff} proves\nProposition~\\ref{thm:handoff}, and the remaining sections prove the results of\nSections~\\ref{sec:price} and~\\ref{sec:two-channels}.")
app_proofs = rep(app_proofs, "realization and therefore in expectation. The ``if and only if'' statements\nare restatements of the identities.", "realization and therefore in expectation.")
oi = app_proofs.index('\\begin{proof}[Proof of Proposition~\\ref{prop:transfer}]'); oj = app_proofs.index('\\end{proof}', oi) + len('\\end{proof}')
app_proofs = app_proofs[:oi] + PROP6_PROOF_NEW + app_proofs[oj:]
app_proofs = app_proofs.rstrip('\n') + '\n\n\\subsection{Proof of Corollary~\\ref{cor:diverge}}\n\\label{app:proof-diverge}\n\n' + cor_diverge_proof + '\n\n'
app_catproofs = acut('\\section{Proofs for the Catalog of Section', '\\section{Numerical Validation of the Catalog}')
app_numval = acut('\\section{Numerical Validation of the Catalog}', '\\section{Coverage and the Transfer: Full Results}')
app_twochan = acut('\\section{Coverage and the Transfer: Full Results}', '\\section{The Dominant Category on AmazonCat-13K}')
app_twochan = app_twochan.replace('\\section{Coverage and the Transfer: Full Results}', '\\subsection{Coverage and the Transfer: Full Numbers}', 1)
app_rest = APPX[APPX.index('\\section{The Dominant Category on AmazonCat-13K}'):]
app_rest = rep(app_rest, 'This appendix expands the brief description of\nSection~\\ref{sec:handoff} into', 'This appendix expands the brief description of\nSection~\\ref{sec:setup} into')

# ---------------- assemble ----------------
MAIN = (PRE + ABSTRACT + '\n\n' + INTRO + RELATED +
  '\\section{Setup}\n\\label{sec:setup}\n\\label{sec:preliminaries}\n\n' + prelim.rstrip('\n') + '\n' + KEYTERMS + '\n\n' +
  DECOMP_HEAD + decomp_prop + '\n\n' + decomp_expl + '\n' + DECOMP_REMARK + width_A + '\n' + WIDTH_AFTER +
  PRICE_HEAD + price_def + setting_thm1 + '\n\n' + thm1_proofpara + '\n\n' + LEFTOVER + cor_diverge + '\n' + PRICE_TAIL +
  CATALOG_HEAD + canon_def + '\n\n' + prop_threeway + '\n' + CANON_MID + prop_onepass + '\n' + CANON_AFTER +
  template + CONVENTIONS_SHORT + '\n' + row5intro + thm2 + '\n' + THM2_DISC + '\n' + catalog11 + '\n\n' + cat_table + '\n' + CATALOG_AFTER +
  TRANSFER_HEAD + prop5 + '\n\n' + prop5_after + '\n\n' + cor_complete + '\n' + PREDICTION + misspec + '\n' + PROP6_NEW + '\n' + PROP7_SENTENCE +
  THM3_LEAD + INSTANCE + THM3 + '\n\n' + MEANING + '\n\n' + REGIME + '\n' + TRANSFER_CLOSE +
  EXP_HEAD + twoscales + exp_setup + AMAZON_HEAD + tab_res + '\n\n' + fig_two + '\n\n' + EQUAL_COVERAGE + p_carries + BGE_SHORT + WIKI_SHORT +
  CONCLUSION + broader)
APPENDIX = (app_head + app_related +
  APP_EXTRA_HEAD + width_B + handoff + APP_PROP4_HEAD + prop4 + '\n' + APP_PROP7_HEAD + prop7 + '\n' + APP_PROP7_AFTER +
  APP_RELAX_HEAD + remark_relax + '\n' + APP_PROBES_HEAD + cheap +
  APP_CAT_HEAD + '\\subsection{Standing Conventions}\n\\label{sec:template}\n\n' + conventions + row3 + '\n\n' + row4 +
  '\\paragraph{Row 5: discussion of Theorem~\\ref{thm:price}.}\n' + thm2_disc + remark_cert + '\n\n' + row5p + rows6_10 + where +
  app_proofs + app_catproofs + APP_F + app_numval +
  APP_MORE_EXP_HEAD + tab_decomp + '\n\n' + p_mult + '\n' + fig_dyn + '\n\n' +
  '\\subsection{Coverage and the Transfer}\n\\label{app:amazon-transfer}\n\n' + p_s34 + p_remove + p_regret + p_final + app_twochan +
  '\\subsection{BGE Features}\n\\label{app:bge}\n\n' + tab_bge + '\n\n' + p_bge +
  wiki_full + '\\subsection{Summary Across Configurations}\n\\label{app:summary}\n\n' + p_summary +
  '\\subsection{Other Routers}\n\\label{app:routers}\n\n' + p_across + tab_base + '\n\n' + p_three + app_rest)
out = MAIN + APPENDIX
out = re.sub(r'\n{4,}', '\n\n\n', out)
open(OUT, 'w').write(out)
print('wrote', OUT, len(out))
