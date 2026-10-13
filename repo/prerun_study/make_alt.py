"""Build the ALT 2027 version of the paper from v7.tex (NeurIPS format).

Reads v7.tex, writes alt_submission/main.tex and copies the class files,
references.bib and the figures next to it. v7.tex is not modified.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "v7.tex"
OUT = ROOT / "alt_submission"
TEMPLATE = Path.home() / "Downloads" / "alt2027-template"

s = SRC.read_text()


def sub1(old, new, text):
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)


# ---------------------------------------------------------------- preamble
body_start = s.index("\\begin{document}")
old_pre = s[:body_start]
macros_start = old_pre.index("% Blackboard bold sets")
macros = old_pre[macros_start:]
# the conjecture environment is defined after the macros in v7; keep it
# with the other theorem environments instead
macros = sub1("\\theoremstyle{remark}\n\\newtheorem{conjecture}[theorem]{Conjecture}\n",
              "", macros)
title_at = macros.index("\\title{")
macros = macros[:title_at].rstrip() + "\n"

preamble = r"""% ALT 2027 submission. Keep alt2027.cls, jmlr.cls and jmlrutils.sty next to
% this file. Compiler: pdfLaTeX (Overleaf default).
\documentclass[anon]{alt2027} % submission (anonymized)
%\documentclass[final]{alt2027} % camera-ready (de-anonymized)

% Loaded by the class: amsmath, amssymb, natbib, graphicx, url, xcolor,
% hyperref, algorithm2e (environment algorithm2e), and theorem, lemma,
% proposition, corollary, definition, remark, conjecture, example and proof
% environments. The class refuses amsthm.
\usepackage{times}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{microtype}
\usepackage{enumitem}

% The class numbers theorems, lemmas, propositions, corollaries, definitions
% and remarks with one shared counter. Give lemmas, propositions,
% corollaries, definitions and remarks their own counters, so that the
% numbering reads Theorem 1, 2, 3, Lemma 1, 2, ... Delete this block to go
% back to the shared counter.
\makeatletter
\newcommand{\altrenumber}[2]{%
  \expandafter\let\csname #1\endcsname\relax
  \expandafter\let\csname end#1\endcsname\relax
  \newtheorem{#1}{#2}}
\makeatother
\altrenumber{lemma}{Lemma}
\altrenumber{proposition}{Proposition}
\altrenumber{corollary}{Corollary}
\theorembodyfont{\upshape}
\altrenumber{definition}{Definition}
\altrenumber{remark}{Remark}
\theorembodyfont{\itshape}
\providecommand{\qed}{\hfill\jmlrBlackBox}
% allow slightly loose lines rather than lines that run into the margin
\setlength{\emergencystretch}{3em}

% The class puts a theorem's title inside the list label, which cannot break
% across lines, so long titles ran into the margin. Same look, but the title
% is set as ordinary text and can wrap.
\makeatletter
\renewcommand*{\@opargbegintheorem}[3]{%
  \ifdef{\@jmlr@currentthm}%
  {%
    \letcs{\jmlr@this@theoremheader}{jmlr@thm@\@jmlr@currentthm @header@font}%
    \letcs{\jmlr@this@theorembody}{jmlr@thm@\@jmlr@currentthm @body@font}%
    \letcs{\jmlr@this@theoremsep}{jmlr@thm@\@jmlr@currentthm @sep}%
    \letcs{\jmlr@this@theorempostheader}%
       {jmlr@thm@\@jmlr@currentthm @postheader}%
  }%
  {%
    \let\jmlr@this@theorembody\@theorembodyfont
    \let\jmlr@this@theoremheader\@theoremheaderfont
    \let\jmlr@this@theoremsep\@theoremsep
    \let\jmlr@this@theorempostheader\@theorempostheader
  }%
  \trivlist
   \item[\hskip\labelsep{\jmlr@this@theoremheader #1\ #2}]%
   {\jmlr@this@theoremheader (#3)\jmlr@this@theorempostheader}\ %
   \mbox{}\jmlr@this@theoremsep
   \jmlr@this@theorembody
}
\makeatother

"""

front = r"""
\title[Information Value Is Not Policy Value]{Information Value Is Not Policy Value:\\ When LLM Warm-Starts Help Linear Contextual Bandits}

% Hidden by the [anon] option. Fill in for the camera-ready version.
\altauthor{%
 \Name{Author Name1} \Email{abc@sample.com}\\
 \addr Address 1
 \AND
 \Name{Author Name2} \Email{xyz@sample.com}\\
 \addr Address 2%
}

"""

doc = s[body_start:]

# ---------------------------------------------------------------- front matter
abs_end = doc.index("\\end{abstract}") + len("\\end{abstract}")
doc = (doc[:abs_end]
       + "\n\n\\begin{keywords}%\n"
         "linear contextual bandits, LinUCB, warm start, large language models, "
         "exploration, regret\n\\end{keywords}"
       + doc[abs_end:])

# ---------------------------------------------------------------- back matter
bi_start = doc.index("\\paragraph{Broader impact.}")
bib_at = doc.index("\\bibliography{references}")
doc = (doc[:bi_start]
       + r"""% ALT strongly encourages an anonymous AI Disclosure section when
% generative AI was used substantively. Uncomment and edit:
% \section*{AI Disclosure}
% We used [tool] to [briefly explain how it was used].

% Hidden by the [anon] option.
\acks{We thank ...}

"""
       + doc[bib_at:])
doc = sub1("\\bibliography{references}\n\\bibliographystyle{abbrv}\n",
           "\\bibliography{references}\n", doc)

chk = doc.index("\\section*{NeurIPS Paper Checklist}")
doc = doc[:chk].rstrip() + "\n\n\\end{document}\n"

# ---------------------------------------------------------------- citations
# natbib author-year: \citet when the authors are part of the sentence,
# \citep when the citation is parenthetical, \citealp inside parentheses.
TEXTUAL = {6, 7, 8, 9, 20, 21, 22, 25, 29, 47, 48, 53, 55, 56, 60, 64, 65,
           66, 67, 71, 72, 73}
cites = list(re.finditer(r"\\cite(\[[^\]]*\])?\{[^}]*\}", doc))
assert len(cites) == 75, len(cites)
out, last = [], 0
for i, m in enumerate(cites):
    cmd = "\\citet" if i in TEXTUAL else "\\citep"
    out.append(doc[last:m.start()])
    out.append(cmd + m.group(0)[len("\\cite"):])
    last = m.end()
out.append(doc[last:])
doc = "".join(out)

# inside parentheses (no nested parentheses in natbib style)
doc = sub1(r"(in particular for the radius of \citep{abbasi2011improved})",
           r"(in particular for the radius of \citealp{abbasi2011improved})", doc)
doc = sub1(r"\begin{lemma}[Subgaussian concentration, Corollary 5.5 of \citep{lattimore2020bandit}]",
           r"\begin{lemma}[Subgaussian concentration; {\citealp[Corollary~5.5]{lattimore2020bandit}}]",
           doc)

# ---------------------------------------------------------------- proofs
# the class's proof environment takes no title; titled proofs use the
# class's proofof environment ("Proof of Theorem 2")
out, pos, n = [], 0, 0
for m in re.finditer(r"\\begin\{proof\}\[Proof of ([^\]]*)\]", doc):
    if m.start() < pos:
        continue
    e = doc.index("\\end{proof}", m.end())
    assert "\\begin{proof}" not in doc[m.end():e]
    out.append(doc[pos:m.start()])
    out.append("\\begin{proofof}{" + m.group(1) + "}" + doc[m.end():e] + "\\end{proofof}")
    pos = e + len("\\end{proof}")
    n += 1
out.append(doc[pos:])
doc = "".join(out)
assert n == 5, n
assert "\\begin{proof}[" not in doc
doc = sub1(r"\sqrt s/(\lambda+s)$.\hfill$\square$", r"\sqrt s/(\lambda+s)$. \qed", doc)

# ---------------------------------------------------------------- layout
# displays and tables that were wider than the 6in / 11pt text block
doc = sub1(r"""\[
\kappa_x:=\frac{\Delta(x)}{\|x\|_2^2},\qquad
C^\star_x:=\frac{4\beta^2\|x\|_2^2}{\Delta(x)^2},\qquad
n^\star_x:=""", r"""\begin{gather*}
\kappa_x:=\frac{\Delta(x)}{\|x\|_2^2},\qquad
C^\star_x:=\frac{4\beta^2\|x\|_2^2}{\Delta(x)^2},\\
n^\star_x:=""", doc)
doc = sub1(r"""N_x:=\#\{t\in[T]:x\in\mathcal A_t\}.
\]""", r"""N_x:=\#\{t\in[T]:x\in\mathcal A_t\}.
\end{gather*}""", doc)
doc = sub1(r"""\[
\kappa^\pi_U(t_0):=\min\Bigl\{\frac{\Delta(x^\pi_t)}{\mathrm{cov}_U(x^\pi_t)}:
\ t\le t_0,\ \Delta(x^\pi_t)>0,\ \mathrm{cov}_U(x^\pi_t)>0\Bigr\},
\qquad
C^{\mathrm{free}}_U(t_0):=\sum_{t\le t_0:\,\Delta(x^\pi_t)=0}\mathrm{cov}_U(x^\pi_t),
\]""", r"""\begin{gather*}
\kappa^\pi_U(t_0):=\min\Bigl\{\frac{\Delta(x^\pi_t)}{\mathrm{cov}_U(x^\pi_t)}:
\ t\le t_0,\ \Delta(x^\pi_t)>0,\ \mathrm{cov}_U(x^\pi_t)>0\Bigr\},
\\
C^{\mathrm{free}}_U(t_0):=\sum_{t\le t_0:\,\Delta(x^\pi_t)=0}\mathrm{cov}_U(x^\pi_t),
\end{gather*}""", doc)
doc = sub1("\\label{tab:amazon-results}\n\\small\n",
           "\\label{tab:amazon-results}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n", doc)
doc = sub1(r"""\[
U_t(b'_{j'})\ \ge\ 1+\tfrac38\cdot2\sqrt\lambda\cdot\frac{L'}{\sqrt{2\lambda}}=1+\frac{3L'}{4\sqrt2}>7.3,
\qquad
U_t(g)\ \le\ 1+\tfrac{13}8\cdot2\sqrt\lambda\cdot\frac1{\sqrt\lambda}=4.25,
\qquad
U_t(p_j)\ \le\ 1-\eps+\tfrac{13}8\cdot2\sqrt\lambda\cdot\frac{\sqrt2}{\sqrt\lambda}<5.6,
\]""", r"""\begin{gather*}
U_t(b'_{j'})\ \ge\ 1+\tfrac38\cdot2\sqrt\lambda\cdot\frac{L'}{\sqrt{2\lambda}}=1+\frac{3L'}{4\sqrt2}>7.3,
\\
U_t(g)\ \le\ 1+\tfrac{13}8\cdot2\sqrt\lambda\cdot\frac1{\sqrt\lambda}=4.25,
\qquad
U_t(p_j)\ \le\ 1-\eps+\tfrac{13}8\cdot2\sqrt\lambda\cdot\frac{\sqrt2}{\sqrt\lambda}<5.6,
\end{gather*}""", doc)
def fit_table(doc, header):
    """Scale the tabular whose header row starts with `header` to the text width."""
    i0 = doc.index(header)
    t0 = doc.rindex("\\begin{tabular}", 0, i0)
    t1 = doc.index("\\end{tabular}", i0) + len("\\end{tabular}")
    return doc[:t0] + "\\resizebox{\\linewidth}{!}{%\n" + doc[t0:t1] + "}" + doc[t1:]


doc = fit_table(doc, "\\# & Instance & $A_\\tau$ & $I_\\tau$")          # catalog
doc = fit_table(doc, "Expert & Change & Expert acc. & Router acc. & Mult. &")  # 20k injection
doc = sub1(r"""By the elliptical potential lemma \citep[Lemma~11]{abbasi2011improved},
$\sum_{t>\tau}\min\{1,\|x_{t,A_t}\|^2_{V_t^{-1}}\}\le2\log\det V_{T+1}/\det V_{\tau+1}$;
since""", r"""By the elliptical potential lemma \citep[Lemma~11]{abbasi2011improved},
\[
\sum_{t>\tau}\min\{1,\|x_{t,A_t}\|^2_{V_t^{-1}}\}\le2\log\frac{\det V_{T+1}}{\det V_{\tau+1}};
\]
since""", doc)
# the 50k injection table: scale to the text width
i0 = doc.index("Change & Expert acc. & Router acc. & Mult.\\ (regret ratio)")
t0 = doc.rindex("\\begin{tabular}", 0, i0)
t1 = doc.index("\\end{tabular}", i0) + len("\\end{tabular}")
doc = doc[:t0] + "\\resizebox{\\linewidth}{!}{%\n" + doc[t0:t1] + "}" + doc[t1:]

# ---------------------------------------------------------------- algorithm
a0 = doc.index("\\begin{algorithm}[t]")
a1 = doc.index("\\end{algorithm}") + len("\\end{algorithm}")
alg = r"""\begin{algorithm2e}[!htb]
\caption{Confidence-Gated Master over Frozen Policies and LinUCB}
\label{alg:confidence-gated}
\DontPrintSemicolon
\LinesNumbered
\KwIn{Frozen estimators $\{(\widehat{\theta}_i,V^{(i)})\}_{i=1}^m$, regularization $\lambda>0$, confidence parameters $\{\alpha_t\}_{t=1}^T$}
Initialize LinUCB state: $V_1=\lambda I_d$, $c_1=0$, $\widehat{\theta}_1=0$\;
Set $\textsc{handover}\gets \mathrm{False}$\;
\For{$t=1,\dots,T$}{
  Observe action set $\mathcal{A}_t$\;
  Compute LinUCB proposal
  $a_t^{(0)} \in \arg\max_{a\in\mathcal{A}_t}\bigl\{\langle \widehat{\theta}_t,x_{t,a}\rangle+ \alpha_t \|x_{t,a}\|_{V_t^{-1}}\bigr\}$\;
  Set $u_t^{(0)} \gets \|x_{t,a_t^{(0)}}\|_{V_t^{-1}}$\;
  \For{$i=1,\dots,m$}{
    Compute frozen proposal $a_t^{(i)} \in \arg\max_{a\in\mathcal{A}_t}\langle \widehat{\theta}_i,x_{t,a}\rangle$\;
    Set $u_t^{(i)} \gets \|x_{t,a_t^{(i)}}\|_{(V^{(i)})^{-1}}$\;
  }
  \eIf{$\textsc{handover}=\mathrm{True}$}{
    $A_t \gets a_t^{(0)}$\;
  }{
    Choose $i_t^\star \in \arg\min_{i\in[m]} u_t^{(i)}$\;
    \eIf{$u_t^{(i_t^\star)} < u_t^{(0)}$}{
      $A_t \gets a_t^{(i_t^\star)}$\;
    }{
      $A_t \gets a_t^{(0)}$\;
    }
  }
  Observe reward $r_t$\;
  Update LinUCB:
  $V_{t+1}\gets V_t + x_{t,A_t}x_{t,A_t}^\top$,
  $c_{t+1}\gets c_t + r_t x_{t,A_t}$,
  $\widehat{\theta}_{t+1}\gets V_{t+1}^{-1}c_{t+1}$\;
  Compute $U_t^{(i)}$ for all $i\in[m]$ and $L_t^{(0)}$\;
  \If{$L_t^{(0)} \ge \max_{i\in[m]} U_t^{(i)}$ \textup{for $K$ consecutive rounds}}{
    $\textsc{handover}\gets \mathrm{True}$\;
  }
}
\end{algorithm2e}"""
doc = doc[:a0] + alg + doc[a1:]

main = preamble + macros + front + doc
assert "\\cite{" not in main
assert "neurips" not in main.lower()
assert "Hippocampus" not in main

OUT.mkdir(exist_ok=True)
(OUT / "main.tex").write_text(main)
for f in ["alt2027.cls", "jmlr.cls", "jmlrutils.sty"]:
    shutil.copy2(TEMPLATE / f, OUT / f)
shutil.copy2(ROOT / "references.bib", OUT / "references.bib")
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", main):
    src = ROOT / g
    assert src.exists(), g
    shutil.copy2(src, OUT / g)
print("wrote", OUT / "main.tex", len(main.splitlines()), "lines")
