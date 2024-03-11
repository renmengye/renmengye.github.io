\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{zgroup}[2020/10/07 v1.0 ZGROUP 2020 Style File]
% v2020: J. Lucas & M. Ren

\ProcessOptions\relax

\LoadClass[wcp]{zgroup/jmlr}
% modified so that header doesn't appear
\jmlrvolume{}
\jmlryear{}
\jmlrproceedings{wcp}{}

\newcommand{\zgroupauthor}[1]{\author{#1}}
\editors{}

\def\ps@jmlrtps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{}%
  \def\@evenhead{\hfill {\small\scshape \@shortauthor} \hfill}%
  \def\@oddfoot{\hfill \small\rmfamily \thepage \hfill}%
  \let\@evenfoot\@oddfoot
}

\def\ps@jmlrps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{{\hfill \small \scshape \space\@jmlr@shortauthors.\hfill}}
  \def\@oddfoot{\hfill \small\rmfamily \thepage \hfill}%
  \def\@evenhead{}%
  \def\@evenfoot{\hfill \small\rmfamily \thepage \hfill}%
}
\jmlrworkshop{}
\makeatletter
 \let\Ginclude@graphics\@org@Ginclude@graphics 
\makeatother

\definecolor{red}{RGB}{255,73,92}
\definecolor{blue}{RGB}{37,110,255}
\definecolor{violet}{RGB}{70,35,122}
\definecolor{green}{RGB}{61,220,151}
\setlength\parskip{6pt}
