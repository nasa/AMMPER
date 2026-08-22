%%
%% This is file `asmarticle.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% asm.dtx  (with options: `class')
%% 
%% The "asm" distribution by TNQ Technologies.
%% Please see the file `asm.ins' for information on how you may
%% (re-)distribute the `asm' bundle of files.
%% 
%% \iffalse
%%
%% This is the file `asm.dtx' version 0.7 (initial release)
%% of the `ASM' LaTeX package.
%%
%% Copyright © 2023. The American Society for Microbiology.
%%
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%   https://www.latex-project.org/lppl.txt
%% and version 1.3c or later is part of all distributions of LaTeX
%% version 2008 or later.
%%
%% This work has the LPPL maintenance status `maintained'.
%%
%% The current maintainers of this work are:
%%   Pratap Ramakrishnan (pratap.ramakrishnan@tnqtech.com);
%%   Rathinavelu Sundaramoorthy (rathinnavelu.s@tnqtech.com)
%%   TNQ Technologies
%%
%% This work consists of the files asm.dtx and asm.ins
%% and the derived files asmarticle.cls and asm.bst.
%%
%% \fi
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%
%% LaTeX Templates Journal Article
%% LaTeX Class
%% Version 0.7 (May 14, 2023)
%%
%% Author(s):
%%   Pratap Ramakrishnan (pratap.ramakrishnan@tnqtech.com)
%%   Rathinavelu Sundaramoorthy (rathinnavelu.s@tnqtech.com)
%%
%% License:
%% This software is released under the terms of the
%% LaTeX Project Public License version 1.3c.
%% Copyright © 2023. The American Society for Microbiology.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%----------------------------------------------------------------------------------------
%% CLASS CONFIGURATION
%%----------------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{LTJournalArticle}[2023/02/07 LaTeX Templates Journal Article Class v2.0]

\usepackage{etoolbox} % Required for conditional logic and easily changing commands

\newtoggle{unnumberedsections} % Create toggle for a class option
\settoggle{unnumberedsections}{false} % Default value for the class option
\DeclareOption{unnumberedsections}{\settoggle{unnumberedsections}{true}} % Set the class option toggle if the class option was used in the template

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % Pass through any extra options specified to the base class

\newif\if@DoubleBlind%
\DeclareOption{DoubleBlind}{\@DoubleBlindtrue}

\ProcessOptions\relax % Process class options

\LoadClass{article} % Load the base class

%%%%%%%%%%%%%%%%General Packages Start
\newif\if@draftmode
\RequirePackage[mathlines]{lineno}
\linenumbers

\RequirePackage[T1]{fontenc}
\RequirePackage{microtype}
\renewcommand{\ttdefault}{\sfdefault}

\RequirePackage{graphicx}
\RequirePackage{changepage}
\RequirePackage{enumitem}

\RequirePackage[mathlines]{lineno}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
  \if@draftmode
\usepackage{concrete,concmath} % alternative to MinionPro, mathpazo
  \else
\usepackage{tgpagella,mathpazo} %
  \fi

%% Lists
\RequirePackage{enumitem}
\setlist{nosep,labelindent=\parindent}

%% Tables
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\newcommand{\tnote}[1]{\textsuperscript{\textit{#1}}}
\newlist{tablenotes}{description}{1}
\setlist[tablenotes]{labelsep=0pt,format=\mdseries\itshape\textsuperscript}

%%% Left-, right- and center-aligned auto-wrapping column types
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{C}{>{\centering\arraybackslash}X}

\RequirePackage[figuresright]{rotating}%
\usepackage{ifthen}
\usepackage{color}
\usepackage{colortbl}
\usepackage{gensymb} % for Celsius
\usepackage{siunitx}  % SI units
\usepackage{textcomp}  % required for siunitx/microtype compatibility
\usepackage[version=4]{mhchem}  % chemical formulas
\usepackage{graphics}
\usepackage{epstopdf}
\usepackage[numbers,square,comma]{natbib}
%%\renewcommand\@biblabel[1]{#1.}
\usepackage{setspace}
%%\usepackage{authblk}
\RequirePackage[hyphenbreaks]{breakurl}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}
\usepackage{lettrine}
\RequirePackage{longtable}%
\RequirePackage{caption}
\RequirePackage{epstopdf}
\RequirePackage{xcolor}
\RequirePackage{framed}
\RequirePackage{ragged2e}
\usepackage[none]{hyphenat}

%%%%%%%%%%%%%%%%PaperSize Start
\abovedisplayskip 9\p@ \@plus 3\p@ \@minus1.5\p@%
\abovedisplayshortskip 0pt% \@plus 3\p@%
\belowdisplayskip\abovedisplayskip%
\belowdisplayshortskip 6\p@ \@plus 2\p@ \@minus1\p@%

%%% Default document point size
\renewcommand\normalsize{%
  \@setfontsize\normalsize{12bp}{14pt plus0.01\p@ minus0.01\p@}
  \let\@listi\@listI
\def\textsc##1{\fontsize{7}{12}\selectfont\uppercase{##1}}%
}%
\normalsize%

\let\@bls\baselineskip%

%%% Stretchable vertical spaces used for \smallskip, \medskip and \bigskip
\setlength\smallskipamount{.25\@bls \@plus 1\p@ \@minus 1\p@} % quarter-line
\setlength\medskipamount{.5\@bls \@plus 2\p@ \@minus 2\p@}    % half-line
\setlength\bigskipamount{1\@bls \@plus 3\p@ \@minus 3\p@}     % one-line

   \setlength{\paperheight}{279.4truemm}  % trimmed page height
   \setlength{\paperwidth}{215.9truemm}    % trimmed page width
   \setlength\textwidth{470pt}%text measure excluding margins%%%
   \setlength{\textheight}{648pt}%
   \addtolength\textheight{\topskip}
   \addtolength{\textheight}{-10pt}%
   \columnwidth=\textwidth%
%%%   \fullwidth=\textwidth%

\setlength\columnsep{0pt}% space between columns for
\setlength\columnseprule{0\p@}% width of rule between two columns

   \setlength\topmargin{0pt}% head margin
   \addtolength\topmargin{-1in}% subtract out the 1 inch driver margin
   \addtolength\topmargin{72bp}% subtract out the 1 inch driver margin

%%%   \addtolength\topmargin{-1pt}% subtract out the 1 inch driver margin
   \setlength\oddsidemargin{72bp}%%{96pt}%
   \addtolength\oddsidemargin{-1in}%
   \addtolength\oddsidemargin{0pt}%
%%   \setlength\evensidemargin{\oddsidemargin}%Updated by Devi as per mail received from Karthick on 14022022-JIRA  ID: LTX-2261
   \setlength\evensidemargin{72bp}%
   \addtolength\evensidemargin{-1in}%
   \addtolength\evensidemargin{0pt}%
   \settoheight\headheight{12pt}% height of running head
   \setlength\topskip{7\p@}% height of first line of text
   \setlength\headsep{10pt}% space below running head --
   \addtolength\headsep{-\topskip}%   base to base with first line of text
   \addtolength\headsep{18pt}%   base to base with first line of text
%%%   \setlength\footins{24\p@}% space above footer line
\addtolength{\skip\footins}{15.87pt}%
   \setlength\footskip{32\p@}% space above footer line
   \setlength\maxdepth{.5\topskip}% pages can be deep by half a line
   \addtolength\topmargin{-\headheight}% subtract Headheight and headsep
   \addtolength\topmargin{-\headsep}% subtract Headheight and headsep
%%%%%%%%%%%%%%%%PaperSize END

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Standard Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% (Do not modify below) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Line spacing
\setlength\lineskip{1\p@} \setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}

%%% Allow some loose lines
\tolerance=1000

%%% Page break penalties
\@lowpenalty   51 \@medpenalty  151 \@highpenalty 301

%%% Set badness and tolerance
\vbadness=9999 \tolerance=9999

%%% Don't allow consecutive hyphens and hyphens at the bottom of the page.
\doublehyphendemerits 100000 \finalhyphendemerits  1000000

%%% Disallow widows and orphans
\clubpenalty 10000 \widowpenalty 10000

%%% Disable page breaks before equations, allow pagebreaks after
%%% equations and discourage widow lines before equations.
\displaywidowpenalty 500 \predisplaypenalty   10000 \postdisplaypenalty  0

%%% Allow breaking the page in the middle of a paragraph
\interlinepenalty 0

%%% Disallow breaking the page after a hyphenated line
\brokenpenalty 10000

%%% Hyphenation; don't split words into less than three characters
\lefthyphenmin=3 \righthyphenmin=3

%%% Float placement parameters

%%% The total number of floats that can be allowed on a page.
\setcounter{totalnumber}{10}
%%% The maximum number of floats at the top and bottom of a page.
\setcounter{topnumber}{5} \setcounter{bottomnumber}{5}
%%% The maximum part of the top or bottom of a text page that can be
%%% occupied by floats. This is set so that at least four lines of text
%%% fit on the page.
\renewcommand\topfraction{.921}
\renewcommand\bottomfraction{.921}
%%% The minimum amount of a text page that must be occupied by text.
%%% This should accomodate four lines of text.
\renewcommand\textfraction{.079}
%%% The minimum amount of a float page that must be occupied by floats.
\renewcommand\floatpagefraction{.887}

%%% The same parameters repeated for double column output
\renewcommand\dbltopfraction{.88}
\renewcommand\dblfloatpagefraction{.88}

%%% Space between floats
\setlength\floatsep    {24\p@ \@plus 2\p@ \@minus 2\p@}
%%% Space between floats and text
\setlength\textfloatsep{24\p@ \@plus 2\p@ \@minus 4\p@}
%%% Space above and below an inline figure
\setlength\intextsep   {18\p@ \@plus 2\p@ \@minus 2\p@}

%%% For double column floats
\setlength\dblfloatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}

%%% Space left at top, bottom and inbetween floats on a float page.
\setlength\@fptop{-2\p@}        % no space above float page figures
\setlength\@fpsep{12\p@ \@plus 2fil} \setlength\@fpbot{0\p@ \@plus 1fil}

%%% The same for double column
\setlength\@dblfptop{-2\p@} \setlength\@dblfpsep{12\p@ \@plus 1fil}
\setlength\@dblfpbot{0\p@ \@plus 2fil}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% End of Standard Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%Runnind Head
\newbox\rhfootbox%
\setbox\rhfootbox=\hbox{\fontsize{8bp}{9bp}\selectfont{American Society for Microbiology \rule{0.25pt}{5pt} www.asm.org}}

\newdimen\rhfootdimen%
\rhfootdimen=\textwidth%
\advance\rhfootdimen by -\wd\rhfootbox

\gdef\ps@headings{%
\let\@oddhead\relax
\let\@evenhead\@oddhead
\def\@oddfoot{\hbox to \textwidth{\fontsize{8bp}{9bp}\selectfont\hfill{\textbf{\thepage}}\hfill}}
\def\@evenfoot{\hbox to \textwidth{\fontsize{8bp}{9bp}\selectfont\hfill{\textbf{\thepage}}\hfill}}}
%%
%%%%%%%%%%%%%%%%General Packages End

\renewcommand{\singlespacing}{%
  \setstretch {\setspace@singlespace}%  normally 1
  \vskip \baselineskip  % Correction for coming into singlespace
}

\newif\ifonehalfspacing
\onehalfspacingfalse

\renewcommand{\onehalfspacing}{%
\global\onehalfspacingtrue
  \setstretch{1.5}%  default
  \ifcase \@ptsize \relax % 10pt
    \setstretch {1.5}%
  \or % 11pt
    \setstretch {1.238}%
  \or % 12pt
    \setstretch {1.266}%
  \fi
}

\renewcommand{\doublespacing}{%
  \setstretch {1.667}%  default
  \ifcase \@ptsize \relax % 10pt
    \setstretch {1.667}%
  \or % 11pt
    \setstretch {1.618}%
  \or % 12pt
    \setstretch {1.655}%
  \fi
}

\onehalfspacing

\renewcommand*{\LettrineTextFont}{}
\gdef\dropcap#1{\noindent#1}

\def\@linenumber{}
\def\linenumber#1{\gdef\@linenumber{#1}}
\def\linenumberfont{\normalfont\normalsize}

%%%%CMYK and RGB Color
\newcommand{\Secondcolorstyle}[1]{
\ifthenelse{\equal{#1}{cmyk}}{\definecolor{SecondColor}{cmyk}{0,.90,.77,0.28}}{%
\ifthenelse{\equal{#1}{rgb}}{%\definecolor{SecondColor}{rgb}{.73,.27,.25}
    \definecolor{SecondColor}{RGB}{183,19,42}
}{%
\plese_provide_Secondcolorstyle}}}

\Secondcolorstyle{rgb}

\gdef\Secondcolor#1{\textcolor{SecondColor}{#1}}
%%%%%%%%%

\pagestyle{headings}
%%%%%%%%%%%%%%%%Running Heda End

\gdef\sauthfn#1{#1}
\gdef\authfn#1{${}^{#1}$}

\def\@ArticleType{}%
\long\gdef\papertype#1{\def\@ArticleType{#1}}

\long\def\@corraddress{}
\long\def\corraddress#1{\def\@corraddress{\textsuperscript{*}#1}\par}

\long\def\@presentaddress{}
\long\gdef\presentaddress#1{\def\@presentaddress{#1}}

\def\@corresp{}%
\long\gdef\corresp#1#2#3{\begingroup%
\long\gdef\@corresp{\normalsize{#1}~\textbf{#2}: #3\vphantom{yg}\par}
\endgroup}

\def\@vol{}
\gdef\vol#1{\begingroup\gdef\@rhfootvol{#1}%
\gdef\@vol{#1}\endgroup}

\def\@articleno{}
\gdef\articleno#1{\begingroup\gdef\@rhfootvol{#1}%
\gdef\@articleno{#1}\endgroup}

\gdef\titelocabbrev#1{\gdef\@titelocabbrev{\ifthenelse{\equal{true}{#1}}{Eloc.}{Elocation}}}
\def\@elocationid{}
\def\elocationid#1{\begingroup%
\gdef\@elocationid{\@titelocabbrev~#1}\endgroup}

\gdef\MonthVal#1{%
\ifthenelse{\equal{#1}{1}}{January}{%
\ifthenelse{\equal{#1}{2}}{February}{%
\ifthenelse{\equal{#1}{3}}{March}{%
\ifthenelse{\equal{#1}{4}}{April}{%
\ifthenelse{\equal{#1}{5}}{May}{%
\ifthenelse{\equal{#1}{6}}{June}{%
\ifthenelse{\equal{#1}{7}}{July}{%
\ifthenelse{\equal{#1}{8}}{August}{%
\ifthenelse{\equal{#1}{9}}{September}{%
\ifthenelse{\equal{#1}{01}}{January}{%
\ifthenelse{\equal{#1}{02}}{February}{%
\ifthenelse{\equal{#1}{03}}{March}{%
\ifthenelse{\equal{#1}{04}}{April}{%
\ifthenelse{\equal{#1}{05}}{May}{%
\ifthenelse{\equal{#1}{06}}{June}{%
\ifthenelse{\equal{#1}{07}}{July}{%
\ifthenelse{\equal{#1}{08}}{August}{%
\ifthenelse{\equal{#1}{09}}{September}{%
\ifthenelse{\equal{#1}{10}}{October}{%
\ifthenelse{\equal{#1}{11}}{November}{%
\ifthenelse{\equal{#1}{12}}{December}{%
\ifthenelse{\equal{#1}{XX}}{XX}{%
}}}}}}}}}}}}}}}}}}}}}}}%

%%%%%%%%%%%%%%%%%%%%%%%%Title Page Journal Details %%%%%%%%%%%%%%%

\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width6pc
  \kern2.6\p@}

%%%%%%%%%%%%%%%%%%Author and Affiliation Definitions
\def\@asmauthor{}
\def\author#1{\def\@asmauthor{\let\authfn\sauthfn\fontsize{12pt}{14pt}\selectfont\bfseries\boldmath#1}}

\newcounter{affilnumcnt}%
\newcount\affilnumcnt%
\affilnumcnt=\theaffilnumcnt%

\def\@asmaffil{}%
\def\affil#1{\g@addto@macro\@asmaffil{\advance\affilnumcnt by1%
\fontsize{12pt}{14pt}\selectfont\mdseries\textsuperscript{\the\affilnumcnt}#1\vphantom{py}\par}}

%%%%%%%%%%%%%%%%%%Author And Affiliations definitions
%%%%%%%%%%%%%%%%%%Article Title Start
\renewcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{headings}
 \@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

\def\@articlesubtitle{}
\def\subtitle#1{\def\@articlesubtitle{#1}}

\def\@equalcontrib{}
\def\equalcontrib#1{\def\@equalcontrib{#1}}

  \let\footnotesize\normalsize
      \long\def\@makefntext#1{\parindent 1em\noindent
              \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}\onehalfspacing#1}

\def\@GeneralInstruction{}
\def\GeneralInstruction{\def\@GeneralInstruction{\vbox{\fboxsep=6pt\fbox{\vbox{\hsize=458pt[Use this template to prepare research and review papers in Overleaf for submission to any ASM journal except Microbiology Resource Announcements (MRA) and the Journal of Microbiology and Biology Education (JMBE). Replace all instructional wording below with the actual text of your manuscript, following the format as laid out here and deleting any elements that do not apply. Different article types have different formatting requirements and not all variations are captured in this template. Refer to the \href{https://journals.asm.org/links-to-journal-author-instructions}{\underline{specific instructions for individual journals}}, especially the pages on Submission and Review Process and Article Types, to understand the relevant formatting elements.]}}}\par\nointerlineskip\vspace*{2.5pc}
}}

\def\@maketitle{%
\setlength{\footnotesep}{9.9pt}
  \null\thispagestyle{headings}
\removelastskip\vskip0pt\vspace*{-44pt}\vbox{\hbox to\hsize{\smash{\lower24pt\hbox{\includegraphics[scale=2.1]{journals_logo_w_asm_logo_stacked_(1).jpg}}}\hfill\hbox{\fontsize{12}{0}\selectfont\Secondcolor{\bfseries\@ArticleType}}}}
\begingroup \removelastskip\vspace*{4.5em}
  \begin{flushleft}\parskip=0pt%
\@GeneralInstruction%
  \let \footnote \thanks\lineskip=12pt%
    {\fontsize{22}{24}\selectfont\bfseries\boldmath\Secondcolor{\@title}\par}%
\if!\@articlesubtitle!\relax\par\nointerlineskip%
\else\par\nointerlineskip\vskip2pc\bgroup\lineskip=12pt\fontsize{18}{20}\selectfont\textbf{\@articlesubtitle}\egroup\par\nointerlineskip\fi
    \vskip 2.5em
\if@DoubleBlind\relax\else    {\large
      \lineskip .5em%
{\fontsize{12pt}{14pt}\selectfont\bfseries\boldmath\@asmauthor}\par
{\fontsize{12pt}{14pt}\selectfont\mdseries\@asmaffil}\par
}\par%\lineskip .5em\@equalcontrib%
\if!\@equalcontrib!\relax\vskip1pc\else\vskip.5pc\lineskip.5em\normalsize\@equalcontrib\vskip1pc\fi%
\normalsize
\if!\@corraddress!\relax\vskip1pc\else\vskip.5pc\lineskip.5em\normalsize\@corraddress\vskip1pc\fi%
\if!\@presentaddress!\relax\vskip1pc\else\vskip.5pc\lineskip.5em\normalsize\@presentaddress\vskip1pc\fi%
\if!\@corresp!\relax\vskip1pc\else\vskip.5pc\lineskip.5em\normalsize\@corresp\vskip1pc\fi%
\fi
  \end{flushleft}%
\endgroup}

\def\Articleraggedright{%  non-hyphenated raggedright
  \@flushglue=0pt plus 1fil%
  \@rightskip\@flushglue \rightskip\@rightskip%
  \leftskip\z@skip%
  \parindent\z@}%

\Articleraggedright%Flush Left Paragraph
\setlength{\parindent}{18pt}
\setlength{\parskip}{.5\baselineskip}
%%%%%%%%%%%%%%%%ArticleTitle End
  \renewenvironment{abstract}{\begingroup\parskip=.5\baselineskip\pagestyle{headings}%
      \if@twocolumn%
        \section*{\abstractname}%
      \else%
{\raggedright\bfseries\Secondcolor{\abstractname}\vphantom{py}}\par\nointerlineskip\ifonehalfspacing\vspace*{12pt}\else\vspace*{6pt}\fi\noindent%
      \fi}%
      {\endgroup\if@twocolumn\else\vskip\baselineskip\fi}%

  \newenvironment{importance}{%
      \if@twocolumn%
        \section*{\importancename}%
      \else%
\removelastskip\par\nointerlineskip\vskip18pt%
{\raggedright\bfseries\Secondcolor{\importancename}\vphantom{yp}}\par\nointerlineskip\ifonehalfspacing\vspace*{12pt}\else\vspace*{6pt}\fi\noindent%
      \fi}%
      {\if@twocolumn\else\fi}%

  \newenvironment{keywords}{\begingroup\parindent=0pt\parskip=0pt%
      \if@twocolumn
        \section*{\keywordname}%
      \else
\removelastskip\par\nointerlineskip\vskip14pt%
{\raggedright\bfseries\Secondcolor{\keywordname}\vphantom{yp}}\par\nointerlineskip\ifonehalfspacing\vspace*{12pt}\else\vspace*{6pt}\fi\noindent%
      \fi}
      {\endgroup\if@twocolumn\else
\par\nointerlineskip\vskip12pt\noindent\rule{\textwidth}{0.5pt}
\fi}

  \long\def\keywords#1{\bgroup\parindent=0pt\parskip=0pt%
      \if@twocolumn
        \section*{\keywordname}%
      \else
\removelastskip\par\nointerlineskip\vskip14pt%
{\raggedright\bfseries\Secondcolor{\keywordname}\vphantom{yp}}\par\nointerlineskip\ifonehalfspacing\vspace*{12pt}\else\vspace*{6pt}\fi\noindent%
      \fi
#1\nobreak\par\nobreak\nointerlineskip\vskip12pt\nobreak\noindent\rule{\textwidth}{0.5pt}
\egroup}

%%%%%%%%%%%%%%%%%Heading-Level%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\raggedcenter{\leftskip=0pt plus 0.5fil\rightskip=0pt plus 0.5fil%
\parfillskip=0pt\let\hb=\break}%
\def\@HIIIsepwithspace{}

\def\log{\mathop{\rm log}\nolimits}

\def\@startsection#1#2#3#4#5#6{%
  \if@noskipsec\leavevmode\fi%
  \par%
  \@tempskipa #4\relax%
  \@afterindenttrue%
  \ifdim \@tempskipa <\z@%
    \@tempskipa -\@tempskipa \@afterindenttrue%
  \fi%
  \if@nobreak%
    \everypar{}%
  \else%
    \addpenalty\@secpenalty\addvspace\@tempskipa%
  \fi%
  \@ifstar%
    {\@ssect{#3}{#4}{#5}{#6}}%
    {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}}%

\setcounter{secnumdepth}{0}

\newcount\appseccnt%
\appseccnt=0%
\def\@seccntformat#1{\Secondcolor{\csname the#1\endcsname}\ifnum\appseccnt=1\relax\hskip0.5em\else\ignorespaces\hskip6pt\ignorespaces\fi}

\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi
\@tempskipa #5\relax
\ifdim \@tempskipa>\z@
\ifnum #2=1%
\begingroup%
\ifnum\appseccnt=1\relax
#6{{\hskip #3\relax\@svsec}%
\interlinepenalty \@M {\Secondcolor{#8}}\vphantom{\llap{yqlh}}\@@par\nobreak}%
\else
#6{{\hskip #3\relax\@svsec}%
\interlinepenalty \@M {\Secondcolor{#8}}\vphantom{\llap{yqlh}}\@@par\nobreak}\fi%
\endgroup
\else%
\ifnum #2=2%
\begingroup%
#6{{\hskip #3\relax\@svsec}%
\interlinepenalty \@M \Secondcolor{#8}\vphantom{\llap{yqlh}}\@@par\nobreak}%
\endgroup
\else%
\ifnum #2=3%
\begingroup
#6{{\hskip#3\relax\@svsec}%
\interlinepenalty\@M{\Secondcolor{#8}}\vphantom{\llap{yqlh}}\@@par\nobreak}%
\endgroup
\else%
\ifnum #2=4%
\begingroup
#6{{\hskip #3\relax\@svsec}%
\interlinepenalty\@M{\Secondcolor{#8}}\vphantom{\llap{yqlh}}\@@par\nobreak}%
\endgroup
\else
\ifnum #2=5%
\begingroup
#6{\@hangfrom{\hskip #3\relax\@svsec}%
\interlinepenalty \@M {\Secondcolor{#8}}\vphantom{\llap{yqlh}}\@@par\nobreak}%
\endgroup
\else%
  \undefine
\fi\fi\fi\fi\fi%
\else%
\def\@svsechd{\ifnum#2=3%
#6{{\hskip#3\relax\@svsec}{\Secondcolor{#8}}\vphantom{\llap{yqlh}}\@@par\nobreak}%
\else%
\ifnum #2=4%
#6{\hskip #3\relax%
\@svsec{\Secondcolor{#8}}\@@par\nobreak}%
\else%
\ifnum #2=5%
#6{\hskip #3\relax%
\@svsec \Secondcolor{#8}\@@par\nobreak}%
\else
#6{\hskip #3\relax\@svsec #8\@@par}%
\fi\fi\fi%
}%
\fi%
\@xsect{#5}}%

\def\@xsect#1{%
\@tempskipa#1\relax%
\ifdim \@tempskipa>\z@%
\par\nobreak%
\vskip\@tempskipa%
\@afterheading%
\else%
\@nobreakfalse%
\global\@noskipsectrue%
\everypar{%
\if@noskipsec%
\global\@noskipsecfalse%
{\setbox\z@\lastbox}%
\clubpenalty\@M%
\begingroup\hskip-2.5pt\@svsechd\endgroup%
\unskip%
\@tempskipa#1\relax%
\hskip-\@tempskipa%
\else%
\clubpenalty\@clubpenalty%
\everypar{}%
\fi}%
\fi\ignorespaces}

\let\sectitle=Z%
\let\subsectitle=Z%
\let\subsubsectitle=Z%
\let\paratitle=Z%
\let\subparatitle=Z%

\def\@ssect#1#2#3#4#5{%
\@tempskipa #3\relax
\ifdim \@tempskipa>\z@
\begingroup
#4{%
\@hangfrom{\hskip#1}%
\ifx A\sectitle%%
{\interlinepenalty \@M{\Secondcolor{#5}}\@@par}
\else
\ifx A\subsectitle%
{\interlinepenalty \@M \Secondcolor{#5}\@@par}
\else
\ifx A\subsubsectitle%
{\interlinepenalty \@M{\Secondcolor{#5}}\@@par}
\else
\ifx A\paratitle
{\interlinepenalty \@M{\Secondcolor{#5}}\@@par}
\else
\ifx A\subparatitle%
{\interlinepenalty\@M\Secondcolor{#5}\@@par}
\else
{\interlinepenalty \@M\Secondcolor{#5}\@@par}
\fi\fi\fi\fi\fi%
}%
\endgroup%
\else
\def\@svsechd{#4{\hskip #1\relax \Secondcolor{#5}\@HIIIsepwithspace}}%
\fi
\@xsect{#3}}

\newskip\hibelowskip
\hibelowskip=2.3ex \@plus.2ex
\advance\hibelowskip by -.5\baselineskip
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {\hibelowskip}%
                                   {\parindent=0pt\fontsize{14}{16}\selectfont\bfseries\boldmath}}%

\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {\hibelowskip}%
                                     {\parindent=0pt\normalfont\bfseries\boldmath}}

\newskip\hiiibelowskip
\hiiibelowskip=1.3ex \@plus.2ex
\advance\hiiibelowskip by -.5\baselineskip
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {\hibelowskip}%
                                     {\parindent=0pt\normalfont\normalsize\bfseries\itshape\boldmath}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                     {\hiiibelowskip}%
                                    {\parindent=0pt\normalfont\normalsize\bfseries\boldmath\noindent}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\z@}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {\hiiibelowskip}%
                                      {\parindent=0pt\normalfont\normalsize\bfseries\boldmath}}

%%%%%%%%%%%%%%%%%Heading-Level End%%%%%%%%%%%%%%%%
%%%%%%%%%%Appendix

%%%%%%%%%%%%%%%%% custom commands to italicize gene and species names
\newcommand{\Btheta}{\textit{B.~theta}}
\newcommand{\Ecoli}{\textit{E.~coli}}
\newcommand{\Bacteroides}{\textit{Bacteroides}}
\newcommand{\Proteobacteria}{\textit{Proteobacteria}}
\newcommand{\oriV}{\textit{oriV}}
\newcommand{\trfA}{\textit{trfA}}
\newcommand{\oriT}{\textit{oriT}}
\newcommand{\ermF}{\textit{ermF}}
\newcommand{\repA}{\textit{repA}}
\newcommand{\chuR}{\textit{chuR}}
\newcommand{\anSME}{\textit{anSME}}
\newcommand{\recA}{\textit{recA}}
\newcommand{\tdk}{\textit{tdk}}
\newcommand{\rnpB}{\textit{rnpB}}
\newcommand{\ilvGEDA}{\textit{ilvGEDA}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%% custom commands to italicize gene and species names  END

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Miscellaneous Start
\newenvironment{fullwidth}{\begingroup}{\endgroup}
\newenvironment{acknowledgments}{\begingroup\if@DoubleBlind\setbox0\vbox\bgroup\fi}{\if@DoubleBlind\egroup\fi\endgroup}
\newenvironment{funding}{\begingroup\if@DoubleBlind\setbox0\vbox\bgroup\fi}{\if@DoubleBlind\egroup\fi\endgroup}
\newenvironment{conflictsinterest}{\begingroup\if@DoubleBlind\setbox0\vbox\bgroup\fi}{\if@DoubleBlind\egroup\fi\endgroup}
\newenvironment{authorbios}{\begingroup\if@DoubleBlind\setbox0\vbox\bgroup\fi}{\if@DoubleBlind\egroup\fi\endgroup}
\newenvironment{authorfootnotes}{\begingroup\if@DoubleBlind\setbox0\vbox\bgroup\fi}{\if@DoubleBlind\egroup\fi\endgroup}
\newenvironment{authorcontributions}{\begingroup\if@DoubleBlind\setbox0\vbox\bgroup\fi}{\if@DoubleBlind\egroup\fi\endgroup}
\renewcommand\textsc{}
\renewcommand\sc{}

\def\headrowfillerT{}
\def\headrow{}
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Miscellaneous End

\renewcommand{\figurename}{\textbf{FIG}}
\renewcommand\thefigure{\textbf{\@arabic\c@figure}}

%%*************************** Table *********************************

\renewenvironment{table}[1][]{%
\renewcommand*{\arraystretch}{1.5}
\@float{table}\fontsize{10pt}{18pt}\selectfont}{\end@float}

\renewcommand{\tablename}{\textbf{TABLE}}
\renewcommand\thetable{\textbf{\@arabic\c@table}}

\belowbottomsep=4pt%
\belowcaptionskip=8pt%
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\leftskip2pc\rightskip2pc plus 1fil\parfillskip=0pt\relax\baselineskip=18pt#1. #2}%
  \ifdim \wd\@tempboxa >\hsize
\leftskip2pc\rightskip2pc plus 1fil\parfillskip=0pt\relax%
\baselineskip=18pt#1. #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

\long\def\nifigmakecaption#1{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\leftskip2pc\rightskip2pc plus 1fil\parfillskip=0pt\relax\baselineskip=18pt\fnum@figure\quad#1}%
  \ifdim \wd\@tempboxa >\hsize
\leftskip2pc\rightskip2pc plus 1fil\parfillskip=0pt\relax%
\baselineskip=18pt\fnum@figure\quad#1\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

\long\def\nifigcaption#1{%
  \par
  \begingroup
    \@parboxrestore
    \normalsize
    \nifigmakecaption{#1}\par
  \endgroup}

\renewenvironment{figure}[1][]{\refstepcounter{figure}\let\caption\nifigcaption}{}

\long\def\nitabmakecaption#1{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\leftskip2pc\rightskip2pc plus 1fil\parfillskip=0pt\relax\baselineskip=18pt\fnum@table\quad#1}%
  \ifdim \wd\@tempboxa >\hsize
\leftskip2pc\rightskip2pc plus 1fil\parfillskip=0pt\relax%
\baselineskip=18pt\fnum@table\quad#1\par\nobreak
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\box\@tempboxa\hfil}%
  \fi
  \vspace*{\belowcaptionskip}}

\long\def\nitabcaption#1{%
  \par
  \begingroup
    \@parboxrestore
    \normalsize
    \nitabmakecaption{#1}\par
  \endgroup}

\renewenvironment{table}[1][]{\refstepcounter{table}\let\caption\nitabcaption}{\par\nointerlineskip\vspace*{18pt}}

%%*************************** BOX *********************************

\definecolor{shadecolor}{cmyk}{0,0,0,0.10}

%%***********************************Bibliograpy%%%%%%%%%%%%%%%%%

\newcommand{\JournalTitle}[1]{\textit{#1\/}}

%%***********************************Bibliograpy%%%%%%%%%%%%%%%%%
\urlstyle{rm}
\def\asmafn#1{${}^{\mathrm{#1}}$}
\let\afn\asmafn

\newcommand{\R}[1]{\label{#1}\linelabel{#1}}

\usepackage[plain]{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{natbib}
\usepackage[acronym,nomain]{glossaries}
\usepackage{comment}

\renewcommand\abstractname{ABSTRACT}
\newcommand\importancename{IMPORTANCE}
\newcommand\keywordname{Keywords}
\def\verbatim@font{\normalfont%
\hyphenchar\font\m@ne
\@noligs}

\endinput
%%
%% End of file `asmarticle.cls'.
