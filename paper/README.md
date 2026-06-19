# Paper Policy

Do not draft results or conclusions before paper-facing artifacts exist. Paper
claims must map to `docs/claim_evidence_matrix.md` and pass ARIS claim audit.

## Building the PDF (reproducible)

The paper is `main.tex` + `references.bib` (BibTeX, `plain` style). Build:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # resolve cross-refs
```

### TeX environment / known compile pitfall

`microtype` is loaded with its default **font expansion** enabled, which pdfTeX
allows only with **scalable** (Type1/OpenType) fonts. On a minimal TeX install whose
default Computer Modern is the bitmap (Type3) variant, the first pass aborts with:

```
! pdfTeX error (font expansion): auto expansion is only possible with scalable fonts.
```

The fix is in the preamble: `\usepackage{lmodern}` is loaded **before**
`\usepackage[T1]{fontenc}`, so the document uses the scalable Latin Modern Type1
fonts and font expansion succeeds. Do not remove `lmodern`, or compile with a TeX
distribution whose default CM fonts are scalable.

Verified to build cleanly (3 passes + BibTeX, no undefined citations/references) with
MiKTeX (`pdfTeX`, MiKTeX `x64`) on Windows; the only residual messages are cosmetic
(`T1/lmr/m/scit` small-caps-italic font-shape substitutions and BibTeX
`volume`+`number` style notes), neither of which affects the output.

Build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.out`, `*.pdf`, `*.toc`) are
git-ignored; only `main.tex` and `references.bib` are tracked.
