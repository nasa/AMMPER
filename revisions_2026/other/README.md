# AMMPER-2 revision, 2026 — what is in here and how to use it

This folder is the deliverable for the current revision round. It contains the
corrected analysis code, the corrected figure panels as drop-in replacements for
the ones in the manuscript, and copies of `main.tex` / `supplemental.tex` with
the text and figure paths updated so the revised manuscript can be compiled
directly.

```
revisions_2026/
  code/          corrected model, fitter, figure generators
  figures/
    main_figure_panels/       Figure 2 and its two intermediate panels
    supplementary_material/   Figure S1 panels
    bug_illustration/         the before/after panel for BUGS_AND_CORRECTIONS.md,
                              which is not a manuscript figure
  other/         this README, REVISION_NUMBERS.md, BUGS_AND_CORRECTIONS.md
  Panels/        everything main.tex needs, ready for Overleaf
  Figures/       everything supplemental.tex needs, ready for Overleaf
  main.tex, supplemental.tex, references.bib, asmarticle.cls, asm.bst
```

## Compiling the revised manuscript

Upload the contents of `revisions_2026/` to Overleaf as-is and compile `main.tex`
and `supplemental.tex` separately. `Panels/` and `Figures/` already hold every
image both files reference — corrected figures where they changed, and the
originals where they did not — so no paths need editing.

`pdflatex` is not installed in the environment where this was prepared, so the
LaTeX was verified structurally rather than by compiling: all `\includegraphics`
targets resolve, all `\ref` targets exist, there are no duplicate labels, all
environments and braces balance, all `\cite` keys are present in
`references.bib`, and the tabular in Text S4 has consistent column counts. It has
not been through a real TeX run — please check the PDF once before circulating.

## What changed, in one paragraph

The kinetic model of alamarBlue reduction takes a population growth curve from an
AMMPER simulation as its only input. The code that assembled that curve tabulated
per-generation cell counts with `pandas.value_counts()`, which sorts by frequency
rather than by generation index, and then discarded the index. Because the
simulated population grows monotonically, this fed the ODE a trajectory running
backwards in time. Correcting it, and accounting for the density of the
experimental inoculum with one offset parameter, reduces the wild type mean
absolute error from 0.241 to 0.030. One smaller bug was fixed at the same time:
two transposed dose labels. A third apparent bug — a normalization step that
reuses its own output in its denominator — turned out to be the deliberate
published convention and was left alone; see below.

## The two bugs

**Bug 1 — growth curve reversed in time (critical).** `value_counts()` sorts by
count, descending; for a monotonically growing population that equals descending
generation order. Verified against raw simulation output rather than inferred:

```
correct  (gen 0→15): [   1    2    4    8 ... 3887 4093 4096]
what the code fed:   [4096 4093 3887 2991 ...    4    2    1]
```

Consequence: the full metabolic capacity of a saturated population is present
from t=0, so the predicted blue species collapses almost linearly from the first
timestep instead of following a sigmoid — which is the shape in the submitted
Figure 2. It also flattened the dose response, because under the reversed
ordering every dose starts from the same saturated population. That is the
mechanical explanation for the reviewer's observation that the predictions looked
identical across doses.

Present in 11 analysis scripts, including the SMAC3 fitting scripts, which is why
the published parameters are specific to the reversed trajectory and had to be
re-estimated rather than carried over.

**Bug 2 — 5 Gy and 10 Gy panel labels transposed.** Folder `WT_Basic_10` holds
`10Gy.txt` and pairs with the 10 Gy CSV — data consistent — but was labeled
"5 Gy"; `WT_Basic_50` is the mirror case. Simulation and experiment were paired
correctly, so the fit was unaffected; two published panels carried the wrong dose
label. Fixed structurally: dose now lives in one `Condition` record shared by the
folder name, the CSV name, and the panel label.

## One modeling change, which is not a bug fix

AMMPER starts from a single cell; the plate reader starts from a dense inoculum.
Fitting to the full simulated trajectory forces the model to explain ~9
generations of near-zero signal the experiment never sees. We added one
parameter, `g0`, giving the point on the growth curve corresponding to
experimental t=0, fit once per strain and held fixed across all six doses. For
the wild type `g0 = 8.81` generations (~463 cells at t=0), and for rad51Δ 6.21
(~77 cells). Refitting it per dose moves it by less than one generation for the
wild type (8.24–9.11), so a single shared value is justified; the mutant is looser
(4.97–7.27) and a single value is kept there too rather than spending five more
parameters.

## Results

| Configuration | WT mean MAE | rad51Δ mean MAE |
|---|---|---|
| As published (reversed input, published params) | 0.241 | 0.380 |
| Ordering fixed, published params | 0.396 | 0.152 |
| Ordering fixed, re-estimated | **0.030** | **0.072** |

Fixing the ordering while keeping the published parameters makes things *worse*.
That is expected — those parameters encode the bug — and the informative
comparison is first row against last.

**The 0.522 in the original submission is not reproducible, and the metric it was
quoted under was not the one the manuscript defined.** The submitted manuscript
reports a wild type error of 0.522; a faithful replay of the reversed-input
configuration gives 0.241. The cause is that the published number came from
`aBFinalplotsSMAC.py`, whose `accuracy_ML()` returns a *sum of squared residuals
over the first eight timepoints* — not a mean absolute error, despite the
manuscript's equation defining one. Evaluated on the same trajectories that
objective returns ≈1.93, so 0.522 cannot be recovered from it either; it
presumably came from an intermediate variant of the script that no longer exists.

The revision uses mean absolute error throughout — in the fitting objective, the
figures, and every number in both `.tex` files — and the manuscript's equation now
matches it. Where the Grid Search / Bayesian Optimization comparison is reported,
the Methods now state explicitly that *that* search minimized a sum of squared
residuals and is not on the same scale as the model errors reported elsewhere, so
0.586 cannot be read against 0.030. The 0.522 is not carried forward as a
baseline; `verify_manuscript_numbers.py` asserts it is absent from both files.

The rad51Δ result is deliberately the most constrained test available: all six
kinetic constants are held at their wild type values and only `g0` is refit, so
one free parameter carries the entire strain difference. The fitted value, 6.21
generations against the wild type's 8.81, says the mutant behaves like a wild
type population ~2.6 generations behind.

Per-dose tables, the identifiability checks, and the population-level numbers are
in `REVISION_NUMBERS.md`, regenerated by `code/generate_revision_numbers.py`.

## The experimental data is unchanged — including a normalization that looks wrong

The experimental series are processed exactly as in the original submission:
absorbances converted to concentrations with the same chemistry, then normalized
against the total dye present at t = 0. The published code does this with

```python
B_C = B_C / (B_C[0] + P_C[0])
P_C = P_C / (B_C[0] + P_C[0])   # B_C[0] is already normalized on this line
```

so the two series are divided by *different* denominators — blue by the raw
initial total (0.3545 for WT 0 Gy), pink by 0.9595. Read as source code that is
an aliasing bug, and partway through this revision it was treated as one and
rewritten to compute the denominator once. **That was wrong and has been
reverted.** Under the shared denominator pink is scaled up 2.7× and measured
blue + pink reaches 1.065, which two fractions of a conserved dye pool cannot do
and which no three-species model can match. Under the published form the sum is
at most 1, peaks at t = 0 in all twelve conditions, and falls monotonically; its
deficit rises from 0.03–0.04 to 0.18–0.58, behaving exactly like the unmeasured
colorless species. That is what puts the measurements in the same units as
predictions normalized by their own three-species total.

Two conditions (rad51Δ at 0 and 2.5 Gy) exceed 1 at t = 0 by 0.3% and 0.08% —
first-timepoint absorbance noise. Reported, not rescaled away.

An earlier draft also considered normalizing each timepoint by its own blue +
pink sum (option a). Also rejected: no publication using alamarBlue reports the
two species as fractions of each other.

### What the reverted change had cost

The wrongly normalized data was the basis of several conclusions in the previous
draft of this revision, none of which survive:

| Claim in the previous draft | Status now |
|---|---|
| WT MAE 0.060, rad51Δ 0.109, all 12 0.085 | **0.030 / 0.072 / 0.051** |
| Measured pink reaches 1.055; sum drifts up 9% | No drift; sum is maximal at t = 0 and falls |
| A ~0.05 residual is *structural* and unreachable | No floor; the earlier residual was the rescaling |
| A690 turbidity leak explains the excess (slope 0.087) | Nothing to explain |
| Optical observation equation, 4-arm ablation, 0.085 → 0.078 | **Withdrawn**; code moved to `revisions_2026/retired/` |
| WT-only kinetics do not transfer (0.378 on rad51Δ) | They transfer fine: 0.0722 vs 0.0716 joint |
| K3 alone is non-identifiable | **K2 and K3** are; K1 is constrained |

The joint fit is still what gets reported, but now because it is the more
constrained arrangement rather than because the alternative fails. The optical
leak has been removed from `predict()` entirely rather than retained with a
zero default, since keeping it would mean carrying two parameters that existed
only to undo a self-inflicted error. `revisions_2026/retired/README.md` records
what was tried.

## What did not get fixed, and why

**Two half-saturation constants are not identifiable.** `K2` and `K3` both land
far above the amount of the corresponding substrate, putting the reversible
pink↔colorless pair in an effectively first-order regime where only the ratios
`v2/K2` and `v3/K3` are constrained. Scaling either pair together by 100× leaves
the WT mean MAE at 0.0302 to four decimal places, whereas the same scaling on
`(v1,K1)` gives 0.059 — nearly double — so the flat response is a property of the
data, not of the search bounds. Which pairs behave this way is read off the scan
in `generate_revision_numbers.py` rather than assumed, and it *changed* when the
normalization was reverted: the earlier round found `K1` and `K3` flat and `K2`
constrained. Which Michaelis-Menten term the data constrain depends on the
amplitude of the series being fit. Reported as non-identifiable in Text S4 rather
than quoted as measured values.

**The dose response is still too weak, by 4–5× (WT) and 10–12× (rad51Δ).** The
direction is now right — for rad51Δ the predicted ordering is perfectly
monotonic in dose — but the magnitude is compressed. This is a real limitation,
not a bug, and it is traceable: the simulated healthy-cell count varies by only
5.5% (WT) and 7.9% (rad51Δ) across the whole 0–30 Gy range, the fitted weight on
damaged cells is small (`k = 0.034`), and damaged rad51Δ cells are assigned
directly to health state 3 (nonviable) rather than 2 in `cellDefinition.py`, so
that strain has *no* health-2 cells at any dose. Its entire dose response
therefore travels through the healthy-cell count alone. A graded health state
with a per-cell metabolic rate is the fix, and that is a model extension rather
than a correction.

**The gamma supplement (Figs. S8–S10) has not been rechecked, and this is the
next open item.** `aBFinalplotsGAMMA.py` has the same `value_counts()` bug, so the
poor gamma fit may be the bug rather than a modeling gap — that question is not yet
answered. Rerunning it is not straightforward: the simulation output for the rad51
gamma conditions (`rad51_0`, `rad51_25`, `rad51_300`) is not present anywhere in
the `updated_figures` branch — only `WT_25`, `WT_250` and `WT_25k50` exist — and
the script as written also pairs against the *proton* rad51 experimental CSVs, so
answering it may require re-running the gamma simulations.

The supplement's treatment of gamma is unchanged from the submitted version and
does not claim a quantitative fit: Text S3 presents the mode as exploratory and
explicitly not a validated capability, and the Discussion says the aB model did not
reproduce the gamma data. So nothing there is wrong as written. But if the fit
improves once the ordering is fixed, gamma becomes a candidate for the main text
and those passages would need rewriting. Tracked in
`other/BUGS_AND_CORRECTIONS.md`.

**Figure 1 is unaffected.** Worth recording because it is easy to assume
otherwise. `generate_growth_curves.py` does contain the same bug, and the stale
asset `results/figures_updated_figures_branch/growth_curves_panel.png` shows
visibly reversed curves. But the panel actually used in the manuscript,
`Panels/comprehensive_2row_panel.png`, shows correct ascending sigmoids in B and
C. It was produced by a different route and needs no change. (Diagnostic detail:
in that script the *unhealthy* counts go through an explicit reindexing loop and
are correct, while the healthy counts go through raw `value_counts()` — which is
exactly what the stale asset shows, healthy descending and unhealthy rising.)

**SMAC3 was not rerun.** The re-estimation here uses
`scipy.optimize.differential_evolution`, seeded, to avoid adding a heavyweight
dependency. The optimizer is not what is being tested, and the Methods text now
says so. If the reviewers want the SMAC3 numbers on the corrected model,
`analysis/aB/aBFinalplotsSMAC.py` can be pointed at
`code/ammper_ab_model_fixed.py`.

## Reproducing everything

From `revisions_2026/code/`, in order:

```bash
python3 fit_ab_model_fixed.py         # ~20 min; writes fitted_parameters.json
python3 make_figure2_panels.py        # ~1 min; writes all figure panels
python3 generate_revision_numbers.py  # writes other/REVISION_NUMBERS.md
python3 verify_manuscript_numbers.py  # checks the .tex against the above
```

`fit_ab_model_fixed.py` is seeded and its fitted values are already recorded as
constants in `ammper_ab_model_fixed.py`, so the later scripts can be run on
their own without refitting.

`verify_manuscript_numbers.py` is the guard against stale numbers: it reads the
generated JSON and greps `main.tex` and `supplemental.tex` for every value they
quote, comparing at the precision at which each is written, and it also checks
`\ref` targets, `\cite` keys against `references.bib`, `\includegraphics` targets
against the files on disk, and tabular column counts. It exits non-zero on any
failure, so it can be run after each edit. It reports 98/98 as committed. It does
not compile LaTeX; `pdflatex` was not available in the environment where this
revision was prepared, so the structural checks parse the sources directly and a
real compile on Overleaf is still the final word.

One thing to know if you re-run the two report scripts: a predicted curve can be
computed either by averaging the predictions of the replicate simulations or by
predicting once from the averaged growth curve. The dye kinetics are nonlinear in
the cell count, so the two differ slightly --- under 0.0005 in MAE, but enough to
straddle a rounding boundary at 10 Gy and about ten percent on the between-dose
spread. Both scripts now use the first route, because that is the curve plotted
in Figure 2 and Figure S1 and therefore the one a reader can measure against. The
fitting objective still uses the second, which is why `fitted_parameters.json`
and the figures can disagree in the fourth decimal place.

Requires numpy, pandas, scipy and matplotlib, and expects the reorganized tree
(it imports `ammper_paths` from the repository root to locate the experimental
data and simulation output).

## Figure substitution map

| Manuscript file | Replace with | Status |
|---|---|---|
| `comprehensive_alamarblue_stacked.{png,pdf,svg,tiff}` | `figures/main_figure_panels/` same names | **corrected** |
| `BlueCurveaBWTpredictionsall.png` | `figures/supplementary_material/` same name | **corrected** |
| `pinkcurveswtpredictionall.png` | `figures/supplementary_material/` same name | **corrected** |
| — | `figures/supplementary_material/aB_predictions_only_WT.png` | **new** (Fig. S1 cont.) |
| — | `figures/supplementary_material/aB_predictions_only_rad51.png` | **new** (Fig. S1 cont.) |
| `comprehensive_2row_panel.png` | unchanged | not affected |
| `Gammarad51_*.png` | unchanged | cannot rerun, flagged in Text S3 |

`WT_panel_all_doses.png` and `rad51_panel_all_doses.png` are the intermediate
per-strain panels that get stitched into Figure 2 as B and C. Neither document
includes them directly, so they are build products rather than manuscript
figures: `make_figure2_panels.py` writes them, `stacked_figure()` consumes them,
and they are kept out of the manuscript's `figures/` folder (the copies there are
in `legacy/intermediate_panels/`).

### Figure 2A is the reduction scheme, not the cell cartoon

Panel A is the alamarBlue reduction scheme — resazurin, resorufin and
dihydroresorufin drawn as structures with `(v1,K1)` and `(v2,K2)/(v3,K3)` on the
arrows — which is what the Figure 2 caption describes and what the current
submission shows. An intermediate rebuild of the composite pulled in an older
cell cartoon (`results/figures_updated_figures_branch/aBcartoon.PNG`) instead,
which contradicted the caption. The scheme is restored and the script is
commented so it does not get swapped back.

The scheme survives only as a raster. It was recovered at 403×503 from the base64
payload embedded in the submitted `comprehensive_alamarblue_stacked.svg` — where
matplotlib had stored it bottom-up, so it is flipped on extraction — and is now
committed as `code/assets/aB_chem_scheme.png` so the figure script has no
dependency on the Overleaf zip. Searching the repository, `results/`, and
`legacy/` turned up no higher-resolution copy and no vector original; the
submitted PDF embeds the same 403×503 raster. **That is the resolution ceiling of
Figure 2A.** If the original artwork (ChemDraw, Illustrator, BioRender) still
exists, dropping it into `code/assets/` and repointing `PANEL_A` is a one-line
change and worth doing before production, since ASM will want ≥300 dpi at final
column width.

## Manuscript text that changed

- **Results**, aB section: new MAE values (0.030 / 0.072), the `g0` explanation,
  and the rad51Δ paragraph rewritten — the old text said the model missed the
  *timing*, which is no longer true; the residual misfit is now a plateau effect
  at low dose.
- **Results**, dose response: both paragraphs rewritten. The previous versions
  described the weak dose response as an inherent limitation with placeholder
  numbers read off a figure; those numbers were `%TODO` placeholders and are now
  measured, and the mechanism is attributed correctly.
- **Methods**: new subsection "Parameter fitting for the kinetic model" (the
  fitting arrangement, the optimizer, the fitted values and the identifiability
  finding); Equation (error) now defines the mean absolute error that is actually
  reported, and the Grid Search / BO values are marked as being on a different,
  sum-of-squares scale.
- **Discussion**: the latency paragraph rewritten around what the model does *not*
  require, rather than around what we previously believed; the dynamic-range
  paragraph rewritten around the damaged-cell representation.
- **Supplement**: Fig. S1 caption updated, two new figures added, and new
  Text S4 — "Fitting the alamarBlue kinetics model" — with the error metric
  defined explicitly, the twelve-condition error table, the fitted values, the
  inoculum offset, the identifiability scan, the mass-conservation constraints and
  the generation time as a fixed hyperparameter rather than a measured property.

### The manuscript states the corrected results; it does not narrate the bugs

Neither `.tex` file tells the story of what the submitted version got wrong. There
are no before/after tables, no "we previously reported", and no account of the
reverted normalization change: a paper that narrates its own corrections is harder
to read than one that states its results, and the comparison matters to us rather
than to a reader of the paper.

That record is `other/BUGS_AND_CORRECTIONS.md` — the file to share with the author
team and to draw on for the response to the editor. `verify_manuscript_numbers.py`
enforces the split in both directions: the pre-correction baselines must be
**absent** from the two `.tex` files and **present** in that note, and the note's
headline numbers must match the current fit, so it cannot go stale against the
paper it explains.
- **References**: unchanged (81 entries). Three alamarBlue references were
  gathered and briefly cited while we were considering per-timepoint
  normalization (option a, where blue + pink = 1 at every timepoint). We went
  with option b instead — the published convention, where blue + pink + colorless
  sums to 1 against the t = 0 total — so those citations were removed along with
  the sentences that introduced them. The literature survey is not lost; it is
  what established that no published study normalizes the way option a would
  have, which is why option b was kept.

Both `%TODO` placeholder numbers that were in the Results are now resolved.
