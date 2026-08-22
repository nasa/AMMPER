# AMMPER-2 aB model: the bugs we found, and what changed

**Status:** internal note for the author team. None of this appears in the
manuscript, which reports only the corrected results. This file is the record of
*why* the numbers changed between the submitted version and the revision, so that
anyone on the team can reconstruct it, and so it can be summarized for the
editor in the response letter if we choose to.

**Headline:** the fit is much better than what we submitted. Wild-type mean
absolute error goes from 0.241 to **0.030**, and *rad51*Δ — which we previously
excluded from the comparison as unfittable — now comes in at **0.072** using the
same dye chemistry and one strain-specific parameter.

| | Submitted | Corrected |
|---|---|---|
| Wild type | 0.241 | **0.030** |
| *rad51*Δ | 0.380 (not reported; excluded as too poor) | **0.072** |
| All twelve conditions | — | **0.051** |

All errors are mean absolute error in concentration-fraction units, averaged over
the blue and pink series and over all timepoints. See "The metric" below — this is
not the metric the submitted manuscript's equation described.

---

## Bug 1 — the growth curve was fed to the ODE backwards (critical)

The kinetic model takes a population growth curve from an AMMPER simulation as
its only input. The code that assembled that curve tabulated per-generation cell
counts with `pandas.value_counts()`, which sorts by **frequency**, descending —
and then discarded the index. For a monotonically growing population, sorting by
descending count *is* sorting by descending generation. So the ODE was integrated
over a trajectory running backwards in time: it started at the final saturated
population and decayed toward a single cell.

Verified against raw simulation output rather than inferred:

```
correct  (gen 0 -> 15): [   1    2    4    8 ... 3887 4093 4096]
what the code fed:      [4096 4093 3887 2991 ...    4    2    1]
```

**Why this produced exactly the artifacts the reviewer noticed.** With the full
metabolic capacity of a saturated population present from *t* = 0:

1. The predicted blue species collapses almost linearly from the first timestep
   instead of following a sigmoid — which is the shape in the submitted Figure 2.
2. Every dose starts from the same saturated population, so the predicted curves
   are nearly indistinguishable across doses. That is the mechanical explanation
   for the reviewer's observation that the predictions looked identical at every
   dose. It was not a modeling limitation; it was this.

**Scope.** The bug was in the assembly of the input curve only. It did not touch
the simulation engine, the experimental data, the kinetic equations, or the
pairing of simulated with measured conditions. But it was present in **11
analysis scripts**, including the SMAC3 fitting scripts — which is why the
published parameters are specific to the reversed trajectory and had to be
re-estimated rather than carried over.

**The fix.** Tabulate on the generation index and reindex onto the full range of
generations, so a generation with no recorded cells contributes a zero instead of
being dropped (which would shift everything after it).

Correcting the ordering while *keeping* the published parameters makes the fit
worse (0.241 → 0.396). That is expected — those parameters encode the bug — so
the informative comparison is submitted-vs-refit, not the middle column.

| Dose | Reversed, published params | Corrected, published params | Corrected, refit |
|---|---|---|---|
| **WT** 0 Gy | 0.226 | 0.420 | **0.029** |
| 2.5 Gy | 0.223 | 0.427 | **0.030** |
| 5 Gy | 0.238 | 0.403 | **0.023** |
| 10 Gy | 0.251 | 0.379 | **0.027** |
| 20 Gy | 0.247 | 0.385 | **0.025** |
| 30 Gy | 0.261 | 0.359 | **0.048** |
| **mean** | **0.241** | **0.396** | **0.030** |
| ***rad51*Δ** 0 Gy | 0.352 | 0.233 | **0.097** |
| 2.5 Gy | 0.353 | 0.230 | **0.091** |
| 5 Gy | 0.391 | 0.110 | **0.053** |
| 10 Gy | 0.405 | 0.087 | **0.064** |
| 20 Gy | 0.416 | 0.073 | **0.075** |
| 30 Gy | 0.364 | 0.176 | **0.050** |
| **mean** | **0.380** | **0.152** | **0.072** |

## Bug 2 — 5 Gy and 10 Gy panel labels transposed

Folder `WT_Basic_10` holds `10Gy.txt` and pairs with the 10 Gy CSV — data
consistent — but was labeled "5 Gy" on the figure; `WT_Basic_50` is the mirror
case. Simulation and experiment were paired correctly, so **the fit was
unaffected**; two published panels simply carried the wrong dose label.

Fixed structurally rather than by hand: dose now lives in one `Condition` record
shared by the folder name, the CSV name, and the panel label, so the three cannot
drift apart again.

## The metric — the equation in the submitted manuscript described the wrong thing

Worth being careful about, because it affects how the old numbers read.

The submitted manuscript gives an equation for **mean absolute error** and calls
the reported errors by that name. But the objective the code actually minimized
(`accuracy_ML()` in `aBFinalplotsSMAC.py` and its siblings) was:

```python
delta  = sum((Experimental_B - Predicted_B) ** 2)
delta1 = sum((Experimental_P - Predicted_P) ** 2)
return delta + delta1
```

— a **sum of squared residuals** over the first eight timepoints. Not a mean, not
absolute. So the equation in the manuscript and the number next to it were not
the same quantity.

This is also why **0.522 is not reproducible.** A faithful replay of the
reversed-input configuration gives 0.241 under MAE; the SSR objective on the same
trajectories gives ≈1.93. Neither is 0.522, so it presumably came from an
intermediate variant of the script that no longer exists.

**What we did about it.** The revision uses MAE throughout — in the code, the
figures, and every number in the manuscript — and the manuscript's equation now
matches it. Where the Grid Search vs. Bayesian Optimization comparison is
reported, the manuscript now states explicitly that that comparison minimized SSR
and is not on the same scale as the reported model errors. The 0.522 is gone
rather than being carried forward as a baseline.

## Not a bug — the experimental normalization that looks like one

Recording this because it cost us a round, and because the next person to read
the code will reach for the same "fix."

The conversion from absorbance to concentration fraction does:

```python
B_C = B_C / (B_C[0] + P_C[0])
P_C = P_C / (B_C[0] + P_C[0])   # <-- B_C[0] is ALREADY normalized on this line
```

so blue is divided by the raw initial total (0.3545 for WT 0 Gy) and pink by
0.9595 — different denominators, a factor of 2.7 apart. Read as source code that
is an aliasing bug, and partway through this revision we treated it as one and
rewrote it to compute the denominator once.

**That was wrong.** Under the shared denominator, pink scales up 2.7× and
measured blue + pink reaches **1.065** — two fractions of a conserved dye pool
cannot sum to more than 1, and no three-species model can match it. Under the
published form the sum is at most 1, is maximal at *t* = 0 in all twelve
conditions, and falls monotonically; its deficit rises from 0.03–0.04 to
0.18–0.58, which is exactly how the unmeasured colorless species should behave.
The published convention is what puts the measurements in the same units as
predictions normalized by their own three-species total.

The experimental data in the revision is therefore **byte-identical in
processing** to the original submission. Only the modeling side was re-estimated.

Two conditions (*rad51*Δ at 0 and 2.5 Gy) exceed 1 at *t* = 0 by 0.3% and 0.08%
— first-timepoint absorbance noise. Reported, not rescaled away.

We also considered normalizing each timepoint by its own blue + pink sum
("option a"). Rejected: no publication using alamarBlue reports the two species
as fractions of each other. The three alamarBlue references gathered while
weighing that option were removed from the bibliography along with it.

### What the normalization detour had cost

Everything in the left column was in an intermediate draft of this revision and
is **withdrawn**. All of it was an artifact of our own rescaling, not a property
of the data:

| Claim in the intermediate draft | Status |
|---|---|
| WT MAE 0.060, *rad51*Δ 0.109, all 12 0.085 | **0.030 / 0.072 / 0.051** |
| Measured pink reaches 1.055; sum drifts up 9% | No drift; sum is maximal at *t* = 0 and falls |
| A ~0.05 residual is *structural* and unreachable | No floor; that residual *was* the rescaling |
| An A690 turbidity leak explains the excess (slope 0.087) | Nothing to explain |
| Optical observation equation, 4-arm ablation, 0.085 → 0.078 | **Withdrawn**; code in `revisions_2026/retired/` |
| WT-only kinetics do not transfer (0.378 on *rad51*Δ) | They transfer fine: 0.0722 vs 0.0716 joint |
| K3 alone is non-identifiable | **K2 and K3** are; K1 is constrained |

The optical leak was removed from `predict()` outright rather than kept with a
zero default — keeping it would mean carrying two parameters that existed only to
undo our own error. `revisions_2026/retired/README.md` records what was tried.

---

## One modeling addition, which is not a bug fix

AMMPER starts from a single cell; the plate reader starts from a dense inoculum.
Fitting to the full simulated trajectory forces the model to explain ~9
generations of near-zero signal the experiment never sees.

We added **one** parameter, `g0`: the point on the growth curve corresponding to
experimental *t* = 0, fit once per strain and held fixed across all six doses.

- Wild type: `g0` = 8.81 generations (~463 cells at *t* = 0). Per-dose refits span
  8.24–9.11, under one generation, so a single shared value is justified.
- *rad51*Δ: `g0` = 6.21 (~77 cells) — the mutant behaves like a wild type
  population **~2.6 generations behind**, consistent with a repair-deficient
  strain expanding more slowly. Per-dose refits are looser (4.97–7.27); we still
  use one shared value rather than spending five more parameters.

Six kinetic constants + `k` + one `g0` per strain = **9 parameters for 12 curves**,
with the dye chemistry shared across both strains because it is a property of
alamarBlue, not of the genotype.

Fitted values: `v1,v2,v3 = 3.22, 18.6, 16.0`; `K1,K2,K3 = 1.39e4, 218, 303`;
`k = 0.034`.

## Two things that are real limitations, not bugs

**K2 and K3 are not identifiable.** Both land far above the corresponding
substrate, putting the reversible pink↔colorless pair in an effectively
first-order regime where only the ratios `v2/K2` and `v3/K3` are constrained.
Scaling either pair together by 100× leaves the WT MAE at 0.0302 to four decimals;
the same scaling on `(v1,K1)` gives 0.059, nearly double. So the flat response is
a property of the data, not of the search bounds. Reported as non-identifiable
rather than quoted as measured values.

**The dose response is still too weak** — by 4–5× (WT) and 10–12× (*rad51*Δ). The
*direction* is now right, and for *rad51*Δ the predicted ordering is perfectly
monotonic in dose, but the magnitude is compressed. This is traceable, not
mysterious: the simulated healthy-cell count varies by only 5.5% (WT) and 7.9%
(*rad51*Δ) across the whole 0–30 Gy range, `k` is small (0.034), and damaged
*rad51*Δ cells are assigned directly to health state 3 (nonviable) rather than 2
in `cellDefinition.py`, so that strain has **no** health-2 cells at any dose and
its entire dose response travels through the healthy-cell count alone. The fix is
a graded health state with a per-cell metabolic rate — a model extension, not a
correction. This is in the manuscript's Discussion.

## Other things worth knowing

**Mass conservation.** An earlier version of the re-estimation clamped each
species at zero, which silently *created* dye when the reversible term ran
backwards — total dye grew >250% and the model degenerated into two-species
blue→pink with colorless pinned near zero. The clamp is replaced by two
admissibility constraints checked before scoring: colorless may not go negative,
and it must actually accumulate (>0.02). The search evaded weaker versions of
these three separate times, each time by driving a parameter to a bound rather
than fitting better. Found by inspecting trajectories, not error values.

**Generation time is a hyperparameter, not a measurement.** 198 min, implied by
the experiment duration and the simulated generation count — longer than the
80–120 min doubling time typical of *S. cerevisiae* in rich medium, which is
expected since the assay medium isn't optimized for growth. It trades off against
`g0` and we did not try to fit both.

**SMAC3 was not rerun.** The re-estimation uses seeded
`scipy.optimize.differential_evolution` to avoid a heavyweight dependency. The
optimizer isn't what's being tested. If reviewers want SMAC3 numbers on the
corrected model, `analysis/aB/aBFinalplotsSMAC.py` can be pointed at
`code/ammper_ab_model_fixed.py`.

**Figure 1 is unaffected** — worth recording because it's easy to assume
otherwise. `generate_growth_curves.py` does contain the same bug, and the stale
asset `results/figures_updated_figures_branch/growth_curves_panel.png` shows
visibly reversed curves. But the panel actually used,
`comprehensive_2row_panel.png`, shows correct ascending sigmoids and was produced
by a different route.

**Figure 2A** is the alamarBlue reduction scheme (molecular structures), not the
older cell cartoon. It survives only as a 403×503 raster, recovered from the
base64 payload inside the submitted SVG and committed at
`code/assets/aB_chem_scheme.png`. If anyone has the original vector artwork
(ChemDraw/Illustrator/BioRender), that is the resolution ceiling of the panel and
worth replacing before production.

**Gamma (Figs. S8--S10 in the supplement) has not been rechecked yet.** It was produced
by `aBFinalplotsGAMMA.py`, which has the same `value_counts()` bug. Whether the
poor gamma fit was the bug or a genuine modeling gap is an open question we are
looking at next; the simulation output for the *rad51* gamma conditions is not in
the archived results, so answering it may require re-running those simulations.

---

## Reproducing it

From `code/`:

```bash
python3 fit_ab_model_fixed.py         # ~20 min; writes fitted_parameters.json
python3 make_figure2_panels.py        # ~1 min; writes all figure panels
python3 generate_revision_numbers.py  # writes REVISION_NUMBERS.md
python3 verify_manuscript_numbers.py  # checks the .tex against the above (99/99)
```

`fit_ab_model_fixed.py` is seeded and its output is already recorded as constants
in `ammper_ab_model_fixed.py`, so the later scripts run standalone.

`verify_manuscript_numbers.py` is the guard against stale numbers: it reads the
generated JSON and checks every value the two `.tex` files quote, at the precision
each is written, plus `\ref` targets, `\cite` keys, `\includegraphics` targets and
tabular column counts. It exits non-zero on failure. It does **not** compile
LaTeX — `pdflatex` was unavailable in the environment where this was prepared, so
a real Overleaf compile is still the final check.

Note if you re-run the reports: a predicted curve can be computed either by
averaging the replicate predictions or by predicting once from the averaged growth
curve. The kinetics are nonlinear in cell count, so these differ — under 0.0005
in MAE, but enough to straddle a rounding boundary at 10 Gy. The figures and
reports use the first (it's what's plotted, so a reader can measure it); the
fitting objective uses the second, which is why `fitted_parameters.json` and the
figures can disagree in the fourth decimal.

Per-dose tables, identifiability scans and population-level numbers:
`REVISION_NUMBERS.md`, regenerated by `code/generate_revision_numbers.py`.
