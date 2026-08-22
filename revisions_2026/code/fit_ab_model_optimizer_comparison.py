"""
Compare global optimizers for the joint aB kinetics fit (comparison figure).

All three optimizers -- differential_evolution, dual_annealing, and CMA-ES --
are actively run here, each across multiple independent seeds (see
run_multi_seed), each left to its own default convergence criteria rather
than capped to a shared evaluation budget. This is a change from an earlier
version of this module, which loaded a previously-fit differential_evolution
result from disk rather than rerunning it, specifically so the reported
number matched exactly what appears elsewhere in the revision. That
constraint has been dropped in favor of putting all three optimizers through
the same multi-seed, convergence-tracked treatment, since a single loaded fit
couldn't participate in the seed-stability and convergence-trajectory
comparisons the other two now get. The prior on-disk fit is still loaded and
printed for reference at the start of a run (see main()), so a large
divergence between it and the newly-run best-of-N result would be visible
immediately rather than silently.

That is a deliberate choice, not an oversight: the point of this comparison is
not "which optimizer is most efficient under a fixed allowance," it is
"does the fit depend on which search algorithm was used." Capping every
optimizer to a shared nfev would answer the first question, and would leave
open the objection that a competing optimizer looked worse only because it
was cut off before its natural stopping point rather than because it
converges to a worse answer. Letting each algorithm run to its own
convergence (from several seeds, keeping the best) and then comparing the
resulting parameters and MAE answers the second, stronger question: three
structurally different search strategies -- population with crossover, an
annealed trajectory, and covariance-matrix adaptation -- converging to
compatible fits is evidence the result is a property of the data and the
physicality constraint, not an artifact of one optimizer's settings or one
lucky seed. nfev and wall time are still recorded and shown on each panel,
but as descriptive output, not as an equalized input.

Two settings are not left at library defaults, for a resource-safety reason
rather than a fairness one: the polish/local-search refinement step is turned
off for all three optimizers that have one (see run_differential_evolution
and run_dual_annealing), and dual_annealing/CMA-ES carry a generous
evaluation-count cap (MAX_FEVALS_SAFETY_CAP) while differential_evolution's
budget is instead bounded via DE_POPSIZE/DE_MAXITER. The objective has a
hard flat penalty plateau wherever a parameter set is chemically inadmissible,
and that shape can in principle keep an algorithm's own stopping check
from ever triggering -- observed in practice as an uncapped dual_annealing run
being killed by the OS rather than converging. The caps are set well above
what any of the three needs on this problem, so they are a backstop against a
runaway process, not the thing determining the reported result.

Why these three
----------------
differential_evolution (scipy): population-based search with mutation and
crossover. Now run fresh from multiple seeds rather than loaded from a prior
fit -- see the note above.

dual_annealing (scipy, already a dependency of this codebase): simulated-
annealing family, and a substantively different search strategy from DE --
a single annealed trajectory with re-annealing rather than a population of
candidate vectors evolved by mutation/crossover.

CMA-ES (`cma` package, pip install cma --break-system-packages): the standard
comparison point for DE in the continuous global-optimization literature.
Also population based, but adapts a full covariance matrix over the search
distribution rather than DE's fixed mutation/crossover scheme, so it is a
meaningfully different algorithm rather than a restatement of DE.

Not on by default: SMAC3 (the published pipeline's optimizer). It's
implemented as run_smac and can be added as a fourth reference panel with
--with-smac, but it's a fixed-trial-budget Bayesian method with no equivalent
to "run until it stops improving," so its inclusion isn't on quite the same
footing as the other three -- see run_smac's docstring for why, and its
n_trials is reported on the panel so that asymmetry stays visible rather than
implicit.

Every panel uses the same objective, bounds, and physicality constraint
(clear_forms_appreciably via _joint_objective) as the reported fit, and each
strain's g0 is polished afterward with the same exhaustive _scan_g0 used in
fit_ab_model_fixed.py -- so any difference between panels is attributable to
the search algorithm, not to a different downstream refinement step.

Statistics
----------
The per-dose MAE figure now carries a formal test of "does optimizer choice
matter," not just an eyeball comparison. Every optimizer refits the exact
same set of dose/strain conditions, so the per-condition MAEs are *paired*
across optimizers (matched blocks), not independent samples -- which rules
out an unpaired test like a plain Kruskal-Wallis or one-way ANOVA across
pooled values. A Friedman test (the non-parametric repeated-measures
equivalent of a one-way ANOVA) is used instead: each dose/strain condition is
a block, each optimizer is a treatment. If that omnibus test is significant,
pairwise post-hoc Wilcoxon signed-rank tests (one per optimizer pair, on the
matched per-condition MAEs) are run with Holm-Bonferroni correction for the
multiple comparisons. Results are written to optimizer_comparison_stats.json,
annotated on the error-bar figure, and printed as a small table -- intended
to support a couple of manuscript sentences describing the comparison.

That comparison is only as trustworthy as the individual optimizer runs it's
built on, though, and dual_annealing and CMA-ES are each stochastic: a single
seed could land in an unusually good or bad basin and never reveal that
through its own internal convergence check (see run_dual_annealing and
run_cma_es for why stopping is a heuristic judgment, not a certificate of
optimality). run_multi_seed addresses this directly: each of these two
optimizers is run from --n-seeds (default 5) independent seeds, the best-of-N
result is what's reported as that optimizer's panel (for consistency with how
differential_evolution -- itself population-based -- already reports its
single best), and the full per-seed spread is kept as panel['seed_runs'] and
summarized in optimizer_comparison_stats.json's seed_stability section. A
small sd across seeds means the earlier Friedman/Wilcoxon comparison is
comparing a representative result for each algorithm; a large sd would mean
the "dual_annealing is worse" conclusion needs to be qualified as "in this
one run" rather than stated as a property of the algorithm. See
optimizer_comparison_seed_stability.png.

Plot styling
------------
Both the per-dose-MAE comparison and the per-strain radiation grids go
through publiplots (https://github.com/jorgebotas/publiplots) for consistent
manuscript styling where available, with a plain-matplotlib fallback so the
script still runs end-to-end if publiplots isn't installed.

Outputs
-------
    optimizer_comparison.json          per-algorithm params, errors, nfev, wall
                                       time, and every seed's result under
                                       seed_runs -- now for all three
                                       optimizers, since differential_evolution
                                       is run fresh rather than loaded (see
                                       module docstring's opening note)
    optimizer_comparison_stats.json    Friedman + pairwise Wilcoxon/Holm
                                       results, plus a seed_stability section
                                       (per-seed spread, sd, best seed) for
                                       all three optimizers
    optimizer_comparison.png           2x3 fitted-curve grid (strain x optimizer,
                                       highest dose only -- kept as a quick-look
                                       summary)
    optimizer_comparison_errors.png     per-dose MAE, grouped by optimizer, one
                                        subplot per strain, publiplots styling,
                                        with Friedman/pairwise annotation
    optimizer_comparison_stats.png      dedicated 3-panel statistics figure:
                                        (A) per-dose MAE table (WT/rad51-delta
                                        sub-columns per optimizer, bolded mean
                                        row), (B) rank distribution per
                                        optimizer across the 12 paired
                                        conditions, (C) Holm-corrected
                                        pairwise p-value heatmap localizing
                                        which pair(s) differ
    optimizer_comparison_convergence.png  best-objective-so-far vs. function
                                          evaluations. Every seed's trajectory
                                          is drawn faded with the best-of-N
                                          seed bold, for all three optimizers
                                          now that differential_evolution is
                                          run fresh with tracking rather than
                                          loaded (see module docstring)
    optimizer_comparison_seed_stability.png  one point per seed per optimizer
                                             (overall mean MAE), best-of-N
                                             starred, sd annotated -- the
                                             figure that answers "is the
                                             reported result representative
                                             or a lucky/unlucky draw"
    optimizer_comparison_doses_<label>_WT.png      one figure per optimizer,
    optimizer_comparison_doses_<label>_rad51.png   per strain: 2x3 dose grid,
                                                   styled after the manuscript's
                                                   own Fig. 2 panels B/C (dose
                                                   label top-left, shared legend
                                                   at bottom, one strain and one
                                                   optimizer per figure). Uses
                                                   each optimizer's best-of-N
                                                   seed result.

Supplementary material
-----------------------
Add to the supplementary list:
  - Supplementary Fig: optimizer_comparison_errors.png
  - Supplementary Fig: optimizer_comparison_stats.png
  - Supplementary Fig: optimizer_comparison_convergence.png
  - Supplementary Fig: optimizer_comparison_seed_stability.png
  - Supplementary Figs: optimizer_comparison_doses_<optimizer>_WT.png and
    ..._rad51.png, one pair per optimizer
  - Supplementary Table: optimizer_comparison_stats.json, formatted as a table
    of pairwise Holm-corrected p-values and per-optimizer seed stability
"""

import argparse
import itertools
import json
import os
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, dual_annealing
from scipy.stats import friedmanchisquare, wilcoxon

import ammper_ab_model_fixed as M
from fit_ab_model_fixed import (
    _joint_objective, BOUNDS, I_G0, N_KINETIC, SEED,
    _scan_g0, per_dose_errors, WT_KEYS, RAD_KEYS,
    OUT_JSON as DE_JSON,
)

try:
    import cma
except ImportError as exc:
    raise ImportError(
        "the CMA-ES panel requires the 'cma' package: "
        "pip install cma --break-system-packages") from exc

# --- publiplots: confirmed API ------------------------------------------
#     import publiplots as pp
#     fig, axes = pp.subplots(nrows, ncols, axes_size=(w, h))
#     pp.scatterplot(data=df, x=..., y=..., hue=..., ax=ax)   # and barplot etc.
#     pp.legend(axes[row], side='top')   # shared legend for one row of axes
#     pp.savefig(path)
# axes is indexed [row] first, matching plt.subplots(..., squeeze=False).
try:
    import publiplots as pp
    HAS_PUBLIPLOTS = True
except ImportError:
    pp = None
    HAS_PUBLIPLOTS = False


def _apply_publiplots_style():
    """publiplots handles styling per-figure via pp.subplots/pp.savefig
    rather than a global rcParams-style call, so there is nothing to set
    globally here. Kept as a hook (and a plain-matplotlib fallback for when
    publiplots isn't installed) in case that changes."""
    if not HAS_PUBLIPLOTS:
        plt.rcParams.update({
            'figure.dpi': 200, 'savefig.dpi': 200, 'font.size': 9,
            'axes.spines.top': False, 'axes.spines.right': False,
        })


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, 'optimizer_comparison.json')
OUT_STATS_JSON = os.path.join(HERE, 'optimizer_comparison_stats.json')
OUT_FIG = os.path.join(HERE, 'optimizer_comparison.png')
OUT_ERR_FIG = os.path.join(HERE, 'optimizer_comparison_errors.png')
OUT_STATS_FIG = os.path.join(HERE, 'optimizer_comparison_stats.png')
OUT_CONVERGENCE_FIG = os.path.join(HERE, 'optimizer_comparison_convergence.png')
OUT_SEED_FIG = os.path.join(HERE, 'optimizer_comparison_seed_stability.png')

#: Shared color per optimizer across every figure in this module, so a color
#: means the same thing in the MAE bars, the rank plot, and the slope plot.
OPTIMIZER_COLORS = {
    'differential_evolution': '#4C72B0',
    'dual_annealing': '#DD8452',
    'CMA-ES': '#55A868',
}


def _color_for(label):
    """Falls back to matplotlib's default cycle for any label not in
    OPTIMIZER_COLORS (e.g. the optional SMAC3 panel)."""
    if label in OPTIMIZER_COLORS:
        return OPTIMIZER_COLORS[label]
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return cycle[hash(label) % len(cycle)]


def _slug(label):
    """Filesystem-safe version of a panel label, e.g. 'SMAC3 (n_trials=5000)'
    -> 'SMAC3_n_trials-5000'."""
    slug = re.sub(r'[^\w]+', '_', label).strip('_')
    return slug


def _strain_dose_fig_path(label, strain):
    return os.path.join(HERE, f'optimizer_comparison_doses_{_slug(label)}_{strain}.png')


#: Joint search space: the 8 shared/kinetic bounds plus a second copy of the
#: g0 bound for the rad51-delta offset -- identical to fit_ab_model_fixed.py.
JOINT_BOUNDS = BOUNDS + [BOUNDS[I_G0]]
LOWER = np.array([b[0] for b in JOINT_BOUNDS])
UPPER = np.array([b[1] for b in JOINT_BOUNDS])

#: Safety cap on function evaluations for dual_annealing and CMA-ES. Not a
#: fairness constraint between optimizers (see the earlier discussion in this
#: module) -- this is a resource guard. Both algorithms lack a hard stop the
#: way this is normally reasoned about: without it, a run that never satisfies
#: the internal convergence check can spin indefinitely, each iteration paying
#: for a 12-condition ODE integration, until something (usually the OS
#: OOM-killer) ends the process uncleanly instead of scipy/cma ending it
#: cleanly. Set well above what either optimizer should need to converge on
#: this problem, so it is a backstop, not the thing actually determining the
#: reported result.
MAX_FEVALS_SAFETY_CAP = 50000

#: Default number of independent seeds run_multi_seed uses for dual_annealing
#: and CMA-ES. Each optimizer's reported panel is the best-of-N across these
#: seeds (see run_multi_seed's docstring for why that's the fair comparison
#: point against differential_evolution, which is itself population-based and
#: already reports its single best internally). The full per-seed spread is
#: retained on the panel (seed_runs) so seed-to-seed stability -- "was the
#: reported result a fluke, or reliably reachable" -- can be reported and
#: plotted rather than assumed.
DEFAULT_N_SEEDS = 5

#: differential_evolution is now actually run (see run_differential_evolution)
#: rather than loaded from a prior fit, so it can participate in the same
#: multi-seed / convergence-tracking / seed-stability treatment as the other
#: two. popsize and maxiter are capped for tractability rather than left at
#: scipy's defaults or run to full convergence: total function evaluations
#: for differential_evolution scale roughly as
#: popsize * len(bounds) * (maxiter + 1), so with the 9-dimensional
#: JOINT_BOUNDS used here, popsize=15 and maxiter=150 gives ~20,000
#: evaluations per seed -- the same order of magnitude dual_annealing and
#: CMA-ES land on with their own stopping criteria (see MAX_FEVALS_SAFETY_CAP
#: and the docstrings on those two run_* functions), rather than an
#: unconstrained run that could take substantially longer with no comparable
#: benefit given the diminishing returns already observed near this budget.
DE_POPSIZE = 15
DE_MAXITER = 150


class ProgressTracker:
    """Wraps _joint_objective so a long-running optimizer reports live
    progress instead of going silent until it finishes, and records the
    full (nfev, best-so-far) convergence trajectory as it goes.

    Uses tqdm if it's installed (pip install tqdm --break-system-packages);
    otherwise falls back to a plain print every `report_every` seconds, so
    this works with no new dependency if you'd rather not install one.
    """

    def __init__(self, growth, experiment, label, total=None, report_every=3.0):
        self.growth = growth
        self.experiment = experiment
        self.label = label
        self.total = total
        self.nfev = 0
        self.best = np.inf
        # (nfev, best-so-far) at every evaluation -- the convergence
        # trajectory for this optimizer. Thinned with _thin_history before
        # being stored on a panel/written to JSON, but kept at full
        # resolution here in case anything else wants it during the run.
        self.history = []
        self._report_every = report_every
        self._last_report = time.time()
        self._start = time.time()
        self._bar = None
        try:
            from tqdm import tqdm
            self._bar = tqdm(total=total, desc=label, unit='eval')
        except ImportError:
            pass

    def __call__(self, vector, seed=None):
        # seed is accepted and ignored so this can also serve as SMAC3's
        # target function, which calls func(config, seed=...).
        value = _joint_objective(np.asarray(vector), self.growth, self.experiment)
        self.nfev += 1
        if value < self.best:
            self.best = value
        # float() here, not np.float64: this tuple round-trips through
        # json.dump via optimizer_comparison.json, which chokes on numpy
        # scalar types.
        self.history.append((self.nfev, float(self.best)))
        if self._bar is not None:
            self._bar.update(1)
            self._bar.set_postfix(best=f'{self.best:.4f}')
        else:
            now = time.time()
            if now - self._last_report >= self._report_every:
                self._last_report = now
                suffix = f'/{self.total}' if self.total else ''
                print(f'  [{self.label}] {self.nfev}{suffix} evals, '
                     f'{now - self._start:.0f}s elapsed, '
                     f'best MAE so far = {self.best:.4f}', flush=True)
        return value

    def close(self):
        if self._bar is not None:
            self._bar.close()


def _thin_history(history, max_points=2000):
    """Evenly subsample a (nfev, best) trajectory to at most max_points
    entries, always keeping the first and last point. Convergence plots don't
    need every one of e.g. 18,000 evaluations to look right, and this keeps
    optimizer_comparison.json (which the full history gets written into) a
    sane size regardless of how long a given optimizer ran."""
    if len(history) <= max_points:
        return history
    idx = np.linspace(0, len(history) - 1, max_points).astype(int)
    idx = sorted(set(idx.tolist()) | {0, len(history) - 1})
    return [history[i] for i in idx]

#: Names for the same 9 dimensions, in the same order, for SMAC3's
#: ConfigurationSpace (which is keyed by name rather than position).
SMAC_PARAM_NAMES = ['v1', 'v2', 'v3', 'K1', 'K2', 'K3', 'k',
                    'g0_wt', 'g0_rad51']

#: Representative conditions shown in each panel: the highest dose in each
#: strain, where dose-response separation (and therefore fit difficulty) is
#: greatest. Row 0 = wild type, row 1 = rad51-delta, since that mutant is the
#: harder fit (0.072 vs 0.030 in the reported DE run) and is worth showing
#: rather than folding into a single overall-MAE number.
DISPLAY_KEYS = ('WT_30', 'rad51_30')
DISPLAY_G0_KEYS = ('g0_wt', 'g0_rad51')
ROW_LABELS = ('Wild Type', 'rad51Δ')


def _finalize(vector, growth, experiment, nfev, elapsed, label,
             convergence_history=None):
    """Turn a raw parameter vector into the same shape fit_ab_model_fixed's
    report uses, so every panel is directly comparable.

    convergence_history, if given, is the (nfev, best-so-far) trajectory
    recorded by a ProgressTracker -- present for dual_annealing and CMA-ES,
    absent for the loaded differential_evolution panel (see _load_de_panel),
    which predates this tracking and is not rerun here."""
    params = tuple(vector[:N_KINETIC])
    wt_g0 = _scan_g0(growth, experiment, params, WT_KEYS)
    rad_g0 = _scan_g0(growth, experiment, params, RAD_KEYS)
    err_wt = per_dose_errors(growth, experiment, params, wt_g0, WT_KEYS)
    err_rad = per_dose_errors(growth, experiment, params, rad_g0, RAD_KEYS)
    all_errors = dict(err_wt, **err_rad)
    result = {
        'label': label,
        'params': {'v1': params[0], 'v2': params[1], 'v3': params[2],
                   'K1': params[3], 'K2': params[4], 'K3': params[5],
                   'k': params[6]},
        'g0_wt': wt_g0, 'g0_rad51': rad_g0,
        'wt_per_dose': err_wt, 'rad51_per_dose': err_rad,
        'overall_mean': float(np.mean(list(all_errors.values()))),
        'nfev': int(nfev), 'wall_seconds': float(elapsed),
    }
    if convergence_history is not None:
        result['convergence_history'] = convergence_history
    return result


def run_differential_evolution(growth, experiment, seed=SEED,
                               popsize=DE_POPSIZE, maxiter=DE_MAXITER):
    """differential_evolution to its own tol-based convergence or maxiter,
    whichever comes first, with polish disabled for the same reason
    dual_annealing's local-search step is disabled (see that function's
    docstring): the objective's flat PENALTY=10.0 plateau wherever a
    parameter set is chemically inadmissible is exactly the shape a
    finite-difference gradient estimate (which is what polish's L-BFGS-B step
    would use) handles badly -- ~0 gradient on the plateau, a sharp jump at
    its edge -- and this keeps the comparison a clean test of each
    algorithm's own population/annealing/covariance mechanism, with no
    bolted-on gradient step on any of the three panels.

    popsize and maxiter are capped rather than left at scipy's defaults or
    run to unconstrained convergence -- see DE_POPSIZE/DE_MAXITER's comment
    for the budget this targets and why. This does mean a given seed's run
    could in principle be stopped by maxiter before its internal tol check
    would have fired; that's an accepted tractability trade-off here, and is
    exactly why this optimizer, like the other two, is run across multiple
    seeds (run_multi_seed) rather than trusted from a single run.
    """
    start = time.time()
    tracker = ProgressTracker(growth, experiment, 'differential_evolution',
                              total=popsize * len(JOINT_BOUNDS) * (maxiter + 1))
    result = differential_evolution(tracker, JOINT_BOUNDS, seed=seed,
                                    popsize=popsize, maxiter=maxiter,
                                    polish=False)
    tracker.close()
    elapsed = time.time() - start
    return _finalize(result.x, growth, experiment, result.nfev, elapsed,
                     'differential_evolution',
                     convergence_history=_thin_history(tracker.history))


def run_dual_annealing(growth, experiment, seed=SEED,
                       maxfun=MAX_FEVALS_SAFETY_CAP):
    """dual_annealing with its own annealing-schedule stopping criteria, but
    with the local-search refinement step turned off and a finite evaluation
    cap applied -- both for a concrete reason, not just to avoid the crash
    this hit last run.

    The default no_local_search=False periodically calls a gradient-based
    local optimizer (L-BFGS-B) on top of the annealing schedule. This
    objective has a hard flat plateau (PENALTY=10.0) wherever
    clear_forms_appreciably rejects a parameter set, with a sharp edge into
    the real landscape elsewhere -- exactly the shape a finite-difference
    gradient estimate handles badly: on the plateau the estimated gradient is
    ~0, and near the edge it can be very large, either of which can send
    L-BFGS-B into a long or unbounded sequence of evaluations. Turning off
    the local-search step keeps this panel a clean test of the annealing
    schedule on its own, which is also the more informative comparison point
    against DE and CMA-ES (neither of which uses a bolted-on gradient method
    either). The maxfun cap is a separate resource backstop in case the
    annealing schedule itself doesn't trigger its convergence check quickly.
    """
    start = time.time()
    tracker = ProgressTracker(growth, experiment, 'dual_annealing', total=maxfun)
    result = dual_annealing(tracker, JOINT_BOUNDS, seed=seed,
                            no_local_search=True, maxfun=maxfun)
    tracker.close()
    elapsed = time.time() - start
    return _finalize(result.x, growth, experiment, result.nfev, elapsed,
                     'dual_annealing',
                     convergence_history=_thin_history(tracker.history))


def run_cma_es(growth, experiment, seed=SEED,
               maxfevals=MAX_FEVALS_SAFETY_CAP):
    """CMA-ES to its own tolfun/tolx stopping criteria, with the same
    evaluation-count safety cap as dual_annealing and for the same reason:
    the flat PENALTY plateau in this objective can in principle keep the
    population's spread from shrinking the way tolx expects, so a hard cap
    prevents an unbounded run rather than changing what "converged" means."""
    start = time.time()
    x0 = (LOWER + UPPER) / 2.0
    sigma0 = float(np.mean(UPPER - LOWER) * 0.3)
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {'bounds': [LOWER.tolist(), UPPER.tolist()],
         'maxfevals': maxfevals, 'seed': seed, 'verbose': -9})
    tracker = ProgressTracker(growth, experiment, 'CMA-ES', total=maxfevals)
    es.optimize(tracker)
    tracker.close()
    elapsed = time.time() - start
    return _finalize(np.asarray(es.result.xbest), growth, experiment,
                     es.result.evaluations, elapsed, 'CMA-ES',
                     convergence_history=_thin_history(tracker.history))


def run_multi_seed(run_fn, growth, experiment, seeds, label):
    """Run run_fn once per seed in `seeds`, keep the best-performing run
    (lowest overall_mean) as the reported panel, and attach every run's
    summary as panel['seed_runs'] so seed-to-seed stability can be reported
    and plotted rather than assumed from a single draw.

    Best-of-N, not mean-of-N or median-of-N, is the reported result, for
    consistency with how differential_evolution is already being compared:
    DE is itself a population-based search that returns its single best
    candidate at convergence, and the DE panel here is one such single run
    loaded from disk (see _load_de_panel), not an average over repeated DE
    runs. Reporting dual_annealing and CMA-ES the same way -- "best result
    this algorithm can reliably produce," represented by its best observed
    outcome -- keeps the three-way comparison apples-to-apples rather than
    comparing DE's single best against an average for the other two.

    The seed-to-seed *spread* (seed_runs, and the derived sd in
    optimizer_comparison_seed_stability.png) is what actually answers the
    robustness question, though: if that spread is small, the reported
    best-of-N is close to what any single seed would give you and the
    earlier "dual_annealing is worse" conclusion is a property of the
    algorithm rather than one unlucky run. If the spread is large, the
    opposite is true and that matters just as much as the headline number.
    """
    runs = []
    for i, seed in enumerate(seeds):
        print(f'  [{label}] seed {i + 1}/{len(seeds)} (seed={seed})...')
        panel = run_fn(growth, experiment, seed=seed)
        runs.append((seed, panel))

    best_seed, best_panel = min(runs, key=lambda sp: sp[1]['overall_mean'])
    best_panel = dict(best_panel)
    best_panel['best_seed'] = best_seed
    best_panel['seed_runs'] = [
        {
            'seed': seed,
            'overall_mean': panel['overall_mean'],
            'nfev': panel['nfev'],
            'wall_seconds': panel['wall_seconds'],
            'convergence_history': panel.get('convergence_history'),
        }
        for seed, panel in runs
    ]
    return best_panel


def run_smac(growth, experiment, seed=SEED, n_trials=5000):
    """Bayesian optimization via SMAC3 -- the optimizer the published pipeline
    used. Included as a reference point, not as a fourth optimizer on equal
    footing with the other three -- see the note on budget below.

    Unlike DE, dual_annealing, and CMA-ES, SMAC3 has no notion of "run until
    the population/trajectory stops improving": it is a fixed-budget Bayesian
    method that spends every trial fitting and querying a surrogate model, and
    tolerance-based stopping isn't part of its default behavior the way it is
    for the scipy optimizers or CMA-ES's tolfun/tolx. So a trial budget
    (n_trials) has to be chosen explicitly here rather than left at a library
    default -- there isn't one to leave it at. This is a real asymmetry with
    the other three panels, not an oversight; report n_trials alongside the
    result so anyone reading the figure can see the comparison isn't fully
    apples-to-apples for this one panel.

    Requires: pip install smac ConfigSpace --break-system-packages
    """
    try:
        from ConfigSpace import ConfigurationSpace, Float
        from smac import HyperparameterOptimizationFacade, Scenario
    except ImportError as exc:
        raise ImportError(
            "the SMAC3 panel requires 'smac' and 'ConfigSpace': "
            "pip install smac ConfigSpace --break-system-packages") from exc

    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        Float(name, (lo, hi)) for name, (lo, hi) in zip(SMAC_PARAM_NAMES, JOINT_BOUNDS)
    ])

    tracker = ProgressTracker(growth, experiment, 'SMAC3', total=n_trials)

    def target(config, seed=0):
        vector = np.array([config[name] for name in SMAC_PARAM_NAMES])
        return tracker(vector)

    scenario = Scenario(cs, deterministic=True, n_trials=n_trials, seed=seed,
                        output_directory=os.path.join(HERE, 'smac_output'))
    smac_opt = HyperparameterOptimizationFacade(scenario, target, overwrite=True)

    start = time.time()
    incumbent = smac_opt.optimize()
    tracker.close()
    elapsed = time.time() - start

    vector = np.array([incumbent[name] for name in SMAC_PARAM_NAMES])
    nfev = len(smac_opt.runhistory)
    return _finalize(vector, growth, experiment, nfev, elapsed,
                     f'SMAC3 (n_trials={n_trials})',
                     convergence_history=_thin_history(tracker.history))


def _load_de_panel():
    """The differential_evolution fit previously produced by
    fit_ab_model_fixed.py, loaded for reference/sanity-check purposes only.

    Not used as the reported differential_evolution panel in the current
    comparison (see the module docstring's opening note): that panel now
    comes from run_differential_evolution via run_multi_seed instead, so it
    can participate in the same seed-stability and convergence-trajectory
    treatment as dual_annealing and CMA-ES. Kept and printed at the start of
    main() purely so a large divergence between the freshly-run best-of-N
    result and this prior fit would be visible immediately.
    """
    with open(DE_JSON) as handle:
        report = json.load(handle)
    params = report['shared_kinetics']['params']
    return {
        'label': 'differential_evolution',
        'params': params,
        'g0_wt': report['wt']['corrected_refit']['g0'],
        'g0_rad51': report['rad51']['corrected_refit']['g0'],
        'wt_per_dose': report['wt']['corrected_refit']['per_dose'],
        'rad51_per_dose': report['rad51']['corrected_refit']['per_dose'],
        'overall_mean': report['overall_mean'],
        'nfev': report['shared_kinetics']['optimizer_nfev'],
        'wall_seconds': None,  # not recorded by the original run
    }


def _plot_panel(ax, growth, experiment, panel, condition_key, g0_key, row_label):
    """One panel: predicted vs experimental blue/pink for condition_key, using
    that panel's own g0 for the strain condition_key belongs to."""
    params = tuple(panel['params'][k] for k in
                   ('v1', 'v2', 'v3', 'K1', 'K2', 'K3', 'k'))
    prediction = M.predict(growth[condition_key], params, g0=panel[g0_key])
    exp = experiment[condition_key]
    ax.plot(prediction.time, prediction.blue, color='tab:blue', label='blue (pred)')
    ax.plot(prediction.time, prediction.pink, color='tab:red', label='pink (pred)')
    ax.scatter(exp.time, exp.blue, color='tab:blue', s=14, alpha=0.6, label='blue (meas)')
    ax.scatter(exp.time, exp.pink, color='tab:red', s=14, alpha=0.6, label='pink (meas)')
    title = f"{panel['label']} -- {row_label}\nMAE={panel['overall_mean']:.4f}  nfev={panel['nfev']}"
    if panel['wall_seconds'] is not None:
        title += f"  {panel['wall_seconds']:.0f}s"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Concentration fraction')
    ax.set_ylim(-0.05, 1.05)


# --- statistics: is the fit sensitive to which optimizer was used? -------

def _holm_correct(pvalues):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    same order as the input, monotonically enforced (each adjusted p-value is
    at least as large as the previous one in sorted order, standard Holm
    behavior so p-values don't become non-monotonic after adjustment)."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (n - rank) * pvalues[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _collect_paired_mae(panels):
    """Build a (n_conditions x n_optimizers) matrix of per-condition MAE,
    where a "condition" is one dose within one strain. Every panel must
    report the same set of condition keys, since they all fit the same
    experimental conditions -- that shared key set is what makes the Friedman
    test's pairing valid."""
    keys = (sorted(panels[0]['wt_per_dose'].keys())
           + sorted(panels[0]['rad51_per_dose'].keys()))
    matrix = []
    for panel in panels:
        row = ([panel['wt_per_dose'][k] for k in sorted(panel['wt_per_dose'].keys())]
              + [panel['rad51_per_dose'][k] for k in sorted(panel['rad51_per_dose'].keys())])
        matrix.append(row)
    # transpose so rows = conditions (blocks), columns = optimizers (treatments)
    return keys, np.array(matrix).T


def _optimizer_statistics(panels):
    """Friedman test across optimizers (treatments) over matched dose/strain
    conditions (blocks), plus Holm-corrected pairwise Wilcoxon signed-rank
    post-hoc tests. See the module docstring's Statistics section for why
    this pairing-aware test is the right one here, rather than pooling all
    per-condition MAEs into an unpaired test."""
    labels = [panel['label'] for panel in panels]
    condition_keys, matrix = _collect_paired_mae(panels)

    friedman_stat, friedman_p = friedmanchisquare(*matrix.T)

    pairs = list(itertools.combinations(range(len(labels)), 2))
    raw_p = []
    stats = []
    for i, j in pairs:
        diff = matrix[:, i] - matrix[:, j]
        if np.allclose(diff, 0):
            # identical paired values: wilcoxon errors out on an all-zero
            # difference vector, and "no measurable difference" is exactly
            # p=1 here, not an error condition worth propagating.
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(matrix[:, i], matrix[:, j])
        raw_p.append(p)
        stats.append(stat)
    adjusted_p = _holm_correct(np.array(raw_p)) if pairs else np.array([])

    pairwise = []
    for (i, j), stat, p_raw, p_adj in zip(pairs, stats, raw_p, adjusted_p):
        pairwise.append({
            'a': labels[i], 'b': labels[j],
            'wilcoxon_stat': float(stat),
            'p_raw': float(p_raw), 'p_holm': float(p_adj),
            'significant_holm_0.05': bool(p_adj < 0.05),
        })

    return {
        'n_conditions': len(condition_keys),
        'optimizers': labels,
        'friedman_statistic': float(friedman_stat),
        'friedman_p': float(friedman_p),
        'friedman_significant_0.05': bool(friedman_p < 0.05),
        'pairwise_holm': pairwise,
    }


def _stats_annotation_text(stats):
    """Short figure-footer summary of the omnibus + post-hoc results."""
    line1 = (f"Friedman: chi2={stats['friedman_statistic']:.2f}, "
            f"p={stats['friedman_p']:.3f} "
            f"({'significant' if stats['friedman_significant_0.05'] else 'n.s.'} "
            f"at alpha=0.05, n={stats['n_conditions']} paired conditions)")
    if stats['friedman_significant_0.05'] and stats['pairwise_holm']:
        sig_pairs = [f"{p['a']} vs {p['b']} (p_holm={p['p_holm']:.3f})"
                    for p in stats['pairwise_holm'] if p['significant_holm_0.05']]
        line2 = ('Holm-corrected pairwise differences: '
                + ('; '.join(sig_pairs) if sig_pairs else 'none survive correction'))
    else:
        line2 = 'Pairwise post-hoc not shown: omnibus test was not significant.'
    return line1 + '\n' + line2


def _seed_stability_summary(panels):
    """For every panel that carries seed_runs (dual_annealing, CMA-ES): the
    per-seed overall_mean values, their spread (sd), and which seed produced
    the reported best-of-N result. differential_evolution has no seed_runs
    (single run, loaded from disk -- see _load_de_panel) and is omitted here
    rather than reported with a meaningless single-point "sd" of zero."""
    summary = {}
    for panel in panels:
        seed_runs = panel.get('seed_runs')
        if not seed_runs:
            continue
        vals = [r['overall_mean'] for r in seed_runs]
        summary[panel['label']] = {
            'n_seeds': len(vals),
            'seeds': [r['seed'] for r in seed_runs],
            'overall_means': vals,
            'best_overall_mean': panel['overall_mean'],
            'best_seed': panel.get('best_seed'),
            'mean_overall_mean': float(np.mean(vals)),
            'sd_overall_mean': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'range_overall_mean': [float(min(vals)), float(max(vals))],
        }
    return summary


def _mae_long_dataframe(panels):
    """Long-form dataframe of per-condition MAE, one row per
    (strain, dose, optimizer) -- the shape publiplots' seaborn-style
    plotting functions expect via their `data=` argument."""
    rows = []
    field_by_strain = {'Wild Type': 'wt_per_dose', 'rad51Δ': 'rad51_per_dose'}
    conds_by_strain = {'Wild Type': M.conditions('WT'), 'rad51Δ': M.conditions('rad51')}
    for panel in panels:
        for strain, field in field_by_strain.items():
            for cond in conds_by_strain[strain]:
                rows.append({
                    'strain': strain,
                    'dose': cond.dose_gy,
                    'optimizer': panel['label'],
                    'mae': panel[field][cond.key],
                })
    return pd.DataFrame(rows)


def _plot_error_bars_publiplots(panels, stats, out_path, suptitle):
    """Per-dose MAE by optimizer, via publiplots' subplots/barplot/legend/
    savefig, with the Friedman/Wilcoxon summary as a figure footer."""
    df = _mae_long_dataframe(panels)
    labels = [panel['label'] for panel in panels]
    palette = {label: _color_for(label) for label in labels}

    fig, axes = pp.subplots(1, 2, axes_size=(32, 26))
    for ax, strain in zip(axes[0], ('Wild Type', 'rad51Δ')):
        sub = df[df['strain'] == strain]
        try:
            pp.barplot(data=sub, x='dose', y='mae', hue='optimizer', ax=ax,
                      palette=palette)
        except TypeError:
            pp.barplot(data=sub, x='dose', y='mae', hue='optimizer', ax=ax)
        ax.set_title(strain, fontsize=13, fontweight='bold')
        ax.set_xlabel('Dose (Gy)')
        ax.set_ylabel('Mean absolute error')
        ax.tick_params(labelbottom=True, labelleft=True)
    pp.legend(axes[0], side='top')
    fig.suptitle(suptitle, fontsize=15, fontweight='bold')
    pp.savefig(out_path, transparent=False, facecolor='white')


def _plot_error_bars_matplotlib(panels, stats, out_path, suptitle):
    """Fallback for when publiplots isn't installed. Publication styling:
    shared optimizer color palette, spines trimmed, horizontal gridlines only,
    numbered ticks forced on both subplots (sharey=False, since the shared-
    axis default hides the right subplot's y tick labels)."""
    wt_doses = [c.dose_gy for c in M.conditions('WT')]
    rad_doses = [c.dose_gy for c in M.conditions('rad51')]
    wt_keys = [c.key for c in M.conditions('WT')]
    rad_keys = [c.key for c in M.conditions('rad51')]
    labels = [panel['label'] for panel in panels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    all_vals = []
    for ax, doses, keys, dose_field, title in (
            (axes[0], wt_doses, wt_keys, 'wt_per_dose', 'Wild Type'),
            (axes[1], rad_doses, rad_keys, 'rad51_per_dose', 'rad51Δ')):
        groups = [[panel[dose_field][key] for key in keys] for panel in panels]
        all_vals.extend(v for g in groups for v in g)
        x = np.arange(len(doses))
        width = 0.8 / len(groups)
        for i, (vals, label) in enumerate(zip(groups, labels)):
            ax.bar(x + (i - (len(groups) - 1) / 2) * width, vals, width,
                  label=label, color=_color_for(label),
                  edgecolor='white', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d:g}' for d in doses])
        ax.set_xlabel('Dose (Gy)', fontsize=10)
        ax.set_ylabel('Mean absolute error', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    ymax = max(all_vals) * 1.15
    for ax in axes:
        ax.set_ylim(0, ymax)
    axes[0].legend(fontsize=9, frameon=False, loc='upper left')
    fig.suptitle(suptitle, fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches='tight', transparent=False, facecolor='white')


def plot_error_bars(panels, stats, out_path, suptitle='Per-dose MAE by optimizer'):
    """Dispatches to the publiplots version if available, else matplotlib."""
    if HAS_PUBLIPLOTS:
        _plot_error_bars_publiplots(panels, stats, out_path, suptitle)
    else:
        _plot_error_bars_matplotlib(panels, stats, out_path, suptitle)


# --- a dedicated, information-dense statistics figure ---------------------
#
# The bar chart answers "how big are the MAE differences." This figure
# answers "why did the Friedman test fire" -- it shows the same result three
# ways: the raw per-dose MAE table, the average rank per optimizer (which is
# literally what Friedman tests), and the Holm-corrected pairwise p-values
# that localize the effect to specific pairs.

def _p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _draw_mae_table(ax, panels):
    """Booktabs-style table: dose rows, WT/rad51-delta sub-columns per
    optimizer, bolded overall-mean row -- the raw numbers the rank plot and
    p-value heatmap are summarizing, laid out the same way as the
    manuscript's own results tables."""
    ax.set_axis_off()
    doses = [c.dose_gy for c in M.conditions('WT')]
    wt_keys = [c.key for c in M.conditions('WT')]
    rad_keys = [c.key for c in M.conditions('rad51')]
    n_panels = len(panels)

    dose_col_w = 0.20
    sub_w = (1.0 - dose_col_w) / (2 * n_panels)
    col_x = []
    x = dose_col_w
    for _ in range(n_panels):
        col_x.append((x + sub_w / 2, x + sub_w + sub_w / 2))
        x += 2 * sub_w

    n_rows = len(doses) + 1  # + mean row
    top, bottom = 0.80, 0.04
    row_ys = np.linspace(top, bottom, n_rows)
    row_h = (row_ys[0] - row_ys[1]) if n_rows > 1 else 0.1

    for panel, (wt_c, rad_c) in zip(panels, col_x):
        span_c = (wt_c + rad_c) / 2
        ax.text(span_c, 0.95, panel['label'], transform=ax.transAxes,
               ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(wt_c, 0.87, 'WT', transform=ax.transAxes,
               ha='center', va='center', fontsize=8)
        ax.text(rad_c, 0.87, 'rad51\u0394', transform=ax.transAxes,
               ha='center', va='center', fontsize=8)
    ax.text(0.0, 0.87, 'Dose', transform=ax.transAxes,
           ha='left', va='center', fontsize=8, fontweight='bold')

    ax.plot([0, 1], [0.99, 0.99], color='black', lw=1.3,
           transform=ax.transAxes, clip_on=False)
    ax.plot([0, 1], [0.82, 0.82], color='black', lw=0.7,
           transform=ax.transAxes, clip_on=False)

    for i, (dose, wt_key, rad_key) in enumerate(zip(doses, wt_keys, rad_keys)):
        y = row_ys[i]
        ax.text(0.0, y, f'{dose:g} Gy', transform=ax.transAxes,
               ha='left', va='center', fontsize=8)
        for panel, (wt_c, rad_c) in zip(panels, col_x):
            ax.text(wt_c, y, f"{panel['wt_per_dose'][wt_key]:.3f}",
                   transform=ax.transAxes, ha='center', va='center', fontsize=8)
            ax.text(rad_c, y, f"{panel['rad51_per_dose'][rad_key]:.3f}",
                   transform=ax.transAxes, ha='center', va='center', fontsize=8)

    mean_y = row_ys[-1]
    rule_y = mean_y + row_h / 2
    ax.plot([0, 1], [rule_y, rule_y], color='black', lw=0.7,
           transform=ax.transAxes, clip_on=False)
    ax.text(0.0, mean_y, 'mean', transform=ax.transAxes,
           ha='left', va='center', fontsize=8, fontweight='bold')
    for panel, (wt_c, rad_c) in zip(panels, col_x):
        wt_mean = float(np.mean(list(panel['wt_per_dose'].values())))
        rad_mean = float(np.mean(list(panel['rad51_per_dose'].values())))
        ax.text(wt_c, mean_y, f'{wt_mean:.3f}', transform=ax.transAxes,
               ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(rad_c, mean_y, f'{rad_mean:.3f}', transform=ax.transAxes,
               ha='center', va='center', fontsize=8, fontweight='bold')
    ax.plot([0, 1], [bottom - row_h / 2, bottom - row_h / 2], color='black',
           lw=1.3, transform=ax.transAxes, clip_on=False)

    ax.set_title('A. Mean absolute error per dose', fontsize=11,
                fontweight='bold')


def _plot_rank_panel(ax, labels, ranks):
    """Violin plot of rank distributions per optimizer across the 12 paired
    conditions (rank 1 = best fit). Violins show the per-condition rank
    distribution; medians and mean+SEM are overplotted for clarity. The
    dashed line marks the null expected rank under no systematic difference."""
    # ranks: rows = conditions, columns = optimizers
    data = [ranks[:, i] for i in range(ranks.shape[1])]
    x = np.arange(len(labels))

    violins = ax.violinplot(data, positions=x, widths=0.6,
                             showmeans=False, showmedians=False,
                             showextrema=False)
    for body, label in zip(violins['bodies'], labels):
        body.set_facecolor(_color_for(label))
        body.set_edgecolor('black')
        body.set_alpha(0.35)

    # overlay individual condition points (slightly jittered), median line,
    # and mean + SEM marker for each optimizer
    for i, col in enumerate(data):
        jitter = (np.random.rand(len(col)) - 0.5) * 0.12
        ax.scatter(np.full(len(col), x[i]) + jitter, col,
                   color=_color_for(labels[i]), edgecolor='black', s=30,
                   linewidth=0.4, alpha=0.8)
        median = np.median(col)
        mean = np.mean(col)
        sem = np.std(col, ddof=1) / np.sqrt(len(col))
        ax.hlines(median, x[i] - 0.22, x[i] + 0.22, color='black', linewidth=1.2)
        ax.errorbar(x[i], mean, yerr=sem, color='black', fmt='o', capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_title('B. Rank distribution per optimizer', fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelbottom=True, labelleft=True)


def _plot_pvalue_heatmap(ax, labels, pairwise_holm):
    """Symmetric matrix of Holm-corrected pairwise p-values, annotated with
    the value and significance stars in each cell -- localizes exactly which
    pair(s) the omnibus Friedman result is coming from."""
    n = len(labels)
    p_matrix = np.full((n, n), np.nan)
    index = {label: i for i, label in enumerate(labels)}
    for entry in pairwise_holm:
        i, j = index[entry['a']], index[entry['b']]
        p_matrix[i, j] = p_matrix[j, i] = entry['p_holm']

    display = np.where(np.isnan(p_matrix), 1.0, p_matrix)
    im = ax.imshow(display, cmap='viridis_r', vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, '\u2014', ha='center', va='center', fontsize=10,
                       color='0.5')
                continue
            p = p_matrix[i, j]
            stars = _p_to_stars(p)
            text_color = 'white' if p < 0.5 else 'black'
            ax.text(j, i, f'{p:.3f}\n{stars}', ha='center', va='center',
                   fontsize=9, color=text_color, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title('C. Holm-corrected pairwise p-values', fontsize=11,
                fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('p-value (Holm-corrected)', fontsize=8)
    ax.tick_params(labelbottom=True, labelleft=True)


def plot_optimizer_statistics_figure(panels, stats, out_path):
    """Three-panel figure: (A) per-dose MAE table -- the raw numbers --
    (B) rank distribution per optimizer, and (C) the Holm-corrected pairwise
    p-values that localize which pair(s) the omnibus Friedman result comes
    from. Meant to stand as the single figure a reader needs to go from "here
    are the numbers" to "is that difference real" to "which optimizers does
    it involve.\""""
    labels = [panel['label'] for panel in panels]
    condition_keys, matrix = _collect_paired_mae(panels)
    ranks = np.array([pd.Series(row).rank().to_numpy() for row in matrix])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4),
                             gridspec_kw={'width_ratios': [1.3, 1, 1]})
    _draw_mae_table(axes[0], panels)
    _plot_rank_panel(axes[1], labels, ranks)
    _plot_pvalue_heatmap(axes[2], labels, stats['pairwise_holm'])

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, bbox_inches='tight', transparent=False, facecolor='white')


def plot_convergence_figure(panels, out_path):
    """Best-value-so-far vs. function evaluations. For any panel that ran
    multiple seeds (seed_runs present -- see run_multi_seed), every seed's
    trajectory is drawn faded, with the seed that produced the reported
    best-of-N result drawn bold and in the legend -- so a reader can see at a
    glance whether the reported curve is representative of the algorithm or
    an outlier among its own seeds. Panels without seed_runs (currently just
    differential_evolution, loaded rather than rerun -- see _load_de_panel)
    fall back to a single curve, or a dashed reference line at the final
    value if no trajectory was recorded at all.

    Note the dashed reference line is not quite the same quantity as the
    y-axis for the seeded curves (their y-axis is the joint objective during
    the search, before the post-hoc g0 polish that per-dose MAE reflects) --
    close enough to be a useful reference, not exact enough to treat as a
    fourth trajectory.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for panel in panels:
        color = _color_for(panel['label'])
        seed_runs = panel.get('seed_runs')
        if seed_runs:
            for run in seed_runs:
                history = run.get('convergence_history')
                if not history:
                    continue
                nfevs, bests = zip(*history)
                is_best = (run['seed'] == panel.get('best_seed'))
                ax.plot(nfevs, bests, color=color,
                       linewidth=2.2 if is_best else 0.8,
                       alpha=1.0 if is_best else 0.30,
                       label=(f"{panel['label']} "
                              f"(best of {len(seed_runs)} seeds)")
                             if is_best else None,
                       zorder=3 if is_best else 2)
        else:
            history = panel.get('convergence_history')
            if history:
                nfevs, bests = zip(*history)
                ax.plot(nfevs, bests, color=color, linewidth=1.6,
                       label=panel['label'])
            else:
                ax.axhline(panel['overall_mean'], color=color, linestyle='--',
                          linewidth=1.6,
                          label=f"{panel['label']} "
                                "(final per-dose MAE mean; trajectory not recorded)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Function evaluations')
    ax.set_ylabel('Best objective value so far')
    ax.set_title('Optimizer convergence', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')


def plot_seed_stability_figure(panels, out_path):
    """One point per seed per optimizer, showing overall_mean MAE, so
    seed-to-seed variability can be judged directly rather than only trusting
    a single best-of-N headline number. The best-of-N result actually
    reported for that optimizer is marked with a star; a small spread around
    it means the headline number is representative of what the algorithm
    reliably achieves, a large spread means it isn't.

    differential_evolution has no seed_runs (single run, loaded from disk --
    see _load_de_panel) and is shown as a single diamond marker labeled
    accordingly, rather than a distribution it doesn't have."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    xt_positions, xt_labels = [], []

    for x, panel in enumerate(panels):
        label = panel['label']
        color = _color_for(label)
        seed_runs = panel.get('seed_runs')
        xt_positions.append(x)
        xt_labels.append(label)

        if seed_runs:
            vals = np.array([r['overall_mean'] for r in seed_runs])
            rng = np.random.default_rng(0)
            jitter = (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(np.full(len(vals), x) + jitter, vals, color=color,
                      edgecolor='black', linewidth=0.5, s=45, alpha=0.85,
                      zorder=3)
            ax.scatter([x], [panel['overall_mean']], color=color,
                      edgecolor='black', linewidth=0.8, s=180, marker='*',
                      zorder=4,
                      label=f'{label} (best of {len(vals)} seeds)')
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            ax.text(x, vals.max() + (vals.max() - vals.min() + 1e-6) * 0.12,
                   f'sd={sd:.4f}', ha='center', va='bottom', fontsize=7.5)
        else:
            ax.scatter([x], [panel['overall_mean']], color=color,
                      edgecolor='black', linewidth=0.8, s=160, marker='D',
                      zorder=4, label=f'{label} (single run, not seeded)')

    ax.set_xticks(xt_positions)
    ax.set_xticklabels(xt_labels, rotation=15, ha='right')
    ax.set_ylabel('Overall mean MAE')
    ax.set_title('Seed-to-seed stability per optimizer', fontsize=13,
                fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')


# --- manuscript-style per-strain, per-optimizer dose grids ----------------
#
# Matches the visual language of the manuscript's own Fig. 2 panels B/C:
# dose label bold in the top-left corner of each subplot (no boxed
# annotation), shared legend centered at the bottom of the figure reading
# "Experimental (Blue/Pink)" / "Predicted (Blue/Pink)", and a bold figure
# title naming the strain. One such figure is produced per optimizer per
# strain, rather than one figure mixing multiple optimizers, so each output
# is a drop-in replacement for the existing panel with a different fit
# underneath it.

_LEGEND_ORDER = ('Experimental (Blue)', 'Experimental (Pink)',
                 'Predicted (Blue)', 'Predicted (Pink)')


def _plot_manuscript_dose_cell(ax, growth, experiment_err, params, g0,
                               condition_key, dose_gy):
    """One subplot styled after the manuscript's Fig. 2 panels: measured
    points with error bars, predicted curves, dose label bold top-left."""
    measurement, bands = experiment_err[condition_key]
    lower_blue, lower_pink = bands['lower']
    upper_blue, upper_pink = bands['upper']
    blue_err = np.abs(upper_blue - lower_blue) / 2.0
    pink_err = np.abs(upper_pink - lower_pink) / 2.0

    prediction = M.predict(growth[condition_key], params, g0=g0)

    ax.errorbar(measurement.time, measurement.blue, yerr=blue_err, fmt='o',
               color='tab:blue', ms=3, elinewidth=0.8, capsize=2,
               label='Experimental (Blue)')
    ax.errorbar(measurement.time, measurement.pink, yerr=pink_err, fmt='o',
               color='tab:pink', ms=3, elinewidth=0.8, capsize=2,
               label='Experimental (Pink)')
    ax.plot(prediction.time, prediction.blue, color='tab:blue', lw=1.5,
           label='Predicted (Blue)')
    ax.plot(prediction.time, prediction.pink, color='tab:pink', lw=1.5,
           label='Predicted (Pink)')

    # Dose label as a left-aligned subplot title (above the axes) rather than
    # an in-axes text box: at t=0 the blue series sits right at ~1.0, which
    # collided with a top-left in-axes label. Living above the plot avoids
    # that collision entirely rather than just nudging the overlap smaller.
    ax.set_title(f'{dose_gy:g} Gy', loc='left', fontsize=11,
                fontweight='bold', pad=6)
    ax.set_xlim(0, M.TRUNCATE_HOURS)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    # Every subplot gets its own numbered ticks. sharex/sharey are off at the
    # figure level for this reason (see _plot_optimizer_strain_grid), so this
    # is just making that explicit/robust rather than relying on defaults.
    ax.tick_params(labelbottom=True, labelleft=True)


def _plot_optimizer_strain_grid(growth, experiment_err, panel, strain):
    """One figure for one (optimizer, strain) pair: all doses for that strain
    laid out the way the manuscript's own panel is (3 columns; as many rows
    as needed for the strain's dose count, 2 rows for the usual 6 doses),
    using only that panel's fit. Produces the direct visual replacement for
    Fig. 2 panel B (WT) or C (rad51), once per optimizer."""
    conds = M.conditions(strain)
    doses = [c.dose_gy for c in conds]
    keys = [c.key for c in conds]
    dose_field = 'wt_per_dose' if strain == 'WT' else 'rad51_per_dose'
    g0_field = 'g0_wt' if strain == 'WT' else 'g0_rad51'
    strain_title = 'Wild Type' if strain == 'WT' else 'rad51\u0394'

    n = len(conds)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    # sharex/sharey are off deliberately: matplotlib's shared-axis mode hides
    # tick labels on interior subplots, and every subplot needs its own
    # numbered axes here. Scale consistency across panels is instead enforced
    # explicitly (set_xlim/set_ylim inside _plot_manuscript_dose_cell) rather
    # than relying on axis sharing to provide it.
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.3 * n_rows),
                             sharex=False, sharey=False, squeeze=False)

    params = tuple(panel['params'][k] for k in
                   ('v1', 'v2', 'v3', 'K1', 'K2', 'K3', 'k'))
    g0 = panel[g0_field]

    for i, (key, dose) in enumerate(zip(keys, doses)):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        _plot_manuscript_dose_cell(ax, growth, experiment_err, params, g0,
                                   key, dose)
        # annotate the per-dose MAE for this optimizer/subplot (top-right)
        dose_mae = panel[dose_field].get(key, None)
        if dose_mae is not None:
            ax.text(0.98, 0.92, f'MAE={dose_mae:.3f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.85, edgecolor='0.8'))
        if col == 0:
            ax.set_ylabel('Concentration fraction')
        if row == n_rows - 1:
            ax.set_xlabel('Time (h)')

    for j in range(n, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].set_axis_off()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    ordered = [(h, l) for l in _LEGEND_ORDER for h, ll in zip(handles, labels) if ll == l]
    if ordered:
        handles, labels = zip(*ordered)
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.02 / n_rows), frameon=False)

    mae = panel[dose_field]
    overall = float(np.mean(list(mae.values())))
    fig.suptitle(f'Alamarblue Assay: {strain_title} Strain Response to '
                f"Radiation -- {panel['label']} (MAE={overall:.3f})",
                fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return fig


def _all_doses(strains=('WT', 'rad51')):
    """Union of dose values across the given strains, sorted ascending."""
    doses = set()
    for strain in strains:
        for c in M.conditions(strain):
            doses.add(round(float(c.dose_gy), 6))
    return sorted(doses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-smac', action='store_true',
                        help='also run the SMAC3 reference panel (requires '
                             "'smac' and 'ConfigSpace'; fixed trial budget, "
                             'not directly on equal footing with the other '
                             'three -- see run_smac docstring)')
    parser.add_argument('--smac-trials', type=int, default=5000,
                        help='SMAC3 trial budget if --with-smac is set '
                             '(default: 5000)')
    parser.add_argument('--n-seeds', type=int, default=DEFAULT_N_SEEDS,
                        help='number of independent seeds to run '
                             'dual_annealing and CMA-ES from; the best-of-N '
                             'result is reported as that optimizer\'s panel '
                             'and the full spread is retained for a seed-'
                             'stability figure (default: '
                             f'{DEFAULT_N_SEEDS}). Runtime scales linearly '
                             'with this -- each additional seed re-runs both '
                             'optimizers to their own convergence.')
    args = parser.parse_args()

    _apply_publiplots_style()

    growth = M.load_growth_curves()
    experiment = M.load_experimental()
    # load_experimental(with_error=True) returns {key: (Measurement, bands)},
    # which is exactly what _plot_manuscript_dose_cell expects -- there's no
    # separate "with_error_bands" function in ammper_ab_model_fixed.
    experiment_err = M.load_experimental(with_error=True)

    print('Loading prior on-disk differential_evolution fit for reference '
         '(not used in the comparison below)...')
    _reference_de_panel = _load_de_panel()
    print(f"  prior fit reported MAE={_reference_de_panel['overall_mean']:.4f} "
         f"nfev={_reference_de_panel['nfev']}")

    seeds = [SEED + i for i in range(args.n_seeds)]

    print(f'\nRunning differential_evolution across {len(seeds)} seeds '
         f'{seeds}, each to its own convergence...')
    panel_a = run_multi_seed(run_differential_evolution, growth, experiment,
                             seeds, 'differential_evolution')
    de_vals = [r['overall_mean'] for r in panel_a['seed_runs']]
    print(f"  best overall mean MAE = {panel_a['overall_mean']:.4f} "
         f"(seed={panel_a['best_seed']})")
    print(f"  across seeds: mean={np.mean(de_vals):.4f} "
         f"sd={np.std(de_vals, ddof=1):.4f} "
         f"range=[{min(de_vals):.4f}, {max(de_vals):.4f}]")
    print(f"  (prior on-disk fit for comparison: "
         f"MAE={_reference_de_panel['overall_mean']:.4f})")

    print(f'\nRunning dual_annealing across {len(seeds)} seeds {seeds}, '
         'each to its own convergence...')
    panel_b = run_multi_seed(run_dual_annealing, growth, experiment, seeds,
                             'dual_annealing')
    da_vals = [r['overall_mean'] for r in panel_b['seed_runs']]
    print(f"  best overall mean MAE = {panel_b['overall_mean']:.4f} "
         f"(seed={panel_b['best_seed']})")
    print(f"  across seeds: mean={np.mean(da_vals):.4f} "
         f"sd={np.std(da_vals, ddof=1):.4f} "
         f"range=[{min(da_vals):.4f}, {max(da_vals):.4f}]")

    print(f'\nRunning CMA-ES across {len(seeds)} seeds {seeds}, '
         'each to its own convergence...')
    panel_c = run_multi_seed(run_cma_es, growth, experiment, seeds, 'CMA-ES')
    cma_vals = [r['overall_mean'] for r in panel_c['seed_runs']]
    print(f"  best overall mean MAE = {panel_c['overall_mean']:.4f} "
         f"(seed={panel_c['best_seed']})")
    print(f"  across seeds: mean={np.mean(cma_vals):.4f} "
         f"sd={np.std(cma_vals, ddof=1):.4f} "
         f"range=[{min(cma_vals):.4f}, {max(cma_vals):.4f}]")

    panels = [panel_a, panel_b, panel_c]

    if args.with_smac:
        print(f'\nRunning SMAC3 (n_trials={args.smac_trials}, reference '
             'panel, fixed budget -- see run_smac docstring)...')
        panel_d = run_smac(growth, experiment, n_trials=args.smac_trials)
        print(f"  overall mean MAE = {panel_d['overall_mean']:.4f}  "
             f"nfev={panel_d['nfev']}  {panel_d['wall_seconds']:.0f}s")
        panels.append(panel_d)

    report = {'display_conditions': DISPLAY_KEYS, 'panels': panels}
    with open(OUT_JSON, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nwrote {OUT_JSON}')

    print('\nComputing optimizer statistics (Friedman + pairwise Wilcoxon/Holm)...')
    stats = _optimizer_statistics(panels)
    stats['seed_stability'] = _seed_stability_summary(panels)
    with open(OUT_STATS_JSON, 'w') as handle:
        json.dump(stats, handle, indent=2)
    print(f"  Friedman: chi2={stats['friedman_statistic']:.2f} "
         f"p={stats['friedman_p']:.4f} "
         f"({'sig' if stats['friedman_significant_0.05'] else 'n.s.'})")
    for pair in stats['pairwise_holm']:
        print(f"    {pair['a']} vs {pair['b']}: p_holm={pair['p_holm']:.4f}")
    for label, summary in stats['seed_stability'].items():
        print(f"  {label} seed stability: n={summary['n_seeds']} "
             f"best={summary['best_overall_mean']:.4f} "
             f"(seed {summary['best_seed']})  "
             f"mean={summary['mean_overall_mean']:.4f} "
             f"sd={summary['sd_overall_mean']:.4f}")
    print(f'wrote {OUT_STATS_JSON}')

    fig, axes = plt.subplots(2, len(panels), figsize=(4.3 * len(panels), 8.0),
                             sharey=True, squeeze=False)
    for row, (condition_key, g0_key, row_label) in enumerate(
            zip(DISPLAY_KEYS, DISPLAY_G0_KEYS, ROW_LABELS)):
        for col, panel in enumerate(panels):
            _plot_panel(axes[row, col], growth, experiment, panel,
                       condition_key, g0_key, row_label)
    axes[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle('Optimizer comparison across 30 Gy' +
                (', SMAC3 at a fixed trial budget' if args.with_smac else ''))
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, transparent=False, facecolor='white')
    print(f'wrote {OUT_FIG}')

    plot_error_bars(panels, stats, OUT_ERR_FIG)
    print(f'wrote {OUT_ERR_FIG}')

    plot_optimizer_statistics_figure(panels, stats, OUT_STATS_FIG)
    print(f'wrote {OUT_STATS_FIG}')

    plot_convergence_figure(panels, OUT_CONVERGENCE_FIG)
    print(f'wrote {OUT_CONVERGENCE_FIG}')

    plot_seed_stability_figure(panels, OUT_SEED_FIG)
    print(f'wrote {OUT_SEED_FIG}')

    print('\nWriting per-optimizer, per-strain dose grids '
         '(manuscript Fig. 2 B/C style)...')
    for panel in panels:
        for strain in ('WT', 'rad51'):
            fig = _plot_optimizer_strain_grid(growth, experiment_err, panel, strain)
            path = _strain_dose_fig_path(panel['label'], strain)
            fig.savefig(path, dpi=200, transparent=False, facecolor='white')
            plt.close(fig)
            print(f'  wrote {path}')

    return report, stats


if __name__ == '__main__':
    main()