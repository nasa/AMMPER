"""
Compare global optimizers for the joint aB kinetics fit (comparison figure).

Panel A reuses the differential_evolution fit already produced by
fit_ab_model_fixed.py (loaded from fitted_parameters.json, not rerun -- so the
number shown here is exactly the number reported elsewhere in the revision,
not a second DE run that happens to agree with the first). Panels B and C run
two additional global optimizers -- dual_annealing and CMA-ES -- on the same
_joint_objective, each left to its own default convergence criteria rather
than capped to a shared evaluation budget.

That is a deliberate choice, not an oversight: the point of this comparison is
not "which optimizer is most efficient under a fixed allowance," it is
"does the fit depend on which search algorithm was used." Capping every
optimizer to DE's nfev would answer the first question, and would leave open
the objection that a competing optimizer looked worse only because it was cut
off before its natural stopping point rather than because it converges to a
worse answer. Letting each algorithm run to its own convergence and then
comparing the resulting parameters and MAE answers the second, stronger
question: three structurally different search strategies -- population with
crossover, an annealed trajectory, and covariance-matrix adaptation --
converging to compatible fits is evidence the result is a property of the data
and the physicality constraint, not an artifact of one optimizer's settings.
nfev and wall time are still recorded and shown on each panel, but as
descriptive output, not as an equalized input.

Two settings are not left at library defaults, for a resource-safety reason
rather than a fairness one: dual_annealing's local-search refinement is turned
off (see run_dual_annealing), and both dual_annealing and CMA-ES carry a
generous evaluation-count cap (MAX_FEVALS_SAFETY_CAP). The objective has a
hard flat penalty plateau wherever a parameter set is chemically inadmissible,
and that shape can in principle keep either algorithm's own stopping check
from ever triggering -- observed in practice as an uncapped dual_annealing run
being killed by the OS rather than converging. The cap is set well above what
either optimizer needs on this problem, so it is a backstop against a runaway
process, not the thing determining the reported result.

Why these two
-------------
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

Outputs
-------
    optimizer_comparison.json          per-algorithm params, errors, nfev, wall time
    optimizer_comparison.png           2x3 fitted-curve grid (strain x optimizer,
                                       highest dose only -- kept as a quick-look
                                       summary)
    optimizer_comparison_errors.png    per-dose MAE, grouped by optimizer, one
                                       subplot per strain
    optimizer_comparison_doses_WT.png       full dose grid, wild type: rows =
    optimizer_comparison_doses_rad51.png    optimizer, columns = all six doses,
                                            styled after the manuscript's own
                                            per-dose figures (measured points
                                            with error bars, predicted curves,
                                            dose + MAE annotation per panel).
                                            This is the one that actually
                                            answers "does the optimizer choice
                                            change the fit at every dose,"
                                            not just at the hardest one.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing

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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, 'optimizer_comparison.json')
OUT_FIG = os.path.join(HERE, 'optimizer_comparison.png')
OUT_ERR_FIG = os.path.join(HERE, 'optimizer_comparison_errors.png')
OUT_DOSE_FIG = {
    'WT': os.path.join(HERE, 'optimizer_comparison_doses_WT.png'),
    'rad51': os.path.join(HERE, 'optimizer_comparison_doses_rad51.png'),
}

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


class ProgressTracker:
    """Wraps _joint_objective so a long-running optimizer reports live
    progress instead of going silent until it finishes.

    Uses tqdm if it's installed (pip install tqdm --break-system-packages);
    otherwise falls back to a plain print every `report_every` seconds, so
    this works with no new dependency if you'd rather not install one. Either
    way you get evaluation count and best-so-far MAE while the optimizer is
    still running, which matters here because a killed or genuinely slow run
    otherwise looks identical to a healthy one from the terminal.
    """

    def __init__(self, growth, experiment, label, total=None, report_every=3.0):
        self.growth = growth
        self.experiment = experiment
        self.label = label
        self.total = total
        self.nfev = 0
        self.best = np.inf
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
ROW_LABELS = ('wild type', 'rad51-delta')


def _finalize(vector, growth, experiment, nfev, elapsed, label):
    """Turn a raw parameter vector into the same shape fit_ab_model_fixed's
    report uses, so every panel is directly comparable."""
    params = tuple(vector[:N_KINETIC])
    wt_g0 = _scan_g0(growth, experiment, params, WT_KEYS)
    rad_g0 = _scan_g0(growth, experiment, params, RAD_KEYS)
    err_wt = per_dose_errors(growth, experiment, params, wt_g0, WT_KEYS)
    err_rad = per_dose_errors(growth, experiment, params, rad_g0, RAD_KEYS)
    all_errors = dict(err_wt, **err_rad)
    return {
        'label': label,
        'params': {'v1': params[0], 'v2': params[1], 'v3': params[2],
                   'K1': params[3], 'K2': params[4], 'K3': params[5],
                   'k': params[6]},
        'g0_wt': wt_g0, 'g0_rad51': rad_g0,
        'wt_per_dose': err_wt, 'rad51_per_dose': err_rad,
        'overall_mean': float(np.mean(list(all_errors.values()))),
        'nfev': int(nfev), 'wall_seconds': float(elapsed),
    }


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
                     'dual_annealing')


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
                     es.result.evaluations, elapsed, 'CMA-ES')


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
                     f'SMAC3 (n_trials={n_trials})')


def _load_de_panel():
    """Panel A: the differential_evolution fit already on disk. Not rerun, so
    this is exactly the number reported elsewhere in the revision."""
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
    ax.set_xlabel('hours')
    ax.set_ylabel('fraction')
    ax.set_ylim(-0.05, 1.05)


def _plot_error_bars(fig, growth, experiment, panels):
    """Separate panel: per-dose MAE, grouped by optimizer, one subplot per
    strain. Complements the fitted-curve grid -- that shows fit *shape* at the
    single hardest dose per strain, this shows fit *magnitude* across every
    condition, which is the more complete quantitative comparison."""
    wt_doses = [c.dose_gy for c in M.conditions('WT')]
    rad_doses = [c.dose_gy for c in M.conditions('rad51')]
    wt_keys = [c.key for c in M.conditions('WT')]
    rad_keys = [c.key for c in M.conditions('rad51')]

    axes = fig.subplots(1, 2, sharey=True)
    width = 0.8 / len(panels)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax, doses, keys, dose_field, title in (
            (axes[0], wt_doses, wt_keys, 'wt_per_dose', 'wild type'),
            (axes[1], rad_doses, rad_keys, 'rad51_per_dose', 'rad51-delta')):
        x = np.arange(len(doses))
        for i, panel in enumerate(panels):
            errors = [panel[dose_field][key] for key in keys]
            ax.bar(x + (i - (len(panels) - 1) / 2) * width, errors, width,
                  label=panel['label'], color=color_cycle[i % len(color_cycle)])
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d:g}' for d in doses])
        ax.set_xlabel('dose (Gy)')
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel('mean absolute error')
    axes[0].legend(fontsize=8)


def _plot_dose_cell(ax, growth, experiment_err, params, g0, condition_key,
                    dose_gy, mae):
    """One cell of the per-optimizer dose grid: measured (with error bars)
    vs predicted blue/pink for one condition, styled after the manuscript's
    own per-dose figures -- dose + MAE annotated in a corner box rather than
    in the subplot title, so the grid reads the same way those figures do."""
    measurement, bands = experiment_err[condition_key]
    lower_blue, lower_pink = bands['lower']
    upper_blue, upper_pink = bands['upper']
    blue_err = np.abs(upper_blue - lower_blue) / 2.0
    pink_err = np.abs(upper_pink - lower_pink) / 2.0

    prediction = M.predict(growth[condition_key], params, g0=g0)

    ax.errorbar(measurement.time, measurement.blue, yerr=blue_err, fmt='o',
               color='tab:blue', ms=3, elinewidth=0.8, capsize=2,
               label='Blue, measured')
    ax.errorbar(measurement.time, measurement.pink, yerr=pink_err, fmt='s',
               color='tab:red', ms=3, elinewidth=0.8, capsize=2,
               label='Pink, measured')
    ax.plot(prediction.time, prediction.blue, color='tab:blue', lw=1.5,
           label='Blue, predicted')
    ax.plot(prediction.time, prediction.pink, color='tab:red', lw=1.5,
           label='Pink, predicted')

    ax.text(0.96, 0.95, f'{dose_gy:g} Gy\nMAE {mae:.3f}',
           transform=ax.transAxes, ha='right', va='top', fontsize=7,
           bbox=dict(boxstyle='round', facecolor='white',
                     edgecolor='0.7', alpha=0.9))
    ax.set_ylim(-0.05, 1.05)


def _plot_optimizer_dose_grid(growth, experiment_err, panels, strain):
    """Full comparison grid for one strain: rows = optimizer, columns = every
    dose for that strain (not just the highest one). This is the figure that
    actually answers whether the optimizer choice changes the fit across the
    whole dose range, rather than at a single representative point."""
    conds = M.conditions(strain)
    doses = [c.dose_gy for c in conds]
    keys = [c.key for c in conds]
    dose_field = 'wt_per_dose' if strain == 'WT' else 'rad51_per_dose'
    g0_field = 'g0_wt' if strain == 'WT' else 'g0_rad51'
    strain_label = 'wild type' if strain == 'WT' else 'rad51-delta'

    n_rows, n_cols = len(panels), len(doses)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 2.3 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    for row, panel in enumerate(panels):
        params = tuple(panel['params'][k] for k in
                       ('v1', 'v2', 'v3', 'K1', 'K2', 'K3', 'k'))
        g0 = panel[g0_field]
        for col, (key, dose) in enumerate(zip(keys, doses)):
            ax = axes[row, col]
            mae = panel[dose_field][key]
            _plot_dose_cell(ax, growth, experiment_err, params, g0, key,
                           dose, mae)
            if row == 0:
                ax.set_title(f'{dose:g} Gy', fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{panel['label']}\nconcentration fraction",
                             fontsize=7.5)
            if row == n_rows - 1:
                ax.set_xlabel('time (h)', fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, -0.015 / n_rows))
    fig.suptitle(f'Gamma radiation, {strain_label}: '
                f'predicted vs measured across all doses, by optimizer')
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


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
    args = parser.parse_args()

    growth = M.load_growth_curves()
    experiment = M.load_experimental()

    panel_a = _load_de_panel()
    print(f"differential_evolution (existing fit): "
         f"MAE={panel_a['overall_mean']:.4f}  nfev={panel_a['nfev']}")

    print('\nRunning dual_annealing to its own convergence...')
    panel_b = run_dual_annealing(growth, experiment)
    print(f"  overall mean MAE = {panel_b['overall_mean']:.4f}  "
         f"nfev={panel_b['nfev']}  {panel_b['wall_seconds']:.0f}s")

    print('\nRunning CMA-ES to its own convergence...')
    panel_c = run_cma_es(growth, experiment)
    print(f"  overall mean MAE = {panel_c['overall_mean']:.4f}  "
         f"nfev={panel_c['nfev']}  {panel_c['wall_seconds']:.0f}s")

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

    fig, axes = plt.subplots(2, len(panels), figsize=(4.3 * len(panels), 8.0),
                             sharey=True, squeeze=False)
    for row, (condition_key, g0_key, row_label) in enumerate(
            zip(DISPLAY_KEYS, DISPLAY_G0_KEYS, ROW_LABELS)):
        for col, panel in enumerate(panels):
            _plot_panel(axes[row, col], growth, experiment, panel,
                       condition_key, g0_key, row_label)
    axes[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle('Optimizer comparison, highest dose per strain '
                '(each run to its own convergence' +
                (', SMAC3 at a fixed trial budget' if args.with_smac else '') +
                ')')
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    print(f'wrote {OUT_FIG}')

    err_fig = plt.figure(figsize=(10, 4.2))
    _plot_error_bars(err_fig, growth, experiment, panels)
    err_fig.suptitle('Per-dose MAE by optimizer')
    err_fig.tight_layout()
    err_fig.savefig(OUT_ERR_FIG, dpi=200)
    print(f'wrote {OUT_ERR_FIG}')
    return report


if __name__ == '__main__':
    main()
