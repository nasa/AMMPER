"""
Supplementary gamma figures: the recovered gamma alamarBlue comparison, and the
evidence for the hit-rate recalibration that made it possible.

The submitted supplement carried a single gamma figure -- three rad51-delta dose
panels whose predicted curves fell away from the first timepoint at every dose --
and a caption stating that the approach "failed to recapitulate the experimental
data behavior under gamma radiation". Two defects produced that figure, and the
figures here are organized to let a reader check each one separately rather than
take the corrected version on trust:

    gamma_ab_predictions_WT.*        the recovered comparison, wild type
    gamma_ab_predictions_rad51.*     the recovered comparison, rad51-delta
    gamma_hit_rate_calibration.*     why the dose axis had to be recalibrated:
                                     simulated against measured survival, and the
                                     dose-rate degeneracy that licenses the fix
    gamma_error_decomposition.*      sensitivity of the MAE to the hit rate and
                                     to refitting the kinetics on gamma. NOT a
                                     supplementary figure: the three numbers it
                                     carried (0.172 uncalibrated, 0.049
                                     transferred, 0.042 refit) are stated in
                                     Supplementary Text S3 instead, which is all
                                     the argument needs. Kept because the
                                     sensitivity is worth being able to redraw.
    gamma_vs_proton.*                the two radiation qualities at matched dose

Every panel is drawn through figure_style, hence through publiplots
(Botas 2025, https://github.com/jorgebotas/publiplots), which sizes axes in
millimeters so that panels are physically comparable across figures.

    python3 make_gamma_figures.py

Requires fit_gamma_ab_model.py and calibrate_gamma_hit_rate.py to have been run.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import publiplots as pp

import ammper_ab_model_fixed as M
import ammper_ab_model_gamma as G
import calibrate_gamma_hit_rate as C
import extract_gamma_experimental as E
import figure_style as S
import run_gamma_simulations as R

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import ammper_paths as P  # noqa: E402

SUPP_OUT, _MAIN_OUT, _BUG_OUT = S.output_dirs()

FIT_JSON = os.path.join(HERE, 'fitted_parameters_gamma.json')
CALIB_JSON = os.path.join(HERE, 'gamma_hit_rate_calibration.json')


def _load(path, script):
    if not os.path.exists(path):
        raise SystemExit(f'{os.path.basename(path)} missing -- run {script} first')
    with open(path) as handle:
        return json.load(handle)


def _by_dose(mapping):
    """JSON dose-keyed dict back to float keys.

    json.dump stringifies float keys as repr, so 1.0 Gy comes back as '1.0'
    while ``f'{1.0:g}'`` is '1'. Converting once here avoids that mismatch
    being rediscovered at each use site.
    """
    return {float(key): value for key, value in mapping.items()}


# ---------------------------------------------------------------------------
# The recovered comparison
# ---------------------------------------------------------------------------

def prediction_panels(strain):
    """Six dose panels: measured points with error bars, predicted curves.

    Laid out to match the proton figure exactly -- same panel size, same colors,
    same marker shapes, same axis limits -- so the gamma comparison can be held
    against the proton one without mentally rescaling anything.

    The prediction band is the spread over replicate simulations, and it is wider
    here than in the proton figure for a reason worth seeing: at the calibrated
    dose the number of deposition events per plane is small (5 at 2.5 Gy nominal,
    56 at 30 Gy), so which cells get hit is genuinely stochastic. Twelve
    replicates per condition are used rather than three for that reason.
    """
    fit = _load(FIT_JSON, 'fit_gamma_ab_model.py')
    params = tuple(fit['transferred']['params'])
    g0 = fit['transferred']['g0'][strain]

    per_run = G.load_growth_curves_per_run(strain)
    experiment = G.load_experimental(strain, with_error=True)
    conditions = G.conditions(strain)

    fig, axes = pp.subplots(2, 3, axes_size=S.PANEL_SMALL, sharex=True,
                            sharey=True, hspace=7, wspace=6)
    flat = np.ravel(axes)
    errors = {}
    for ax, cond in zip(flat, conditions):
        measurement, bands = experiment[cond.key]
        predictions = [G.predict(curve, params, g0=g0)
                       for curve in per_run[cond.key]]
        time = predictions[0].time
        blue = np.stack([p.blue for p in predictions])
        pink = np.stack([p.pink for p in predictions])
        n = len(predictions)

        for values, color in ((blue, S.BLUE), (pink, S.PINK)):
            mean = values.mean(0)
            sem = values.std(0) / np.sqrt(n)
            ax.plot(time, mean, '-', linewidth=1.4, color=color, zorder=3)
            ax.fill_between(time, mean - sem, mean + sem, color=color,
                            alpha=0.18, linewidth=0, zorder=2)

        upper_blue, upper_pink = bands['upper']
        lower_blue, lower_pink = bands['lower']
        for series, low, high, color, marker in (
                (measurement.blue, lower_blue, upper_blue, S.BLUE, 'o'),
                (measurement.pink, lower_pink, upper_pink, S.PINK, 's')):
            ax.errorbar(measurement.time, series,
                        yerr=[np.abs(series - low), np.abs(series - high)],
                        fmt=marker, markersize=2.6, capsize=1.2, linewidth=0.7,
                        elinewidth=0.7, color=color, ecolor=color, alpha=0.75,
                        zorder=4)

        errors[cond.key] = M.mean_absolute_error(
            M.Prediction(time, blue.mean(0), pink.mean(0)), measurement)
        # Upper right: the blue series has decayed to near zero and the pink has
        # plateaued at about 0.38 by 15 h, so it is the one corner of these
        # panels that no curve passes through at any dose.
        S.inset_label(ax, f'{cond.dose_gy:g} Gy\nMAE {errors[cond.key]:.3f}',
                      loc='upper right')
        S.concentration_axes(ax, G.TRUNCATE_HOURS)

    for ax in flat[3:]:
        pp.set_axis_labels(ax, xlabel='Time (h)')
    for ax in (flat[0], flat[3]):
        pp.set_axis_labels(ax, ylabel='Concentration fraction')

    # One legend for the whole grid, below the bottom row: the species coding is
    # the same in all six panels, so repeating it six times would only cost
    # plotting area. Anchored to the figure rather than to an axes, since an
    # axes-anchored legend on a shared-axis grid lands between the rows.
    # bbox_inches='tight' at save time expands the canvas to include this, so the
    # anchor sits just below the figure rather than inside the bottom row.
    fig.legend(handles=S.species_legend(), loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=6.5)
    pp.suptitle(f'Gamma radiation, {S.STRAIN_LABEL[strain]}')
    written = S.save(fig, SUPP_OUT, f'gamma_ab_predictions_{strain}', tight=True)
    return errors, written


# ---------------------------------------------------------------------------
# Why the dose axis was recalibrated
# ---------------------------------------------------------------------------

def calibration_figure():
    """Three panels documenting the hit-rate calibration.

    (a) Simulated survival against measured survival on the published dose axis.
        The two curves are on completely different scales, which is the finding:
        no dye chemistry downstream of a population trajectory that has already
        gone extinct could have reproduced the measurements.
    (b) The same simulated curve against events per plane, with the archived
        k-sweep runs overlaid. Those three runs are the same 2.5 Gy dose at
        k = 50, 100 and 1000, so if survival depends on dose and k only through
        their product they must land on the curve traced by the k = 100 dose
        series -- which is what licenses recalibrating k by rescaling the dose
        axis rather than editing the simulator.
    (c) The calibration objective against k, showing the minimum and the
        published value.
    """
    calib = _load(CALIB_JSON, 'calibrate_gamma_hit_rate.py')
    shared = calib['shared']

    fig, axes = pp.subplots(1, 3, axes_size=S.PANEL_MEDIUM, wspace=12)
    ax_dose, ax_events, ax_cost = np.ravel(axes)

    # -- (a) simulated vs measured, published dose axis ---------------------
    doses = np.array(R.DOSES)
    for strain, color in (('WT', S.WT_COLOR), ('rad51', S.RAD51_COLOR)):
        simulated = _by_dose(calib['simulated_survival'][strain])
        ax_dose.plot(doses, [simulated[d] for d in doses],
                     '-o', color=color, markersize=2.6, linewidth=1.2,
                     label=f'{S.STRAIN_LABEL[strain]}, simulated')
        measured = _by_dose(calib['measured_survival'][strain])
        inside = sorted(d for d in measured if 0 < d <= max(doses))
        ax_dose.plot(inside, [measured[d] for d in inside],
                     '--s', color=color, markersize=2.6, linewidth=1.2,
                     markerfacecolor='white',
                     label=f'{S.STRAIN_LABEL[strain]}, measured')
    ax_dose.set_xscale('log')
    ax_dose.set_yscale('log')
    pp.set_axis_labels(ax_dose, xlabel='Dose (Gy), uncalibrated hit rate',
                       ylabel='Survival, fraction of 0 Gy')
    S.tidy(ax_dose)
    S.panel_letter(ax_dose, 'A')
    ax_dose.legend(frameon=False, fontsize=5.5, loc='lower left')

    # -- (b) the dose-rate degeneracy --------------------------------------
    simulated = _by_dose(calib['simulated_survival']['WT'])
    events = doses * C.PUBLISHED_HIT_RATE
    ax_events.plot(events, [simulated[d] for d in doses], '-', color='0.35',
                   linewidth=1.2, label='0.01-30 Gy at $k=100$')
    # The k=50 and k=100 points are a factor of two apart on a log axis spanning
    # three decades, so labels at a common offset overlap. Staggering the offsets
    # separates them without moving the markers.
    label_offsets = ((0, 7), (5, -8), (5, 3))
    for check, offset in zip(calib['degeneracy_check'], label_offsets):
        ax_events.plot(check['events_per_plane'], check['observed_survival'],
                       'D', markersize=4, color=S.CORRECTED,
                       markeredgecolor='white', markeredgewidth=0.5, zorder=5)
        ax_events.annotate(f"$k={check['hit_rate']:g}$",
                           (check['events_per_plane'],
                            check['observed_survival']),
                           textcoords='offset points', xytext=offset,
                           fontsize=5.5, color=S.CORRECTED)
    ax_events.plot([], [], 'D', markersize=4, color=S.CORRECTED,
                   label='2.5 Gy at $k=50,100,1000$')
    ax_events.set_xscale('log')
    ax_events.set_yscale('log')
    pp.set_axis_labels(ax_events, xlabel='Deposition events per plane',
                       ylabel='Survival, fraction of 0 Gy')
    S.tidy(ax_events)
    S.panel_letter(ax_events, 'B')
    ax_events.legend(frameon=False, fontsize=5.5, loc='lower left')

    # -- (c) the objective against k ---------------------------------------
    grid = np.exp(np.linspace(np.log(0.2), np.log(2 * C.PUBLISHED_HIT_RATE), 400))
    measured = {s: C.measured_survival(s) for s in ('WT', 'rad51')}
    sim = {s: _by_dose(calib['simulated_survival'][s]) for s in ('WT', 'rad51')}
    cost = [np.sqrt(np.mean(np.concatenate(
                [C._residuals(rate, measured[s], sim[s])**2
                 for s in ('WT', 'rad51')]))) for rate in grid]
    ax_cost.plot(grid, cost, '-', color='0.2', linewidth=1.3)
    ax_cost.axvline(shared, color=S.CORRECTED, linewidth=1.1,
                    label=f'calibrated, $k={shared:.2f}$')
    ax_cost.axvline(C.PUBLISHED_HIT_RATE, color=S.PUBLISHED, linewidth=1.1,
                    linestyle='--', label='uncalibrated, $k=100$')
    for strain, color in (('WT', S.WT_COLOR), ('rad51', S.RAD51_COLOR)):
        ax_cost.axvline(calib['per_strain'][strain], color=color,
                        linewidth=0.8, linestyle=':',
                        label=f"{S.STRAIN_LABEL[strain]} alone, "
                              f"$k={calib['per_strain'][strain]:.2f}$")
    ax_cost.set_xscale('log')
    pp.set_axis_labels(ax_cost, xlabel='Hit rate $k$ (events Gy$^{-1}$ plane$^{-1}$)',
                       ylabel='RMS log-survival residual')
    S.tidy(ax_cost)
    S.panel_letter(ax_cost, 'C')
    ax_cost.legend(frameon=False, fontsize=5.5, loc='upper left')

    return S.save(fig, SUPP_OUT, 'gamma_hit_rate_calibration')


# ---------------------------------------------------------------------------
# What each correction is worth
# ---------------------------------------------------------------------------

#: Configurations in fitted_parameters_gamma.json that are reportable modeling
#: choices, with the label used in the figure. The file also carries two
#: configurations that only exist as intermediate states of the analysis; those
#: belong in notes/BUGS_AND_CORRECTIONS.md and not in the manuscript, so they are
#: deliberately not plotted here.
STAGES = [
    ('published_rate', 'Uncalibrated rate,\n$k=100$'),
    ('transferred', 'Calibrated rate,\nproton kinetics'),
    ('refit', 'Calibrated rate,\nkinetics refit'),
]


def error_decomposition():
    """Sensitivity of the gamma comparison to the two modeling choices.

    Two things this is meant to make checkable. First, that the hit rate is what
    the comparison turns on: at the uncalibrated rate the error is
    several times larger, whatever the dye chemistry does downstream. Second,
    that refitting the dye chemistry on the gamma data buys almost nothing over
    transferring it from the proton fit, which is what makes the transferred
    configuration reportable as a held-out test rather than as a second fitting
    exercise.
    """
    fit = _load(FIT_JSON, 'fit_gamma_ab_model.py')

    rows = []
    for key, label in STAGES:
        block = fit[key]
        per_dose = (block.get('per_dose')
                    or dict(block['wt_per_dose'], **block['rad51_per_dose']))
        for condition, error in per_dose.items():
            strain = 'rad51' if 'rad51' in condition else 'WT'
            rows.append({'stage': label, 'condition': condition,
                         'strain': S.STRAIN_LABEL[strain], 'mae': error})
    frame = pd.DataFrame(rows)

    fig, axes = pp.subplots(1, 2, axes_size=S.PANEL_MEDIUM, sharey=True,
                            wspace=10)
    ax_stage, ax_dose = np.ravel(axes)

    order = [label for _key, label in STAGES]
    offsets = {'Wild type': -0.14, S.STRAIN_LABEL['rad51']: 0.14}
    for strain_label, offset in offsets.items():
        subset = frame[frame['strain'] == strain_label]
        color = S.WT_COLOR if strain_label == 'Wild type' else S.RAD51_COLOR
        x = [order.index(s) + offset for s in subset['stage']]
        ax_stage.scatter(x, subset['mae'], s=6, color=color, alpha=0.55,
                         linewidth=0, zorder=3)
        means = [subset[subset['stage'] == s]['mae'].mean() for s in order]
        ax_stage.plot([i + offset for i in range(len(order))], means, '-',
                      color=color, linewidth=1.2, marker='_', markersize=9,
                      markeredgewidth=1.6, label=strain_label, zorder=4)
    ax_stage.set_xticks(range(len(order)))
    ax_stage.set_xticklabels(order, rotation=35, ha='right', fontsize=5.5)
    ax_stage.set_ylim(bottom=0)
    pp.set_axis_labels(ax_stage, ylabel='Mean absolute error')
    S.tidy(ax_stage)
    S.panel_letter(ax_stage, 'A')
    ax_stage.legend(frameon=False, fontsize=6, loc='upper right')

    # -- per dose, uncalibrated rate against the reported configuration -----
    for key, style, alpha in (('published_rate', '--', 0.9),
                              ('transferred', '-', 1.0)):
        block = fit[key]
        per_dose = (block.get('per_dose')
                    or dict(block['wt_per_dose'], **block['rad51_per_dose']))
        for strain, color in (('WT', S.WT_COLOR), ('rad51', S.RAD51_COLOR)):
            doses = [c.dose_gy for c in G.conditions(strain)]
            values = [per_dose[c.key] for c in G.conditions(strain)]
            label = (f'{S.STRAIN_LABEL[strain]}, '
                     + ('$k=100$' if key == 'published_rate' else 'calibrated'))
            ax_dose.plot(doses, values, style, color=color, linewidth=1.2,
                         marker='o' if key == 'transferred' else 's',
                         markersize=2.6, alpha=alpha,
                         markerfacecolor=color if key == 'transferred' else 'white',
                         label=label)
    pp.set_axis_labels(ax_dose, xlabel='Dose (Gy)',
                       ylabel='Mean absolute error')
    ax_dose.set_ylim(bottom=0)
    S.tidy(ax_dose)
    S.panel_letter(ax_dose, 'B')
    ax_dose.legend(frameon=False, fontsize=5.5, loc='center right')

    return S.save(fig, SUPP_OUT, 'gamma_error_decomposition')


# ---------------------------------------------------------------------------
# Gamma against proton at matched dose
# ---------------------------------------------------------------------------

def gamma_vs_proton():
    """The two radiation qualities compared at the same nominal dose.

    (a) The measured aB response. The gamma series is shifted later than the
        proton series at both extremes of dose, by more than the two doses differ
        from each other within either quality -- which is why the gamma fit needs
        its own generation time (205.62 min against 198) and its own inoculum
        offset, and why transferring only the dye chemistry is the meaningful
        transfer to attempt.
    (b) The simulated healthy-cell count. Here the qualities separate sharply, and
        in a way specific to the mutant: under protons both strains lose under 10%
        of the population across 0-30 Gy, whereas under gamma at the calibrated
        rate rad51-delta falls to about a third of the control while the wild type
        stays above 85%. Gamma damage is spatially uniform, so it reaches cells
        that ion tracks miss entirely, and a repair-deficient strain has no way to
        absorb it. This is the dose response the proton simulations were too
        compressed to supply, and it is why the gamma comparison is worth
        reporting rather than only the proton one.
    """
    fig, axes = pp.subplots(1, 2, axes_size=S.PANEL_MEDIUM, wspace=11)
    ax_ab, ax_pop = np.ravel(axes)

    # Only the two extreme doses. All six on one axes is twelve curves whose
    # dose ordering is the subject of Figs. S1-S2, not of this panel; the
    # comparison here is between radiation qualities, and the extremes bracket it.
    SHOWN = (0.0, 30.0)
    proton = M.load_experimental('WT')
    gamma = G.load_experimental('WT')
    colors = S.dose_colors(len(SHOWN))
    for dose, color in zip(SHOWN, colors):
        p = proton[f'WT_{dose:g}']
        g = gamma[f'gWT_{dose:g}']
        ax_ab.plot(p.time, p.blue, '-', color=color, linewidth=1.4,
                   label=f'{dose:g} Gy, proton')
        ax_ab.plot(g.time, g.blue, '--', color=color, linewidth=1.4,
                   label=f'{dose:g} Gy, gamma')
    # Short y-labels on both panels: a label long enough to run past the top of
    # the axes collides with the panel letter, which sits above it.
    S.concentration_axes(ax_ab, M.TRUNCATE_HOURS, xlabel='Time (h)',
                         ylabel='Measured blue fraction')
    S.panel_letter(ax_ab, 'A')
    ax_ab.legend(frameon=False, fontsize=5.5, loc='lower left')

    # -- simulated population, both qualities, both strains -----------------
    for strain, color in (('WT', S.WT_COLOR), ('rad51', S.RAD51_COLOR)):
        proton_curves = M.load_growth_curves(strain)
        gamma_curves = G.load_growth_curves(strain)
        control = proton_curves[f'{strain}_0'].healthy[-1]
        ax_pop.plot(G.DOSES,
                    [proton_curves[f'{strain}_{d:g}'].healthy[-1] / control
                     for d in G.DOSES], '-o', color=color, markersize=2.6,
                    linewidth=1.2, label=f'{S.STRAIN_LABEL[strain]}, proton')
        ax_pop.plot(G.DOSES,
                    [gamma_curves[f'g{strain}_{d:g}'].healthy[-1] / control
                     for d in G.DOSES], '--s', color=color, markersize=2.6,
                    linewidth=1.2, markerfacecolor='white',
                    label=f'{S.STRAIN_LABEL[strain]}, gamma')
    pp.set_axis_labels(ax_pop, xlabel='Dose (Gy)',
                       ylabel='Simulated healthy cells / 0 Gy')
    S.tidy(ax_pop)
    S.panel_letter(ax_pop, 'B')
    ax_pop.legend(frameon=False, fontsize=5.5, loc='lower left')

    return S.save(fig, SUPP_OUT, 'gamma_vs_proton')


def main():
    S.init()
    written = []
    for strain in ('WT', 'rad51'):
        errors, paths = prediction_panels(strain)
        written += paths
        print(f'{S.STRAIN_LABEL[strain]}: mean MAE = '
              f'{np.mean(list(errors.values())):.4f}')
    written += calibration_figure()
    written += error_decomposition()
    written += gamma_vs_proton()
    for path in written:
        print('wrote', os.path.relpath(path, os.path.dirname(HERE)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
