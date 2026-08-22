"""
Rebuild every non-gamma supplementary figure from its source data, through
publiplots.

The submitted supplement's figures were made by several independent scripts over
several years, in different fonts, at different marker sizes, with axis labels in
different conventions, and in three cases from data that no longer existed
anywhere in the repository. Since the reviewer's objection is about how carefully
the model was matched to the measurements, every figure carrying that claim is
regenerated here from the archived simulation output and the archived
experimental workbooks, in one style, by one script.

    Botas, J. (2025). PubliPlots: Publication-ready plotting for Python.
    https://github.com/jorgebotas/publiplots

FIGURES WRITTEN, and what changed from the submitted version

    ab_dose_trends_WT             S1. Predictions across all six doses on shared
    ab_dose_trends_rad51              axes, blue and pink side by side. Replaces
                                      BlueCurveaBWTpredictionsall.png and
                                      pinkcurveswtpredictionall.png, which were
                                      two separate single-species files for the
                                      wild type only; the mutant now gets the
                                      same treatment.
    ab_predicted_vs_measured_WT   S2 and S3, one figure per strain. Predicted
    ab_predicted_vs_measured_rad51    against measured on matched axes, with the
                                      between-dose spread quantified on the figure
                                      rather than only in the caption. Split into
                                      two figures because the two strain blocks
                                      together overfull a page.
    diffusion_propagator          S4. Analytic; regenerated from Eq. S_prop
                                      rather than from the undated original.
    damage_distribution           S5. Regenerated from results/bulk_aB/. The
                                      submitted version was a histogram over
                                      "four different simulations"; this one uses
                                      all 380 archived runs, shows the pooled
                                      distribution per strain, and separates the
                                      two damage states, which is what makes the
                                      wild-type / mutant asymmetry visible.
    ros_optimization              S6. Re-measured (benchmark_ros_assignment.py).
                                      The submitted figure's source timings do
                                      not survive and its unoptimized curve was
                                      partly polynomial extrapolation.
    smac_convergence              S7. Regenerated from results/smac3_output/
                                      runhistory.json.

    bo_vs_grid_search             NOT in the supplement any more. It compared
                                      manual grid search against SMAC3 Bayesian
                                      optimization on the legacy objective (sum of
                                      squared residuals over the first eight
                                      timepoints), so neither arm produced a
                                      number the manuscript now reports; the
                                      figure invited the reader to compare it with
                                      errors it is not on the same scale as. The
                                      function is kept because the comparison is
                                      still recomputable and the numbers appear in
                                      supplementary_figure_numbers.json.

Figures NOT regenerated here, and why: the gamma figures are built by
make_gamma_figures.py; the two AMMPER rendering screenshots (AMMPERgamma,
AMMPERiontrack) and the gamma geometry sketch (Gammaradiationm) are
visualizations of the simulator rather than analyses of data, and are kept as
submitted.

    python3 make_supplementary_figures.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import publiplots as pp
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, ttest_rel

import ammper_ab_model_fixed as M
import figure_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import ammper_paths as P  # noqa: E402

SUPP_OUT, _MAIN_OUT, _BUG_OUT = S.output_dirs()

BENCHMARK_JSON = os.path.join(HERE, 'ros_assignment_benchmark.json')

#: Health flags in the simulator output. Read off cellDefinition: cellRad and
#: cellROS set health = 2 for a cell with unrepaired strand breaks that is still
#: alive, and health = 3 for a dead one. The distinction matters for the damage
#: distribution figure, because the two strains populate different states:
#: cellROS_rad51 sets health = 3 directly, under the comment "any form of damage
#: results in death to rad51 cells", so a rad51-delta run contains no state-2
#: cells at all. The submitted figure pooled these and so could not show that.
HEALTHY, DAMAGED, DEAD = 1, 2, 3


# ---------------------------------------------------------------------------
# S1 / S2 / S3: the alamarBlue dose response
# ---------------------------------------------------------------------------

def _predicted_and_measured(strain):
    """Replicate-mean predictions and measurements for every dose of a strain."""
    per_run = M.load_growth_curves_per_run(strain)
    experiment = M.load_experimental(strain)
    params, g0 = M.params_for(strain)
    out = []
    for cond in M.conditions(strain):
        curves = [M.predict(c, params, g0=g0) for c in per_run[cond.key]]
        out.append((cond,
                    M.Prediction(curves[0].time,
                                 np.mean([c.blue for c in curves], axis=0),
                                 np.mean([c.pink for c in curves], axis=0)),
                    experiment[cond.key]))
    return out


def dose_trends(strain):
    """Predictions for all six doses on shared axes, one panel per species.

    This is the figure that answers the reviewer's question of whether the
    predictions distinguish the doses at all: in the submitted version they were
    nearly superimposed, because a reversed growth curve starts every dose from
    the same saturated population. Both species are shown because the dose
    ordering reverses between them, and a reader can check that reversal here
    without cross-referencing two separate files as the submitted supplement
    required.
    """
    series = _predicted_and_measured(strain)
    colors = S.dose_colors(len(series))

    fig, axes = pp.subplots(1, 2, axes_size=S.PANEL_MEDIUM, sharey=True,
                            wspace=10)
    ax_blue, ax_pink = np.ravel(axes)

    spreads = {}
    for ax, species, letter, name in ((ax_blue, 'blue', 'A', 'Blue (oxidized)'),
                                      (ax_pink, 'pink', 'B', 'Pink (reduced)')):
        stack = []
        for (cond, prediction, _measurement), color in zip(series, colors):
            values = getattr(prediction, species)
            ax.plot(prediction.time, values, '-', linewidth=1.5, color=color,
                    label=f'{cond.dose_gy:g} Gy')
            stack.append(values)
        # Peak-to-peak across the six doses at the timepoint where they are
        # furthest apart: the single number that says whether the predictions
        # resolve dose at all.
        spreads[species] = float(np.ptp(np.stack(stack), axis=0).max())
        S.concentration_axes(ax, M.TRUNCATE_HOURS, xlabel='Time (h)',
                             ylabel='Concentration fraction' if letter == 'A'
                             else None)
        pp.set_axis_labels(ax, title=name)
        S.panel_letter(ax, letter)
        # Blue falls from 1 and pink rises from 0, so the corner that stays empty
        # is not the same one in the two panels: a pink label in the lower left
        # sits on the rising curve's origin.
        # Blue falls from 1 and pink rises from 0, so the corner left empty is not
        # the same one in the two panels.
        S.inset_label(ax, f'between-dose spread\n{spreads[species]:.3f}',
                      loc='lower left' if species == 'blue' else 'upper right')

    # Dose legend in the pink panel's upper left. Pink stays below 0.4 for both
    # strains, so that corner is free in either; the blue panel's free corner
    # depends on how far the strain's curve has fallen by 15 h, and the
    # rad51-delta curve is still above 0.25 there.
    ax_pink.legend(frameon=False, fontsize=5.5, ncol=2, loc='upper left')
    pp.suptitle(f'Predicted aB response across dose, {S.STRAIN_LABEL[strain]}')
    return spreads, S.save(fig, SUPP_OUT, f'ab_dose_trends_{strain}')


def predicted_vs_measured(strain):
    """Predictions above, measurements below, on matched axes.

    The compression of the predicted dose separation relative to the measured one
    is the substantive limitation reported in the main text, and it is a
    statement about two numbers -- the predicted spread and the measured spread --
    so both are computed here and printed on the figure. Putting them on matched
    axes rather than in a caption is the difference between a reader being able to
    check the claim and having to take it.
    """
    series = _predicted_and_measured(strain)
    colors = S.dose_colors(len(series))

    fig, axes = pp.subplots(2, 2, axes_size=S.PANEL_SMALL, sharex=True,
                            sharey=True, hspace=8, wspace=8)
    grid = np.reshape(axes, (2, 2))

    spreads = {}
    for column, (species, name) in enumerate((('blue', 'Blue (oxidized)'),
                                              ('pink', 'Pink (reduced)'))):
        predicted, measured = [], []
        for (cond, prediction, measurement), color in zip(series, colors):
            grid[0, column].plot(prediction.time, getattr(prediction, species),
                                 '-', linewidth=1.5, color=color,
                                 label=f'{cond.dose_gy:g} Gy')
            grid[1, column].plot(measurement.time, getattr(measurement, species),
                                 'o-', markersize=2.2, linewidth=1.1,
                                 color=color)
            predicted.append(getattr(prediction, species))
            measured.append(getattr(measurement, species))
        spreads[species] = {
            'predicted': float(np.ptp(np.stack(predicted), axis=0).max()),
            'experimental': float(np.ptp(np.stack(measured), axis=0).max())}
        ratio = spreads[species]['experimental'] / spreads[species]['predicted']
        pp.set_axis_labels(grid[0, column], title=f'{name}, predicted')
        pp.set_axis_labels(grid[1, column], title=f'{name}, measured')
        # The two species run in opposite directions -- blue falls from 1, pink
        # rises from 0 -- so the empty corner is not the same one in both.
        S.inset_label(grid[1, column],
                      f"spread {spreads[species]['experimental']:.3f}\n"
                      f"vs {spreads[species]['predicted']:.3f} predicted\n"
                      f'({ratio:.1f}$\\times$)',
                      loc='lower left' if species == 'blue' else 'upper left')

    for ax in np.ravel(grid):
        S.concentration_axes(ax, M.TRUNCATE_HOURS)
    for ax in grid[1, :]:
        pp.set_axis_labels(ax, xlabel='Time (h)')
    for ax in grid[:, 0]:
        pp.set_axis_labels(ax, ylabel='Concentration fraction')
    S.panel_letter(grid[0, 0], 'A')
    S.panel_letter(grid[0, 1], 'B')

    # The dose legend goes in the predicted-pink panel, upper left: pink rises
    # from zero in both strains, so that corner is empty for both, whereas the
    # blue panel's free corner depends on how far the strain's blue curve falls
    # within the window (the rad51-delta curve is still above 0.25 at 15 h and
    # runs straight through a legend placed there).
    grid[0, 1].legend(frameon=False, fontsize=5.5, ncol=2, loc='upper left')
    pp.suptitle(f'Predicted against measured dose response, '
                f'{S.STRAIN_LABEL[strain]}')
    return spreads, S.save(fig, SUPP_OUT,
                           f'ab_predicted_vs_measured_{strain}')


# ---------------------------------------------------------------------------
# S4: the diffusion propagator
# ---------------------------------------------------------------------------

def diffusion_propagator():
    """Equation S_prop at three times, plus the width scaling it implies.

    Analytic, so it is regenerated exactly rather than recovered. The submitted
    figure showed three unlabeled profiles; the second panel here adds the
    quantity the figure exists to support -- that the width grows as sqrt(4Dt),
    which is what sets how far ROS spread from a deposition event between
    generations, and therefore how far from a track a cell can be and still be
    damaged.
    """
    fig, axes = pp.subplots(1, 3, axes_size=S.PANEL_MEDIUM, wspace=12)
    ax_profile, ax_log, ax_width = np.ravel(axes)

    # D = 1 in the arbitrary units of the original figure; only the shape and the
    # sqrt(t) scaling are being illustrated, and both are independent of D.
    D = 1.0
    r = np.linspace(-6, 6, 601)
    times = (0.2, 1.0, 5.0)
    colors = S.dose_colors(len(times), cmap='plasma')
    for t, color in zip(times, colors):
        # 3D propagator, evaluated along a radial line for visualization.
        profile = (4 * np.pi * D * t) ** (-1.5) * np.exp(-r**2 / (4 * D * t))
        ax_profile.plot(r, profile, '-', linewidth=1.5, color=color,
                        label=f'$t = {t:g}$')
        # The same three curves on a log density axis. On the linear axis the
        # t = 5 profile is flat against zero, because the peak height falls as
        # t^(-3/2) -- a factor of 125 between the first and last time shown -- so
        # the linear panel alone cannot show the broadening the figure exists to
        # illustrate. Both are kept: the linear panel is the honest picture of
        # relative magnitude, the log panel the readable picture of shape.
        ax_log.plot(r, profile, '-', linewidth=1.5, color=color,
                    label=f'$t = {t:g}$')
    pp.set_axis_labels(ax_profile, xlabel='Radial distance $r$ ($\\mu$m)',
                       ylabel='Probability density')
    S.tidy(ax_profile)
    S.panel_letter(ax_profile, 'A')
    ax_profile.legend(frameon=False, fontsize=6, loc='upper right')

    ax_log.set_yscale('log')
    ax_log.set_ylim(1e-6, 1.0)
    pp.set_axis_labels(ax_log, xlabel='Radial distance $r$ ($\\mu$m)',
                       ylabel='Probability density (log)')
    S.tidy(ax_log)
    S.panel_letter(ax_log, 'B')
    ax_log.legend(frameon=False, fontsize=6, loc='lower center')

    grid = np.linspace(0.02, 6, 300)
    ax_width.plot(grid, np.sqrt(4 * D * grid), '-', linewidth=1.5, color='0.2')
    for t, color in zip(times, colors):
        ax_width.plot([t], [np.sqrt(4 * D * t)], 'o', markersize=4, color=color,
                      zorder=4)
    # The AMMPER lattice spacing is the resolution limit on this: spreading
    # narrower than one lattice unit cannot be represented at all, which is the
    # implementation constraint the section discusses.
    ax_width.axhline(1.0, color=S.PUBLISHED, linewidth=1.0, linestyle='--',
                     label='AMMPER lattice unit')
    pp.set_axis_labels(ax_width, xlabel='Time $t$',
                       ylabel='Width $\\sqrt{4Dt}$ ($\\mu$m)')
    S.tidy(ax_width)
    S.panel_letter(ax_width, 'C')
    ax_width.legend(frameon=False, fontsize=6, loc='lower right')

    pp.suptitle('Diffusion propagator used for ROS spreading')
    return S.save(fig, SUPP_OUT, 'diffusion_propagator')


# ---------------------------------------------------------------------------
# S5: the damage distribution
# ---------------------------------------------------------------------------

def _damage_ratios():
    """Per-run damaged- and dead-to-healthy ratios, over every archived run.

    Walks results/bulk_aB/ directly rather than going through the growth-curve
    loader, because the loader collapses the two damage states into one
    "unhealthy" series and the point of this figure is that the two strains
    populate different states.
    """
    out = {}
    for cond in M.CONDITIONS:
        damaged, dead = [], []
        for path in sorted(glob.glob(os.path.join(
                P.bulk_aB(cond.sim_folder), '*', '*Gy.txt'))):
            frame = pd.read_csv(path,
                               names=['Generation', 'x', 'y', 'z', 'Health'])
            counts = frame['Health'].value_counts()
            healthy = max(counts.get(float(HEALTHY), 0), 1)
            damaged.append(counts.get(float(DAMAGED), 0) / healthy)
            dead.append(counts.get(float(DEAD), 0) / healthy)
        out[cond.key] = {'damaged': np.array(damaged, dtype=float),
                         'dead': np.array(dead, dtype=float),
                         'dose_gy': cond.dose_gy, 'strain': cond.strain}
    return out


def damage_distribution():
    """The non-parametric damage distribution, over all archived runs.

    (a) Per-run damage ratio against dose, every run plotted, with the fraction
        of runs registering no damage at all annotated. The distribution is
        zero-inflated and strongly non-normal, which is the justification for the
        rank-based tests used in the main text; showing every run rather than a
        histogram of four of them makes the zero inflation legible.
    (b) The distribution itself, pooled over dose: a histogram of the per-run
        damage ratio, one series per strain. Panel (a) shows every run against
        dose, which is where the dose trend is legible, but the shape of the
        distribution has to be read off the density of the cloud. The histogram
        states it directly -- a spike in the first bin and a short right tail --
        which is what the main text calls zero-inflated and non-parametric, and is
        the justification for the rank-based tests.
    (c) The two damage states separately. The wild type accumulates state-2
        cells (damaged, alive) and never a state-3 cell in any of 206 runs, while
        rad51-delta accumulates state-3 cells (dead) and never a state-2 cell in
        any of 174. That is not a data feature: cellROS_rad51 assigns death
        directly for any damage, so the mutant has no repairable-damage state by
        construction. It is worth showing because it is exactly the binary
        health-state limitation the Discussion identifies, made quantitative.
    """
    ratios = _damage_ratios()

    fig, axes = pp.subplots(1, 3, axes_size=S.PANEL_MEDIUM, wspace=14)
    ax_scatter, ax_hist, ax_states = np.ravel(axes)

    rng = np.random.default_rng(20260804)
    for strain, color, shift in (('WT', S.WT_COLOR, -0.62),
                                 ('rad51', S.RAD51_COLOR, +0.62)):
        for cond in M.conditions(strain):
            block = ratios[cond.key]
            # Either strain has damage in exactly one of the two states, so the
            # sum is the total damage ratio without double counting.
            values = block['damaged'] + block['dead']
            # Horizontal jitter and a per-strain shift, so the two strains'
            # clouds sit side by side instead of overlapping. Both are horizontal
            # only: the y values are the measurement and must not be perturbed.
            # The shift plus jitter stays well inside the 2.5 Gy dose spacing.
            jitter = rng.uniform(-0.5, 0.5, size=len(values))
            ax_scatter.scatter(cond.dose_gy + shift + jitter, values, s=4,
                               color=color, alpha=0.45, linewidth=0, zorder=3)
        means = [ratios[c.key]['damaged'].mean() + ratios[c.key]['dead'].mean()
                 for c in M.conditions(strain)]
        ax_scatter.plot([c.dose_gy for c in M.conditions(strain)], means, '-',
                        color=color, linewidth=1.4, marker='D', markersize=3.2,
                        markeredgecolor='white', markeredgewidth=0.4,
                        label=S.STRAIN_LABEL[strain], zorder=4)
    pp.set_axis_labels(ax_scatter, xlabel='Dose (Gy)',
                       ylabel='Damaged cells / healthy cells')
    ax_scatter.set_ylim(bottom=-0.003)
    S.tidy(ax_scatter)
    S.panel_letter(ax_scatter, 'A')
    ax_scatter.legend(frameon=False, fontsize=6, loc='upper left')

    total_runs = sum(len(b['damaged']) for b in ratios.values())
    zero_runs = sum(int(np.sum((b['damaged'] + b['dead']) == 0))
                    for b in ratios.values())
    S.inset_label(ax_scatter,
                  f'{zero_runs}/{total_runs} runs\nregister no damage',
                  loc='lower right')

    # -- (b) the pooled distribution ----------------------------------------
    # The zero-damage runs get a bin of their OWN, separated from the nonzero
    # ratios by a visible gap, rather than sharing the first bin of a uniform
    # grid with them. With uniform bins of width max/18 = 0.0051 the first bin
    # holds 95 wild type runs where only 74 are actually zero, so the bar a
    # reader sees does not equal the fraction the annotation states (46% against
    # 36%). That mismatch is the reason this panel read as inconsistent with its
    # own legend. Splitting the bin makes the annotated number the height of the
    # bar it annotates.
    pooled = np.concatenate([b['damaged'] + b['dead'] for b in ratios.values()])
    nonzero = pooled[pooled > 0]
    edges = np.linspace(float(nonzero.min()), float(pooled.max()), 15)
    zero_width = (edges[1] - edges[0]) * 0.8
    fractions = {}
    counts = {}
    for index, (strain, color) in enumerate((('WT', S.WT_COLOR),
                                             ('rad51', S.RAD51_COLOR))):
        values = np.concatenate([ratios[c.key]['damaged'] + ratios[c.key]['dead']
                                 for c in M.conditions(strain)])
        ax_hist.hist(values[values > 0], bins=edges, color=color, alpha=0.55,
                     edgecolor=color, linewidth=0.7,
                     label=S.STRAIN_LABEL[strain], zorder=3)
        # The zero bar, drawn to the left of the nonzero axis at half width so
        # the two strains sit side by side instead of occluding each other.
        n_zero = int(np.sum(values == 0))
        ax_hist.bar(-1.6 * zero_width + index * 0.5 * zero_width, n_zero,
                    width=0.5 * zero_width, color=color, alpha=0.55,
                    edgecolor=color, linewidth=0.7, align='edge', zorder=3)
        fractions[strain] = float(np.mean(values == 0))
        counts[strain] = {'n_runs': int(len(values)), 'zero': n_zero}
    # Mark the break between the zero bar and the nonzero histogram, so the gap
    # is read as a deliberate separation rather than an empty first bin.
    ax_hist.axvline(-0.35 * zero_width, color='0.6', linewidth=0.7,
                    linestyle=':', zorder=2)
    pp.set_axis_labels(ax_hist, xlabel='Damaged cells / healthy cells',
                       ylabel='Number of simulations')
    # Headroom above the zero bar, which is several times taller than any other,
    # so that the legend and the annotation clear it. The zero bars carry the
    # counts as direct labels rather than in an inset: an inset large enough to
    # hold both strains' counts collides with the strain legend in a panel this
    # width, and a number written on the bar it describes needs no cross-
    # reference anyway.
    ax_hist.set_ylim(0, ax_hist.get_ylim()[1] * 1.30)
    for index, strain in enumerate(('WT', 'rad51')):
        block = counts[strain]
        ax_hist.annotate(f"{block['zero']}/{block['n_runs']}\n"
                         f"({fractions[strain]:.0%})",
                         xy=(-1.35 * zero_width + index * 0.5 * zero_width,
                             block['zero']),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=5, color='0.35',
                         zorder=4)
    ax_hist.annotate('no\ndamage', xy=(-1.1 * zero_width, 0),
                     xytext=(0, -10), textcoords='offset points',
                     ha='center', va='top', fontsize=5, color='0.45',
                     annotation_clip=False)
    S.tidy(ax_hist)
    S.panel_letter(ax_hist, 'B')
    ax_hist.legend(frameon=False, fontsize=6, loc='upper right')

    # -- (c) the two damage states, per strain ------------------------------
    width = 0.9
    bar_heights = []
    for offset, (state, name) in ((-0.5 * width, ('damaged', 'Damaged, alive')),
                                  (+0.5 * width, ('dead', 'Dead'))):
        for index, strain in enumerate(('WT', 'rad51')):
            values = [ratios[c.key][state].mean()
                      for c in M.conditions(strain)]
            color = S.WT_COLOR if strain == 'WT' else S.RAD51_COLOR
            bar_heights.append(float(np.mean(values)))
            ax_states.bar(index * 2.4 + offset, np.mean(values), width=width,
                          color=color, alpha=0.5 if state == 'dead' else 1.0,
                          edgecolor=color, linewidth=0.8,
                          hatch='///' if state == 'dead' else None, zorder=3)
    # Headroom for the legend: without it the tallest bar runs to the top of the
    # axes and the legend sits on top of it.
    ax_states.set_ylim(0, max(bar_heights) * 1.45)
    # Each strain has exactly one nonzero bar -- the other state is empty in
    # every run -- so a bare bar chart shows two bars and a legend describing
    # four, which reads as a missing series. Label the absent bar explicitly at
    # its own x position, so "zero in all runs" is drawn rather than inferred
    # from blank space. This is the point of the panel.
    for index, (strain, empty_state) in enumerate((('WT', 'dead'),
                                                   ('rad51', 'damaged'))):
        offset = +0.5 * width if empty_state == 'dead' else -0.5 * width
        ax_states.annotate('0 in all\nruns', xy=(index * 2.4 + offset, 0),
                           xytext=(0, 4), textcoords='offset points',
                           ha='center', va='bottom', fontsize=5,
                           color='0.45', zorder=4)
    ax_states.set_xticks([0, 2.4])
    ax_states.set_xticklabels([S.STRAIN_LABEL['WT'], S.STRAIN_LABEL['rad51']],
                              fontsize=6.5)
    pp.set_axis_labels(ax_states,
                       ylabel='Mean ratio to healthy cells,\naveraged over dose')
    S.tidy(ax_states)
    S.panel_letter(ax_states, 'C')
    # Legend keyed on the fill style rather than the color, since color already
    # carries strain here: solid = damaged-but-alive, hatched = dead.
    ax_states.legend(handles=[
        Line2D([0], [0], marker='s', color='none', markerfacecolor='0.35',
               markeredgecolor='0.35', markersize=5,
               label='Damaged, alive (health 2)'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='none',
               markeredgecolor='0.35', markeredgewidth=1.0, markersize=5,
               label='Dead (health 3), hatched')],
        frameon=False, fontsize=5.5, loc='upper center')

    pp.suptitle('Distribution of radiation damage across simulations')
    written = S.save(fig, SUPP_OUT, 'damage_distribution')
    summary = {
        'total_runs': total_runs, 'zero_damage_runs': zero_runs,
        'zero_fraction': fractions,
        'per_condition': {key: {'n_runs': len(block['damaged']),
                                'zero_runs': int(np.sum(
                                    (block['damaged'] + block['dead']) == 0)),
                                'mean_damaged': float(block['damaged'].mean()),
                                'mean_dead': float(block['dead'].mean()),
                                'max_total': float((block['damaged']
                                                    + block['dead']).max())}
                          for key, block in ratios.items()}}
    return summary, written


# ---------------------------------------------------------------------------
# S6: the run-time optimization, re-measured
# ---------------------------------------------------------------------------

def ros_optimization():
    """Measured run time of the two damage-assignment implementations.

    Replaces the submitted run-time figure, whose source timings do not survive
    and whose unoptimized curve was partly a polynomial extrapolation. What is
    plotted here is measured over the whole range shown; the naive arm simply
    stops where it was no longer run, and the figure says so rather than
    continuing the line.

    The measured result is more specific, and less flattering, than the submitted
    claim of "minutes to seconds": the vectorized implementation is slower below
    about a thousand ROS points, because a vectorized filter pays a fixed
    per-call cost that an interpreted loop over a few points does not. It wins by
    a factor that grows linearly above that. This is the regime that matters,
    since the diffusion ROS model emits 82 lattice points per deposition event
    against one for the naive model, so enabling diffusion moves a simulation
    from the left of the crossover to well right of it.
    """
    if not os.path.exists(BENCHMARK_JSON):
        raise SystemExit('ros_assignment_benchmark.json missing -- run '
                         'benchmark_ros_assignment.py first')
    with open(BENCHMARK_JSON) as handle:
        report = json.load(handle)
    frame = pd.DataFrame(report['rows'])

    fig, axes = pp.subplots(1, 2, axes_size=S.PANEL_MEDIUM, wspace=12)
    ax_time, ax_speedup = np.ravel(axes)

    counts = sorted(frame['n_cells'].unique())
    colors = S.dose_colors(len(counts), cmap='cividis')
    for n_cells, color in zip(counts, colors):
        subset = frame[frame['n_cells'] == n_cells].sort_values('n_ros')
        measured = subset.dropna(subset=['naive_seconds'])
        ax_time.plot(measured['n_ros'], measured['naive_seconds'], 's--',
                     color=color, markersize=3, linewidth=1.2,
                     markerfacecolor='white',
                     label=f'{n_cells} cells, unoptimized')
        ax_time.plot(subset['n_ros'], subset['filtered_seconds'], 'o-',
                     color=color, markersize=3, linewidth=1.2,
                     label=f'{n_cells} cells, optimized')
    # Panel B was three curves, one per cell count, and they overlapped almost
    # exactly -- which is the result, not a plotting problem: the speedup is set
    # by the number of ROS points and is independent of population size, because
    # the vectorized arm does one pass per cell either way. Three
    # indistinguishable curves say that badly. Plot the mean across cell counts
    # with the observed range as a band, and state the width of the band.
    ratios = (frame.dropna(subset=['naive_seconds'])
              .assign(ratio=lambda f: f['naive_seconds'] / f['filtered_seconds'])
              .pivot_table(index='n_ros', columns='n_cells', values='ratio'))
    speedup_spread = float((ratios.max(axis=1) / ratios.min(axis=1)).max())
    ax_speedup.fill_between(ratios.index, ratios.min(axis=1), ratios.max(axis=1),
                            color=colors[len(colors) // 2], alpha=0.30,
                            linewidth=0, zorder=2,
                            label=f'range over {counts[0]} to {counts[-1]} cells')
    ax_speedup.plot(ratios.index, ratios.mean(axis=1), 'o-',
                    color=colors[len(colors) // 2], markersize=3.2,
                    linewidth=1.4, markeredgecolor='white',
                    markeredgewidth=0.4, zorder=3,
                    label='mean over cell counts')

    limit = report['naive_limit']
    ax_time.axvline(limit, color='0.55', linewidth=0.9, linestyle=':')
    # Said on the figure, not only in the caption: the submitted version
    # continued its unoptimized curve past the last measurement by fitting a
    # polynomial, and the distinction between measured and extrapolated is the
    # whole reason this figure was rebuilt.
    ax_time.annotate('unoptimized arm\nnot run beyond here', (limit, 0.02),
                     textcoords='offset points', xytext=(-4, 0), ha='right',
                     fontsize=5.5, color='0.35')
    ax_time.set_xscale('log')
    ax_time.set_yscale('log')
    pp.set_axis_labels(ax_time, xlabel='ROS lattice points',
                       ylabel='Run time (s)')
    S.tidy(ax_time)
    S.panel_letter(ax_time, 'A')
    # Six entries in two columns need a decade of headroom above the tallest
    # curve, or the legend sits on top of the 1024-cell lines.
    low, high = ax_time.get_ylim()
    ax_time.set_ylim(low, high * 8)
    ax_time.legend(frameon=False, fontsize=5, ncol=2, loc='upper left')

    ax_speedup.axhline(1.0, color='0.55', linewidth=0.9, linestyle='--')
    ax_speedup.annotate('equal cost', (frame['n_ros'].min(), 1.0),
                        textcoords='offset points', xytext=(2, 3), fontsize=5.5,
                        color='0.35')
    ax_speedup.set_xscale('log')
    ax_speedup.set_yscale('log')
    # Short label: one long enough to run past the top of the axes collides with
    # the panel letter, which sits above it.
    pp.set_axis_labels(ax_speedup, xlabel='ROS lattice points',
                       ylabel='Speedup, unoptimized/optimized')
    S.tidy(ax_speedup)
    S.panel_letter(ax_speedup, 'B')
    ax_speedup.legend(frameon=False, fontsize=5.5, loc='upper left')
    S.inset_label(ax_speedup,
                  f'Cell count changes the speedup\nby at most '
                  f'{(speedup_spread - 1) * 100:.0f}%',
                  loc='lower right')

    pp.suptitle('Run time of the ROS damage assignment, measured')
    written = S.save(fig, SUPP_OUT, 'ros_optimization')

    # The crossover is the number the caption quotes, so derive it here rather
    # than reading it off the plot: the smallest measured size at which the
    # optimized arm is already faster, taken over all cell counts.
    measured = frame.dropna(subset=['naive_seconds'])
    faster = measured[measured['naive_seconds'] > measured['filtered_seconds']]
    crossover = int(faster['n_ros'].min()) if len(faster) else None
    best = measured.assign(
        ratio=measured['naive_seconds'] / measured['filtered_seconds'])
    summary = {'crossover_ros_points': crossover,
               'max_speedup': float(best['ratio'].max()),
               'speedup_spread_over_cell_counts': speedup_spread,
               'max_speedup_at': int(best.loc[best['ratio'].idxmax(), 'n_ros']),
               'machine': report['machine']}
    return summary, written


# ---------------------------------------------------------------------------
# S7: SMAC convergence
# ---------------------------------------------------------------------------

def smac_convergence():
    """Best-so-far loss against trial, for every archived SMAC3 run.

    Regenerated from results/smac3_output/*/0/runhistory.json, whose ``data``
    field is a list of per-trial records with the cost at index 4. The submitted
    figure plotted the loss per iteration for a single run; plotting the running
    minimum for all six archived runs is the quantity that actually supports the
    caption's claim about where the loss converges, and it shows that two of the
    six never returned a finite cost at all.
    """
    runs = []
    for path in sorted(glob.glob(os.path.join(
            P.SMAC3_OUTPUT, '*', '0', 'runhistory.json'))):
        with open(path) as handle:
            history = json.load(handle)
        costs = np.array([row[4] for row in history['data']], dtype=float)
        runs.append({'run_id': os.path.basename(
                         os.path.dirname(os.path.dirname(path)))[:8],
                     'costs': costs,
                     'finite': bool(np.isfinite(costs).any())})
    if not runs:
        raise SystemExit('no SMAC3 runhistory found under results/smac3_output')

    fig, axes = pp.subplots(1, 2, axes_size=S.PANEL_MEDIUM, wspace=12)
    ax_trace, ax_best = np.ravel(axes)

    finite = [r for r in runs if r['finite']]
    colors = S.dose_colors(len(finite), cmap='viridis')
    # Label runs by their trial budget rather than by the SMAC3 output directory
    # hash: the hash is an implementation detail that means nothing to a reader,
    # whereas the budget is the variable the panel is about. Where two runs share
    # a budget they are distinguished by a suffix, since it matters that the same
    # budget was run more than once and reached a different loss.
    _budget_counts = {}
    for run in finite:
        _budget_counts[len(run['costs'])] = \
            _budget_counts.get(len(run['costs']), 0) + 1
    _seen = {}
    for run in finite:
        budget = len(run['costs'])
        _seen[budget] = _seen.get(budget, 0) + 1
        run['label'] = (f'{budget} trials'
                        + (f' (run {_seen[budget]})'
                           if _budget_counts[budget] > 1 else ''))
    for run, color in zip(sorted(finite, key=lambda r: -len(r['costs'])),
                          colors):
        costs = run['costs']
        trials = np.arange(1, len(costs) + 1)
        # Running minimum: what the optimizer would return if stopped here. The
        # raw per-trial cost is the sampling behavior, not the convergence, and
        # plotting it for six runs at once is unreadable.
        best = np.minimum.accumulate(np.where(np.isfinite(costs), costs, np.inf))
        ax_trace.plot(trials, best, '-', color=color, linewidth=1.4,
                      label=run['label'])
        ax_trace.plot(trials[np.isfinite(costs)], costs[np.isfinite(costs)], '.',
                      color=color, markersize=1.8, alpha=0.4)
    ax_trace.set_xscale('log')
    pp.set_axis_labels(ax_trace, xlabel='Trial',
                       ylabel='Loss (best so far)')
    S.tidy(ax_trace)
    S.panel_letter(ax_trace, 'A')
    ax_trace.legend(frameon=False, fontsize=5, loc='upper right')

    # -- (b) final loss against budget --------------------------------------
    # Finite runs first, so that the axis limits are set by real values before
    # the non-converging runs are placed relative to them.
    for run in runs:
        if run['finite']:
            ax_best.plot([len(run['costs'])],
                         [np.min(run['costs'][np.isfinite(run['costs'])])],
                         'o', color=S.CORRECTED, markersize=4.5,
                         markeredgecolor='white', markeredgewidth=0.5, zorder=4)
    # Runs that never returned a finite cost are shown, not dropped: two of the
    # six is a result about the optimizer's budget, and omitting them would make
    # the trend look cleaner than it is. They are drawn as open triangles in a
    # band above every finite value, and the axis is extended to hold them, so
    # that their height cannot be misread as a loss.
    low, high = ax_best.get_ylim()
    band = high + 0.16 * (high - low)
    # Both non-converging runs used the same budget, so without a nudge they
    # would draw exactly on top of each other and read as one run. The offset is
    # multiplicative because the axis is logarithmic.
    nudges = (0.93, 1.07)
    non_finite = [r for r in runs if not r['finite']]
    for run, nudge in zip(non_finite, nudges * len(non_finite)):
        ax_best.plot([len(run['costs']) * nudge], [band], '^', color='0.5',
                     markersize=4.5, markerfacecolor='white',
                     clip_on=False, zorder=4)
    ax_best.set_ylim(low, high + 0.30 * (high - low))
    ax_best.axhline(high + 0.06 * (high - low), color='0.75', linewidth=0.7,
                    linestyle=':')
    ax_best.set_xscale('log')
    pp.set_axis_labels(ax_best, xlabel='Trial budget',
                       ylabel='Best loss attained')
    S.tidy(ax_best)
    S.panel_letter(ax_best, 'B')
    ax_best.legend(handles=[
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=S.CORRECTED, markeredgecolor=S.CORRECTED,
               markersize=4.5, label='converged'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='white',
               markeredgecolor='0.5', markersize=4.5,
               label='no finite loss found')],
        frameon=False, fontsize=5.5, loc='upper right')

    pp.suptitle('SMAC3 Bayesian optimization convergence')
    written = S.save(fig, SUPP_OUT, 'smac_convergence')
    summary = {'runs': [{'run_id': r['run_id'], 'n_trials': len(r['costs']),
                         'best': (float(np.min(r['costs'][
                             np.isfinite(r['costs'])])) if r['finite'] else None)}
                        for r in runs]}
    return summary, written


# ---------------------------------------------------------------------------
# Bayesian optimization against grid search -- no longer a supplementary figure
# ---------------------------------------------------------------------------

#: Per-dose wild-type errors from the archived comparison, verbatim from
#: analysis/aB/erroranalysis.py. These are the errors of the *legacy* kinetic
#: model under the two parameter-search strategies, retained unchanged because
#: the claim they support is about the two search strategies relative to each
#: other, not about the absolute quality of that model. The corrected model's
#: errors are an order of magnitude smaller and are reported separately; putting
#: the two on one axes would invite reading the search comparison as a model
#: comparison.
GRID_SEARCH_ERRORS = [0.6373249026708974, 0.6955181692388223,
                      0.5471709792283441, 0.5957490859964685,
                      0.5725566036371199, 0.4696130438287181]
BO_ERRORS = [0.576673099854368, 0.6262511595988866, 0.4820317214608146,
             0.5312940527466247, 0.5057455993190662, 0.410623842402046]
BO_DOSES = [0.0, 2.5, 5.0, 10.0, 20.0, 30.0]


def bo_vs_grid_search():
    """Bayesian optimization against manual grid search, per dose.

    (a) The two error series against dose, paired by dose.
    (b) The paired differences, with the two tests reported in the main text
        annotated on the panel. Bayesian optimization is better at every one of
        the six doses, which is what makes the paired test the appropriate one
        and is more informative than the mean difference alone.
    """
    grid = np.array(GRID_SEARCH_ERRORS)
    bo = np.array(BO_ERRORS)
    doses = np.array(BO_DOSES)

    _t_statistic, t_p = ttest_rel(grid, bo)
    _u_statistic, u_p = mannwhitneyu(grid, bo, alternative='two-sided')

    fig, axes = pp.subplots(1, 2, axes_size=S.PANEL_MEDIUM, wspace=12)
    ax_series, ax_delta = np.ravel(axes)

    ax_series.plot(doses, grid, 's--', color=S.PUBLISHED, markersize=3.4,
                   linewidth=1.3, markerfacecolor='white',
                   label='Manual grid search')
    ax_series.plot(doses, bo, 'o-', color=S.CORRECTED, markersize=3.4,
                   linewidth=1.3, label='Bayesian optimization')
    # Paired-by-dose connectors, so the reader sees that the comparison is within
    # dose rather than between two independent samples.
    for dose, a, b in zip(doses, grid, bo):
        ax_series.plot([dose, dose], [a, b], '-', color='0.6', linewidth=0.6,
                       zorder=1)
    pp.set_axis_labels(ax_series, xlabel='Dose (Gy)',
                       ylabel='Mean absolute error')
    S.tidy(ax_series)
    S.panel_letter(ax_series, 'A')
    ax_series.legend(frameon=False, fontsize=6, loc='upper right')

    delta = grid - bo
    ax_delta.bar(doses, delta, width=1.9, color=S.CORRECTED, alpha=0.85,
                 edgecolor=S.CORRECTED, linewidth=0.8, zorder=3)
    ax_delta.axhline(0, color='0.3', linewidth=0.9)
    ax_delta.axhline(delta.mean(), color=S.PUBLISHED, linewidth=1.0,
                     linestyle='--', label=f'mean {delta.mean():.3f}')
    pp.set_axis_labels(ax_delta, xlabel='Dose (Gy)',
                       ylabel='Error reduction (grid $-$ BO)')
    # Headroom for the statistics box: the bars are all within a factor of 1.2 of
    # each other, so without it the annotation has nowhere to sit that is not
    # over a bar.
    ax_delta.set_ylim(0, delta.max() * 1.75)
    S.tidy(ax_delta)
    S.panel_letter(ax_delta, 'B')
    S.inset_label(ax_delta,
                  f'paired $t$: $p = {t_p:.2e}$\n'
                  f'Mann-Whitney $U$: $p = {u_p:.3f}$\n'
                  f'BO better at {int((delta > 0).sum())}/{len(delta)} doses',
                  loc='upper left')
    ax_delta.legend(frameon=False, fontsize=6, loc='upper right')

    pp.suptitle('Parameter search strategy, wild type')
    written = S.save(fig, SUPP_OUT, 'bo_vs_grid_search')
    summary = {'grid_mean': float(grid.mean()), 'bo_mean': float(bo.mean()),
               'mean_reduction': float(delta.mean()),
               'ttest_p': float(t_p), 'mannwhitney_p': float(u_p),
               'bo_better_at': int((delta > 0).sum())}
    return summary, written


def main():
    S.init()
    written, numbers = [], {}

    print('S1: dose trends')
    for strain in ('WT', 'rad51'):
        spreads, paths = dose_trends(strain)
        written += paths
        numbers[f'dose_trend_spread_{strain}'] = spreads
        print(f'  {strain:<6s} between-dose spread '
              + '  '.join(f'{k} {v:.4f}' for k, v in spreads.items()))

    print('S2 and S3: predicted against measured')
    for strain in ('WT', 'rad51'):
        spreads, paths = predicted_vs_measured(strain)
        written += paths
        numbers[f'spread_{strain}'] = spreads
        for species, values in spreads.items():
            print(f'  {strain:<6s} {species:<5s} predicted '
                  f'{values["predicted"]:.4f}  measured '
                  f'{values["experimental"]:.4f}  ratio '
                  f'{values["experimental"] / values["predicted"]:.1f}x')

    print('S4: diffusion propagator')
    written += diffusion_propagator()

    print('S5: damage distribution')
    damage, paths = damage_distribution()
    written += paths
    numbers['damage'] = damage
    print(f"  {damage['zero_damage_runs']}/{damage['total_runs']} runs "
          f'register no damage')

    print('S6: run-time optimization')
    timing, paths = ros_optimization()
    written += paths
    numbers['ros_timing'] = timing
    print(f"  crossover at {timing['crossover_ros_points']} ROS points; "
          f"max speedup {timing['max_speedup']:.1f}x at "
          f"{timing['max_speedup_at']} points")

    print('S7: SMAC convergence')
    smac, paths = smac_convergence()
    written += paths
    numbers['smac'] = smac
    for run in smac['runs']:
        print(f"  {run['run_id']}  n={run['n_trials']:<4d} best="
              + ('none' if run['best'] is None else f"{run['best']:.5f}"))

    # Not a supplementary figure any more (see the module docstring); still run so
    # that supplementary_figure_numbers.json keeps a record of the comparison.
    print('BO against grid search (not in the supplement)')
    search, paths = bo_vs_grid_search()
    written += paths
    numbers['search'] = search
    print(f"  grid {search['grid_mean']:.4f} vs BO {search['bo_mean']:.4f}; "
          f"paired t p={search['ttest_p']:.3e}, "
          f"Mann-Whitney p={search['mannwhitney_p']:.3f}")

    out = os.path.join(HERE, 'supplementary_figure_numbers.json')
    with open(out, 'w') as handle:
        json.dump(numbers, handle, indent=2)
    print(f'\nwrote {out}')
    for path in written:
        print('wrote', os.path.relpath(path, os.path.dirname(HERE)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
