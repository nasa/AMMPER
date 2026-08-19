"""
Regenerate main text Figure 2 and supplementary Figure S1 from the corrected model.

Outputs drop-in replacements for the files currently in the manuscript's
Panels/ and Figures/ directories, at the same sizes and in the same formats, so
they can be substituted without touching the LaTeX geometry:

  figures/main_figure_panels/
      WT_panel_all_doses.{png,pdf}          Figure 2B  (intermediate)
      rad51_panel_all_doses.{png,pdf}       Figure 2C  (intermediate)
      comprehensive_alamarblue_stacked.{png,pdf,svg,tiff}   <<< Figure 2
  figures/supplementary_material/
      BlueCurveaBWTpredictionsall.png       Figure S1A
      pinkcurveswtpredictionall.png         Figure S1B
      aB_predictions_only_WT.png            Figure S1 new panels: predictions
      aB_predictions_only_rad51.png           alone, overplotted across doses
  figures/bug_illustration/
      aB_published_vs_corrected_WT.png      before/after the ordering fix; not a
                                            manuscript figure, see
                                            notes/BUGS_AND_CORRECTIONS.md

Panel geometry, colors, marker styles, fonts and legend placement are carried
over unchanged from analysis/aB/ab_final_plots_panel.py and
analysis/aB/stack_ab_figures.py. What changes is the numbers going in, plus the
two corrected dose labels (Bug 3).
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

import ammper_ab_model_fixed as M

HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = os.path.dirname(HERE)
MAIN_OUT = os.path.join(REVISION, 'figures', 'main_figure_panels')
SUPP_OUT = os.path.join(REVISION, 'figures', 'supplementary_material')
#: The before/after panel documents the growth-curve ordering bug. It is not a
#: manuscript figure -- the manuscript reports only the corrected results -- so it
#: is written outside the supplementary directory, which holds only figures the
#: supplement actually includes. See notes/BUGS_AND_CORRECTIONS.md.
BUG_OUT = os.path.join(REVISION, 'figures', 'bug_illustration')
for d in (MAIN_OUT, SUPP_OUT, BUG_OUT):
    os.makedirs(d, exist_ok=True)

#: Panel A of Figure 2 is the alamarBlue reduction scheme -- molecular structures
#: for resazurin, resorufin and dihydroresorufin with the rate constants on the
#: arrows -- unaffected by the bugs. This is the version in the current
#: submission, and it is what the Figure 2 caption describes. An older cell
#: cartoon (results/figures_updated_figures_branch/aBcartoon.PNG) was used at one
#: point and does not match the caption; do not substitute it back.
#:
#: The scheme exists only as a raster. It was recovered at 403x503 from the
#: base64 payload embedded in the submitted comprehensive_alamarblue_stacked.svg
#: (matplotlib stores such images bottom-up, so it is flipped on extraction) and
#: is committed under code/assets/ so this script does not depend on the zip. If
#: the original vector artwork turns up, drop it in as code/assets/ and point
#: PANEL_A at it -- 403x503 is the resolution limit of Figure 2A as it stands.
PANEL_A = os.path.join(HERE, 'assets', 'aB_chem_scheme.png')

COLOR_BLUE = '#2E5EAA'
COLOR_PINK = '#D64161'


# ---------------------------------------------------------------------------
# Figure 2B / 2C: six per-dose panels per strain
# ---------------------------------------------------------------------------

def dose_panel(ax, cond, growth_runs, measurement, bands, params, g0):
    """One dose panel: experimental points with error bars + predicted curves."""
    predictions = [M.predict(curve, params, g0=g0) for curve in growth_runs]
    time = predictions[0].time
    blue = np.stack([p.blue for p in predictions])
    pink = np.stack([p.pink for p in predictions])
    # Standard error over replicate simulations, as in the published figure.
    blue_mean, blue_sem = blue.mean(0), blue.std(0) / np.sqrt(len(predictions))
    pink_mean, pink_sem = pink.mean(0), pink.std(0) / np.sqrt(len(predictions))

    upper_blue, upper_pink = bands['upper']
    lower_blue, lower_pink = bands['lower']
    ax.errorbar(measurement.time, measurement.blue,
                yerr=[np.abs(measurement.blue - lower_blue),
                      np.abs(measurement.blue - upper_blue)],
                fmt='o', markersize=5, capsize=3, linewidth=1.5,
                color=COLOR_BLUE, ecolor=COLOR_BLUE, alpha=0.7)
    ax.errorbar(measurement.time, measurement.pink,
                yerr=[np.abs(measurement.pink - lower_pink),
                      np.abs(measurement.pink - upper_pink)],
                fmt='s', markersize=5, capsize=3, linewidth=1.5,
                color=COLOR_PINK, ecolor=COLOR_PINK, alpha=0.7)

    ax.plot(time, blue_mean, '-', linewidth=2.5, color=COLOR_BLUE, alpha=0.9)
    ax.fill_between(time, blue_mean - blue_sem, blue_mean + blue_sem,
                    color=COLOR_BLUE, alpha=0.15)
    ax.plot(time, pink_mean, '-', linewidth=2.5, color=COLOR_PINK, alpha=0.9)
    ax.fill_between(time, pink_mean - pink_sem, pink_mean + pink_sem,
                    color=COLOR_PINK, alpha=0.15)

    ax.set_xlim([0, M.TRUNCATE_HOURS])
    ax.set_ylim([-0.05, 1.05])
    # Dose label comes from the Condition record -- the fix for Bug 3.
    label = f'{cond.dose_gy:g} Gy'
    ax.text(0.05, 0.95, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='gray'))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return M.mean_absolute_error(M.Prediction(time, blue_mean, pink_mean),
                                 measurement)


LEGEND = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_BLUE,
           markersize=8, label='Experimental (Blue)', markeredgecolor=COLOR_BLUE),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_PINK,
           markersize=8, label='Experimental (Pink)', markeredgecolor=COLOR_PINK),
    Line2D([0], [0], color=COLOR_BLUE, linewidth=2.5, label='Predicted (Blue)'),
    Line2D([0], [0], color=COLOR_PINK, linewidth=2.5, label='Predicted (Pink)'),
]


def strain_panel(strain, title, outfile, with_legend):
    per_run = M.load_growth_curves_per_run(strain)
    experiment = M.load_experimental(strain, with_error=True)
    params, g0 = M.params_for(strain)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    errors = {}
    for index, cond in enumerate(M.conditions(strain)):
        ax = axes[index // 3, index % 3]
        measurement, bands = experiment[cond.key]
        errors[cond.key] = dose_panel(ax, cond, per_run[cond.key],
                                      measurement, bands, params, g0)

    for ax in axes[1, :]:
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    for ax in axes[:, 0]:
        ax.set_ylabel('Concentration Fraction', fontsize=12, fontweight='bold')
    if with_legend:
        fig.legend(handles=LEGEND, loc='upper center', bbox_to_anchor=(0.5, 0.01),
                   ncol=4, frameon=True, fontsize=11, edgecolor='gray')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(MAIN_OUT, f'{outfile}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    return errors


# ---------------------------------------------------------------------------
# Figure 2: the stacked composite
# ---------------------------------------------------------------------------

def stacked_figure():
    """Assemble schematic + WT + rad51 into the main text figure."""
    fig = plt.figure(figsize=(16, 22))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1.5, 1.5], hspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    images = [PANEL_A,
              os.path.join(MAIN_OUT, 'WT_panel_all_doses.png'),
              os.path.join(MAIN_OUT, 'rad51_panel_all_doses.png')]
    for ax, image in zip(axes, images):
        ax.imshow(mpimg.imread(image))
        ax.axis('off')

    # No title over panel A: the scheme is self-labeling and the submitted figure
    # has none. B and C carry their own titles from the panel scripts.

    box = dict(boxstyle='round', facecolor='white', alpha=0.8,
               edgecolor='black', linewidth=2)
    for ax, letter, x, y in ((axes[0], 'A', 0.01, 1.00),
                             (axes[1], 'B', 0.01, 1.00),
                             (axes[2], 'C', -0.02, 0.98)):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=20,
                fontweight='bold', va='top', ha='left', bbox=box)

    plt.tight_layout()
    base = os.path.join(MAIN_OUT, 'comprehensive_alamarblue_stacked')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{base}.{ext}', bbox_inches='tight',
                    **({'dpi': 300} if ext != 'svg' else {}))
    # ASM asks for TIFF at submission; the published Panels/ directory has one.
    fig.savefig(f'{base}.tiff', dpi=300, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure S1: predictions overplotted across doses
# ---------------------------------------------------------------------------
# The reviewer asked whether the predicted curves differ between doses at all.
# In the submitted figure they were nearly indistinguishable -- a consequence of
# Bug 1, since a reversed growth curve starts every dose at the same saturated
# population. These panels show the predictions alone, on shared axes, so the
# dose ordering can be read directly.

def _dose_colors(n):
    return plt.cm.viridis(np.linspace(0, 0.9, n))


def predictions_only(strain, outfile):
    per_run = M.load_growth_curves_per_run(strain)
    experiment = M.load_experimental(strain)
    params, g0 = M.params_for(strain)
    conds = M.conditions(strain)
    colors = _dose_colors(len(conds))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    strain_label = 'Wild type' if strain == 'WT' else r'$rad51\Delta$'
    fig.suptitle(f'{strain_label}: dose dependence of the predicted and measured '
                 f'aB response', fontsize=15, fontweight='bold')

    spreads = {}
    for column, (species, name) in enumerate((('blue', 'Blue (oxidized)'),
                                              ('pink', 'Pink (reduced)'))):
        predicted, measured = [], []
        for cond, color in zip(conds, colors):
            curves = [M.predict(c, params, g0=g0) for c in per_run[cond.key]]
            mean = np.mean([getattr(c, species) for c in curves], axis=0)
            axes[0, column].plot(curves[0].time, mean, '-', linewidth=2.2,
                                 color=color, label=f'{cond.dose_gy:g} Gy')
            series = getattr(experiment[cond.key], species)
            axes[1, column].plot(experiment[cond.key].time, series, 'o-',
                                 markersize=4, linewidth=1.6, color=color,
                                 label=f'{cond.dose_gy:g} Gy')
            predicted.append(mean)
            measured.append(series)

        # Between-dose spread: how far apart the six curves get, at the time of
        # maximum separation. This is the quantity the reviewer asked for.
        spreads[species] = {
            'predicted': float(np.ptp(np.stack(predicted), axis=0).max()),
            'experimental': float(np.ptp(np.stack(measured), axis=0).max()),
        }

        axes[0, column].set_title(f'{name} — AMMPER prediction', fontsize=12,
                                  fontweight='bold')
        axes[1, column].set_title(f'{name} — experiment', fontsize=12,
                                  fontweight='bold')

    for ax in axes.flat:
        ax.set_xlim([0, M.TRUNCATE_HOURS])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, ncol=2, frameon=True, edgecolor='gray')
    for ax in axes[1, :]:
        ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    for ax in axes[:, 0]:
        ax.set_ylabel('Concentration Fraction', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(SUPP_OUT, f'{outfile}.png'), dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(SUPP_OUT, f'{outfile}.pdf'), bbox_inches='tight')
    plt.close(fig)
    return spreads


def legacy_s1_panels():
    """Drop-in replacements for the two existing Figure S1 image files.

    S1A is the blue predictions across all WT doses, S1B the pink ones. The
    manuscript includes them under those filenames, so they keep their names.
    """
    per_run = M.load_growth_curves_per_run('WT')
    params, g0 = M.params_for('WT')
    conds = M.conditions('WT')
    colors = _dose_colors(len(conds))

    for species, name, outfile in (
            ('blue', 'Blue (oxidized) alamarBlue', 'BlueCurveaBWTpredictionsall'),
            ('pink', 'Pink (reduced) alamarBlue', 'pinkcurveswtpredictionall')):
        fig, ax = plt.subplots(figsize=(8, 6))
        for cond, color in zip(conds, colors):
            curves = [M.predict(c, params, g0=g0) for c in per_run[cond.key]]
            mean = np.mean([getattr(c, species) for c in curves], axis=0)
            ax.plot(curves[0].time, mean, '-', linewidth=2.4, color=color,
                    label=f'{cond.dose_gy:g} Gy')
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Concentration Fraction', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}: predictions, wild type, all doses',
                     fontsize=13, fontweight='bold')
        ax.set_xlim([0, M.TRUNCATE_HOURS])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10, frameon=True, edgecolor='gray')
        plt.tight_layout()
        fig.savefig(os.path.join(SUPP_OUT, f'{outfile}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close(fig)


def before_after():
    """Published vs corrected, side by side, for the 0 Gy wild type series.

    Written to figures/bug_illustration/ for the internal bug note, not to the
    supplement: the manuscript presents the corrected results directly rather
    than a before/after comparison. Left: the growth curve as the published code
    fed it to the ODE, with the published parameters. Right: the corrected model.
    """
    growth = M.load_growth_curves('WT')
    experiment = M.load_experimental('WT')
    curve = growth['WT_0']
    reversed_curve = M.GrowthCurve(curve.healthy[::-1].copy(),
                                   curve.unhealthy[::-1].copy(), curve.n_runs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    axes[0].plot(range(M.NGEN + 1), curve.healthy, 'o-', color='#1b7837',
                 linewidth=2.4, label='corrected (ascending)')
    axes[0].plot(range(M.NGEN + 1), reversed_curve.healthy, 's--', color='#762a83',
                 linewidth=2.4, label='as published (value_counts)')
    axes[0].set_xlabel('Generation', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Healthy cell count', fontsize=11, fontweight='bold')
    axes[0].set_title('Growth curve fed to the ODE', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    for ax, (cur, params, g0, title) in zip(axes[1:], (
            (reversed_curve, (0.7799990799515666, 1.679928455577914,
                              0.10002078747628415, 450.0, 6601.0, 9994.0, 0.5),
             0.0, 'As published'),
            (curve, M.WT_PARAMS, M.WT_G0, 'Corrected and refit'))):
        prediction = M.predict(cur, params, g0=g0)
        measurement = experiment['WT_0']
        ax.plot(measurement.time, measurement.blue, 'o', color=COLOR_BLUE,
                markersize=5, alpha=0.75, label='Experimental (Blue)')
        ax.plot(measurement.time, measurement.pink, 's', color=COLOR_PINK,
                markersize=5, alpha=0.75, label='Experimental (Pink)')
        ax.plot(prediction.time, prediction.blue, '-', color=COLOR_BLUE, linewidth=2.5)
        ax.plot(prediction.time, prediction.pink, '-', color=COLOR_PINK, linewidth=2.5)
        mae = M.mean_absolute_error(prediction, measurement)
        ax.set_title(f'{title} (MAE = {mae:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Concentration Fraction', fontsize=11, fontweight='bold')
        ax.set_xlim([0, M.TRUNCATE_HOURS])
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=9, loc='center right')

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Effect of the growth-curve ordering correction (wild type, 0 Gy)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(BUG_OUT, 'aB_published_vs_corrected_WT.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    plt.style.use('seaborn-v0_8-paper')

    print('Figure 2B/2C: per-dose panels')
    wt_errors = strain_panel(
        'WT', 'Alamarblue Assay: Wild Type Strain Response to Radiation',
        'WT_panel_all_doses', with_legend=False)
    rad_errors = strain_panel(
        'rad51', 'Alamarblue Assay: rad51Δ Strain Response to Radiation',
        'rad51_panel_all_doses', with_legend=True)
    for key, value in list(wt_errors.items()) + list(rad_errors.items()):
        print(f'    {key:<12s} MAE {value:.4f}')

    print('Figure 2: stacked composite')
    stacked_figure()

    print('Figure S1: dose-dependence panels')
    legacy_s1_panels()
    wt_spread = predictions_only('WT', 'aB_predictions_only_WT')
    rad_spread = predictions_only('rad51', 'aB_predictions_only_rad51')
    before_after()

    print('\nBetween-dose spread (max over time, concentration-fraction units)')
    for strain, spread in (('WT', wt_spread), ('rad51', rad_spread)):
        for species, values in spread.items():
            print(f'    {strain:<6s} {species:<5s} predicted {values["predicted"]:.4f}'
                  f'   experimental {values["experimental"]:.4f}')

    import json
    with open(os.path.join(HERE, 'figure_numbers.json'), 'w') as handle:
        json.dump({'wt_mae': wt_errors, 'rad51_mae': rad_errors,
                   'spread_wt': wt_spread, 'spread_rad51': rad_spread},
                  handle, indent=2)
    print(f'\nwrote {os.path.join(HERE, "figure_numbers.json")}')
    print(f'panels -> {MAIN_OUT}')
    print(f'supplement -> {SUPP_OUT}')
    print(f'bug note figure -> {BUG_OUT}')


if __name__ == '__main__':
    main()
