"""
Produce every number the revised manuscript quotes, as one auditable table.

Run after fit_ab_model_fixed.py. Writes other/REVISION_NUMBERS.md plus a machine
readable revision_numbers.json, so that no value in the manuscript has to be
read off a figure by eye.

Sections:
  1. Mean absolute error, before and after the fix, per dose and overall
  2. Fitted parameters, with an identifiability check on K1 and K2
  3. Inoculum offset g0, per strain and per dose
  4. Between-dose spread: predicted vs experimental (the reviewer's question)
  5. Dose-response direction: correlation of each curve with dose
  6. Where the residual error lives: healthy vs unhealthy cell contributions
"""

import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

import ammper_ab_model_fixed as M
from fit_ab_model_fixed import PUBLISHED_PARAMS, _reversed_growth

HERE = os.path.dirname(os.path.abspath(__file__))
OTHER = os.path.join(os.path.dirname(HERE), 'other')
os.makedirs(OTHER, exist_ok=True)

TIME_OF_INTEREST = None  # computed per species: the time of maximum spread


def _mean_prediction(runs, params, g0):
    """Prediction averaged over replicate simulations, as plotted in Figure 2.

    The alternative -- predicting once from the averaged growth curve -- is what
    the fitting objective uses, and the two differ slightly because the dye
    kinetics are nonlinear in the cell count. The difference is below 0.0005 MAE
    in every condition, but it straddles a rounding boundary at 10 Gy, so the
    reported table and the figures must agree on one route. We report the route
    the figures plot, since that is the curve a reader can measure against.
    """
    predictions = [M.predict(run, params, g0) for run in runs]
    return M.Prediction(
        time=predictions[0].time,
        blue=np.mean([p.blue for p in predictions], axis=0),
        pink=np.mean([p.pink for p in predictions], axis=0))


def section_errors(growth, experiment):
    """MAE for each configuration, per dose."""
    out = {}
    for strain in ('WT', 'rad51'):
        keys = [c.key for c in M.conditions(strain)]
        params, g0 = M.params_for(strain)
        per_run = M.load_growth_curves_per_run(strain)
        reversed_runs = {k: [_reversed_growth(run) for run in per_run[k]]
                         for k in keys}
        rows = {}
        for key in keys:
            rows[key] = {
                'published': M.mean_absolute_error(
                    _mean_prediction(reversed_runs[key], PUBLISHED_PARAMS, 0.0),
                    experiment[key]),
                'corrected_published_params': M.mean_absolute_error(
                    _mean_prediction(per_run[key], PUBLISHED_PARAMS, 0.0),
                    experiment[key]),
                'corrected_refit': M.mean_absolute_error(
                    _mean_prediction(per_run[key], params, g0), experiment[key]),
            }
        out[strain] = rows
    return out


#: The three (v, K) pairs, by index into the parameter tuple.
VK_PAIRS = {'v1/K1': (0, 3), 'v2/K2': (1, 4), 'v3/K3': (2, 5)}


def section_identifiability(growth, experiment):
    """Scale each (v, K) pair together and see whether the fit notices.

    If a Michaelis-Menten term is operating far below saturation, then v and K
    enter only through their ratio, and scaling both by the same factor leaves the
    prediction unchanged -- so a flat row means that K is not identifiable from
    these data and its absolute magnitude should not be interpreted. Each pair is
    tested separately, since which of them are saturating is a property of the
    fitted values and not something to assume: under the current fit K1 and K3 are
    large relative to the substrate amounts while K2 is small.
    """
    keys = [c.key for c in M.conditions('WT')]
    out = {}
    for name, (v_index, k_index) in VK_PAIRS.items():
        rows = {}
        for scale in (1, 2, 10, 100):
            params = list(M.SHARED_PARAMS)
            params[v_index] *= scale
            params[k_index] *= scale
            rows[scale] = float(np.mean([
                M.mean_absolute_error(
                    M.predict(growth[k], tuple(params), M.WT_G0), experiment[k])
                for k in keys]))
        out[name] = rows
    return out


def section_offsets(growth, experiment):
    """Shared vs per-dose g0. A stable per-dose g0 means one value suffices."""
    out = {}
    for strain in ('WT', 'rad51'):
        params, shared = M.params_for(strain)
        rows = {}
        for cond in M.conditions(strain):
            best = minimize_scalar(
                lambda x: M.mean_absolute_error(
                    M.predict(growth[cond.key], params, x), experiment[cond.key]),
                bounds=(0.0, 14.5), method='bounded')
            rows[cond.key] = {'per_dose_g0': float(best.x),
                              'per_dose_mae': float(best.fun),
                              'shared_mae': M.mean_absolute_error(
                                  M.predict(growth[cond.key], params, shared),
                                  experiment[cond.key])}
        out[strain] = {'shared_g0': shared, 'per_dose': rows}
    return out


def section_dose_response(growth, experiment):
    """Spread across doses and its direction, predicted vs experimental.

    The predicted curve for a dose is the mean over that dose's replicate
    simulations, matching exactly what Figure S1 plots: predicting once from the
    averaged growth curve is not the same thing, because the dye kinetics are
    nonlinear in the cell count, and the two routes differ by about ten percent
    in the spread. The figure defines the quantity, so the figure's route wins.
    """
    out = {}
    for strain in ('WT', 'rad51'):
        params, g0 = M.params_for(strain)
        per_run = M.load_growth_curves_per_run(strain)
        conds = M.conditions(strain)
        doses = [c.dose_gy for c in conds]
        rows = {}
        predictions = {c.key: [M.predict(run, params, g0) for run in per_run[c.key]]
                       for c in conds}
        for species in ('blue', 'pink'):
            predicted = np.stack([
                np.mean([getattr(p, species) for p in predictions[c.key]], axis=0)
                for c in conds])
            measured = np.stack([getattr(experiment[c.key], species) for c in conds])
            # Evaluate at each series' own time of maximum between-dose separation.
            pi = int(np.argmax(np.ptp(predicted, axis=0)))
            mi = int(np.argmax(np.ptp(measured, axis=0)))
            time = M.predict(growth[conds[0].key], params, g0).time
            rows[species] = {
                'predicted_spread': float(np.ptp(predicted[:, pi])),
                'experimental_spread': float(np.ptp(measured[:, mi])),
                'predicted_at_hours': float(time[pi]),
                'experimental_at_hours': float(experiment[conds[0].key].time[mi]),
                'predicted_r_with_dose': float(np.corrcoef(doses, predicted[:, pi])[0, 1]),
                'experimental_r_with_dose': float(np.corrcoef(doses, measured[:, mi])[0, 1]),
                'predicted_values': [float(v) for v in predicted[:, pi]],
                'experimental_values': [float(v) for v in measured[:, mi]],
            }
            rows[species]['spread_ratio'] = (
                rows[species]['experimental_spread'] / rows[species]['predicted_spread'])
        out[strain] = {'doses': doses, 'species': rows}
    return out


def section_population(growth):
    """The dose signal available in the simulation, before any dye kinetics."""
    out = {}
    for strain in ('WT', 'rad51'):
        conds = M.conditions(strain)
        healthy = [float(growth[c.key].healthy[-1]) for c in conds]
        unhealthy = [float(growth[c.key].unhealthy[-1]) for c in conds]
        out[strain] = {
            'doses': [c.dose_gy for c in conds],
            'final_healthy': healthy,
            'final_unhealthy': unhealthy,
            'healthy_spread_percent': float(100 * np.ptp(healthy) / np.mean(healthy)),
            'healthy_r_with_dose': float(
                np.corrcoef([c.dose_gy for c in conds], healthy)[0, 1]),
            'replicates': {c.key: growth[c.key].n_runs for c in conds},
        }
    return out


def write_markdown(report):
    lines = ['# AMMPER-2 revision: every number, and where it comes from', '',
             'Generated by `code/generate_revision_numbers.py`. All values are',
             'mean absolute error (MAE) in concentration-fraction units unless',
             'stated otherwise.', '',
             '## 1. Mean absolute error', '',
             'Three configurations: `published` = the growth curve reversed as in the',
             'submitted code with the published parameters; `corr+pub` = ordering fixed',
             'but parameters unchanged; `corr+refit` = ordering fixed and refit.', '']

    for strain in ('WT', 'rad51'):
        label = 'Wild type' if strain == 'WT' else 'rad51-delta'
        lines += [f'### {label}', '',
                  '| Dose | published | corr+pub | corr+refit |',
                  '|---|---|---|---|']
        rows = report['errors'][strain]
        for cond in M.conditions(strain):
            r = rows[cond.key]
            lines.append(f'| {cond.dose_gy:g} Gy | {r["published"]:.4f} | '
                         f'{r["corrected_published_params"]:.4f} | '
                         f'**{r["corrected_refit"]:.4f}** |')
        means = {k: np.mean([r[k] for r in rows.values()])
                 for k in ('published', 'corrected_published_params', 'corrected_refit')}
        lines += [f'| **mean** | **{means["published"]:.4f}** | '
                  f'**{means["corrected_published_params"]:.4f}** | '
                  f'**{means["corrected_refit"]:.4f}** |', '']

    lines += ['Note that fixing the ordering while keeping the published parameters',
              'makes the fit *worse*. That is expected: those parameters were',
              'optimized against the reversed curve, so they encode the bug. The',
              'comparison that matters is `published` vs `corr+refit`.', '',
              '## 2. Fitted parameters (shared by both strains, 12 curves '
              'jointly)', '',
              '| Parameter | Value |', '|---|---|']
    names = ('v1', 'v2', 'v3', 'K1', 'K2', 'K3', 'k')
    for name, value in zip(names, M.SHARED_PARAMS):
        lines.append(f'| {name} | {value:.4g} |')
    lines += [f'| g0 (WT) | {M.WT_G0:.3f} generations |',
              f'| g0 (rad51) | {M.RAD51_G0:.3f} generations |', '',
              '### Identifiability of the half-saturation constants', '',
              'Each (v, K) pair is scaled together. Where a term operates far',
              'below saturation, v and K enter only through their ratio, so the',
              'row is flat and that K is not identifiable from these data.',
              'Which pairs behave this way is read off the table rather than',
              'assumed:', '',
              '| Scale | ' + ' | '.join(report['identifiability']) + ' |',
              '|---|' + '---|' * len(report['identifiability'])]
    scales = sorted(next(iter(report['identifiability'].values())),
                    key=lambda s: int(s))
    for scale in scales:
        row = [f'{report["identifiability"][pair][scale]:.4f}'
               for pair in report['identifiability']]
        lines.append(f'| {scale}x | ' + ' | '.join(row) + ' |')
    flat = [pair for pair, rows in report['identifiability'].items()
            if abs(rows[scales[-1]] - rows[scales[0]]) < 0.01]
    lines += ['', ('Flat to within 0.01 across a hundredfold scaling: '
                   + (', '.join(flat) if flat else 'none')
                   + '. The corresponding half-saturation constants should be'
                     ' reported as not individually identifiable from these'
                     ' data, not as measured quantities.'), '',
              '## 3. Inoculum offset g0', '',
              'One shared g0 per strain is used for the figures. Refitting g0',
              'separately per dose is shown for comparison only.', '']
    for strain in ('WT', 'rad51'):
        block = report['offsets'][strain]
        label = 'Wild type' if strain == 'WT' else 'rad51-delta'
        lines += [f'### {label} (shared g0 = {block["shared_g0"]:.2f})', '',
                  '| Dose | shared-g0 MAE | best per-dose g0 | per-dose MAE |',
                  '|---|---|---|---|']
        for cond in M.conditions(strain):
            r = block['per_dose'][cond.key]
            lines.append(f'| {cond.dose_gy:g} Gy | {r["shared_mae"]:.4f} | '
                         f'{r["per_dose_g0"]:.2f} | {r["per_dose_mae"]:.4f} |')
        lines.append('')
    lines += ['For the wild type the per-dose optimum varies over less than one',
              'generation, so a single shared value costs little. For rad51-delta it',
              'varies non-monotonically with dose, mirroring the non-monotonic',
              'experimental curves.', '',
              '## 4. Between-dose spread (the reviewer\'s question)', '',
              'Spread = the largest gap between the six dose curves, measured at the',
              'time of maximum separation.', '',
              '| Strain | Species | predicted | experimental | ratio |',
              '|---|---|---|---|---|']
    for strain in ('WT', 'rad51'):
        for species, values in report['dose_response'][strain]['species'].items():
            lines.append(f'| {strain} | {species} | {values["predicted_spread"]:.4f} | '
                         f'{values["experimental_spread"]:.4f} | '
                         f'{values["spread_ratio"]:.1f}x |')
    lines += ['', '## 5. Direction of the dose response', '',
              'Correlation between dose and the curve value at the time of maximum',
              'separation. Blue should rise with dose and pink should fall.', '',
              '| Strain | Species | predicted r | experimental r |',
              '|---|---|---|---|']
    for strain in ('WT', 'rad51'):
        for species, values in report['dose_response'][strain]['species'].items():
            lines.append(f'| {strain} | {species} | '
                         f'{values["predicted_r_with_dose"]:+.3f} | '
                         f'{values["experimental_r_with_dose"]:+.3f} |')
    lines += ['', 'The corrected model gets the *direction* right (and for',
              'rad51-delta it is perfectly monotonic), but the *magnitude* is',
              'several-fold too small. That is the honest remaining limitation.', '',
              '## 6. Dose signal available in the simulation', '',
              'Before any dye kinetics: how much the simulated population itself',
              'changes with dose. This bounds what the aB model can express.', '',
              '| Strain | final healthy counts (0 to 30 Gy) | spread | r with dose |',
              '|---|---|---|---|']
    for strain in ('WT', 'rad51'):
        block = report['population'][strain]
        counts = ', '.join(f'{v:.0f}' for v in block['final_healthy'])
        lines.append(f'| {strain} | {counts} | '
                     f'{block["healthy_spread_percent"]:.1f}% of mean | '
                     f'{block["healthy_r_with_dose"]:+.3f} |')
    lines += ['', 'The simulated healthy population varies only ~5-8% across a',
              '0-30 Gy range, and the fitted weight on unhealthy cells is small',
              f'(k = {M.WT_PARAMS[6]:.3f}). Note also that rad51-delta damaged cells are',
              'assigned health code 3 (nonviable) rather than 2 in',
              '`cellDefinition.py`, so that strain has *no* health-2 cells at any',
              'dose and its dose response travels almost entirely through the',
              'healthy-cell count. This, not the dye chemistry, is why the',
              'predicted dose separation is compressed.', '',
              '| Strain | Condition | replicate simulations |', '|---|---|---|']
    for strain in ('WT', 'rad51'):
        for key, n in report['population'][strain]['replicates'].items():
            lines.append(f'| {strain} | {key} | {n} |')
    lines.append('')
    return '\n'.join(lines)


def main():
    growth = M.load_growth_curves()
    experiment = M.load_experimental()

    report = {
        'errors': section_errors(growth, experiment),
        'identifiability': section_identifiability(growth, experiment),
        'offsets': section_offsets(growth, experiment),
        'dose_response': section_dose_response(growth, experiment),
        'population': section_population(growth),
        'fitted': {'wt_params': list(M.WT_PARAMS), 'wt_g0': M.WT_G0,
                   'rad51_g0': M.RAD51_G0},
    }

    with open(os.path.join(HERE, 'revision_numbers.json'), 'w') as handle:
        json.dump(report, handle, indent=2)
    markdown = write_markdown(report)
    with open(os.path.join(OTHER, 'REVISION_NUMBERS.md'), 'w') as handle:
        handle.write(markdown)

    print(markdown)
    print(f'\nwrote {os.path.join(OTHER, "REVISION_NUMBERS.md")}')


if __name__ == '__main__':
    main()
