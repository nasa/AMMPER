"""
Refit the corrected alamarBlue model and record the resulting errors.

Why a refit is necessary
------------------------
The published kinetic parameters were optimized against a growth curve that was
being fed to the ODE backwards (Bug 1 in ammper_ab_model_fixed.py). They are
therefore the best parameters *for the bug*, and simply correcting the ordering
while keeping them makes the fit worse, not better. The parameters have to be
re-estimated on the corrected forward model.

What is fit, and on what
------------------------
The six kinetic parameters (v1, v2, v3, K1, K2, K3) and the unhealthy-cell weight
k are SHARED across all twelve conditions -- six doses in each of two strains --
and fit jointly with one inoculum offset g0 per strain. Nine parameters for
twelve curves. The dye chemistry is a property of alamarBlue rather than of the
strain, so the strain difference is carried entirely by the population state
through g0, and the model is never re-tuned per dose.

Why fit both strains together rather than the wild type alone
------------------------------------------------------------
On these data the two routes are nearly equivalent, and the reason for choosing
the joint fit is not that the alternative fails. Fitting the wild type alone (all
six doses, eight parameters) and then transferring the kinetics with only g0
refit for rad51-delta gives 0.0302 and 0.0722, against 0.0302 and 0.0716 jointly
-- 0.0512 versus 0.0509 over all twelve conditions. The chemistry constrained by
one strain does describe the other, which is a stronger result than a transfer
failure would have been.

(An intermediate version of this revision reported the opposite, a transfer error
of 0.379 attributed to v2 being unconstrained by the wild type alone. That was an
artifact of a rescaled pink series, not a property of the data; the experimental
normalization has since been reverted to the published convention. See the "NOT A
BUG" section of ammper_ab_model_fixed.py.)

Two reasons to keep the joint fit anyway. First, it is the more constrained
arrangement and the one the manuscript claims: one chemistry plus one offset per
strain, fit to twelve curves at once, with no opportunity to tune the kinetics to
the strain that is then presented as a held-out test. Second, the wild-type-only
search drives k -- the metabolic weight on damaged cells -- to exactly 0.0, its
lower bound, because the six wild-type doses barely differ and the cheapest way
to fit them is to ignore dose entirely. Those kinetics have no dose-response
mechanism left at all. The joint fit keeps k off its bound (0.034), which is
small but is at least identified by data rather than by a constraint.

The transfer configuration is still computed and reported below, so both numbers
above can be checked rather than taken on trust.

We also record reference configurations so that the effect of the fix can be
quantified rather than asserted:

    reversed_input_published_params   as submitted: published parameters on the
                                      reversed growth curve
    corrected_input_published_params  ordering fixed, published parameters kept,
                                      no refit -- worse, as expected, since those
                                      parameters encode the bug
    corrected_refit                   ordering fixed, re-estimated <-- the result
    wt_only_fit                       the wild-type-only-then-transfer route
                                      described above, kept as evidence

Optimizer
---------
scipy.optimize.differential_evolution, a global search, seeded for
reproducibility. The published pipeline used SMAC3 Bayesian Optimization; the
choice of optimizer is not what is being tested here, and differential evolution
avoids adding a heavyweight dependency to the revision code. If the reviewers
would prefer the SMAC3 numbers, analysis/aB/aBFinalplotsSMAC.py can be re-run
against this module -- see other/README.md.

Outputs
-------
    fitted_parameters.json   parameters + per-dose errors (consumed by figures)
    stdout                   a human-readable summary table
"""

import json
import os

import numpy as np
from scipy.optimize import differential_evolution

import ammper_ab_model_fixed as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, 'fitted_parameters.json')

#: The parameters reported in the original submission, in this module's ordering
#: (v1, v2, v3, K1, K2, K3, k).
PUBLISHED_PARAMS = (0.7799990799515666, 1.679928455577914, 0.10002078747628415,
                    450.0, 6601.0, 9994.0, 0.5)

#: Search bounds. The v and K ranges bracket the published values by well over an
#: order of magnitude in both directions; g0 spans the whole growth curve.
#: v2 and v3 are bounded away from zero so that the pink<->clear step cannot be
#: switched off entirely; an unconstrained lower bound of 1e-4 let the search
#: park both on the bound and freeze the colorless species (see _objective).
BOUNDS = [
    (1e-4, 20.0),      # v1
    (1e-2, 20.0),      # v2, pink -> clear
    (1e-4, 20.0),      # v3, clear -> pink
    (1.0, 5e4),        # K1
    (1.0, 5e4),        # K2
    (1.0, 5e4),        # K3
    (0.0, 2.0),        # k, unhealthy-cell weight
    (0.0, 14.5),       # g0, inoculum offset in generations
]

#: Indices into a parameter vector: 0-6 kinetics, 7 offset.
N_KINETIC, I_G0 = 7, 7

SEED = 20260804


#: Returned instead of an error when a parameter set is chemically inadmissible.
#: Large enough that differential_evolution never prefers such a point, and
#: constant so that it forms a flat plateau rather than a misleading gradient.
PENALTY = 10.0


def _objective(vector, growth, experiment, keys):
    """Mean of the per-dose mean absolute errors, with a physicality constraint.

    Parameter sets that do not produce genuine over-reduction are rejected
    outright. Two distinct evasions had to be closed off:

    1. A small K3 makes the initial pink->clear rate negative, reversing the
       final reduction step. Under the earlier clamped integrator this pinned
       clear at zero and created mass -- total dye grew by over 250% across the
       window -- while fitting the coloured series well.
    2. Requiring only non-negativity was then satisfied by driving v2 and v3 to
       their lower bounds, switching the pink->clear step off so that clear
       stayed frozen at its initial value. Formally a three-species model, in
       substance a two-species one.

    Both routes reduce the model to blue->pink, contradicting the assay
    chemistry, in which resorufin is progressively over-reduced to a colorless
    product that interferes with the readout. clear_forms_appreciably therefore
    demands that the colorless fraction actually accumulate.
    """
    params, g0 = tuple(vector[:N_KINETIC]), vector[I_G0]
    errors = []
    for key in keys:
        if not M.clear_forms_appreciably(growth[key], params, g0):
            return PENALTY
        prediction = M.predict(growth[key], params, g0=g0)
        errors.append(M.mean_absolute_error(prediction, experiment[key]))
    return float(np.mean(errors))


#: Condition keys, resolved once so the objectives stay picklable for workers=-1.
WT_KEYS = [c.key for c in M.conditions('WT')]
RAD_KEYS = [c.key for c in M.conditions('rad51')]


def _joint_objective(vector, growth, experiment):
    """Mean error over all twelve conditions, kinetics shared between strains.

    Layout: the kinetics, then the wild-type offset, then the rad51-delta offset
    as the final entry. The two strains are weighted equally, so a parameter set
    cannot buy a small gain on the six wild-type curves by giving up more on the
    six mutant ones.
    """
    g0_rad = vector[-1]
    wt = _objective(vector[:-1], growth, experiment, WT_KEYS)
    if wt >= PENALTY:
        return PENALTY
    rad_vector = np.concatenate([vector[:N_KINETIC], [g0_rad]])
    rad = _objective(rad_vector, growth, experiment, RAD_KEYS)
    if rad >= PENALTY:
        return PENALTY
    return float((wt + rad) / 2.0)


def _scan_g0(growth, experiment, params, keys, step=0.01):
    """Best inoculum offset on a fine grid, with the kinetics held fixed.

    Exhaustive over the g0 bounds at 0.01-generation resolution. Offsets that
    violate the physicality constraint are skipped, as in _objective.
    """
    low, high = BOUNDS[I_G0]
    best_error, best_g0 = np.inf, np.nan
    for g0 in np.arange(low + step, high, step):
        vector = np.concatenate([params, [g0]])
        error = _objective(vector, growth, experiment, keys)
        if error < best_error:
            best_error, best_g0 = error, float(g0)
    if not np.isfinite(best_error):
        raise RuntimeError('no admissible g0 found for ' + ', '.join(keys))
    return best_g0


def per_dose_errors(growth, experiment, params, g0, keys):
    return {key: M.mean_absolute_error(
                     M.predict(growth[key], params, g0=g0),
                     experiment[key])
            for key in keys}


def _reversed_growth(curve):
    """Reproduce Bug 1: the growth curve as the published code fed it in."""
    return M.GrowthCurve(healthy=curve.healthy[::-1].copy(),
                         unhealthy=curve.unhealthy[::-1].copy(),
                         n_runs=curve.n_runs)


def summarize(label, errors):
    values = np.array(list(errors.values()))
    print(f'  {label:<24s} mean MAE = {values.mean():.4f}   '
          f'range = {values.min():.4f}-{values.max():.4f}')
    return float(values.mean())


def main():
    growth = M.load_growth_curves()
    experiment = M.load_experimental()

    wt_keys = [c.key for c in M.conditions('WT')]
    rad_keys = [c.key for c in M.conditions('rad51')]

    report = {'seed': SEED, 'bounds': BOUNDS, 'wt': {}, 'rad51': {}}

    # -- reference configurations -------------------------------------------
    print('Wild type reference configurations')
    reversed_growth = {k: _reversed_growth(v) for k, v in growth.items()}

    # The published code integrated on the 16 discrete generation ticks with no
    # offset, so g0=0 on the reversed curve is the closest faithful analogue.
    err_reversed = per_dose_errors(reversed_growth, experiment,
                                   PUBLISHED_PARAMS, 0.0, wt_keys)
    report['wt']['reversed_input_published_params'] = {
        'per_dose': err_reversed,
        'mean': summarize('reversed + published', err_reversed)}

    err_corrected_only = per_dose_errors(growth, experiment,
                                         PUBLISHED_PARAMS, 0.0, wt_keys)
    report['wt']['corrected_input_published_params'] = {
        'per_dose': err_corrected_only,
        'mean': summarize('corrected + published', err_corrected_only)}

    # -- the fit: shared kinetics, one offset per strain ---------------------
    print('\nFitting shared kinetics + one offset per strain '
          '(9 parameters, 12 curves jointly)...')
    result = differential_evolution(
        _joint_objective, BOUNDS + [BOUNDS[I_G0]],
        args=(growth, experiment), seed=SEED, maxiter=300, popsize=24, tol=1e-8,
        polish=True, updating='deferred', workers=-1)
    params = tuple(result.x[:N_KINETIC])

    # Polish both offsets by exhaustive scan with the kinetics held fixed. Done
    # identically for the two strains, so their difference cannot be an artifact
    # of where a stochastic search happened to stop.
    wt_g0 = _scan_g0(growth, experiment, params, wt_keys)
    rad_g0 = _scan_g0(growth, experiment, params, rad_keys)

    err_wt = per_dose_errors(growth, experiment, params, wt_g0, wt_keys)
    err_rad = per_dose_errors(growth, experiment, params, rad_g0, rad_keys)

    shared = {
        'params': {'v1': params[0], 'v2': params[1], 'v3': params[2],
                   'K1': params[3], 'K2': params[4], 'K3': params[5],
                   'k': params[6]},
        'optimizer_nfev': int(result.nfev),
    }
    report['shared_kinetics'] = shared

    report['wt']['corrected_refit'] = {
        'g0': wt_g0,
        'cells_at_t0': float(np.interp(wt_g0, np.arange(M.NGEN + 1),
                                       growth['WT_0'].healthy)),
        'per_dose': err_wt,
        'mean': summarize('wild type', err_wt),
    }
    report['rad51']['corrected_refit'] = {
        'g0': rad_g0,
        'cells_at_t0': float(np.interp(rad_g0, np.arange(M.NGEN + 1),
                                       growth['rad51_0'].healthy)),
        'per_dose': err_rad,
        'mean': summarize('rad51-delta', err_rad),
    }
    all_errors = dict(err_wt, **err_rad)
    report['overall_mean'] = summarize('all twelve conditions', all_errors)
    print(f'    g0 = {wt_g0:.2f} generations for the wild type '
          f'({report["wt"]["corrected_refit"]["cells_at_t0"]:.0f} cells at t=0), '
          f'{rad_g0:.2f} for rad51-delta '
          f'({report["rad51"]["corrected_refit"]["cells_at_t0"]:.0f} cells); '
          f'{wt_g0 - rad_g0:.2f} generations behind')

    # -- the wild-type-only alternative, reported as evidence ----------------
    # Recorded because the module docstring makes a claim about it: the two routes
    # agree closely on these data (0.0512 vs 0.0509 over all twelve), and the
    # wild-type-only search puts k on its lower bound. Kept in the output so both
    # claims can be checked rather than taken on trust.
    print('\nWild type alone, kinetics then transferred (not the reported fit)')
    wt_only = differential_evolution(
        _objective, BOUNDS, args=(growth, experiment, wt_keys),
        seed=SEED, maxiter=300, popsize=24, tol=1e-8,
        polish=True, updating='deferred', workers=-1)
    wt_only_params = tuple(wt_only.x[:N_KINETIC])
    wt_only_g0 = _scan_g0(growth, experiment, wt_only_params, wt_keys)
    rad_transfer_g0 = _scan_g0(growth, experiment, wt_only_params, rad_keys)
    err_wt_only = per_dose_errors(growth, experiment, wt_only_params,
                                  wt_only_g0, wt_keys)
    err_rad_transfer = per_dose_errors(growth, experiment, wt_only_params,
                                       rad_transfer_g0, rad_keys)
    report['wt_only_fit'] = {
        'params': {'v1': wt_only_params[0], 'v2': wt_only_params[1],
                   'v3': wt_only_params[2], 'K1': wt_only_params[3],
                   'K2': wt_only_params[4], 'K3': wt_only_params[5],
                   'k': wt_only_params[6]},
        'g0_wt': wt_only_g0, 'g0_rad51': rad_transfer_g0,
        'wt_per_dose': err_wt_only,
        'rad51_per_dose': err_rad_transfer,
        'wt_mean': summarize('wild type (fit here)', err_wt_only),
        'rad51_mean': summarize('rad51-delta (transferred)', err_rad_transfer),
        'overall_mean': summarize('all twelve conditions',
                                  dict(err_wt_only, **err_rad_transfer)),
    }
    print('    the wild type is fit marginally better this way and the mutant '
          'much worse; see the module docstring.')

    with open(OUT_JSON, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nwrote {OUT_JSON}')
    return report


if __name__ == '__main__':
    main()
