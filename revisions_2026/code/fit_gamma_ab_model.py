"""
Fit the corrected alamarBlue model to the gamma radiation data, and separate the
two defects that made the submitted gamma comparison fail.

The submitted supplement reported that this model "failed to recapitulate the
experimental data" under gamma radiation and diagnosed the failure as a
limitation of the binary health-state assumption. This script tests that
diagnosis by walking the two candidate explanations one at a time. There are two
independent things wrong, in two different places, and they have to be undone in
order:

    submitted            published gamma kinetics on the reversed growth curve,
                         at the published hit rate: what Fig. S9 showed
    ordering_fixed       the same kinetics on the correctly ordered curve
    published_rate       proton kinetics transferred, offset refit, but still at
                         the published hit rate k = 100
    transferred          proton kinetics transferred, offset refit, at the
                         calibrated hit rate                <-- the reported result
    refit                kinetics re-estimated on the gamma data, calibrated rate

If the failure were a modeling gap, "transferred" would stay poor and only
"refit" -- or not even that -- would improve. It is not a modeling gap. The
ordering fix recovers part of it; the hit-rate calibration recovers the rest; and
then the proton dye chemistry describes the gamma data with nothing refit but one
inoculum offset per strain.

WHY "TRANSFERRED" IS THE HEADLINE. It holds the dye chemistry completely fixed.
Those constants were estimated from proton data only, so the gamma curves are a
genuine held-out test: twelve gamma curves predicted from two numbers plus a
population trajectory. "refit" is reported alongside to bound how much of the
residual is attributable to the chemistry rather than to AMMPER's population
dynamics -- if refitting buys almost nothing, the transferred constants are not
the limiting factor.

THE HIT RATE IS CALIBRATED ELSEWHERE, AND NOT ON THIS DATA. k is fit in
calibrate_gamma_hit_rate.py against the OD690 growth channel, which is a
turbidity measurement of cell density, not against the blue and pink series this
script predicts and scores. Letting the aB fit choose k would let dye chemistry
decide how many photons a Gray delivers. Because that calibration runs on
separate data, the gamma errors reported here are a held-out consequence of it.
The published-rate configuration is retained above as the evidence for why the
recalibration was necessary at all.

    python3 fit_gamma_ab_model.py

Outputs
-------
    fitted_parameters_gamma.json   parameters, offsets and per-dose errors
    stdout                         a human-readable summary
"""

import json
import os

import numpy as np
from scipy.optimize import differential_evolution

import ammper_ab_model_fixed as M
import ammper_ab_model_gamma as G
import fit_ab_model_fixed as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, 'fitted_parameters_gamma.json')
CALIBRATION_JSON = os.path.join(HERE, 'gamma_hit_rate_calibration.json')

#: Kinetics reported in the submitted gamma supplement, in the legacy gamma
#: script's own two-step scheme: [V1_max, V2_max, K1_M, K2_M, k] =
#: [0.75, 1.65, 500, 8000, 0.5]. That scheme ran blue -> pink -> colorless
#: irreversibly, so it has no v3/K3. Mapped onto this module's three-species
#: parameter vector by setting the reverse step to its smallest admissible value,
#: which is the closest faithful embedding: an irreversible forward step is a
#: reversible one whose reverse rate is negligible.
PUBLISHED_GAMMA_PARAMS = (0.75, 1.65, 1e-4, 500.0, 8000.0, 1.0, 0.5)

SEED = F.SEED

#: Same bounds as the proton fit, so that "refit on gamma" and "fit on protons"
#: are searches over the same space and their outcomes are comparable.
BOUNDS = F.BOUNDS
N_KINETIC, I_G0 = F.N_KINETIC, F.I_G0
PENALTY = F.PENALTY

WT_KEYS = [c.key for c in G.conditions('WT')]
RAD_KEYS = [c.key for c in G.conditions('rad51')]
ALL_KEYS = WT_KEYS + RAD_KEYS


def _objective(vector, growth, experiment, keys, require_admissible=True):
    """Mean per-dose MAE with the physicality constraint, on the gamma clock.

    Mirrors fit_ab_model_fixed._objective; it cannot simply call it because the
    gamma comparison uses a different generation time, which enters predict().

    ``require_admissible=False`` drops the constraint that the colorless species
    genuinely accumulates. That is needed only to put a number on the
    published-hit-rate configuration: there the simulated population is extinct
    over most of the dose range, so no offset produces appreciable
    over-reduction, and refusing to score it would leave the comparison with a
    hole exactly where the diagnosis lives. Nothing reported as a result is scored
    with the constraint off.
    """
    params, g0 = tuple(vector[:N_KINETIC]), vector[I_G0]
    errors = []
    for key in keys:
        if require_admissible and not M.clear_forms_appreciably(
                growth[key], params, g0,
                generation_minutes=G.GENERATION_MINUTES,
                hours=G.TRUNCATE_HOURS):
            return PENALTY
        prediction = G.predict(growth[key], params, g0=g0)
        errors.append(M.mean_absolute_error(prediction, experiment[key]))
    return float(np.mean(errors))


def _joint_objective(vector, growth, experiment):
    """Mean over all gamma conditions, kinetics shared, one offset per strain."""
    wt = _objective(vector[:-1], growth, experiment, WT_KEYS)
    if wt >= PENALTY:
        return PENALTY
    rad_vector = np.concatenate([vector[:N_KINETIC], [vector[-1]]])
    rad = _objective(rad_vector, growth, experiment, RAD_KEYS)
    if rad >= PENALTY:
        return PENALTY
    return float((wt + rad) / 2.0)


def _scan_g0(growth, experiment, params, keys, step=0.01,
             require_admissible=True):
    """Best offset on a fine grid with the kinetics held fixed.

    Returns ``(g0, admissible)``. When no offset satisfies the physicality
    constraint the scan is repeated without it and ``admissible`` comes back
    False, so the caller can report the configuration and label it for what it is
    rather than crashing on it.
    """
    low, high = BOUNDS[I_G0]

    def best(constrained):
        error, chosen = np.inf, np.nan
        for g0 in np.arange(low + step, high, step):
            value = _objective(np.concatenate([params, [g0]]), growth,
                               experiment, keys, require_admissible=constrained)
            if value < error:
                error, chosen = value, float(g0)
        return error, chosen

    error, g0 = best(require_admissible)
    if np.isfinite(error) and error < PENALTY:
        return g0, True
    if require_admissible:
        _error, g0 = best(False)
        return g0, False
    raise RuntimeError('no offset scored at all for ' + ', '.join(keys))


def per_dose_errors(growth, experiment, params, g0, keys):
    return {key: M.mean_absolute_error(G.predict(growth[key], params, g0=g0),
                                       experiment[key])
            for key in keys}


def summarize(label, errors):
    values = np.array(list(errors.values()))
    print(f'  {label:<40s} mean MAE = {values.mean():.4f}   '
          f'range = {values.min():.4f}-{values.max():.4f}')
    return float(values.mean())


def _transfer(growth, experiment, params, note=None):
    """Transfer fixed kinetics, refitting only the offset, per strain."""
    wt_g0, wt_ok = _scan_g0(growth, experiment, params, WT_KEYS)
    rad_g0, rad_ok = _scan_g0(growth, experiment, params, RAD_KEYS)
    err_wt = per_dose_errors(growth, experiment, params, wt_g0, WT_KEYS)
    err_rad = per_dose_errors(growth, experiment, params, rad_g0, RAD_KEYS)
    out = {
        'params': list(params),
        'g0': {'WT': wt_g0, 'rad51': rad_g0},
        'admissible': {'WT': wt_ok, 'rad51': rad_ok},
        'wt_per_dose': err_wt, 'rad51_per_dose': err_rad,
        'wt_mean': summarize('wild type', err_wt),
        'rad51_mean': summarize('rad51-delta', err_rad),
        'mean': summarize('all gamma conditions', dict(err_wt, **err_rad))}
    if note:
        out['note'] = note
    print(f'    g0 = {wt_g0:.2f} generations for the wild type, '
          f'{rad_g0:.2f} for rad51-delta'
          + ('' if wt_ok and rad_ok
             else '   [no admissible offset: the colorless species never '
                  'accumulates, because the population is extinct]'))
    return out


# ---------------------------------------------------------------------------
# Hit-rate sensitivity
# ---------------------------------------------------------------------------

#: The archived 2.5 Gy wild-type gamma runs, which differ only in the GammaRadGen
#: hit rate. Decoded from the folder labels in
#: analysis/gamma/GammaAMMPERParametrization5.py.
ARCHIVED_HIT_RATES = {50: 'WT_25k50', 100: 'WT_25', 1000: 'WT_250'}


def hit_rate_sensitivity(params, g0, experiment):
    """Error at 2.5 Gy as a function of the GammaRadGen hit rate.

    Reads the three archived folders directly, since they are the only runs the
    original authors made that vary k. Reported rather than optimized here: k is
    calibrated in calibrate_gamma_hit_rate.py against the growth channel, and
    letting the aB fit choose it would let dye chemistry determine how many
    photons a Gray delivers.
    """
    import pandas as pd
    import ammper_paths as P

    out = {}
    for rate, folder in sorted(ARCHIVED_HIT_RATES.items()):
        curves = []
        for directory, _dirs, files in os.walk(P.bulk_gamma(folder)):
            if G.GAMMA_DATA_FILE not in files:
                continue
            frame = pd.read_csv(
                os.path.join(directory, G.GAMMA_DATA_FILE),
                names=['Generation', 'x', 'y', 'z', 'Health'])
            curves.append(M.GrowthCurve(M._counts_by_generation(frame, 1),
                                        M._counts_by_generation(frame, 2), 1))
        if not curves:
            continue
        mean = M.GrowthCurve(
            np.mean([c.healthy for c in curves], axis=0),
            np.mean([c.unhealthy for c in curves], axis=0), len(curves))
        error = M.mean_absolute_error(G.predict(mean, params, g0=g0),
                                      experiment['gWT_2.5'])
        damaged = mean.unhealthy.sum() / max(mean.healthy.sum(), 1.0)
        out[rate] = {'mae': float(error), 'n_runs': len(curves),
                     'damaged_to_healthy': float(damaged),
                     'folder': folder}
        print(f'  k = {rate:>4d} ({folder:<10s}, {len(curves)} runs)  '
              f'MAE = {error:.4f}   damaged/healthy = {damaged:.3f}')
    return out


def main():
    experiment = G.load_experimental()

    # Three views of the same six conditions per strain: the calibrated runs, the
    # published-rate runs, and the published-rate runs read the buggy way.
    growth = G.load_growth_curves(calibrated=True)
    uncalibrated = G.load_growth_curves(calibrated=False)
    buggy = {key: M.GrowthCurve(
                 np.mean([r.healthy for r in runs], axis=0),
                 np.mean([r.unhealthy for r in runs], axis=0), len(runs))
             for key, runs in G.load_buggy_growth_curves().items()}

    calibration = None
    if os.path.exists(CALIBRATION_JSON):
        with open(CALIBRATION_JSON) as handle:
            calibration = json.load(handle)

    report = {'seed': SEED, 'bounds': BOUNDS,
              'generation_minutes': G.GENERATION_MINUTES,
              'published_hit_rate': G.PUBLISHED_HIT_RATE,
              'calibrated_hit_rate': calibration and calibration['shared'],
              'doses_gy': G.DOSES,
              'n_runs': {key: curve.n_runs for key, curve in growth.items()}}

    # -- what the submitted figure shows ------------------------------------
    # The published gamma script integrated on the discrete generation ticks with
    # no inoculum offset, so g0 = 0 on the reversed curve is the faithful
    # analogue. Note that the reversed curve is what the *healthy* series became;
    # the damaged series was reindexed correctly even in the published code, and
    # load_buggy_growth_curves reproduces that asymmetry exactly.
    print('As submitted, and with the growth-curve ordering corrected '
          '(published hit rate)')
    err_submitted = per_dose_errors(buggy, experiment,
                                    PUBLISHED_GAMMA_PARAMS, 0.0, ALL_KEYS)
    report['submitted'] = {
        'params': list(PUBLISHED_GAMMA_PARAMS), 'g0': 0.0,
        'hit_rate': G.PUBLISHED_HIT_RATE, 'growth_curve': 'reversed',
        'per_dose': err_submitted,
        'mean': summarize('submitted (reversed curve)', err_submitted)}

    err_ordering = per_dose_errors(uncalibrated, experiment,
                                   PUBLISHED_GAMMA_PARAMS, 0.0, ALL_KEYS)
    report['ordering_fixed'] = {
        'params': list(PUBLISHED_GAMMA_PARAMS), 'g0': 0.0,
        'hit_rate': G.PUBLISHED_HIT_RATE, 'growth_curve': 'corrected',
        'per_dose': err_ordering,
        'mean': summarize('ordering fixed, same kinetics', err_ordering)}

    # -- the ordering fix alone is not enough --------------------------------
    print('\nProton kinetics transferred, offset refit, still at the published '
          'hit rate k = 100')
    report['published_rate'] = _transfer(
        uncalibrated, experiment, M.SHARED_PARAMS,
        note=('The best available fit at the published hit rate. The simulated '
              'population is extinct over most of the measured dose range, so '
              'no offset produces appreciable over-reduction and the '
              'physicality constraint cannot be satisfied at all.'))
    report['published_rate']['hit_rate'] = G.PUBLISHED_HIT_RATE

    # -- the reported result: calibrated rate, proton kinetics, only g0 refit --
    print('\nProton kinetics transferred unchanged at the calibrated hit rate, '
          'only g0 refit per strain')
    transferred = M.SHARED_PARAMS
    report['transferred'] = _transfer(growth, experiment, transferred)
    report['transferred']['hit_rate'] = calibration and calibration['shared']
    report['transferred']['cells_at_t0'] = {
        strain: float(np.interp(report['transferred']['g0'][strain],
                                np.arange(G.NGEN + 1),
                                growth[f'g{strain}_0'].healthy))
        for strain in ('WT', 'rad51')}

    # -- refit on gamma, to bound how much the chemistry is costing -----------
    print('\nRefitting the kinetics on the gamma data '
          '(reported for comparison, not as the result)...')
    result = differential_evolution(
        _joint_objective, BOUNDS + [BOUNDS[I_G0]],
        args=(growth, experiment), seed=SEED, maxiter=300, popsize=24,
        tol=1e-8, polish=True, updating='deferred', workers=-1)
    refit_params = tuple(result.x[:N_KINETIC])
    report['refit'] = _transfer(growth, experiment, refit_params)
    report['refit']['hit_rate'] = calibration and calibration['shared']
    report['refit']['optimizer_nfev'] = int(result.nfev)

    # -- hit-rate sensitivity ------------------------------------------------
    print('\nSensitivity to the GammaRadGen hit rate at 2.5 Gy '
          '(archived runs, published-rate analysis)')
    report['hit_rate_sensitivity'] = hit_rate_sensitivity(
        M.SHARED_PARAMS, report['published_rate']['g0']['WT'], experiment)

    with open(OUT_JSON, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nwrote {OUT_JSON}')
    return report


if __name__ == '__main__':
    main()
