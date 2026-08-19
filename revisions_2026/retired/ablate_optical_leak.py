"""
Is the optical observation equation justified? An ablation.

The observation equation added to predict() introduces two parameters, so it has
to earn them. This script fits the model three ways and compares the mean
absolute error over all twelve conditions, so that the leak cannot be credited
for a gain on one strain that it pays for on the other:

    A  no leak, two-stage      kinetics + g0 on WT; g0 only for rad51
    B  leak, two-stage         as A, plus the two leak coefficients on WT
    C  leak, joint             kinetics + leak + both offsets on all 12 at once
    D  no leak, joint          C's scheme with both coefficients pinned to zero

The comparison that decides the question is C vs D, which differ *only* in whether
the leak is free: A and C also differ in how the offsets are fit, so C beating A
would not by itself implicate the leak. D is what isolates it. If D matches C, the
gain came from fitting jointly and the leak should be dropped -- predict() would
keep its default of leak=(0, 0) and the two parameters would not be spent.

All arms get the same optimizer budget, since at a reduced budget the no-leak arm
landed on a poor optimum and flattered the leak.

Run time is roughly 25 minutes; results are written to ablation_leak.json.
"""

import json
import os

import numpy as np
from scipy.optimize import differential_evolution

import ammper_ab_model_fixed as M
import fit_ab_model_fixed as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'ablation_leak.json')

WT_KEYS = [c.key for c in M.conditions('WT')]
RAD_KEYS = [c.key for c in M.conditions('rad51')]

KINETIC_BOUNDS = F.BOUNDS[:F.N_KINETIC]
G0_BOUNDS = F.BOUNDS[F.I_G0]
LEAK_BOUNDS = F.BOUNDS[F.I_LEAK]

SEED = F.SEED


#: Both arms get exactly this budget. Matching it matters: at a reduced budget the
#: no-leak arm landed on a poor wild-type optimum whose offset then transferred
#: badly, which would have credited the leak for an optimizer failure.
MAXITER, POPSIZE = 300, 24


def _mean(errors):
    return float(np.mean(list(errors.values())))


def _fit(objective, bounds, args=(), maxiter=MAXITER, popsize=POPSIZE):
    return differential_evolution(objective, bounds, args=args, seed=SEED,
                                  maxiter=maxiter, popsize=popsize, tol=1e-8,
                                  polish=True, updating='deferred', workers=-1)


# Module-level objectives so differential_evolution can pickle them for workers=-1.

def _wt_objective(vector, growth, experiment):
    return F._objective(vector, growth, experiment, WT_KEYS)


def _rad_objective(vector, growth, experiment, params, leak):
    return F._objective(np.concatenate([params, vector, leak]),
                        growth, experiment, RAD_KEYS)


def _joint_objective(vector, growth, experiment):
    """Mean over all twelve conditions, with the leak shared between strains."""
    params = tuple(vector[:F.N_KINETIC])
    g0_wt, g0_rad = vector[F.I_G0], vector[10]
    leak = tuple(vector[F.I_LEAK])
    wt = F._objective(np.concatenate([params, [g0_wt], leak]),
                      growth, experiment, WT_KEYS)
    if wt >= F.PENALTY:
        return F.PENALTY
    rad = F._objective(np.concatenate([params, [g0_rad], leak]),
                       growth, experiment, RAD_KEYS)
    if rad >= F.PENALTY:
        return F.PENALTY
    return float((wt + rad) / 2.0)


def _brute_force_g0(growth, experiment, params, leak, keys, step=0.02):
    """Best offset on a fine grid, as a check on the 1-D search."""
    best = (np.inf, np.nan)
    for x in np.arange(G0_BOUNDS[0] + step, G0_BOUNDS[1], step):
        if not all(M.clear_forms_appreciably(growth[k], params, x) for k in keys):
            continue
        err = _mean(F.per_dose_errors(growth, experiment, params, x, keys, leak))
        if err < best[0]:
            best = (err, float(x))
    return best


def two_stage(growth, experiment, use_leak):
    """Fit on the wild type, then transfer to rad51-delta with g0 free."""
    bounds = list(KINETIC_BOUNDS) + [G0_BOUNDS]
    if use_leak:
        bounds += list(LEAK_BOUNDS)

    result = _fit(_wt_objective, bounds, args=(growth, experiment))
    params = tuple(result.x[:F.N_KINETIC])
    g0_wt = float(result.x[F.I_G0])
    leak = tuple(float(x) for x in result.x[F.I_LEAK]) if use_leak else M.NO_LEAK

    # The 1-D stage-2 search is cheap enough to verify exhaustively.
    _, g0_rad = _brute_force_g0(growth, experiment, params, leak, RAD_KEYS)

    err_wt = F.per_dose_errors(growth, experiment, params, g0_wt, WT_KEYS, leak)
    err_rad = F.per_dose_errors(growth, experiment, params, g0_rad, RAD_KEYS, leak)
    return params, leak, g0_wt, g0_rad, err_wt, err_rad


def joint(growth, experiment):
    """Fit kinetics, both leak coefficients and both offsets on all 12 at once."""
    bounds = (list(KINETIC_BOUNDS) + [G0_BOUNDS] + list(LEAK_BOUNDS)
              + [G0_BOUNDS])
    result = _fit(_joint_objective, bounds, args=(growth, experiment))
    params = tuple(result.x[:F.N_KINETIC])
    g0_wt, g0_rad = float(result.x[F.I_G0]), float(result.x[10])
    leak = tuple(float(x) for x in result.x[F.I_LEAK])
    err_wt = F.per_dose_errors(growth, experiment, params, g0_wt, WT_KEYS, leak)
    err_rad = F.per_dose_errors(growth, experiment, params, g0_rad, RAD_KEYS, leak)
    return params, leak, g0_wt, g0_rad, err_wt, err_rad


def joint_no_leak(growth, experiment):
    """Arm C's fitting scheme with the leak forced to zero -- the fair control.

    Arm C changes two things at once relative to A: it adds the leak, and it fits
    both offsets jointly over all twelve conditions rather than transferring from
    the wild type. Without this arm, a gain from the second change would be
    misattributed to the first.
    """
    bounds = list(KINETIC_BOUNDS) + [G0_BOUNDS] + [(0.0, 0.0), (0.0, 0.0)] \
        + [G0_BOUNDS]
    result = _fit(_joint_objective, bounds, args=(growth, experiment))
    params = tuple(result.x[:F.N_KINETIC])
    g0_wt, g0_rad = float(result.x[F.I_G0]), float(result.x[10])
    err_wt = F.per_dose_errors(growth, experiment, params, g0_wt, WT_KEYS,
                               M.NO_LEAK)
    err_rad = F.per_dose_errors(growth, experiment, params, g0_rad, RAD_KEYS,
                                M.NO_LEAK)
    return params, M.NO_LEAK, g0_wt, g0_rad, err_wt, err_rad


def main():
    growth = M.load_growth_curves()
    experiment = M.load_experimental()
    report = {}

    runs = [('A_no_leak_two_stage', lambda: two_stage(growth, experiment, False)),
            ('B_leak_two_stage', lambda: two_stage(growth, experiment, True)),
            ('C_leak_joint', lambda: joint(growth, experiment)),
            ('D_no_leak_joint', lambda: joint_no_leak(growth, experiment))]

    for name, run in runs:
        print(f'--- {name} ---', flush=True)
        params, leak, g0_wt, g0_rad, err_wt, err_rad = run()
        all_errors = dict(err_wt, **err_rad)
        entry = {
            'params': list(params), 'leak': list(leak),
            'g0_wt': g0_wt, 'g0_rad51': g0_rad,
            'wt_mean': _mean(err_wt), 'rad51_mean': _mean(err_rad),
            'overall_mean': _mean(all_errors),
            'per_condition': all_errors,
        }
        report[name] = entry
        print(f'  WT {entry["wt_mean"]:.4f}   rad51 {entry["rad51_mean"]:.4f}   '
              f'all 12 {entry["overall_mean"]:.4f}   leak {leak}', flush=True)
        print(f'  g0: WT {g0_wt:.2f}  rad51 {g0_rad:.2f}', flush=True)

    best = min(report, key=lambda k: report[k]['overall_mean'])
    report['best_over_all_twelve'] = best
    print(f'\nbest over all twelve conditions: {best}')

    with open(OUT, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
