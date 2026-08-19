"""
Corrected alamarBlue (aB) kinetics model for AMMPER-2.
2026 revision round.

This module is the bug-fixed replacement for the aB modeling code that produced
the originally submitted Figure 2 and Figure S1. It is written as an importable
module so that every figure and number in the revision comes from one
implementation rather than from several near-duplicate scripts.

-------------------------------------------------------------------------------
THE TWO BUGS THIS FIXES, AND ONE THING THAT LOOKS LIKE A THIRD BUT IS NOT
-------------------------------------------------------------------------------

BUG 1 (critical) -- the growth curve was fed to the ODE time-reversed.

    The original code built the population growth curve with

        Growth_curve = Healthy['Generation'].value_counts()
        Growth_curve = np.array(Growth_curve)

    pandas' value_counts() sorts by *frequency*, descending -- not by the
    generation index. Because the yeast population grows monotonically, the
    per-generation counts are themselves monotonically increasing, so sorting by
    descending count is exactly sorting by descending generation. Discarding the
    index then throws away the only evidence of the ordering.

    The ODE therefore integrated a population that STARTS at ~4096 cells and
    DECAYS to 1 cell over 15 hours -- the time reverse of the simulation. Since
    metabolic capacity is at its maximum from t=0, the predicted blue curve
    collapses almost linearly from the first timestep, which is the shape that
    appears in the submitted Figure 2. It also flattens the dose response: every
    dose begins from the same saturated ~4096-cell state, so the predicted
    curves for different doses lie nearly on top of one another.

    Fixed here by grouping on the generation index and reindexing onto the full
    0..NGEN range, so that a generation in which no cells were recorded becomes
    a zero rather than a silently dropped row:

        healthy = frame.groupby('Generation').size().reindex(range(NGEN+1), fill_value=0)

NOT A BUG -- the experimental normalization, which looks like an aliasing error
but is the convention the manuscript is written around.

    The published code normalizes the two measured species with

        B_C = B_C / (B_C[0] + P_C[0])
        P_C = P_C / (B_C[0] + P_C[0])   # <-- B_C[0] is already normalized here

    The second line reuses B_C after it has been reassigned, so the two series
    are divided by different denominators: blue by the raw initial total
    (0.3545 for the WT 0 Gy series) and pink by 1 + P_C[0]/(B_C[0]+P_C[0])
    = 0.9595. An earlier pass in this revision treated that as an aliasing bug
    and computed the denominator once for both. That was wrong, and it is worth
    recording why, because the "fix" is superficially the obvious one.

    Dividing both series by the raw initial total scales pink up by 2.7x, and
    the recovered blue + pink then reaches 1.065 -- above unity, which is not
    physically admissible for fractions of a conserved dye pool, and which no
    three-species model can reproduce. Under the published convention the sum
    stays at or below 1 (max 0.961 for this series) and its deficit from 1
    behaves like the colorless species, rising smoothly from 0.039 to 0.60
    across the window. That is exactly the third species in the blue -> pink ->
    colorless chain, and it is consistent with how predictions are normalized
    (blue, pink and colorless each divided by their per-timepoint sum, so the
    three sum to 1 by construction).

    So the published normalization is not an accident that happens to work: it
    puts the measurements in the same units as the predictions, with the
    unmeasured colorless species carrying the residual. It is retained here
    unchanged. The consequence to keep in mind is that measured "blue" and
    "pink" are fractions of the total dye pool including colorless, not
    fractions of the blue + pink subtotal.

BUG 2 -- the 5 Gy and 10 Gy panels were mislabeled.

    Results folder WT_Basic_10 holds 10Gy.txt and was paired with
    AlamarblueRawdataWT10Gy.csv (both 10 Gy, consistent), but the panel was
    labeled "5 Gy"; WT_Basic_50 holds 5Gy.txt, paired with the 5 Gy CSV, and was
    labeled "10 Gy". Simulation and experiment were paired correctly throughout,
    so the fit is unaffected -- but two published panels carried the wrong dose.

    Fixed here by deriving the dose label from CONDITIONS below, where each
    condition records the dose once and both the simulation folder and the
    experimental CSV are looked up from it.

-------------------------------------------------------------------------------
ONE MODELING CHANGE (not a bug fix)
-------------------------------------------------------------------------------

Once the ordering is fixed, the dominant remaining mismatch is that AMMPER
starts from a SINGLE cell while the plate-reader experiment starts from a dense
inoculum. Fitting the aB model to a simulation that begins at one cell forces it
to explain ~9 generations of near-zero signal that the experiment never
observes.

We therefore introduce one parameter, g0: the point on the AMMPER growth curve
that corresponds to experimental t=0. It is a property of the assay setup (how
many cells were plated), not of the dye chemistry, and it is fit once per strain
and then held fixed across all six doses. Time is integrated continuously rather
than on the 16 discrete generation ticks, so that g0 need not be an integer.

For the wild type this gives g0 = 8.81 generations (~463 cells at t=0), and for
rad51-delta 6.21 (~77 cells), which are physically sensible inocula for this
assay and put the mutant ~2.6 generations behind.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------

    import ammper_ab_model_fixed as M

    gc   = M.load_growth_curves('WT')          # {condition: GrowthCurve}
    exp  = M.load_experimental('WT')           # {condition: Measurement}
    pred = M.predict(gc['WT_Basic_0'], M.WT_PARAMS, g0=M.WT_G0)
    mae  = M.mean_absolute_error(pred, exp['WT_Basic_0'])
"""

import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ammper_paths as P  # noqa: E402

# ---------------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------------

#: Number of simulated generations in the AMMPER runs used for the aB work.
NGEN = 15

#: Hours of each aB curve considered, matching the main text (the blue-to-pink
#: conversion is what matters for most yeast experiments).
TRUNCATE_HOURS = 15

#: Minutes per yeast generation in the 150 MeV proton experiments.
GENERATION_MINUTES = 198.0

#: Absorbance bleed-through / background corrections for converting plate-reader
#: absorbances into blue and pink concentrations. Unchanged from the original.
OD_RATIO_RED = 1.04
OD_RATIO_GREEN = 1.06
PINK_RATIO = 0.06
BLUE_RATIO = 0.7

CSV_COLUMNS = ['Time', 'A570', 'A600', 'A690', 'A750']

Condition = namedtuple('Condition', 'key strain dose_gy sim_folder csv_stem')

#: The single source of truth for dose bookkeeping. Bug 3 existed because the
#: dose appeared independently in the folder name, the CSV name, and the panel
#: label, and the three fell out of sync. Here the dose is recorded once.
#:
#: Note the historical folder naming: the numeric suffix is the dose in units of
#: 0.1 Gy for some conditions (25 -> 2.5 Gy, 50 -> 5 Gy, 200 -> 20 Gy,
#: 300 -> 30 Gy) but the plain dose for others (10 -> 10 Gy). That inconsistency
#: is what made the 5/10 Gy swap easy to miss.
CONDITIONS = [
    Condition('WT_0',    'WT', 0.0,  'WT_Basic_0',   'WTKGy'),
    Condition('WT_2.5',  'WT', 2.5,  'WT_Basic_25',  'WT25Gy'),
    Condition('WT_5',    'WT', 5.0,  'WT_Basic_50',  'WT5Gy'),
    Condition('WT_10',   'WT', 10.0, 'WT_Basic_10',  'WT10Gy'),
    Condition('WT_20',   'WT', 20.0, 'WT_Basic_200', 'WT20Gy'),
    Condition('WT_30',   'WT', 30.0, 'WT_Basic_300', 'WT30Gy'),
    Condition('rad51_0',   'rad51', 0.0,  'rad51_Basic_0',   'rad51KGy'),
    Condition('rad51_2.5', 'rad51', 2.5,  'rad51_Basic_25',  'rad5125Gy'),
    Condition('rad51_5',   'rad51', 5.0,  'rad51_Basic_50',  'rad515Gy'),
    Condition('rad51_10',  'rad51', 10.0, 'rad51_Basic_10',  'rad5110Gy'),
    Condition('rad51_20',  'rad51', 20.0, 'rad51_Basic_200', 'rad5120Gy'),
    Condition('rad51_30',  'rad51', 30.0, 'rad51_Basic_300', 'rad5130Gy'),
]


def conditions(strain=None):
    """Conditions for one strain ('WT' or 'rad51'), in ascending dose order."""
    out = [c for c in CONDITIONS if strain is None or c.strain == strain]
    return sorted(out, key=lambda c: c.dose_gy)


# ---------------------------------------------------------------------------
# Simulation output -> growth curves
# ---------------------------------------------------------------------------

GrowthCurve = namedtuple('GrowthCurve', 'healthy unhealthy n_runs')
Measurement = namedtuple('Measurement', 'time blue pink')
Prediction = namedtuple('Prediction', 'time blue pink')


def _counts_by_generation(frame, health_flag):
    """Per-generation cell count, correctly ordered by generation.

    This is the fix for Bug 1. groupby().size() is indexed by generation, and
    reindexing onto the full 0..NGEN range turns an unobserved generation into a
    zero instead of dropping it and shifting everything after it.
    """
    subset = frame[frame['Health'] == health_flag]
    counts = subset.groupby('Generation').size()
    return counts.reindex(range(NGEN + 1), fill_value=0).to_numpy(dtype=float)


def load_growth_curves_per_run(strain=None, results_root=None):
    """One GrowthCurve per replicate simulation: ``{condition_key: [GrowthCurve]}``.

    Needed for the prediction uncertainty bands in Figure 2, which are the
    spread over replicate simulations rather than a parameter uncertainty.
    """
    out = {}
    for cond in conditions(strain):
        folder = (os.path.join(results_root, cond.sim_folder)
                  if results_root else P.bulk_aB(cond.sim_folder))
        runs = []
        for root, _dirs, files in os.walk(folder):
            data_files = [f for f in files if f.endswith('Gy.txt')]
            if not data_files:
                continue
            frame = pd.read_csv(os.path.join(root, sorted(data_files)[0]),
                                names=['Generation', 'x', 'y', 'z', 'Health'])
            runs.append(GrowthCurve(healthy=_counts_by_generation(frame, 1),
                                    unhealthy=_counts_by_generation(frame, 2),
                                    n_runs=1))
        if not runs:
            raise FileNotFoundError(f'no simulation output under {folder}')
        out[cond.key] = runs
    return out


def load_growth_curves(strain=None, results_root=None):
    """Mean healthy and unhealthy growth curves per condition.

    Averages over every replicate simulation present for the condition. Returns
    ``{condition_key: GrowthCurve}``.
    """
    per_run = load_growth_curves_per_run(strain, results_root)
    return {key: GrowthCurve(
                healthy=np.mean(np.stack([r.healthy for r in runs]), axis=0),
                unhealthy=np.mean(np.stack([r.unhealthy for r in runs]), axis=0),
                n_runs=len(runs))
            for key, runs in per_run.items()}


# ---------------------------------------------------------------------------
# Plate reader -> experimental concentrations
# ---------------------------------------------------------------------------

def _absorbance_to_concentrations(frame):
    """Blue and pink concentrations from raw absorbances (unchanged chemistry)."""
    a690 = frame['A690'].to_numpy()
    a570 = frame['A570'].to_numpy()
    a600 = frame['A600'].to_numpy()
    od600 = a690 * OD_RATIO_RED
    od570 = a690 * OD_RATIO_GREEN
    blue = ((a600 - od600 - PINK_RATIO * (a570 - od570))
            / (1 - PINK_RATIO * BLUE_RATIO))
    pink = a570 - od570 - BLUE_RATIO * blue
    return blue, pink, frame['Time'].to_numpy()


def load_experimental(strain=None, with_error=False):
    """Experimental aB curves, using the published normalization exactly.

    The two series are divided by different denominators, reproducing the
    published code line for line:

        B_C = B_C / (B_C[0] + P_C[0])
        P_C = P_C / (B_C[0] + P_C[0])   # B_C already reassigned above

    This is deliberate and is NOT rewritten to share a denominator; see the
    module docstring for why. Briefly: sharing the denominator scales pink by
    2.7x and drives blue + pink to 1.065, which is not admissible for fractions
    of a conserved dye pool. As written, blue + pink stays at or below 1 and the
    deficit tracks the unmeasured colorless species, which is what makes these
    measurements directly comparable to predictions normalized by their
    per-timepoint three-species total.
    """
    out = {}
    for cond in conditions(strain):
        frame = pd.read_csv(P.ab_experimental(f'AlamarblueRawdata{cond.csv_stem}.csv'),
                            names=CSV_COLUMNS).head(TRUNCATE_HOURS)
        blue, pink, time = _absorbance_to_concentrations(frame)
        # Published normalization, reproduced exactly: blue is divided by the raw
        # initial total, then pink is divided by the ALREADY-NORMALIZED blue[0]
        # plus the raw pink[0]. Do not "simplify" this to a shared denominator.
        blue = blue / (blue[0] + pink[0])
        pink = pink / (blue[0] + pink[0])
        measurement = Measurement(time=time, blue=blue, pink=pink)
        if not with_error:
            out[cond.key] = measurement
            continue
        std = pd.read_csv(P.ab_experimental(f'AlamarblueRawdata{cond.csv_stem}STD.csv'),
                          names=CSV_COLUMNS).head(TRUNCATE_HOURS)
        bands = {}
        for sign, name in ((+1, 'upper'), (-1, 'lower')):
            shifted = frame.copy()
            for col in ('A570', 'A600', 'A690', 'A750'):
                shifted[col] = frame[col].to_numpy() + sign * std[col].to_numpy()
            b, p, _ = _absorbance_to_concentrations(shifted)
            b = b / (b[0] + p[0])
            p = p / (b[0] + p[0])      # same convention as the central series
            bands[name] = (b, p)
        out[cond.key] = (measurement, bands)
    return out


# ---------------------------------------------------------------------------
# The kinetic model
# ---------------------------------------------------------------------------

#: Initial amounts of the three aB species, in the arbitrary units the model
#: was written in. Only their ratios matter, since output is normalized.
BLUE_0, PINK_0, CLEAR_0 = 10000.0, 100.0, 100.0


def predict(growth_curve, params, g0, hours=TRUNCATE_HOURS, n_steps=600,
            generation_minutes=GENERATION_MINUTES, return_clear=False):
    """Integrate the Michaelis-Menten aB model over a growth curve.

    params : (v1, v2, v3, K1, K2, K3, k)
        v/K are the Michaelis-Menten rate and half-saturation constants for
        blue->pink (1) and the reversible pink<->clear pair (2, 3). k scales the
        contribution of unhealthy cells relative to healthy ones.
    g0 : float
        Position on the growth curve corresponding to experimental t=0, in
        generations. Accounts for the dense experimental inoculum.

    The readout is purely kinetic: the three integrated species are reported as
    fractions of their own conserved total, exactly as the published model did.
    There is no observation equation and no optical term.

    The three species are integrated without clamping, so total dye is conserved
    to numerical precision. An earlier version of this function clamped each
    species at zero with max(x, 0). That silently created mass whenever the
    reversible pink<->clear term ran backwards: with parameters for which the
    initial pink->clear rate is negative, the clamp fired on the clear species
    and total dye grew by over 250% across the window, pinning clear near zero
    and turning the model into a two-species blue->pink conversion. Parameter
    sets that drive clear negative are now rejected by the fitter (see
    clear_is_physical) rather than papered over here, which keeps the
    blue->pink->clear chain that the assay chemistry requires.
    """
    v1, v2, v3, k1, k2, k3, k = params

    healthy = interp1d(np.arange(NGEN + 1), growth_curve.healthy, kind='linear')
    unhealthy = interp1d(np.arange(NGEN + 1), growth_curve.unhealthy, kind='linear')

    time = np.linspace(0.0, hours, n_steps)
    generation = np.clip(g0 + time * 60.0 / generation_minutes, 0, NGEN)
    n_healthy = healthy(generation)
    n_unhealthy = unhealthy(generation)

    blue = np.empty(n_steps)
    pink = np.empty(n_steps)
    clear = np.empty(n_steps)
    blue[0], pink[0], clear[0] = BLUE_0, PINK_0, CLEAR_0

    for i in range(n_steps - 1):
        dt = time[i + 1] - time[i]
        cells = n_healthy[i] + k * n_unhealthy[i]

        # irreversible blue -> pink
        rate_bp = v1 * (blue[i] / (k1 + blue[i])) * cells
        d_bp = rate_bp * dt

        # reversible pink <-> clear
        alpha = pink[i] / k2
        pi = clear[i] / k3
        rate_pc = ((v2 * alpha) - (v3 * pi)) / (1.0 + alpha + pi) * cells
        d_pc = rate_pc * dt

        blue[i + 1] = blue[i] - d_bp
        pink[i + 1] = pink[i] + d_bp - d_pc
        clear[i + 1] = clear[i] + d_pc

    total = blue + pink + clear
    blue_frac, pink_frac = blue / total, pink / total

    # The kinetic core conserves dye exactly, so blue + pink + clear = 1 and the
    # two reported species are directly comparable to the measurements, which are
    # normalized the same way (see load_experimental). No observation-equation
    # correction is applied or needed.

    if return_clear:
        return (Prediction(time=time, blue=blue_frac, pink=pink_frac),
                clear / total)
    return Prediction(time=time, blue=blue_frac, pink=pink_frac)


def clear_is_physical(growth_curve, params, g0, **kwargs):
    """True if every species stays non-negative and clear never runs backwards.

    The reversible pink<->clear term can start negative when K3 is small enough
    that the initial clear/K3 ratio dominates v2*pink/K2. Such parameter sets
    reverse the final reduction step, which the assay chemistry does not permit:
    reduction of resorufin to the colorless species is what the third state
    represents, and it should accumulate rather than deplete.
    """
    prediction, clear = predict(growth_curve, params, g0, return_clear=True,
                                **kwargs)
    return bool(np.all(prediction.blue >= 0) and np.all(prediction.pink >= 0)
                and np.all(clear >= 0) and np.all(np.diff(clear) >= -1e-12))


#: Minimum growth in the colorless fraction across the window for a parameter
#: set to count as producing genuine over-reduction. Non-negativity alone is not
#: enough: the search can satisfy it by driving v2 and v3 to their lower bounds,
#: which switches the pink->clear step off and leaves clear pinned at its initial
#: value. That is a two-species model wearing three-species clothing.
MIN_CLEAR_GROWTH = 0.02


def clear_forms_appreciably(growth_curve, params, g0,
                            minimum=MIN_CLEAR_GROWTH, **kwargs):
    """True if the colorless species is physical AND actually accumulates."""
    prediction, clear = predict(growth_curve, params, g0, return_clear=True,
                                **kwargs)
    physical = (np.all(prediction.blue >= 0) and np.all(prediction.pink >= 0)
                and np.all(clear >= 0) and np.all(np.diff(clear) >= -1e-12))
    return bool(physical and (clear[-1] - clear[0]) >= minimum)


def mean_absolute_error(prediction, measurement):
    """Mean absolute error against the experimental timepoints.

    Averaged over the blue and the pink series, so the value is comparable
    across conditions regardless of how many timepoints each has.
    """
    blue_at = interp1d(prediction.time, prediction.blue,
                       bounds_error=False, fill_value='extrapolate')
    pink_at = interp1d(prediction.time, prediction.pink,
                       bounds_error=False, fill_value='extrapolate')
    blue_err = np.mean(np.abs(blue_at(measurement.time) - measurement.blue))
    pink_err = np.mean(np.abs(pink_at(measurement.time) - measurement.pink))
    return float((blue_err + pink_err) / 2.0)


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------
# Produced by fit_ab_model_fixed.py. Recorded here so the figure scripts are
# reproducible without refitting. See other/REVISION_NUMBERS.md for the search
# settings and the resulting per-dose errors.

#: Kinetics as (v1, v2, v3, K1, K2, K3, k), shared by both strains and fit to all
#: twelve conditions at once. Mean MAE 0.030 on the six wild-type doses, 0.072 on
#: the six rad51-delta doses, 0.051 overall.
#:
#: Fit jointly rather than on the wild type alone. On these data the two routes
#: agree closely -- fitting the wild type alone and transferring the kinetics with
#: only g0 refit gives 0.030 and 0.072 as well, 0.0512 over all twelve against
#: 0.0509 here -- so the joint fit is retained because it is the more constrained
#: arrangement rather than because the alternative fails. fit_ab_model_fixed.py
#: records both.
#:
#: K1 lands far above the substrate amounts. In that regime the corresponding
#: Michaelis-Menten term is effectively first order in substrate -- the dye is not
#: saturating the enzyme at these concentrations -- so that half-saturation
#: constant is not individually identifiable from these data and only its ratio to
#: v1 matters. Reported as such rather than quoted as a measured value. See
#: other/REVISION_NUMBERS.md.
SHARED_PARAMS = (3.2206228388864004, 18.595742333383342, 16.009871227112072,
                 13924.153293848001, 217.9361618349067, 302.63211189443365,
                 0.03434380490506569)

#: Retained under the old name: the kinetics are shared, so this is the same
#: tuple. Kept so existing callers and figure scripts keep working.
WT_PARAMS = SHARED_PARAMS

#: Growth-curve offset for the wild type, in generations (~463 cells at t=0).
WT_G0 = 8.81

#: rad51-delta shares the kinetics; only the offset differs. ~77 cells at t=0,
#: i.e. 2.60 generations behind the wild type.
RAD51_G0 = 6.21


def params_for(strain):
    """(params, g0) for 'WT' or 'rad51'. Kinetics are shared by construction."""
    return SHARED_PARAMS, (WT_G0 if strain == 'WT' else RAD51_G0)


def load_fitted(path=None):
    """Load fitted parameters from the JSON written by fit_ab_model_fixed.py."""
    import json
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fitted_parameters.json')
    with open(path) as handle:
        fit = json.load(handle)
    return fit
