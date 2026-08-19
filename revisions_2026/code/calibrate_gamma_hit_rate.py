"""
Calibrate the GammaRadGen hit rate against measured survival, independently of
the alamarBlue dye chemistry.

WHY THIS EXISTS
---------------
The submitted supplement reported that the alamarBlue model "failed to
recapitulate the experimental data" under gamma radiation and blamed the model's
binary health-state assumption. Correcting the growth-curve ordering bug (see
ammper_ab_model_gamma.py) improves the gamma fit but does not rescue it, and
running the dose series for the first time shows why. At the published hit rate
the simulated wild-type population is essentially wiped out well before the top
of the measured dose range:

    dose      simulated healthy cells        measured growth channel
              (fraction of 0 Gy control)     (fraction of 0 Gy control, 15 h)
     2.5 Gy               0.68                         0.87
       5 Gy               0.45                         0.85
      10 Gy               0.21                         0.84
      20 Gy               0.013                        0.76
      30 Gy               0.002                        0.78

At 20 Gy the simulation retains 1.3% of the control population while the
experiment retains 76%. No dye chemistry can reconcile those: the population
trajectory is the aB model's only input, and it has been driven to extinction
before the dose range of interest begins. The submitted gamma figure was
therefore diagnosing the wrong component. The binary health state is a real
limitation of AMMPER, but it is not what made the gamma predictions fail.

WHAT IS ACTUALLY MIS-SET
------------------------
GammaRadGen converts dose to deposition events with

    n_Hits = int(round(dose * k)),  k = 100

emitted independently in each of N = 64 x-planes, so the total is
64 * int(round(100 * dose)) events, each depositing energy = 100000 (1e5 eV in
AMMPER's units, i.e. 100 keV). The source comment on that constant reads

    k = 100 # is arbitrary for now, need physical parameters to calculate the
            # number of events in an area 64 x 64 micrometers

and the line below it, "scan k = [1,2,3,4,5] see which one matches expected
population ratio", records that the calibration was intended and never done. The
three archived gamma folders (WT_25, WT_250, WT_25k50) are that unfinished scan:
the same 2.5 Gy dose at k = 100, 1000 and 50.

An order-of-magnitude check confirms the constant is not physical. A 64 um cube
of water has a mass of 2.6e-10 kg, so 1 Gy deposits 2.6e-10 J. At 100 keV per
event that is about 16,000 events per Gy, against the 6,400 the implementation
emits -- the right order of magnitude, which is what makes k = 100 look
defensible in isolation. But AMMPER's damage rule is not energy-integrating: a
cell is damaged if any deposition event falls within +-2 um of its center
(genROSOld), so what matters is the spatial density of events near cells, not the
total energy. The published k puts 192,000 events into a 64-um cube at 30 Gy,
which saturates the cube geometrically regardless of the per-event energy.

THE ONE DEGENERACY THAT MAKES THIS TRACTABLE
--------------------------------------------
n_Hits depends on dose and k only through their product, so a simulation at dose
D with hit rate k is the same object as a simulation at dose D*k/100 with the
published k = 100. Recalibrating k is exactly a rescaling of the dose axis, and
it needs no modification to the simulator: this module runs the unmodified
GammaRadGen at the equivalent dose instead. That is verified numerically in
verify_dose_rate_degeneracy() below, against the archived k-sweep -- the archived
2.5 Gy run at k = 1000 (2,500 events per plane) falls on the curve traced by the
k = 100 dose series at 20-30 Gy (2,000-3,000 events per plane).

WHAT IS CALIBRATED AGAINST WHAT, AND WHY IT IS NOT CIRCULAR
-----------------------------------------------------------
k is fit to the OD690 growth channel -- how much the cell population grew --
which is a turbidity measurement of cell density. It is not fit to the aB blue
and pink series that the kinetic model predicts and that every reported error is
computed against. The dye chemistry is then transferred from the proton fit with
nothing refit but the inoculum offset, so the resulting gamma alamarBlue errors
are a held-out consequence of a calibration performed on separate data. Letting
the aB fit choose k would let dye chemistry determine how many photons a Gray
delivers, which is why it is not done here or anywhere in the revision.

k is a property of the radiation model, not of the genotype, so a single shared
value is used for both strains. The per-strain values are reported as a
sensitivity: they differ (6.9 for the wild type against 1.9 for rad51-delta),
and that residual disagreement is a real and separate finding about AMMPER's
relative radiosensitivity between strains, discussed in the supplement rather
than absorbed into a fitted parameter.

    python3 calibrate_gamma_hit_rate.py            # calibrate and report
    python3 calibrate_gamma_hit_rate.py --run      # also run the equivalent doses

Outputs
-------
    gamma_hit_rate_calibration.json
"""

import argparse
import json
import os
import subprocess
import sys
import concurrent.futures

import numpy as np
import pandas as pd

import ammper_ab_model_fixed as M
import extract_gamma_experimental as E
import run_gamma_simulations as R

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import ammper_paths as P  # noqa: E402

OUT_JSON = os.path.join(HERE, 'gamma_hit_rate_calibration.json')

#: Published hit rate, and the value the simulator hardcodes.
PUBLISHED_HIT_RATE = 100.0

#: Doses simulated at the published hit rate, which together trace the
#: survival-versus-events curve that the calibration inverts.
SIMULATED_DOSES = R.DOSES

#: Nominal experimental doses the gamma aB comparison reports. These match the
#: proton series so the two radiation qualities can be compared at equal dose.
NOMINAL_DOSES = [0.0, 2.5, 5.0, 10.0, 20.0, 30.0]

#: Hour at which the growth channel is read for the calibration. The aB
#: comparison window is 15 h (M.TRUNCATE_HOURS), and using the same endpoint
#: keeps the calibration on the same stretch of the experiment as the quantity it
#: is meant to make predictable.
GROWTH_HOUR = M.TRUNCATE_HOURS

#: Floor on simulated survival when taking logs. Three replicates cannot resolve
#: a survival fraction below about 1e-3, so a zero is noise, not information.
SURVIVAL_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# Measured survival: the OD690 growth channel
# ---------------------------------------------------------------------------

def measured_survival(strain):
    """``{dose: net growth at GROWTH_HOUR, relative to the 0 Gy control}``.

    The 690 nm channel is the turbidity (cell density) readout, labelled
    "690 nm Growth" in the source workbooks. Net growth is taken as the rise from
    t = 0 rather than the absolute absorbance, so the blank subtracts out.
    """
    out = {}
    for dose in E.DOSES:
        frame = pd.read_csv(P.ab_gamma_experimental(E.csv_name(strain, dose)),
                            names=M.CSV_COLUMNS)
        values = frame['A690'].to_numpy()
        out[dose] = float(values[GROWTH_HOUR] - values[0])
    control = out[0.0]
    return {dose: value / control for dose, value in out.items()}


# ---------------------------------------------------------------------------
# Simulated survival: the healthy-cell count
# ---------------------------------------------------------------------------

def _final_healthy(root, gamma=True):
    """Mean healthy-cell count at the last generation, over replicate runs.

    The gamma branch of the simulator writes Gamma.txt; the proton branch, which
    supplies the 0 Gy control, writes "<dose>Gy.txt". Returns
    ``(mean count, number of runs)``.
    """
    counts = []
    for directory, _dirs, files in os.walk(root):
        wanted = ([f for f in files if f == 'Gamma.txt'] if gamma
                  else [f for f in files if f.endswith('Gy.txt')])
        if not wanted:
            continue
        frame = pd.read_csv(os.path.join(directory, sorted(wanted)[0]),
                            names=['Generation', 'x', 'y', 'z', 'Health'])
        counts.append(M._counts_by_generation(frame, 1)[-1])
    if not counts:
        raise FileNotFoundError(f'no simulation output under {root}')
    return float(np.mean(counts)), len(counts)


def simulated_survival(strain):
    """``{dose: final healthy count / 0 Gy control}`` at the published hit rate.

    The 0 Gy control comes from the proton simulations: GammaRadGen refuses a
    zero dose, and at 0 Gy the two branches of the simulator are the same object
    (no radiation, so no ROS and no repair code runs at all).
    """
    control, _n = _final_healthy(P.bulk_aB(f'{strain}_Basic_0'), gamma=False)
    out = {0.0: 1.0}
    for dose in SIMULATED_DOSES:
        folder = P.bulk_gamma(R.condition_folder(strain, dose))
        healthy, _n = _final_healthy(folder)
        out[dose] = healthy / control
    return out


def verify_dose_rate_degeneracy():
    """Check that survival depends on dose and k only through their product.

    The archived 2.5 Gy folders differ only in k, so each is equivalent to a
    different dose at the published k = 100. If the degeneracy holds, each
    archived point must fall on the curve traced by the k = 100 dose series at
    the same number of events per plane. This is what licenses calibrating k by
    rescaling the dose axis instead of editing the simulator.
    """
    control, _n = _final_healthy(P.bulk_aB('WT_Basic_0'), gamma=False)
    simulated = simulated_survival('WT')
    events = np.log([dose * PUBLISHED_HIT_RATE for dose in SIMULATED_DOSES])
    survival = np.log([max(simulated[d], SURVIVAL_FLOOR)
                       for d in SIMULATED_DOSES])

    checks = []
    for rate, folder in sorted(R_ARCHIVED.items()):
        healthy, n_runs = _final_healthy(P.bulk_gamma(folder))
        observed = healthy / control
        equivalent = 2.5 * rate                      # events per plane
        predicted = float(np.exp(np.interp(np.log(equivalent), events, survival)))
        checks.append({
            'archived_folder': folder, 'hit_rate': rate,
            'events_per_plane': equivalent, 'n_runs': n_runs,
            'observed_survival': observed, 'predicted_survival': predicted,
            'equivalent_dose_at_k100': equivalent / PUBLISHED_HIT_RATE})
        print(f'  {folder:<10s} k={rate:<5g} = {equivalent:>6.0f} events/plane '
              f'= {equivalent / PUBLISHED_HIT_RATE:>5.1f} Gy at k=100   '
              f'survival observed {observed:.4f} vs {predicted:.4f} predicted')
    return checks


#: Archived 2.5 Gy folders, decoded from the sweep in
#: analysis/gamma/GammaAMMPERParametrization5.py: same dose, three hit rates.
R_ARCHIVED = {50: 'WT_25k50', 100: 'WT_25', 1000: 'WT_250'}


# ---------------------------------------------------------------------------
# The calibration itself
# ---------------------------------------------------------------------------

def _log_survival_interpolator(simulated):
    """log survival as a function of log dose, at the published hit rate."""
    doses = np.log(SIMULATED_DOSES)
    survival = np.log([max(simulated[d], SURVIVAL_FLOOR)
                       for d in SIMULATED_DOSES])
    return doses, survival


def _residuals(hit_rate, measured, simulated):
    """log(measured survival) - log(simulated survival) at each measured dose.

    Residuals are taken in the log because survival spans three orders of
    magnitude across the series; an absolute residual would be determined
    entirely by the two or three lowest doses.
    """
    doses, survival = _log_survival_interpolator(simulated)
    out = []
    for dose in E.DOSES:
        if dose == 0 or dose > max(SIMULATED_DOSES):
            continue        # 0 Gy is the normalization; 40-60 Gy not simulated
        equivalent = dose * hit_rate / PUBLISHED_HIT_RATE
        out.append(np.log(measured[dose])
                   - float(np.interp(np.log(equivalent), doses, survival)))
    return np.array(out)


def calibrate(strains=('WT', 'rad51'), n_grid=6000):
    """Least-squares hit rate over a log grid, and the per-strain values.

    Returned rate is shared across strains, since the number of deposition
    events a Gray produces is a property of the radiation and the geometry, not
    of the genotype.
    """
    measured = {s: measured_survival(s) for s in strains}
    simulated = {s: simulated_survival(s) for s in strains}
    grid = np.exp(np.linspace(np.log(0.05), np.log(2 * PUBLISHED_HIT_RATE),
                              n_grid))

    def cost(rate, subset):
        stacked = np.concatenate([_residuals(rate, measured[s], simulated[s])**2
                                  for s in subset])
        return float(np.mean(stacked))

    shared = min(grid, key=lambda r: cost(r, strains))
    per_strain = {s: float(min(grid, key=lambda r: cost(r, (s,))))
                  for s in strains}
    return {
        'shared': float(shared),
        'shared_rms': float(np.sqrt(cost(shared, strains))),
        'per_strain': per_strain,
        'per_strain_rms': {s: float(np.sqrt(cost(per_strain[s], (s,))))
                           for s in strains},
        'published_rms': float(np.sqrt(cost(PUBLISHED_HIT_RATE, strains))),
        'measured_survival': {s: measured[s] for s in strains},
        'simulated_survival': {s: simulated[s] for s in strains},
    }


# ---------------------------------------------------------------------------
# Running the equivalent doses
# ---------------------------------------------------------------------------

def equivalent_dose(nominal, hit_rate):
    """Dose to run at the published k that realizes ``nominal`` Gy at ``hit_rate``."""
    return nominal * hit_rate / PUBLISHED_HIT_RATE


def calibrated_folder(strain, nominal):
    """Folder for a calibrated-dose run, keyed by the NOMINAL dose.

    Keyed by nominal rather than equivalent dose so that the folder name matches
    the dose the manuscript reports; the equivalent dose actually passed to the
    simulator is recorded in the calibration JSON.
    """
    tag = f'{nominal:g}'.replace('.', 'p')
    return os.path.join('revision2026', f'gamma_calib_{strain}_{tag}Gy')


def run_calibrated(hit_rate, replicates=R.REPLICATES, jobs=None):
    """Run the unmodified simulator at the calibrated equivalent doses."""
    todo = []
    for nominal in NOMINAL_DOSES:
        if nominal == 0:
            continue        # 0 Gy control comes from the proton runs
        dose = equivalent_dose(nominal, hit_rate)
        for strain in ('WT', 'rad51'):
            root = P.bulk_gamma(calibrated_folder(strain, nominal))
            have = sum(1 for _r, _d, files in os.walk(root)
                       if 'Gamma.txt' in files)
            for index in range(have, replicates):
                todo.append((strain, nominal, dose, index))

    print(f'{len(todo)} calibrated run(s) needed at k = {hit_rate:.3f}')
    if not todo:
        return []

    def one(item):
        strain, nominal, dose, index = item
        folder = os.path.join(calibrated_folder(strain, nominal),
                              f'rep{index + 1}')
        command = [sys.executable, R.SIMULATOR, 'd', R.STRAIN_FLAG[strain], 'a',
                   f'{dose:.6g}', folder]
        result = subprocess.run(command, cwd=os.path.join(REPO, 'src'),
                                capture_output=True, text=True)
        ok = result.returncode == 0
        print(f'{"done" if ok else "FAIL"} {strain} {nominal:g} Gy nominal '
              f'({dose:.4g} Gy equivalent) rep {index + 1}')
        if not ok:
            print(result.stderr.strip()[-600:])
        return ok

    jobs = jobs or max(1, os.cpu_count() - 1)
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(one, todo))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true',
                        help='also run the simulations at the calibrated doses')
    parser.add_argument('--jobs', type=int, default=None)
    arguments = parser.parse_args()

    print('Dose-rate degeneracy check (archived k-sweep against the k=100 '
          'dose series)')
    degeneracy = verify_dose_rate_degeneracy()

    print('\nCalibrating the hit rate against the measured OD690 growth channel')
    result = calibrate()
    print(f'  shared k = {result["shared"]:.3f}   '
          f'rms log-survival residual {result["shared_rms"]:.3f}')
    for strain, rate in result['per_strain'].items():
        print(f'  {strain:<6s} alone: k = {rate:.3f}   '
              f'rms {result["per_strain_rms"][strain]:.3f}')
    print(f'  published k = {PUBLISHED_HIT_RATE:g}: '
          f'rms {result["published_rms"]:.3f}')
    print(f'  the published rate is {PUBLISHED_HIT_RATE / result["shared"]:.0f}x '
          f'too many events per Gray')

    result['published_hit_rate'] = PUBLISHED_HIT_RATE
    result['degeneracy_check'] = degeneracy
    result['growth_hour'] = GROWTH_HOUR
    result['nominal_doses'] = NOMINAL_DOSES
    result['equivalent_doses'] = {
        f'{d:g}': equivalent_dose(d, result['shared']) for d in NOMINAL_DOSES}

    print('\nEquivalent doses to run at the published k = 100:')
    for nominal in NOMINAL_DOSES:
        if nominal == 0:
            continue
        print(f'  {nominal:>5g} Gy nominal -> '
              f'{equivalent_dose(nominal, result["shared"]):.4f} Gy')

    with open(OUT_JSON, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(f'\nwrote {OUT_JSON}')

    if arguments.run:
        run_calibrated(result['shared'], jobs=arguments.jobs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
