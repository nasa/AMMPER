"""
Run the AMMPER gamma simulations needed for the corrected gamma alamarBlue fit.

The archived gamma output (results/bulk_gamma/WT_25, WT_250, WT_25k50) is not a
dose series: all three folders are the same 2.5 Gy dose run at three values of the
hit-rate constant k in GammaRadGen (100, 1000 and 50 respectively), from the
parametrization sweep in analysis/gamma/GammaAMMPERParametrization5.py. There is
no archived gamma output at any other dose, and none at all for rad51-delta,
which is why the original gamma aB analysis had to pair gamma simulations against
the *proton* rad51 experimental CSVs.

This script generates what was missing: three replicate runs per condition for
both strains across the dose series that the gamma alamarBlue experiment
measured, using the unmodified simulator (src/AMMPERBulk_GAMMAfinal.py) at its
published settings.

    python3 run_gamma_simulations.py                 # run everything missing
    python3 run_gamma_simulations.py --list          # show the plan and exit
    python3 run_gamma_simulations.py --jobs 4        # limit concurrency

Runs are skipped when the target folder already holds the requested number of
replicates, so the script can be re-invoked to top up an interrupted sweep.

THE 0 Gy CONTROL. GammaRadGen refuses a zero dose (it prints an error and returns
nothing), because a gamma run with no radiation events is not a gamma run. The
unirradiated control is instead taken from the existing 0 Gy proton simulations,
which are physically the same object: at 0 Gy the proton branch creates no
radData and no ROSData, so no radiation, ROS or repair code executes and the
simulation reduces to Brownian motion plus replication in a 64-micron cube for 15
generations -- identical to what a hypothetical 0 Gy gamma run would produce.
Reusing them rather than re-running avoids presenting the same computation twice
under two names. This is handled in ammper_ab_model_gamma.py, not here.

COST. Wall time grows with dose, because GammaRadGen emits 64 x (100 x dose)
deposition events and every cell tests itself against all of them at every
generation: about 30 s at 2.5 Gy, 70 s at 10 Gy and several minutes at 30 Gy on
one core. The runs are independent, so they are dispatched across cores.
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import ammper_paths as P  # noqa: E402

SIMULATOR = os.path.join(REPO, 'src', 'AMMPERBulk_GAMMAfinal.py')

#: Doses simulated, in Gy. Eleven of the fifteen doses the gamma alamarBlue
#: experiment read out: every dose from 0.01 to 30 Gy.
#:
#: The five doses that match the proton series (2.5-30 Gy) were run first, and
#: they turned out to be the wrong place to look. At the published hit rate the
#: simulated population is already almost wiped out by 20 Gy -- 1.3% of the
#: unirradiated healthy count -- whereas the measured growth channel at 20 Gy is
#: still at 76% of the control. The high-dose end of the series therefore carries
#: no information about the dye chemistry: it only reports that the radiation
#: model is mis-calibrated, and it reports the same thing at every dose above
#: about 10 Gy. The sub-Gray doses, which are where the measured dose response
#: actually lives, are what constrain the calibration, and they are the cheapest
#: runs in the series (0.01 Gy is 64 deposition events against 192,000 at 30 Gy).
#:
#: 40, 50 and 60 Gy are still excluded: they cost hours per replicate and, being
#: further into the regime where the simulated population is already extinct, add
#: nothing that 30 Gy does not already show.
DOSES = [0.01, 0.05, 0.1, 0.35, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0]

#: Replicates per condition, matching the three runs per condition in the
#: archived proton and gamma output. The spread over replicates is what the
#: prediction uncertainty bands show.
REPLICATES = 3

#: Simulator argument for each strain: cellType 'a' is wild type, 'b' is rad51.
STRAIN_FLAG = {'WT': 'a', 'rad51': 'b'}


def dose_tag(dose):
    """Folder fragment for a dose: 2.5 Gy -> '25', 10 Gy -> '10'.

    Follows the archived convention, in which fractional doses are written with
    the decimal point dropped. Note that this rule is only injective by accident
    once sub-Gray doses are included -- 0.5 Gy becomes '05' and 0.05 Gy becomes
    '005', which differ, but 5 Gy and 0.5 Gy would collide if the leading zero
    were also dropped. The assertion below is what keeps that accident honest;
    the experimental CSVs use a different and unambiguous rule
    (extract_gamma_experimental.dose_tag, which writes 0.05 Gy as '0p05Gy').
    """
    return f'{dose:g}'.replace('.', '') if dose < 10 else f'{dose:g}'


#: The folder names must distinguish the doses, or two conditions would write
#: into the same directory and silently average together.
assert len({dose_tag(d) for d in DOSES}) == len(DOSES), \
    f'dose_tag collides on {DOSES}: {[dose_tag(d) for d in DOSES]}'


def condition_folder(strain, dose):
    """Results folder for a condition, relative to results/bulk_gamma/."""
    return os.path.join('revision2026', f'gamma_{strain}_Basic_{dose_tag(dose)}')


def folder_for(strain, dose, index):
    """Results folder for one replicate, relative to results/bulk_gamma/.

    Each replicate gets its own subfolder. The simulator names its output
    directory by wall-clock minute, so two replicates of the same condition
    starting inside the same minute would otherwise land in the same directory and
    the second would overwrite the first -- which is why the archived runs were
    separated by a 60 s sleep in AMMPERruns_GAMMAFINAL.py. Giving each replicate a
    distinct parent removes the collision instead of waiting it out, so the sweep
    can use every core.
    """
    return os.path.join(condition_folder(strain, dose), f'rep{index + 1}')


def existing_runs(strain, dose):
    """How many completed replicate runs the condition already has on disk."""
    root = P.bulk_gamma(condition_folder(strain, dose))
    if not os.path.isdir(root):
        return 0
    return sum(1 for _root, _dirs, files in os.walk(root)
               if 'Gamma.txt' in files)


def run_once(strain, dose, index):
    """Invoke the simulator once. Returns (label, seconds, ok)."""
    label = f'{strain} {dose:g} Gy rep {index + 1}'
    command = [sys.executable, SIMULATOR, 'd', STRAIN_FLAG[strain], 'a',
               f'{dose:g}', folder_for(strain, dose, index)]
    started = time.time()
    result = subprocess.run(command, cwd=os.path.join(REPO, 'src'),
                            capture_output=True, text=True)
    elapsed = time.time() - started
    if result.returncode != 0:
        print(f'FAIL {label}: exit {result.returncode}\n'
              f'{result.stderr.strip()[-800:]}')
        return label, elapsed, False
    print(f'done {label}  {elapsed:6.1f}s')
    return label, elapsed, True


def plan():
    """Runs still needed, ordered longest-first so the tail does not straggle."""
    todo = []
    for dose in DOSES:
        for strain in STRAIN_FLAG:
            have = existing_runs(strain, dose)
            for index in range(have, REPLICATES):
                todo.append((strain, dose, index))
    return sorted(todo, key=lambda item: -item[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=max(1, os.cpu_count() - 2),
                        help='concurrent simulations (default: cores - 2)')
    parser.add_argument('--list', action='store_true',
                        help='show what would run, then exit')
    arguments = parser.parse_args()

    todo = plan()
    print(f'{len(todo)} run(s) needed '
          f'({len(DOSES)} doses x {len(STRAIN_FLAG)} strains x '
          f'{REPLICATES} replicates, minus what is already on disk)')
    for strain, dose, index in todo:
        print(f'  {strain:<6s} {dose:>5g} Gy  rep {index + 1}')
    if arguments.list or not todo:
        return 0

    started = time.time()
    failures = []
    with concurrent.futures.ThreadPoolExecutor(arguments.jobs) as pool:
        futures = [pool.submit(run_once, *item) for item in todo]
        for future in concurrent.futures.as_completed(futures):
            label, _elapsed, ok = future.result()
            if not ok:
                failures.append(label)

    print(f'\n{len(todo) - len(failures)}/{len(todo)} run(s) completed in '
          f'{(time.time() - started) / 60:.1f} min')
    for label in failures:
        print(f'  FAILED: {label}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
