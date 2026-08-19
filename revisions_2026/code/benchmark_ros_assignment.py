"""
Measure the run-time scaling of the two ROS damage-assignment implementations.

WHY THIS SCRIPT EXISTS. The submitted supplement carried a run-time comparison
figure (Fig. S4, Optimization.png) asserting that the optimized AMMPER reduced
run times "from minutes to seconds and from hours to minutes", with the caption
noting that the last two points of the unoptimized curve were extrapolated with a
second-degree polynomial "due to the excessive computational resources required
for actual runs". No timing measurements survive anywhere in the repository, so
that figure cannot be regenerated and its extrapolated points cannot be checked.
Rather than carry an unreproducible figure through a revision that is about how
carefully the model was checked, this script re-measures the quantity directly.

WHAT IS MEASURED. Both implementations of the same operation: deciding which
cells a set of ROS coordinates damages, by testing each ROS point against each
cell's bounding box.

    naive       the genROSOld / pre-optimization idiom, a Python nested loop over
                every ROS point crossed with every cell: O(R x C) interpreted
                comparisons.
    filtered    the cellDefinition.cellROS idiom in the released code, a pandas
                bounding-box filter evaluated once per cell over the whole ROS
                frame: O(C) vectorized passes.

Both are run on identical inputs and their hit counts are asserted equal, so the
comparison is of two implementations of one function rather than of two different
functions. The naive arm is skipped above NAIVE_LIMIT points, and the figure
built from this output marks where that happens rather than extrapolating across
it -- which is the specific defect in the submitted figure.

WHAT THE RESULT ACTUALLY SHOWS, AND WHY IT IS REPORTED ANYWAY. The vectorized
implementation is *slower* at small ROS counts, because a pandas filter has a
fixed per-call overhead that an interpreted loop over a handful of points does
not pay. It wins by a growing margin above a crossover of a few thousand points.
That is a more useful statement than "minutes to seconds": it says the
optimization matters exactly in the regime the diffusion ROS model creates, since
genROS emits 82 lattice points per deposition event where genROSOld emits one,
and it says the two implementations are interchangeable below the crossover.

    python3 benchmark_ros_assignment.py

Output
------
    ros_assignment_benchmark.json    sizes, timings, hit counts, machine metadata
"""

import json
import os
import platform
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, 'ros_assignment_benchmark.json')

#: Simulation box edge in microns; AMMPER uses a 64^3 cube throughout.
BOX = 64

#: Cell bounding half-width in microns. A cell is a sphere inscribed in a 4 um
#: cube, and both implementations test +/- 2 um about the cell center, so this
#: constant is the one both arms share and must not be varied between them.
HALF_WIDTH = 2

#: Cell counts at which the sweep is run. AMMPER starts from one cell and reaches
#: a few thousand by generation 15, so 64-1024 brackets the range over which the
#: damage assignment is called with a non-trivial population.
CELL_COUNTS = (64, 256, 1024)

#: ROS point counts. genROS emits 82 lattice points per deposition event and a
#: single 150 MeV proton track contributes about 6,900 events above threshold, so
#: the upper end of this sweep is well below a full-dose run: the point is to
#: locate the crossover and the scaling exponent, not to time a whole simulation.
ROS_COUNTS = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000)

#: Above this many ROS points the naive arm is not run. It is quadratic in the
#: product of the two sizes, so at 64,000 points and 1,024 cells it would take
#: tens of minutes for a number whose scaling is already established. The figure
#: shows the measured range and stops, rather than extrapolating.
NAIVE_LIMIT = 8000

#: Repeats per configuration; the reported time is the minimum, which is the
#: standard choice for timing because it is the measurement least contaminated by
#: unrelated load on the machine.
REPEATS = 3

SEED = 20260804


def naive_assign(ros, cells):
    """genROSOld / pre-optimization idiom: interpreted loop over both sets."""
    total = 0
    for i in range(len(ros)):
        px, py, pz = ros[i, 0], ros[i, 1], ros[i, 2]
        for cx, cy, cz in cells:
            if (px <= cx + HALF_WIDTH and px >= cx - HALF_WIDTH
                    and py <= cy + HALF_WIDTH and py >= cy - HALF_WIDTH
                    and pz <= cz + HALF_WIDTH and pz >= cz - HALF_WIDTH):
                total += 1
    return total


def filtered_assign(ros, cells):
    """cellDefinition.cellROS idiom: a vectorized filter per cell.

    The three successive filters and the emptiness short-circuits mirror the
    released code, including the fact that it narrows the frame progressively so
    that the second and third comparisons run on fewer rows than the first.
    """
    frame = pd.DataFrame(ros, columns=['Posx', 'Posy', 'Posz'])
    total = 0
    for cx, cy, cz in cells:
        sub = frame.loc[(frame['Posx'] <= cx + HALF_WIDTH)
                        & (frame['Posx'] >= cx - HALF_WIDTH)]
        if not sub.empty:
            sub = sub.loc[(sub['Posy'] <= cy + HALF_WIDTH)
                          & (sub['Posy'] >= cy - HALF_WIDTH)]
        if not sub.empty:
            sub = sub.loc[(sub['Posz'] <= cz + HALF_WIDTH)
                          & (sub['Posz'] >= cz - HALF_WIDTH)]
        total += len(sub)
    return total


def _time(function, *args):
    """Minimum wall-clock over REPEATS, with the result of the last call."""
    best, value = np.inf, None
    for _ in range(REPEATS):
        start = time.perf_counter()
        value = function(*args)
        best = min(best, time.perf_counter() - start)
    return best, value


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for n_cells in CELL_COUNTS:
        cells = rng.integers(0, BOX, size=(n_cells, 3))
        for n_ros in ROS_COUNTS:
            ros = rng.integers(0, BOX, size=(n_ros, 3)).astype(float)
            filtered_seconds, filtered_hits = _time(filtered_assign, ros, cells)
            if n_ros <= NAIVE_LIMIT:
                naive_seconds, naive_hits = _time(naive_assign, ros, cells)
                # The two arms must agree exactly, or the timing compares two
                # different computations. Assert rather than report, because a
                # mismatch invalidates the measurement rather than qualifying it.
                assert naive_hits == filtered_hits, (naive_hits, filtered_hits)
            else:
                naive_seconds, naive_hits = None, None
            rows.append({'n_cells': int(n_cells), 'n_ros': int(n_ros),
                         'hits': int(filtered_hits),
                         'naive_seconds': naive_seconds,
                         'filtered_seconds': filtered_seconds})
            speedup = ('--' if naive_seconds is None
                       else f'{naive_seconds / filtered_seconds:6.1f}x')
            naive_text = ('  skipped ' if naive_seconds is None
                          else f'{naive_seconds:9.3f}s')
            print(f'  cells {n_cells:5d}  ROS {n_ros:6d}   naive {naive_text}  '
                  f'filtered {filtered_seconds:7.3f}s   speedup {speedup}')

    report = {
        'seed': SEED, 'repeats': REPEATS, 'naive_limit': NAIVE_LIMIT,
        'box_microns': BOX, 'half_width_microns': HALF_WIDTH,
        'cell_counts': list(CELL_COUNTS), 'ros_counts': list(ROS_COUNTS),
        # Recorded because a wall-clock benchmark is only interpretable with the
        # machine attached to it. No timing metadata survives for the submitted
        # figure, which is part of why it could not be checked.
        'machine': {'platform': platform.platform(),
                    'processor': platform.processor(),
                    'python': platform.python_version(),
                    'numpy': np.__version__, 'pandas': pd.__version__},
        'rows': rows}
    with open(OUT_JSON, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nwrote {OUT_JSON}')
    return report


if __name__ == '__main__':
    main()
