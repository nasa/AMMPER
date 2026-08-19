"""
Extract the gamma alamarBlue plate-reader series from the source workbooks.

The proton analysis reads one CSV per strain and dose out of
data/experimental/alamarblue/, in the format

    Time,A570,A600,A690,A750          (no header, one row per hour)

with a matching ...STD.csv holding the standard deviation over the three
replicate wells. The gamma experiment was never exported that way: only a single
dose survived as a CSV (alamarblue_gamma/aBGAMMA1Gy.csv, wild type at 1 Gy), and
everything else lives inside the two analysis workbooks

    Fig1 - KA2-750-NSRL-19A gamma IR QC_wt_041119_analyzed072922 (1).xlsm
    Fig1 - KA2-750-NSRL-19A gamma IR QC_rad51_041119_analyzed072922.xlsm

which between them hold fifteen doses per strain -- 0, 0.01, 0.05, 0.1, 0.35,
0.5, 1, 2.5, 5, 10, 20, 30, 40, 50 and 60 Gy -- at 49 hourly timepoints, in four
channels. This script writes them out in the proton CSV format so the gamma fit
reads its data by exactly the same route as the proton fit.

The experimental values are not altered in any way: absorbances are copied out of
the workbook cells as they stand. Per-dose means come from the four
"<channel> nm" average sheets and the standard deviations are computed from the
three replicate columns on the corresponding "<channel> all" sheets.

Validation: WT 1 Gy is also present as an archived CSV, and the extraction is
checked against it cell by cell. That check runs on every invocation, so a
workbook layout change cannot pass silently.

    python3 extract_gamma_experimental.py            # extract and validate
    python3 extract_gamma_experimental.py --check    # validate only, write nothing

Output goes to data/experimental/alamarblue_gamma/, named to mirror the proton
files:

    AlamarblueGammaWT0Gy.csv        AlamarblueGammaWT0GySTD.csv
    AlamarblueGammarad5125Gy.csv    ...

with the dose written the way the proton files write it (25 for 2.5 Gy, and "K"
for the unirradiated control is NOT used here -- 0 Gy is written as 0Gy, since
the gamma controls are labelled by dose in the workbook).
"""

import argparse
import os
import re
import sys

import numpy as np
import openpyxl
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import ammper_paths as P  # noqa: E402

WORKBOOKS = {
    'WT': 'Fig1 - KA2-750-NSRL-19A gamma IR QC_wt_041119_analyzed072922 (1).xlsm',
    'rad51': 'Fig1 - KA2-750-NSRL-19A gamma IR QC_rad51_041119_analyzed072922.xlsm',
}

#: Workbook sheet holding the per-dose mean for each channel, and the sheet
#: holding the three replicates. The average sheets carry a title in row 1 and
#: the dose labels in row 2; the replicate sheets carry the dose label in row 2
#: (merged across its three columns) and the replicate number in row 3.
CHANNELS = [
    ('A570', '570 nm', '570 all'),
    ('A600', '600 nm', '600 all'),
    ('A690', '690 nm Growth', '690 all'),
    ('A750', '750 nm', '750 all'),
]

#: Doses present in both workbooks, in the order the plate was laid out.
DOSES = [0.0, 0.01, 0.05, 0.1, 0.35, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0,
         40.0, 50.0, 60.0]

REPLICATES = 3


def dose_tag(dose):
    """Filename fragment for a dose, following the proton naming convention.

    The proton files write 2.5 Gy as ``25Gy``, i.e. the dose in tenths with the
    decimal point dropped. Sub-Gray gamma doses would collide under that rule
    (0.05 and 0.5 both becoming ``05``), so fractional doses below 1 Gy keep
    their decimal point written as ``p``: 0.05 Gy -> ``0p05Gy``.
    """
    if dose >= 1 and float(dose).is_integer():
        return f'{int(dose)}Gy'
    if dose >= 1:
        return f'{dose:g}'.replace('.', '') + 'Gy'
    return f'{dose:g}'.replace('.', 'p') + 'Gy'


def csv_name(strain, dose, std=False):
    return (f'AlamarblueGamma{strain}{dose_tag(dose)}'
            f'{"STD" if std else ""}.csv')


def _normalize(label):
    """Comparable form of a workbook dose label.

    The labels are inconsistent in the source: a leading space on
    ``' WT 0.01 Gy'``, and the mutant written as ``rad51Δ`` with a literal
    Greek delta. Only the numeric part is needed to identify the column.
    """
    if label is None:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)\s*Gy', str(label))
    return float(match.group(1)) if match else None


def _dose_columns(header_row):
    """{dose: column index} for a row of dose labels (0-based into the row)."""
    out = {}
    for index, cell in enumerate(header_row):
        dose = _normalize(cell)
        if dose is not None and dose not in out:
            out[dose] = index
    return out


def read_means(workbook):
    """{dose: DataFrame[Time, A570, A600, A690, A750]} of per-dose means."""
    frames = {}
    for column_name, sheet_name, _ in CHANNELS:
        rows = list(workbook[sheet_name].iter_rows(values_only=True))
        columns = _dose_columns(rows[1])
        times = [row[0] for row in rows[2:] if row[0] is not None]
        for dose, index in columns.items():
            values = [row[index] for row in rows[2:] if row[0] is not None]
            frame = frames.setdefault(
                dose, pd.DataFrame({'Time': np.asarray(times, dtype=float)}))
            frame[column_name] = np.asarray(values, dtype=float)
    return frames


def read_stds(workbook):
    """{dose: DataFrame[Time, A570, ...]} of standard deviations over replicates.

    On the "all replicates" sheets the dose label sits above the first of its
    three columns and the other two are blank (a merged cell), so the three
    replicate columns of a dose are the label column and the two after it.
    """
    frames = {}
    for column_name, _, sheet_name in CHANNELS:
        rows = list(workbook[sheet_name].iter_rows(values_only=True))
        columns = _dose_columns(rows[1])
        data_rows = [row for row in rows[3:] if row[0] is not None]
        times = [row[0] for row in data_rows]
        for dose, index in columns.items():
            block = np.array([[row[index + r] for r in range(REPLICATES)]
                              for row in data_rows], dtype=float)
            frame = frames.setdefault(
                dose, pd.DataFrame({'Time': np.asarray(times, dtype=float)}))
            # Sample standard deviation over the three wells, matching the
            # proton STD files (which numpy reproduces with ddof=1).
            frame[column_name] = block.std(axis=1, ddof=1)
    return frames


def validate_against_archive(means):
    """Check the WT 1 Gy extraction against the archived CSV, column by column.

    aBGAMMA1Gy.csv is the one gamma dose that was exported at the time, so it is
    the only external check available on this extraction. If it agrees to
    floating-point tolerance in all four channels, the workbook layout has been
    read correctly.
    """
    archive = pd.read_csv(P.ab_gamma_experimental('aBGAMMA1Gy.csv'),
                          names=['Time', 'A570', 'A600', 'A690', 'A750'])
    extracted = means[1.0]
    if len(archive) != len(extracted):
        return False, (f'row count differs: archive {len(archive)}, '
                       f'extracted {len(extracted)}')
    for column in ('A570', 'A600', 'A690', 'A750'):
        if not np.allclose(archive[column].to_numpy(),
                           extracted[column].to_numpy(), atol=1e-9):
            worst = np.max(np.abs(archive[column].to_numpy()
                                  - extracted[column].to_numpy()))
            return False, f'{column} differs, largest deviation {worst:.2e}'
    return True, f'all four channels agree over {len(archive)} timepoints'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='validate the extraction without writing CSVs')
    arguments = parser.parse_args()

    out_dir = P.ab_gamma_experimental()
    written = 0
    for strain, workbook_name in WORKBOOKS.items():
        path = os.path.join(P.EXPERIMENTAL, workbook_name)
        workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
        means = read_means(workbook)
        stds = read_stds(workbook)
        workbook.close()

        missing = [d for d in DOSES if d not in means]
        if missing:
            print(f'FAIL {strain}: doses absent from the workbook: {missing}')
            return 1

        if strain == 'WT':
            ok, detail = validate_against_archive(means)
            print(f'{"OK  " if ok else "FAIL"} WT 1 Gy vs archived '
                  f'aBGAMMA1Gy.csv: {detail}')
            if not ok:
                return 1

        if arguments.check:
            print(f'     {strain}: {len(means)} doses readable, not written')
            continue

        for dose in DOSES:
            for frame, std in ((means[dose], False), (stds[dose], True)):
                target = os.path.join(out_dir, csv_name(strain, dose, std))
                frame.to_csv(target, header=False, index=False)
                written += 1

    print(f'\n{written} CSV file(s) written to {out_dir}'
          if written else '\ncheck only, nothing written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
