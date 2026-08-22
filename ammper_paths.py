"""
Path resolution for the reorganized AMMPER tree (August 2026).

Before the reorganization every script ran from the repository root and referred
to data with bare relative paths ("abFinalPlots/...", "Results_Bulk_aB",
"radiationData/100/Track0/..."). Files now live in data/ and results/, so those
bare paths no longer resolve.

Import this module instead of hardcoding paths. It works no matter which
directory a script is launched from, because everything is resolved relative to
this file's own location rather than the process working directory:

    import ammper_paths as P
    df = pd.read_csv(P.ab_experimental("AlamarblueRawdataWTKGy.csv"))
    for root, dirs, files in os.walk(P.bulk_aB("WT_Basic_0")):
        ...

Importing this module also puts src/ and src/ammper/ on sys.path, so the bare
intra-package imports in the simulation code ("from cellDefinition import Cell")
keep working.

OLD -> NEW path map:
    abFinalPlots/                    -> data/experimental/alamarblue/
    aBGAMMA/                         -> data/experimental/alamarblue_gamma/
    Biosentineldata/                 -> data/experimental/biosentinel/
    radiationData/                   -> data/radiation_input/ritracks/
    *Fluence*_data.txt               -> data/fluence/
    Results_Bulk_aB/                 -> results/bulk_aB/
    Results_Bulk_GAMMAFINAL/         -> results/bulk_gamma/
    Results/                         -> results/single_runs/
    smac3_output/                    -> results/smac3_output/
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

SRC = os.path.join(ROOT, "src")
AMMPER_PKG = os.path.join(SRC, "ammper")

DATA = os.path.join(ROOT, "data")
EXPERIMENTAL = os.path.join(DATA, "experimental")
AB_EXPERIMENTAL = os.path.join(EXPERIMENTAL, "alamarblue")
AB_GAMMA_EXPERIMENTAL = os.path.join(EXPERIMENTAL, "alamarblue_gamma")
BIOSENTINEL = os.path.join(EXPERIMENTAL, "biosentinel")
RADIATION_INPUT = os.path.join(DATA, "radiation_input", "ritracks")
FLUENCE = os.path.join(DATA, "fluence")

RESULTS = os.path.join(ROOT, "results")
BULK_AB = os.path.join(RESULTS, "bulk_aB")
BULK_GAMMA = os.path.join(RESULTS, "bulk_gamma")
SINGLE_RUNS = os.path.join(RESULTS, "single_runs")
SMAC3_OUTPUT = os.path.join(RESULTS, "smac3_output")

FIGURES = os.path.join(ROOT, "figures")


def _join(base, *parts):
    return os.path.join(base, *parts) if parts else base


def ab_experimental(*parts):
    """alamarBlue plate-reader CSVs (proton experiments)."""
    return _join(AB_EXPERIMENTAL, *parts)


def ab_gamma_experimental(*parts):
    """alamarBlue plate-reader data for the gamma experiments."""
    return _join(AB_GAMMA_EXPERIMENTAL, *parts)


def biosentinel(*parts):
    """BioSentinel/LEIA source spreadsheets."""
    return _join(BIOSENTINEL, *parts)


def radiation_input(*parts):
    """RITRACKS-generated proton track data, keyed by proton energy."""
    return _join(RADIATION_INPUT, *parts)


def fluence(*parts):
    """Deep-space and GCRSim fluence tables."""
    return _join(FLUENCE, *parts)


def bulk_aB(*parts):
    """Simulation output for the alamarBlue proton runs."""
    return _join(BULK_AB, *parts)


def bulk_gamma(*parts):
    """Simulation output for the exploratory gamma runs."""
    return _join(BULK_GAMMA, *parts)


def single_runs(*parts):
    """Simulation output for individual (non-bulk) runs."""
    return _join(SINGLE_RUNS, *parts)


def figures(*parts):
    """Output directory for generated figures; created on demand."""
    target = _join(FIGURES, *parts)
    directory = target if not parts else os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return target


def add_src_to_path():
    """Make the bare intra-package simulation imports resolve."""
    for path in (SRC, AMMPER_PKG):
        if path not in sys.path:
            sys.path.insert(0, path)


add_src_to_path()
