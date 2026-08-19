# AMMPER-2 — organized source tree (August 2026)

Agent-Based Model for Microbial Populations Exposed to Radiation, version 2.
This branch (`organized_AMMPER_2_August_2026`) is a **reorganization of
`updated_figures`**: the same code and data, arranged so that the manuscript
figures can be regenerated from a known entry point, and so that the simulation
engine, the analysis scripts, the input data, and the simulation outputs are no
longer interleaved in a single flat directory.

No scientific behavior was changed in the reorganization commits. The figure
scripts reproduce their previous output (verified by re-running them and
comparing the rendered panels). Bug fixes live in **separate, later commits** so
they can be reviewed independently of the file moves — see `docs/BUGFIXES.md`.

## Layout

```
src/                        simulation
  AMMPER.py                   interactive entry point
  AMMPERCLI.py                command-line entry point
  AMMPERBulk_aB.py            batch runner used for the alamarBlue proton runs
  AMMPERBulk_GAMMAfinal.py    batch runner for the exploratory gamma runs
  AMMPERruns_aB.py            driver that loops AMMPERBulk_aB over doses
  AMMPERruns_GAMMAFINAL.py    driver for the gamma runs
  ammper/                     the model itself (imported by the above)
    cellDefinition.py           the Cell agent
    genTraverse_groundTesting.py  proton tracks, ground-test environments
    genTraverse_deepSpace.py      proton tracks, deep-space environment
    genROS.py                     ROS with diffusion and decay ("complex")
    genROSOld.py                  ROS static and eternal ("naive")
    genROSDiffusion.py            standalone diffusion experiments
    cellPlot.py                   per-generation visualization
    cellPlot_deepSpace.py         per-generation visualization, deep space
    GammaRadGen.py                exploratory gamma event generation

analysis/                   everything downstream of a simulation
  aB/                         alamarBlue kinetics model and figures
    ab_final_plots_panel.py       >>> MAIN TEXT FIGURE 2 (per-dose panels)
    stack_ab_figures.py           >>> MAIN TEXT FIGURE 2 (assembles the stack)
    aBFinalplotsSMAC.py           SMAC3 Bayesian Optimization parameter fit
    aBFinalplotsSMAC2.py          SMAC3 variant
    aBFinalplots.py               earlier single-figure version
    aBFinalplotsCombinedAnalysis.py  combined proton + gamma analysis
    aBFinalPlotsMaddie.py         collaborator variant
    AlamarBlueToy16Grid.py        manual Grid Search parameter fit
    AlamarBlueToy17_Statistical.py  statistical version of the toy model
    erroranalysis.py              Grid Search vs BO error comparison
  growth_curves/
    generate_growth_curves.py     growth curves from simulation output
    stack_growth_curves.py        >>> MAIN TEXT FIGURE 1 (assembles the panel)
  ros/ROSDiffusionLifetime.py   ROS half-life / diffusion analysis
  gamma/                        exploratory gamma analysis
  stats/STATS_PAPER.R           CLMM, Kruskal-Wallis, Wilcoxon (R 4.3.1)
  moreplots.py                  assorted supporting plots

gui/                        graphical interface (PySide/Qt) and its assets
data/
  experimental/               plate-reader and BioSentinel source data
    alamarblue/                 proton aB CSVs (mean and STD per dose)
    alamarblue_gamma/           gamma aB data
    biosentinel/                BioSentinel/LEIA spreadsheets
  radiation_input/ritracks/   RITRACKS track data, keyed by proton energy
  fluence/                    deep-space and GCRSim fluence tables
results/
  bulk_aB/                    simulation output, proton aB runs (was Results_Bulk_aB)
  bulk_gamma/                 simulation output, gamma runs
  single_runs/                individual run output (was Results)
  smac3_output/               SMAC3 optimizer run history
  figures_updated_figures_branch/  pre-rendered panel assets used by the compositors
  paper2024_revision_figures/ figures from the 2024 revision round
docs/                       notes, media, and BUGFIXES.md
figures/                    generated output (git-ignored, created on demand)
ammper_paths.py             path resolution — import this, don't hardcode paths
```

## Running it

Requires the packages in `requirements.txt` (numpy, pandas, scipy,
scikit-learn, matplotlib; `smac` only for the SMAC3 fitting scripts).

Scripts resolve their own paths relative to the repository, so they can be
launched from anywhere:

```bash
# one simulation: radType cellType ROSType dose outputFolder
#   radType  a=150 MeV Proton  b=GCRSim  c=Deep Space  d=Gamma
#   cellType a=wild type       b=rad51
#   ROSType  a=Basic (naive)   b=Complex (diffusion+decay)
python3 src/AMMPERBulk_aB.py a a a 2.5 WT_Basic_25

# main text Figure 1
python3 analysis/growth_curves/stack_growth_curves.py

# main text Figure 2 (per-dose panels first, then the stack)
python3 analysis/aB/ab_final_plots_panel.py
python3 analysis/aB/stack_ab_figures.py
```

Output lands in `figures/`. Simulation output lands in `results/bulk_aB/<name>/`.

Note that `src/AMMPERBulk_aB.py` expects the single-letter argument codes above,
not the expanded strings — passing `"150 MeV Proton"` fails with a `NameError`
on `N`, because the argument branches only match the letters. This is
pre-existing upstream behavior and was left as is.

## Paths

Use `ammper_paths` rather than literal relative paths:

```python
import ammper_paths as P
df = pd.read_csv(P.ab_experimental("AlamarblueRawdataWTKGy.csv"))
sim = P.bulk_aB("WT_Basic_0")
out = P.figures("my_panel.png")   # creates figures/ if needed
```

The old-to-new mapping is documented at the top of `ammper_paths.py`. Several
Windows absolute paths (`C:\Users\danie\...`) remain in the older analysis
scripts; they were already dead before the reorganization and are left
untouched so those scripts stay comparable with their published versions.
