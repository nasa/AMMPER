# AMMPER-2

**Agent-Based Model for Microbial Populations Exposed to Radiation**

AMMPER-2 is a research simulation for studying how ionizing radiation affects
microbial populations. It models cell growth, direct radiation damage, reactive
oxygen species (ROS), and DNA repair on a three-dimensional lattice. The model
supports wild-type and `rad51` yeast phenotypes, proton and gamma exposures,
ground-test and deep-space environments, and both basic static and
diffusion-and-decay ROS treatments.

The repository contains the simulation engine, command-line and graphical
interfaces, bundled RITRACKS radiation-track inputs, experimental data,
analysis code, and scripts used to reproduce manuscript figures. AMMPER is
research software and is not intended for clinical or operational radiation
risk decisions.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/` | Simulation entry points and the `ammper` model modules |
| `gui/` | PyQt5 graphical interface and GUI assets |
| `data/` | Experimental data, fluence tables, and radiation-track inputs |
| `analysis/` | Growth-curve, alamarBlue, ROS, gamma, and statistical analyses |
| `results/` | Archived simulation results and figure source assets |
| `figures/` | Generated publication figures |
| `revisions_2026/` | Manuscript-revision code, figures, and source files |
| `ammper_paths.py` | Repository-relative path helpers |

## Installation

AMMPER's supplied environment targets Python 3.10. The versions in
`requirements.txt` are used for the main simulation and GUI.
> continue to next section for Apple Silicon Mac install

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/nasa/AMMPER.git
   cd AMMPER
   ```

2. Create and activate a virtual environment:

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

Run commands from the repository root. Scripts use `ammper_paths.py` to find
bundled inputs and output directories independent of the clone location.

### Apple Silicon Macs:

If you are installing on an Apple Silicon Mac, **`pip install -r requirements.txt` will fail or hang indefinitely while installing `PyQt5==5.15.9`**. This is due to the `PyQt5-Qt5` binary dependency does not incldue a native `arm64` wheel on PyPI, forcing `pip` to compile it from source. This hangs on a license prompt that `pip` hides from the terminal. 

A fix is to use **Conda-Forge** to install a pre-compiled, native `arm64` binary of PyQt5, and use `pip` only for the remaining pure-Python dependencies.

1. **Install Miniconda** (if you do not already have it):

```bash 
brew install --cask miniconda 

conda init zsh
```
Close and reopen your terminal after this step so the Conda configuration loads.

2. **Use the free Conda-Forge channel.** By default, Conda uses Anaconda's commercial repository, which enforces strict rate limits. Run this once to permanently switch to the free, unrestricted community channel:

```bash 
echo "channels:" > ~/.condarc
echo "  - conda-forge" >> ~/.condarc
echo "channel_priority: strict" >> ~/.condarc
conda clean --all --yes
```

3. **Create a dedicated environment for AMMPER**, forcing Conda-Forge with `--override-channels` :

```bash 
conda create --name ammper python=3.10 -y --override-channels -c conda-forge
conda activate ammper
```

4. **Install PyQt5 as a pre-compiled binary from Conda-Forge:**

```bash
conda install pyqt=5.15.9 -y --override-channels -c conda-forge
```

5. **Install the remaining dependencies with `pip`**, skipping the `PyQt5` line since Conda is now managing it:

```bash
pip install -r <(grep -v "PyQt5" requirements.txt)
```

6. **Verify the installation:**

```bash
python -c "import PyQt5; print('PyQt5 successfully imported!')"
```

> **Note:** Once this Conda environment is set up, use `conda activate ammper` instead of `source .venv/bin/activate` for all future work on this repository.
 If you = use `pyenv`, it can silently override Conda's Python run `which python` to confirm it resolves inside the `ammper` environment.

### Dependencies

| Dependency | Version | Purpose |
| Matplotlib | 3.7.2 | Plotting and figure generation |
| --- | ---: | --- |
| MoviePy | 1.0.3 | GUI video generation | 
| NumPy | 1.25.2 | Arrays and numerical simulation |
| pandas | 2.1.0 | Experimental and simulation data handling |
| PyQt5 | 5.15.9 | Graphical interface |
| scikit-learn | 1.3.0 | Data splitting and analysis utilities |
| SciPy | 1.11.2 | Scientific calculations and ROS distributions |

FFmpeg is also needed to export videos through MoviePy. Some specialist or
legacy analysis scripts have dependencies not installed by
`requirements.txt`, including SMAC/ConfigSpace, OpenPyXL, statsmodels,
pingouin, COBRApy, and R packages. Inspect the imports in the particular script
before running it. The core simulation and figure commands below use the pinned
requirements.

## Usage

### Interactive command-line simulation

Start the prompt-driven interface:

```bash
python src/AMMPERCLI.py
```

The program asks for the radiation environment, dose where applicable, cell
type, and ROS model. Interactive runs write their description, cell-state
data, and plots beneath a timestamped `Results/` directory.

### Scripted simulation

For a non-interactive proton run:

```bash
python src/AMMPERBulk_aB.py a a a 2.5 WT_Basic_25
```

The five positional arguments are:

1. radiation: `a` = 150 MeV proton, `b` = GCRSim, `c` = deep space,
   `d` = gamma;
2. cell type: `a` = wild type, `b` = `rad51`;
3. ROS model: `a` = basic, `b` = diffusion and decay;
4. dose in Gy (proton mode supports `0`, `2.5`, `5`, `10`, `20`,
   and `30`); and
5. output-group name.

This example writes timestamped output under
`results/bulk_aB/WT_Basic_25/`. The bulk runner intentionally waits 61 seconds
at the end to prevent timestamp collisions. Pass the single-letter codes shown
above; expanded names are not accepted.


### Graphical interface

```bash
python gui/AMMPERGUI.py
```

A desktop session is required. Video export also requires FFmpeg on the system
path.

### Reproduce the main figure panels

The repository includes the required archived output and panel assets:

```bash
python analysis/growth_curves/stack_growth_curves.py
python analysis/aB/ab_final_plots_panel.py
python analysis/aB/stack_ab_figures.py
```

Generated PDF, PNG, and SVG files are written to `figures/`. The manuscript
and revision-specific reproduction scripts are in `revisions_2026/`; those
scripts may require the optional dependencies noted above.

## Contributing

Contributions that improve correctness, reproducibility, documentation, or
usability are welcome.

1. Open an issue describing the bug or proposed change. For model changes,
   explain the scientific rationale and expected effect on results.
2. Fork the repository, create a focused branch, and keep unrelated changes in
   separate commits.
3. Use four-space indentation, descriptive names, docstrings for reusable
   functions, and repository-relative paths through `ammper_paths.py`. Do not
   introduce machine-specific absolute paths.
4. Update documentation and dependency declarations when setup or behavior
   changes. Do not commit local environments, caches, or newly generated bulk
   results unless they are required reference data.
5. Submit a pull request summarizing the change and validation commands.
   Identify altered numerical output or regenerated figures, and include
   before-and-after output when scientific results change.

## License
This software is released under the **NASA Open Source Agreement (NOSA) Version 1.3**. Reference Number ARC-18739-1

A copy of the full license text should be included in the `LICENSE` file of this repository. You can also view the official terms online at the [Open Source Initiative (OSI)](https://opensource.org). 

## Contact

For scientific or project questions, contact the manuscript's corresponding
author, **Jessica Lee**, at **jessica.a.lee@nasa.gov**.

Additional project contacts:
- **Daniel Palacios** — [Daniel.Palacios@bcm.edu](mailto:Daniel.Palacios@bcm.edu)
- **Pramesh Sharma** — [prameshsharma25@gmail.com](mailto:prameshsharma25@gmail.com)

For bug reports, feature requests, and contribution proposals, use the
[GitHub issue tracker](https://github.com/nasa/AMMPER/issues) so discussion and
resolution remain visible to the project team.
