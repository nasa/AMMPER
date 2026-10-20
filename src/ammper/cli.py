"""
The `ammper` command-line interface.

Wraps the existing entry-point scripts
(`AMMPERCLI.py`, `AMMPERBulk_aB.py`) and the PyQt5 GUI (`gui/AMMPERGUI.py`)
unchanged, using `runpy` to execute them exactly as if invoked directly with
`python path/to/script.py args...`. 

@author: dannyofmiami
"""

import argparse
import difflib
import importlib.metadata
import os
import re
import runpy
import sys
import time

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.traceback import install as _install_rich_traceback

from ammper import data_check, data_fetch
from ammper import paths as P

_install_rich_traceback(show_locals=False)

# 🪱 a must <3.
BANNER = r"""
    _    __  __ __  __ ____  _____ ____
   / \  |  \/  |  \/  |  _ \| ____|  _ \
  / _ \ | |\/| | |\/| | |_) |  _| | |_) |
 / ___ \| |  | | |  | |  __/| |___|  _ <
/_/   \_\_|  |_|_|  |_|_|   |_____|_| \_\
"""

CELL_TYPES = {"wt": "a", "rad51": "b"}
# "complex" is the diffusion/decay ROS model. Above ~10 Gy it needs more RAM than many machines
# have, so runs it cannot hold trigger a warning (ammper.preflight); it is never blocked.
ROS_TYPES = {"basic": "a", "complex": "b"}


def _console():
    return Console()


def _use_utf8_output():
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding == "utf8" or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _print_banner(console):
    try:
        version = importlib.metadata.version("ammper")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    console.print(BANNER, style="bold cyan", highlight=False)
    console.print(
        f"[bold]NASA AMMPER[/bold] v{version} — "
        "Agent-Based Model for Microbial Populations Exposed to Radiation\n"
    )


def _run_module_as_main(module_name, argv):
    """Execute an ammper.* driver script exactly as `python -m` would."""
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = old_argv


def cmd_run(args):
    """Launch the existing interactive single-run CLI (AMMPERCLI.py)."""
    console = _console()
    _print_banner(console)
    console.print("[cyan]Starting interactive simulation setup...[/cyan]\n")
    _run_module_as_main("ammper.AMMPERCLI", ["AMMPERCLI.py"])
    console.print("\n[bold green][✔] Simulation complete.[/bold green]")
    return 0


def cmd_quickstart(args):
    """Run non-interactive tutorial simulation with defaults."""
    console = _console()
    _print_banner(console)

    cell_code = CELL_TYPES[args.cell_type]
    ros_code = ROS_TYPES[args.ros_type]
    dose = str(args.dose)
    folder = args.name or "quickstart"

    console.print(
        f"[cyan]Running tutorial simulation[/cyan] "
        f"(150 MeV Proton, cell type={args.cell_type}, ROS model={args.ros_type}, "
        f"dose={dose} Gy)...\n"
    )

    if args.ros_type == "complex":
        from ammper import preflight
        warning = preflight.complex_ros_warning(preflight.PROTON, float(args.dose))
        if warning:
            console.print(f"[bold yellow][!] Warning:[/bold yellow] {warning}\n")

    start = time.time()
    _run_module_as_main(
        "ammper.AMMPERBulk_aB",
        ["AMMPERBulk_aB.py", "a", cell_code, ros_code, dose, folder],
    )
    elapsed = time.time() - start

    console.print(
        f"\n[bold green][✔] Quickstart complete[/bold green] "
        f"({elapsed:.1f}s)."
    )
    console.print(f"Results written to: [bold]{P.bulk_aB(folder)}[/bold]")
    console.print("[dim]Next: try `ammper gui` to visualize results, or `ammper run` for a full interactive simulation.[/dim]")
    return 0


def cmd_gui(args):
    """Launch the PyQt5 desktop GUI."""
    console = _console()
    _print_banner(console)

    # The GUI ships inside the package (ammper/gui), with its fonts and images, so this works from
    # a clone and from an installed wheel, from any working directory.
    gui_script = os.path.join(P.GUI_DIR, "AMMPERGUI.py")
    if not os.path.isfile(gui_script):
        console.print(f"[bold red][x] GUI script not found at {gui_script}[/bold red]")
        return 1

    console.print("[cyan]Launching AMMPER GUI...[/cyan]")
    old_argv = sys.argv
    sys.argv = [gui_script]
    try:
        runpy.run_module("ammper.gui.AMMPERGUI", run_name="__main__", alter_sys=True)
    finally:
        sys.argv = old_argv
    return 0


def _check(console, table, label, ok, detail=""):
    mark = "[bold green]✔[/bold green]" if ok else "[bold red]x[/bold red]"
    table.add_row(mark, label, detail)
    return ok


def _machine_check(console):
    """This machine's memory vs. Complex ROS run sizes. Informational: never affects the exit status."""
    from ammper import preflight

    specs = preflight.detect_machine()
    console.print("\n[bold]Machine check[/bold] (memory needed by Complex ROS runs)")
    ram = (f"{specs.total_ram_gb:.1f} GB total, {specs.available_ram_gb:.1f} GB free"
           if specs.available_ram_gb is not None else "could not be detected")
    console.print(f"  {specs.system} ({specs.machine}), {specs.cpu_count} CPU cores; memory: {ram}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Complex ROS run")
    table.add_column("Peak memory", justify="right")
    table.add_column("Est. time*", justify="right")
    table.add_column("This machine")
    styles = {"ok": "[green]OK[/green]", "warn": "[yellow]caution[/yellow]",
              "block": "[bold red]too large[/bold red]"}
    levels = []
    for rad_type, dose, est, level in preflight.complex_ros_table(specs):
        levels.append(level)
        table.add_row(f"{rad_type} {dose:g} Gy", f"{est.peak_gb:.1f} GB",
                      f"{est.minutes:.1f} min", styles[level])
    console.print(table)
    console.print("[dim]* measured on an Apple M-series chip; slower CPUs take longer. Memory is "
                  "the limit that crashes machines.[/dim]")

    if specs.available_ram_gb is None:
        console.print("[yellow][!] Free memory could not be detected, so Complex ROS runs are "
                      "unchecked. Keep at least 1 GB of free RAM per Gy.[/yellow]")
    elif any(level != "ok" for level in levels):
        ceiling = preflight.max_safe_proton_dose(specs)
        limit = (f"above {ceiling:g} Gy" if ceiling else "of any dose")
        console.print(f"[bold yellow][!] Warning:[/bold yellow] this machine does not have the "
                      f"free memory for every Complex ROS run. Complex ROS runs {limit} may "
                      "stall, swap or crash it. Close other applications, use a lower dose, "
                      "or use Basic ROS.")
    else:
        console.print("[green][✔] This machine can run every Complex ROS run listed.[/green]")


def cmd_setup(args):
    """Check that the environment is ready to run simulations."""
    console = _console()
    _print_banner(console)

    table = Table(show_header=True, header_style="bold")
    table.add_column("", width=3)
    table.add_column("Check")
    table.add_column("Detail")

    all_ok = True

    for module_name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "PyQt5", "rich"):
        try:
            importlib.import_module(module_name)
            ok = True
            detail = "installed"
        except ImportError as exc:
            ok = False
            detail = str(exc)
        all_ok &= _check(console, table, f"dependency: {module_name}", ok, detail)

    data_dirs = {"fluence tables": P.FLUENCE}
    if P.IN_CLONE:
        # manuscript-analysis inputs: only present in a clone of the repository
        data_dirs.update({
            "alamarBlue experimental data": P.AB_EXPERIMENTAL,
            "BioSentinel experimental data": P.BIOSENTINEL,
        })
    for label, path in data_dirs.items():
        exists = os.path.isdir(path) and bool(os.listdir(path))
        all_ok &= _check(console, table, label, exists, path)

    # RITRACKS is split: the bundled 150 MeV Proton tutorial subset is
    # required for a healthy baseline install (quickstart/150 MeV Proton/
    # Gamma all need only it); the rest of the dataset (GCRSim, Deep Space)
    # is fetched separately via `ammper data pull` and does NOT gate
    # all_ok here -- a fresh clone without it is still a working install.
    tutorial_path = P.radiation_input("150")
    tutorial_ok = os.path.isdir(tutorial_path) and bool(os.listdir(tutorial_path))
    all_ok &= _check(
        console, table, "RITRACKS tutorial data (150 MeV Proton)",
        tutorial_ok, tutorial_path,
    )

    # Informational (does not gate all_ok): can every radiation type run on this machine?
    blocked = {radType: missing for radType, missing in data_check.missing_by_radiation_type().items()
               if missing}
    if blocked:
        detail = "; ".join(f"{radType}: run `ammper data pull {data_check.PULL_TARGETS[radType]}`"
                           for radType in blocked)
    else:
        detail = "all four radiation types can run"
    _check(console, table, "RITRACKS for every radiation type", not blocked, detail)

    try:
        home = P.ensure_writable_home()
        figures_dir = P.figures()  # creates FIGURES/ itself if missing
        writable = os.access(home, os.W_OK) and os.access(figures_dir, os.W_OK)
        detail = home + ("  (this clone)" if P.IN_CLONE else "  (set AMMPER_HOME to change)")
    except OSError as exc:
        writable = False
        detail = str(exc)
    all_ok &= _check(console, table, "results and downloads folder is writable", writable, detail)

    console.print(table)

    _machine_check(console)

    if all_ok:
        console.print(
            "\n[bold green][✔] Environment looks good.[/bold green] "
            "Try [bold]ammper quickstart[/bold] to run a tutorial simulation."
        )
        return 0
    else:
        console.print(
            "\n[bold red][x] Some checks failed.[/bold red] "
            "See CONTRIBUTING.md for setup instructions, or re-run "
            r'`pip install -e ".\[dev]"` from the repository root.'
        )
        return 1


def _pull_shortcuts(manifest):
    """The `ammper data pull ...` commands for the run types that need extra folders, with
    download sizes taken from the manifest (so they cannot drift from the Zenodo record)."""
    parts = []
    for target, radType in (("gcrsim", "GCRSim"), ("deep-space", "Deep Space")):
        megabytes = data_fetch.download_size(
            data_check.required_ritracks_energies(radType), manifest) / 1e6
        parts.append(f"`ammper data pull {target}`" + (f" (~{megabytes:.0f} MB)" if megabytes else ""))
    gigabytes = data_fetch.download_size(manifest["full_energies"], manifest) / 1e9
    parts.append("`ammper data pull` (everything" + (f", ~{gigabytes:.1f} GB" if gigabytes else "") + ")")
    return ", ".join(parts[:-1]) + " or " + parts[-1]


def cmd_data_status(args):
    """Show which RITRACKS energy folders are present locally."""
    console = _console()
    _print_banner(console)

    manifest = P.load_ritracks_manifest()
    bundled = set(manifest["tutorial_energies"])
    needed_by = data_check.preset_energies()
    results = data_fetch.status()

    table = Table(show_header=True, header_style="bold")
    table.add_column("", width=3)
    table.add_column("Energy (MeV)")
    table.add_column("Used by")
    for name in manifest["full_energies"]:
        users = [radType for radType, energies in needed_by.items() if name in energies]
        if name in bundled:
            used_by = "bundled in the package; " + ", ".join(users)
        else:
            used_by = ", ".join(users) if users else "optional: no radiation type reads it"
        if not users and not results[name]:
            table.add_row("[dim]·[/dim]", name, f"[dim]{used_by}[/dim]")   # optional: not a problem
        else:
            _check(console, table, name, results[name], used_by)
    console.print(table)

    present, total = sum(results.values()), len(results)
    console.print(f"\n{present}/{total} RITRACKS energy folders present.")
    blocked = {radType: missing for radType, missing in data_check.missing_by_radiation_type().items()
               if missing}
    if not blocked:
        console.print("[green]All four radiation types (150 MeV Proton, GCRSim, Deep Space, "
                      "Gamma) can run with the data on this machine.[/green]")
    for radType, missing in blocked.items():
        console.print(f"[yellow]{radType} cannot run yet:[/yellow] {data_check.pull_hint(radType, missing)}")
    optional_missing = [name for name in data_check.optional_energies(manifest) if not results[name]]
    if optional_missing:
        megabytes = data_fetch.download_size(optional_missing, manifest) / 1e6
        console.print(
            f"[dim]{len(optional_missing)} optional folder(s) (~{megabytes:.0f} MB) are not read by any "
            "radiation type; `ammper data pull` fetches everything that is missing.[/dim]"
        )
    return 0


def cmd_data_pull(args):
    """Download RITRACKS energy folders from Zenodo (everything, or just what a run type needs)."""
    console = _console()
    _print_banner(console)
    try:
        pulled = data_fetch.pull(console, args.targets, force=args.force)
    except data_fetch.UnknownTargetError as exc:
        console.print(f"\n[bold red][x] {exc}[/bold red]")
        return 2
    except (data_fetch.DatasetNotPublishedError, data_fetch.ChecksumMismatchError) as exc:
        console.print(f"\n[bold red][x] {exc}[/bold red]")
        return 1
    except OSError as exc:
        console.print(f"\n[bold red][x] {exc}[/bold red]")
        console.print("[dim]Check your network connection and try again; folders that "
                      "finished are kept, so re-running continues where it stopped.[/dim]")
        return 1

    if pulled:
        console.print(
            f"\n[bold green][✔] {len(pulled)} RITRACKS folder(s) ready.[/bold green] "
            "Run `ammper data verify` to confirm."
        )
    return 0


def cmd_data_verify(args):
    """Verify the on-disk RITRACKS dataset against the manifest."""
    console = _console()
    _print_banner(console)

    results = data_fetch.verify()
    missing = [name for name, ok in results.items() if not ok]

    if not missing:
        console.print(
            f"[bold green][✔] All {len(results)} RITRACKS energy folders "
            "present and non-empty.[/bold green]"
        )
        return 0

    console.print(
        f"[bold red][x] {len(missing)} of {len(results)} RITRACKS energy "
        "folders missing or empty:[/bold red]"
    )
    console.print(", ".join(missing))
    console.print("[dim]Run `ammper data pull` (or `ammper data pull gcrsim` / "
                  "`deep-space`) to fetch them.[/dim]")
    return 1


def cmd_data(args):
    """`ammper data` with no further subcommand: show its help."""
    args.data_parser.print_help()
    return 0


EPILOG = """\
Examples:
  ammper setup                          check your install is ready to go
  ammper quickstart                     run the ~5s tutorial simulation
  ammper quickstart --dose 2.5          tutorial run at 2.5 Gy instead of 0
  ammper run                            full interactive simulation (prompts you for options)
  ammper gui                            launch the desktop GUI
  ammper data status                    show which RITRACKS energies are present locally
  ammper data pull                      fetch the full RITRACKS dataset from Zenodo (~3.5GB)

Notes:
  `ammper quickstart`, `150 MeV Proton`, and `Gamma` runs work out of the box.
  `GCRSim` and `Deep Space` need the full dataset -- run `ammper data pull` first.

  --debug and --version are global flags: put them before the subcommand,
  e.g. `ammper --debug run`, not `ammper run --debug`.

New to AMMPER? Start with `ammper setup`, then `ammper quickstart`.
"""


_INVALID_CHOICE_RE = re.compile(r"invalid choice: '([^']+)' \(choose from ((?:'[^']+'(?:, )?)+)\)")


class _BrandedArgumentParser(argparse.ArgumentParser):
    """Shows the AMMPER banner whenever help is printed

    A typo like `ammper quickstrat` would otherwise produce
    `invalid choice: 'quickstrat' (choose from 'setup', 'quickstart', ...)`
    —  catch this with a "did you mean...?" suggestion; `error()` does the 
    same thing here for both bad subcommands
    (`quickstrat` -> `quickstart`) and bad option values
    (`--cell-type mutant` -> did you mean `wt`?).
    """

    def print_help(self, file=None):
        _print_banner(_console())
        super().print_help(file)

    def error(self, message):
        console = _console()

        match = _INVALID_CHOICE_RE.search(message)
        if match:
            bad_value, raw_choices = match.groups()
            choices = re.findall(r"'([^']+)'", raw_choices)
            suggestion = difflib.get_close_matches(bad_value, choices, n=1)
            if suggestion:
                console.print(
                    f"[bold red][x] Unknown option '{bad_value}'.[/bold red] "
                    f"Did you mean [bold]'{suggestion[0]}'[/bold]?"
                )
            else:
                console.print(
                    f"[bold red][x] Unknown option '{bad_value}'.[/bold red] "
                    f"Choose from: {', '.join(choices)}"
                )
            self.exit(2)

        if message.startswith("unrecognized arguments:"):
            console.print(f"[bold red][x] {message}[/bold red]")
            console.print(f"[dim]Run `{self.prog} --help` to see the available options.[/dim]")
            self.exit(2)

        super().error(message)


def build_parser():
    parser = _BrandedArgumentParser(
        prog="ammper",
        description="NASA AMMPER — Agent-Based Model for Microbial Populations "
        "Exposed to Radiation.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"ammper {_safe_version()}",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="show the full traceback instead of a short error message",
    )
    subparsers = parser.add_subparsers(
        dest="command", metavar="{setup,quickstart,run,gui,data}",
        parser_class=_BrandedArgumentParser,
    )

    p_setup = subparsers.add_parser(
        "setup",
        help="check that dependencies and data are ready to run simulations",
        description="Check that dependencies and data are ready to run simulations.",
    )
    p_setup.set_defaults(func=cmd_setup)

    p_quickstart = subparsers.add_parser(
        "quickstart",
        help="run a short, non-interactive tutorial simulation",
        description="Run a short (~5s at the default 0 Gy dose), non-interactive "
        "tutorial simulation — a wild-type cell population under 150 MeV proton "
        "radiation with the basic ROS model, unless overridden below.",
    )
    p_quickstart.add_argument(
        "--dose", type=float, default=0.0,
        help="radiation dose in Gy (default: 0, the fastest baseline run)",
    )
    p_quickstart.add_argument(
        "--cell-type", type=str.lower, choices=sorted(CELL_TYPES), default="wt",
        help="cell strain to simulate (default: wt)",
    )
    p_quickstart.add_argument(
        "--ros-type", type=str.lower, choices=sorted(ROS_TYPES), default="basic",
        help="reactive-oxygen-species model (default: basic)",
    )
    p_quickstart.add_argument(
        "--name", default=None,
        help="results subfolder name under results/bulk_aB/ (default: quickstart)",
    )
    p_quickstart.set_defaults(func=cmd_quickstart)

    p_run = subparsers.add_parser(
        "run",
        help="run a full interactive simulation (prompts for options)",
        description="Run a full simulation. You'll be prompted for radiation "
        "type, cell type, ROS model, and dose.",
    )
    p_run.set_defaults(func=cmd_run)

    p_gui = subparsers.add_parser(
        "gui",
        help="launch the AMMPER desktop GUI",
        description="Launch the PyQt5 desktop GUI.",
    )
    p_gui.set_defaults(func=cmd_gui)

    p_data = subparsers.add_parser(
        "data",
        help="manage the RITRACKS dataset (tutorial subset vs. downloads from Zenodo)",
        description="Manage the RITRACKS dataset. 150 MeV Proton and Gamma need only the "
        "150 MeV tutorial folder (~11MB). GCRSim (~119MB download) and Deep Space "
        "(~193MB download) need extra folders; the whole dataset is ~1.3GB.",
    )
    p_data.set_defaults(func=cmd_data, data_parser=p_data)
    data_subparsers = p_data.add_subparsers(
        dest="data_command", metavar="{status,pull,verify}",
        parser_class=_BrandedArgumentParser,
    )

    p_data_status = data_subparsers.add_parser(
        "status",
        help="show which RITRACKS energy folders are present locally",
        description="Show which RITRACKS energy folders are present locally, and "
        "which are bundled tutorial data vs. fetched from Zenodo.",
    )
    p_data_status.set_defaults(func=cmd_data_status)

    p_data_pull = data_subparsers.add_parser(
        "pull",
        help="download RITRACKS folders from Zenodo (all, or just what a run type needs)",
        description="Download RITRACKS energy folders from the pinned Zenodo record, check "
        "each against the manifest's size and sha256, and extract them into "
        "data/radiation_input/ritracks/. Folders you already have are skipped.",
    )
    p_data_pull.add_argument(
        "targets", nargs="*", metavar="TARGET",
        help="gcrsim, deep-space, proton, a proton energy such as 20, or all "
        "(default: all)",
    )
    p_data_pull.add_argument(
        "--force", action="store_true", help="download again even if a folder is already present",
    )
    p_data_pull.set_defaults(func=cmd_data_pull)

    p_data_verify = data_subparsers.add_parser(
        "verify",
        help="verify the on-disk RITRACKS dataset against the manifest",
        description="Check every RITRACKS energy folder in the manifest is "
        "present and non-empty on disk.",
    )
    p_data_verify.set_defaults(func=cmd_data_verify)

    return parser


def _safe_version():
    try:
        return importlib.metadata.version("ammper")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _interactive_menu(parser, console):
    """A guided prompt for `ammper` run with no arguments and a real terminal
    attached — friendlier for a first-time user than a static help dump."""
    _print_banner(console)
    console.print("What would you like to do?\n")
    console.print("  [bold]setup[/bold]       check your install is ready to go")
    console.print("  [bold]quickstart[/bold]  run the ~5s tutorial simulation")
    console.print("  [bold]run[/bold]         full interactive simulation")
    console.print("  [bold]gui[/bold]         launch the desktop GUI")
    console.print("  [bold]data[/bold]        manage the RITRACKS dataset (fetch the full ~3.5GB set)")
    console.print("  [bold]help[/bold]        show the full --help text")
    console.print()

    choice = Prompt.ask(
        "Enter a command",
        choices=["setup", "quickstart", "run", "gui", "data", "help", "quit"],
        default="setup",
        console=console,
    )

    if choice == "quit":
        return 0
    if choice == "help":
        parser.print_help()
        return 0

    args = parser.parse_args([choice])
    return args.func(args)


def main(argv=None):
    _use_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if not getattr(args, "command", None):
            if sys.stdin.isatty():
                return _interactive_menu(parser, _console())
            parser.print_help()
            return 0
        return args.func(args)
    except (KeyboardInterrupt, EOFError):
        _console().print("\n[yellow]Interrupted.[/yellow]")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level UX guard, not a bug swallow
        if args.debug:
            raise
        console = _console()
        console.print(f"\n[bold red][x] {type(exc).__name__}: {exc}[/bold red]")
        console.print("[dim]Re-run with --debug to see the full traceback.[/dim]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
