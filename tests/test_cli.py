"""
Adding Tests to AMMPER project
@author: dannyofmiami
"""

import os

import pytest

from ammper import cli


def test_no_command_prints_help_and_returns_0(capsys):
    exit_code = cli.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "usage: ammper" in captured.out


def test_version_flag_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0


def test_quickstart_maps_flags_to_the_legacy_argv_contract(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "_run_module_as_main", lambda module_name, argv: calls.append((module_name, argv))
    )

    exit_code = cli.main(
        ["quickstart", "--dose", "2.5", "--cell-type", "rad51", "--ros-type", "complex",
         "--name", "my_run"]
    )

    assert exit_code == 0
    assert len(calls) == 1
    module_name, argv = calls[0]
    assert module_name == "ammper.AMMPERBulk_aB"
    # radType, cellType, ROSType, dose, folder -- see README's argument contract
    assert argv == ["AMMPERBulk_aB.py", "a", "b", "b", "2.5", "my_run"]


def _quickstart_complex(monkeypatch, capsys, free_gb, dose):
    from ammper import preflight

    calls = []
    monkeypatch.setattr(
        cli, "_run_module_as_main", lambda module_name, argv: calls.append((module_name, argv))
    )
    monkeypatch.setattr(
        preflight, "detect_machine",
        lambda: preflight.MachineSpecs("Linux", "x86_64", 8, free_gb, free_gb))
    code = cli.main(["quickstart", "--ros-type", "complex", "--dose", str(dose)])
    return code, calls, capsys.readouterr().out


def test_complex_ros_warns_a_small_machine_at_a_high_dose_but_still_runs(monkeypatch, capsys):
    code, calls, out = _quickstart_complex(monkeypatch, capsys, free_gb=8, dose=30)
    assert code == 0 and len(calls) == 1           # a warning, never a block
    assert "Warning" in out and "swap or crash" in out


def test_complex_ros_is_silent_for_machines_and_doses_that_are_fine(monkeypatch, capsys):
    for free_gb, dose in ((64, 30), (8, 2.5)):
        code, calls, out = _quickstart_complex(monkeypatch, capsys, free_gb=free_gb, dose=dose)
        assert code == 0 and len(calls) == 1 and "Warning" not in out


def test_quickstart_defaults_to_the_fast_zero_dose_run(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "_run_module_as_main", lambda module_name, argv: calls.append((module_name, argv))
    )

    cli.main(["quickstart"])

    _, argv = calls[0]
    assert argv == ["AMMPERBulk_aB.py", "a", "a", "a", str(float(0)), "quickstart"]


def test_run_launches_the_interactive_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "_run_module_as_main", lambda module_name, argv: calls.append((module_name, argv))
    )

    exit_code = cli.main(["run"])

    assert exit_code == 0
    assert calls == [("ammper.AMMPERCLI", ["AMMPERCLI.py"])]


def test_gui_launches_the_packaged_gui_module(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli.runpy, "run_module",
        lambda name, run_name, alter_sys=False: calls.append((name, run_name, alter_sys)),
    )

    exit_code = cli.main(["gui"])

    assert exit_code == 0
    # the GUI is a module inside the package, so it runs from a clone and from an installed wheel
    assert calls == [("ammper.gui.AMMPERGUI", "__main__", True)]


def test_the_gui_script_and_its_assets_ship_inside_the_package():
    from ammper import paths

    assert os.path.isfile(os.path.join(paths.GUI_DIR, "AMMPERGUI.py"))
    assert os.path.isfile(os.path.join(paths.GUI_DIR, "fonts", "IBMPlexSans", "IBMPlexSans-Regular.ttf"))
    assert os.path.isfile(os.path.join(paths.GUI_DIR, "images", "AMMPER_Logo1.PNG"))


def test_uncaught_exception_is_reported_and_returns_1(monkeypatch, capsys):
    def boom(args):
        raise ValueError("simulated failure")

    monkeypatch.setattr(cli, "cmd_quickstart", boom)

    exit_code = cli.main(["quickstart"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "ValueError: simulated failure" in captured.out
    assert "--debug" in captured.out


def test_debug_flag_reraises_the_exception(monkeypatch):
    def boom(args):
        raise ValueError("simulated failure")

    monkeypatch.setattr(cli, "cmd_quickstart", boom)

    with pytest.raises(ValueError, match="simulated failure"):
        cli.main(["--debug", "quickstart"])


def test_typo_subcommand_gets_a_did_you_mean_suggestion(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["quickstrat"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "Did you mean 'quickstart'?" in captured.out
    # not argparse's raw jargon
    assert "invalid choice" not in captured.out


def test_bad_choice_value_with_no_close_match_lists_choices(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["quickstart", "--cell-type", "mutant"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "Choose from: rad51, wt" in captured.out


def test_cell_type_and_ros_type_are_case_insensitive(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "_run_module_as_main", lambda module_name, argv: calls.append((module_name, argv))
    )

    exit_code = cli.main(["quickstart", "--cell-type", "WT", "--ros-type", "BASIC"])

    assert exit_code == 0
    _, argv = calls[0]
    assert argv == ["AMMPERBulk_aB.py", "a", "a", "a", str(float(0)), "quickstart"]


def test_unrecognized_flag_gets_a_friendly_hint(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["quickstart", "--cel-type", "wt"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.out
    assert "--help" in captured.out


def test_no_command_falls_back_to_static_help_when_not_a_tty(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    exit_code = cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "usage: ammper" in captured.out


def test_interactive_menu_dispatches_the_chosen_command(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "quickstart")

    calls = []
    monkeypatch.setattr(
        cli, "_run_module_as_main", lambda module_name, argv: calls.append((module_name, argv))
    )

    exit_code = cli.main([])

    assert exit_code == 0
    assert calls and calls[0][0] == "ammper.AMMPERBulk_aB"


def test_interactive_menu_quit_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "quit")

    exit_code = cli.main([])

    assert exit_code == 0


def test_interactive_menu_help_shows_full_help(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "help")

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage: ammper" in captured.out


def test_interactive_menu_eof_is_treated_as_interrupted(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def raise_eof(*a, **k):
        raise EOFError

    monkeypatch.setattr(cli.Prompt, "ask", raise_eof)

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 130
    assert "Interrupted" in captured.out


def test_data_with_no_subcommand_prints_its_own_help(capsys):
    exit_code = cli.main(["data"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage: ammper data" in captured.out


def test_data_status_reports_counts_and_the_pull_hint(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.P, "load_ritracks_manifest",
        lambda: {"tutorial_energies": ["150"], "full_energies": ["150", "20", "42.76"],
                 "files": {"150": {"size": 4_000_000}, "20": {"size": 17_000_000},
                           "42.76": {"size": 26_000_000}}},
    )
    monkeypatch.setattr(
        cli.data_check, "required_ritracks_energies",
        lambda radType: {"150 MeV Proton": {"150"}, "GCRSim": {"20", "150"}, "Deep Space": {"42.76"}}[radType])
    monkeypatch.setattr(cli.data_fetch, "status", lambda: {"150": True, "20": False, "42.76": False})
    monkeypatch.setattr(cli.data_check, "missing_ritracks_energies",
                        lambda names: sorted((n for n in names if n != "150"), key=float))

    exit_code = cli.main(["data", "status"])
    out = " ".join(capsys.readouterr().out.split())        # Rich wraps long lines

    assert exit_code == 0
    assert "1/3 RITRACKS energy folders present" in out
    # what each folder is for, and what is still blocked, with sizes from the manifest
    assert "bundled in the package; 150 MeV Proton, GCRSim" in out
    assert "GCRSim cannot run yet: Run `ammper data pull gcrsim` to download just the 1 missing folder(s) (~17 MB)" in out
    assert "Deep Space cannot run yet: Run `ammper data pull deep-space`" in out


def test_data_status_marks_missing_optional_folders_neutrally(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.P, "load_ritracks_manifest",
        lambda: {"tutorial_energies": ["150"], "full_energies": ["150", "7.65"],
                 "files": {"150": {"size": 4_000_000}, "7.65": {"size": 9_000_000}}})
    monkeypatch.setattr(cli.data_check, "required_ritracks_energies",
                        lambda radType: {"150"} if radType == "150 MeV Proton" else set())
    monkeypatch.setattr(cli.data_fetch, "status", lambda: {"150": True, "7.65": False})
    monkeypatch.setattr(cli.data_check, "missing_ritracks_energies", lambda names: [])

    cli.main(["data", "status"])
    out = capsys.readouterr().out

    row = next(line for line in out.splitlines() if "7.65" in line)
    assert "·" in row and "x" not in row.split("7.65")[0]
    assert "optional: no radiation type reads it" in row


def test_data_pull_reports_dataset_not_published_cleanly(monkeypatch, capsys):
    def boom(console, targets=None, force=False):
        raise cli.data_fetch.DatasetNotPublishedError("not published yet")

    monkeypatch.setattr(cli.data_fetch, "pull", boom)

    exit_code = cli.main(["data", "pull"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "not published yet" in captured.out


def test_data_pull_success_prints_confirmation(monkeypatch, capsys):
    monkeypatch.setattr(cli.data_fetch, "pull", lambda console, targets=None, force=False: ["20", "23"])

    exit_code = cli.main(["data", "pull"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "2 RITRACKS folder(s) ready" in captured.out


def test_data_pull_passes_run_types_and_energies_through(monkeypatch):
    seen = []
    monkeypatch.setattr(cli.data_fetch, "pull",
                        lambda console, targets=None, force=False: seen.append((targets, force)) or [])

    assert cli.main(["data", "pull", "gcrsim", "20"]) == 0
    assert cli.main(["data", "pull", "--force"]) == 0

    assert seen == [(["gcrsim", "20"], False), ([], True)]


def test_data_pull_rejects_an_unknown_target_with_exit_code_2(monkeypatch, capsys):
    def boom(console, targets=None, force=False):
        raise cli.data_fetch.UnknownTargetError("'warp' is not a run type")

    monkeypatch.setattr(cli.data_fetch, "pull", boom)

    assert cli.main(["data", "pull", "warp"]) == 2
    assert "not a run type" in capsys.readouterr().out


def test_data_pull_explains_that_finished_folders_are_kept_after_a_network_error(monkeypatch, capsys):
    def boom(console, targets=None, force=False):
        raise OSError("timed out")

    monkeypatch.setattr(cli.data_fetch, "pull", boom)

    assert cli.main(["data", "pull"]) == 1
    out = capsys.readouterr().out
    assert "timed out" in out and "re-running continues" in out


def test_data_verify_all_present(monkeypatch, capsys):
    monkeypatch.setattr(cli.data_fetch, "verify", lambda: {"150": True, "20": True})

    exit_code = cli.main(["data", "verify"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "All 2 RITRACKS" in captured.out


def test_data_verify_reports_missing_folders(monkeypatch, capsys):
    monkeypatch.setattr(cli.data_fetch, "verify", lambda: {"150": True, "20": False})

    exit_code = cli.main(["data", "verify"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "1 of 2" in captured.out
    assert "20" in captured.out


def test_output_survives_a_windows_code_page(monkeypatch):
    """A cp1252 pipe (as on Windows CI) must not crash on the check mark printed after a run."""
    import io

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", stream)
    cli._use_utf8_output()
    cli._console().print("[bold green][\u2714] Quickstart complete[/bold green]")
    stream.flush()
    assert "\u2714".encode("utf-8") in stream.buffer.getvalue()
