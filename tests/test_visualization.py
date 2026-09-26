"""Regression tests for GUI video export and desktop playback.

Run with QT_QPA_PLATFORM=offscreen to exercise Qt without opening windows.
No scientific simulation or bundled radiation data is required.
"""

import importlib
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication, QMessageBox

ROOT = Path(__file__).resolve().parents[1]
# The legacy direct-script invocation puts gui/ on sys.path.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gui"))
from gui import movieMaker  # noqa: E402


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(application):
    module = importlib.import_module("gui.AMMPERGUI")
    view = module.Widget()
    for index in range(1, 16):
        setattr(view, f"path{index}", f"frame {index}.png")
    yield module, view
    view.close()
    view.deleteLater()
    application.processEvents()


@pytest.mark.parametrize("opens", [True, False])
def test_visualization_uses_desktop_file_url(widget, monkeypatch, tmp_path, opens):
    module, view = widget
    path = str(tmp_path / "résultats #1 visualization.mp4")
    render = MagicMock(return_value=path)
    launch = MagicMock(return_value=opens)
    warning = MagicMock()
    monkeypatch.setattr(module, "mm", render)
    monkeypatch.setattr(QDesktopServices, "openUrl", launch)
    monkeypatch.setattr(QMessageBox, "warning", warning)
    # Linux and Windows do not provide the macOS 'open' command.
    monkeypatch.setattr(
        subprocess, "call", MagicMock(side_effect=FileNotFoundError("open"))
    )

    view.visualization()

    render.assert_called_once_with("", *[f"frame {i}.png" for i in range(1, 16)])
    launch.assert_called_once()
    url = launch.call_args.args[0]
    assert url.isLocalFile()
    assert url.toLocalFile() == path
    assert view.stackedWidget.currentIndex() == 5
    if opens:
        warning.assert_not_called()
    else:
        warning.assert_called_once()
        assert path in warning.call_args.args[2]


def test_visualization_reports_export_error_without_opening_a_player(
    widget, monkeypatch
):
    module, view = widget
    launch = MagicMock()
    warning = MagicMock()
    monkeypatch.setattr(module, "mm", MagicMock(side_effect=OSError("encoder failed")))
    monkeypatch.setattr(QDesktopServices, "openUrl", launch)
    monkeypatch.setattr(QMessageBox, "warning", warning)

    view.visualization()

    launch.assert_not_called()
    warning.assert_called_once()
    assert "encoder failed" in warning.call_args.args[2]


@pytest.mark.parametrize("failure", [None, "image", "encode"])
def test_movie_maker_closes_clips_and_returns_output_path(
    monkeypatch, tmp_path, failure
):
    monkeypatch.chdir(tmp_path)
    clips = [MagicMock() for _ in range(15)]
    for clip in clips:
        clip.set_duration.return_value = clip
    factory = MagicMock(
        side_effect=clips if failure != "image" else [clips[0], OSError("bad image")]
    )
    combined = MagicMock()
    concatenate = MagicMock(return_value=combined)
    if failure == "encode":
        combined.write_videofile.side_effect = OSError("encoder failed")
    monkeypatch.setattr(movieMaker, "ImageClip", factory)
    monkeypatch.setattr(movieMaker, "concatenate_videoclips", concatenate)
    paths = [f"frame{i}.png" for i in range(15)]

    if failure:
        with pytest.raises(OSError, match="bad image|encoder failed"):
            movieMaker.movie_maker("images", *paths)
    else:
        result = movieMaker.movie_maker("images", *paths)
        assert result == str(tmp_path / "visualization.mp4")
        combined.write_videofile.assert_called_once_with(result, fps=24)
    for clip in clips[: 1 if failure == "image" else 15]:
        clip.close.assert_called_once()
    if failure == "image":
        concatenate.assert_not_called()
    else:
        combined.close.assert_called_once()


def test_gui_supports_package_import_from_another_working_directory(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", "import gui.AMMPERGUI; print('imported')"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_movie_maker_exports_a_real_playable_video(tmp_path, monkeypatch):
    from moviepy.editor import VideoFileClip

    monkeypatch.chdir(tmp_path)
    images = tmp_path / "frames with spaces"
    images.mkdir()
    paths = []
    for index in range(15):
        path = images / f"frame {index}.png"
        pixels = np.full((32, 32, 3), index * 12, dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(path.name)

    result = movieMaker.movie_maker(str(images), *paths)

    assert result == str(tmp_path / "visualization.mp4")
    assert Path(result).stat().st_size > 0
    clip = VideoFileClip(result, audio=False)
    try:
        assert clip.size == [32, 32]
        assert clip.fps == pytest.approx(24)
        assert clip.duration == pytest.approx(15, abs=0.05)
        assert clip.get_frame(0.5).mean() < clip.get_frame(14.5).mean()
    finally:
        clip.close()
