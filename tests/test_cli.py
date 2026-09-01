"""End-to-end exercise of the command-line entry points."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from nrrdvis import io as nio
from nrrdvis.cli import _parse_labels, _parse_split, app
from nrrdvis.phantom import torso_phantom

runner = CliRunner()


@pytest.fixture
def segmentation(tmp_path):
    _, labels = torso_phantom(shape=(48, 64, 64), spacing=(2.5, 1.5, 1.5))
    return nio.save(labels, tmp_path / "labels.nrrd")


@pytest.fixture
def scan(tmp_path):
    image, _ = torso_phantom(shape=(48, 64, 64), spacing=(2.5, 1.5, 1.5))
    return nio.save(image, tmp_path / "scan.nrrd")


def test_demo_runs_without_data(tmp_path):
    result = runner.invoke(app, ["demo", "-o", str(tmp_path / "demo.html"), "--lesions", "2"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "demo.html").stat().st_size > 1000


def test_info_reports_geometry(scan):
    result = runner.invoke(app, ["info", str(scan)])
    assert result.exit_code == 0, result.output
    assert "spacing (mm)" in result.output
    assert "isotropic" in result.output


def test_measure_names_labels_and_writes_json(segmentation, tmp_path):
    out = tmp_path / "m.json"
    result = runner.invoke(app, [
        "measure", str(segmentation), "--labels", "1=organ,2=lesion",
        "--split", "2", "--json", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "organ" in result.output

    records = json.loads(out.read_text())
    assert [r["name"] for r in records if r["label"] == 1] == ["organ"]
    assert any(r["name"].startswith("lesion_") for r in records)


def test_measure_min_volume_drops_speckle(segmentation):
    everything = runner.invoke(app, ["measure", str(segmentation), "--split", "2"])
    filtered = runner.invoke(app, [
        "measure", str(segmentation), "--split", "2", "--min-volume", "100000",
    ])
    assert everything.exit_code == filtered.exit_code == 0
    assert filtered.output.count("label_2") < everything.output.count("label_2")


def test_view_writes_a_viewer(segmentation, tmp_path):
    out = tmp_path / "scene.html"
    result = runner.invoke(app, [
        "view", str(segmentation), "-o", str(out),
        "--labels", "1=organ,2=lesion", "--split", "2", "--max-faces", "2000",
    ])
    assert result.exit_code == 0, result.output
    page = out.read_text()
    assert "var SCENE" in page and "organ" in page


def test_prep_preserves_physical_extent(scan, tmp_path):
    out = tmp_path / "prepped.nrrd"
    result = runner.invoke(app, [
        "prep", str(scan), str(out), "--window", "abdomen", "--isotropic", "2.0",
    ])
    assert result.exit_code == 0, result.output
    before, after = nio.load(scan), nio.load(out)
    assert after.extent_mm == pytest.approx(before.extent_mm, rel=0.05)


def test_convert_keeps_spacing_and_origin(segmentation, tmp_path):
    out = tmp_path / "converted.nii.gz"
    result = runner.invoke(app, ["convert", str(segmentation), str(out)])
    assert result.exit_code == 0, result.output
    assert nio.load(out).spacing == pytest.approx(nio.load(segmentation).spacing)


def test_missing_input_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["info", str(tmp_path / "absent.nrrd")])
    assert result.exit_code != 0


# --- option parsing ---------------------------------------------------------


def test_parse_labels():
    assert _parse_labels("1=liver, 2=tumour") == {1: "liver", 2: "tumour"}
    assert _parse_labels(None) == {}


@pytest.mark.parametrize("bad", ["liver", "x=liver"])
def test_parse_labels_rejects_malformed(bad):
    import typer

    with pytest.raises(typer.BadParameter):
        _parse_labels(bad)


def test_parse_split():
    assert _parse_split("2,3") == (2, 3)
    assert _parse_split(None) == ()


def test_parse_split_rejects_non_integers():
    import typer

    with pytest.raises(typer.BadParameter):
        _parse_split("2,liver")


def test_unknown_window_is_reported(scan, tmp_path):
    result = runner.invoke(app, [
        "prep", str(scan), str(tmp_path / "o.nrrd"), "--window", "pancreas_soft",
    ])
    assert result.exit_code != 0
    assert isinstance(result.exception, KeyError)


def test_view_on_an_empty_segmentation(tmp_path):
    """No labels is a legitimate input, not a crash."""
    from nrrdvis.volume import Volume

    empty = nio.save(Volume(np.zeros((16, 16, 16), np.uint8)), tmp_path / "empty.nrrd")
    result = runner.invoke(app, ["view", str(empty), "-o", str(tmp_path / "e.html")])
    assert result.exit_code == 0, result.output
