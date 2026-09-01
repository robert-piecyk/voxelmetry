"""Round-tripping must preserve geometry, including the axis-order flip."""

import numpy as np
import pytest

from voxelmetry import io as nio
from voxelmetry.volume import Volume


@pytest.mark.parametrize("suffix", [".nrrd", ".nii.gz", ".mha"])
def test_roundtrip_preserves_spacing_and_origin(tmp_path, suffix):
    volume = Volume(
        np.random.default_rng(0).random((7, 9, 11)).astype(np.float32),
        spacing=(3.0, 1.5, 0.75),
        origin=(10.0, -20.0, 5.0),
        name="probe",
    )
    path = nio.save(volume, tmp_path / f"probe{suffix}")
    restored = nio.load(path)

    assert restored.shape == volume.shape
    assert restored.spacing == pytest.approx(volume.spacing)
    assert restored.origin == pytest.approx(volume.origin)
    np.testing.assert_allclose(restored.array, volume.array, rtol=1e-5)


def test_anisotropic_spacing_is_not_transposed(tmp_path):
    """SimpleITK reports (x, y, z) while arrays are [z, y, x].

    Getting this backwards is silent on isotropic data and wrong on everything
    else, so the asymmetric case is pinned explicitly.
    """
    volume = Volume(np.zeros((4, 8, 16)), spacing=(5.0, 2.0, 1.0))
    restored = nio.load(nio.save(volume, tmp_path / "aniso.nrrd"))
    assert restored.spacing == pytest.approx((5.0, 2.0, 1.0))
    assert restored.shape == (4, 8, 16)
    assert restored.extent_mm == pytest.approx((20.0, 16.0, 16.0))


def test_labels_survive_roundtrip_as_integers(tmp_path):
    array = np.zeros((6, 6, 6), dtype=np.uint8)
    array[2:4, 2:4, 2:4] = 3
    restored = nio.load(nio.save(Volume(array), tmp_path / "labels.nrrd"))
    assert set(np.unique(restored.array)) == {0, 3}


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        nio.load(tmp_path / "absent.nrrd")


def test_empty_directory_raises_valueerror(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No DICOM files"):
        nio.load(tmp_path / "empty")


def test_compression_shrinks_a_sparse_label_volume(tmp_path):
    array = np.zeros((40, 40, 40), dtype=np.uint8)
    array[10:20, 10:20, 10:20] = 1
    volume = Volume(array)
    big = nio.save(volume, tmp_path / "raw.nrrd", compress=False)
    small = nio.save(volume, tmp_path / "small.nrrd", compress=True)
    assert small.stat().st_size < big.stat().st_size


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.nrrd", True), ("a.nii.gz", True), ("a.nii", True), ("a.png", False), ("a", False)],
)
def test_is_volume_file(name, expected):
    assert nio.is_volume_file(name) is expected
