"""Reading real-shaped DICOM series, including the slice-ordering guarantee."""

import numpy as np
import pytest

from nrrdvis import io as nio

pydicom = pytest.importorskip("pydicom", reason="DICOM support is an optional extra")

from .dicom_fixtures import write_ct_series  # noqa: E402


@pytest.fixture
def series_array():
    """A volume whose every slice is identifiable by its own constant value."""
    array = np.zeros((24, 32, 40), dtype=np.int16)
    for z in range(array.shape[0]):
        array[z] = -1000 + z * 10  # unique, and in air-to-tissue HU range
        array[z, 8:24, 10:30] = 40 + z
    return array


def test_reads_a_dicom_series_with_its_geometry(tmp_path, series_array):
    directory = write_ct_series(tmp_path / "series", series_array, spacing=(2.5, 0.8, 0.8))
    volume = nio.load(directory)

    assert volume.shape == series_array.shape
    assert volume.spacing == pytest.approx((2.5, 0.8, 0.8))
    assert volume.origin[0] == pytest.approx(-400.0)
    assert volume.extent_mm == pytest.approx((60.0, 25.6, 32.0))


def test_slices_are_ordered_by_position_not_filename(tmp_path, series_array):
    """The guarantee that v1 lacked, on data built to break filename sorting.

    Filenames sort cleanly here but are assigned in a shuffled order, which is
    exactly what a real archive looks like: a TCIA liver CT named
    00000001.dcm onward had 46 of its 88 adjacent pairs out of anatomical
    order. Sorting by name yields a scrambled volume and no error.
    """
    ordered = write_ct_series(tmp_path / "ordered", series_array, filename_order="sequential")
    shuffled = write_ct_series(tmp_path / "shuffled", series_array, filename_order="shuffled")

    from_ordered = nio.load(ordered)
    from_shuffled = nio.load(shuffled)

    # Both must reconstruct the same volume, and it must be the original.
    np.testing.assert_array_equal(from_ordered.array, from_shuffled.array)
    np.testing.assert_array_equal(from_shuffled.array, series_array)

    # And the filenames really were scrambled, or this proves nothing.
    names = sorted(p.name for p in shuffled.iterdir())
    positions = []
    for name in names:
        ds = pydicom.dcmread(shuffled / name, stop_before_pixels=True)
        positions.append(float(ds.ImagePositionPatient[2]))
    assert positions != sorted(positions), "fixture failed to scramble the filenames"


def test_dicom_roundtrips_through_nrrd(tmp_path, series_array):
    directory = write_ct_series(tmp_path / "series", series_array, spacing=(3.0, 0.7, 0.7))
    volume = nio.load(directory)
    restored = nio.load(nio.save(volume, tmp_path / "out.nrrd"))

    assert restored.spacing == pytest.approx(volume.spacing)
    assert restored.origin == pytest.approx(volume.origin)
    np.testing.assert_array_equal(restored.array, volume.array)


def test_non_dicom_files_in_the_directory_are_ignored(tmp_path, series_array):
    """Archives ship LICENSE files and checksums alongside the images."""
    directory = write_ct_series(tmp_path / "series", series_array)
    (directory / "LICENSE").write_text("CC-BY 4.0")
    (directory / "checksums.txt").write_text("deadbeef")

    volume = nio.load(directory)
    assert volume.shape == series_array.shape
