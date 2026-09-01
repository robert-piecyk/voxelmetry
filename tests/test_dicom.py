"""Reading real-shaped DICOM series, including the slice-ordering guarantee."""

import numpy as np
import pytest

from voxelmetry import io as nio

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


# --- DICOM SEG --------------------------------------------------------------


@pytest.fixture
def seg_masks():
    """A liver-like segment with a smaller segment fully inside it, plus a tumour.

    Modelled on a real Colorectal-Liver-Metastases SEG, where 'Liver' and
    'Liver Remnant' cover largely the same voxels.
    """
    shape = (12, 40, 40)
    organ = np.zeros(shape, dtype=bool)
    organ[2:10, 8:32, 8:32] = True
    remnant = np.zeros(shape, dtype=bool)
    remnant[3:9, 10:26, 10:26] = True  # entirely inside organ
    tumour = np.zeros(shape, dtype=bool)
    tumour[5:7, 14:18, 14:18] = True   # inside both
    return {1: organ, 2: remnant, 3: tumour}, {1: "Liver", 2: "Remnant", 3: "Tumor"}


def test_reads_a_seg_into_a_labelmap(tmp_path, seg_masks):
    from .dicom_fixtures import write_seg

    masks, labels = seg_masks
    path = write_seg(tmp_path / "seg.dcm", masks, labels, spacing=(5.0, 0.9, 0.9))

    with pytest.warns(UserWarning, match="segments overlap"):
        labelmap, names = nio.load_dicom_seg(path)

    assert names == labels
    assert labelmap.spacing == pytest.approx((5.0, 0.9, 0.9))
    assert set(np.unique(labelmap.array)) == {0, 1, 2, 3}


def test_seg_overlap_warning_names_what_was_lost(tmp_path, seg_masks):
    from .dicom_fixtures import write_seg

    masks, labels = seg_masks
    path = write_seg(tmp_path / "seg.dcm", masks, labels)

    with pytest.warns(UserWarning) as record:
        nio.load_dicom_seg(path)
    message = str(record[0].message)
    assert "Liver" in message and "voxels" in message


def test_seg_priority_controls_which_segment_wins(tmp_path, seg_masks):
    """Flattening is lossy; the caller must be able to choose the loss."""
    from .dicom_fixtures import write_seg

    masks, labels = seg_masks
    path = write_seg(tmp_path / "seg.dcm", masks, labels)

    with pytest.warns(UserWarning):
        organ_wins, _ = nio.load_dicom_seg(path, priority=[3, 2, 1])
    with pytest.warns(UserWarning):
        tumour_wins, _ = nio.load_dicom_seg(path, priority=[1, 2, 3])

    # With the organ last it covers everything; with it first the tumour shows.
    assert (organ_wins.array == 3).sum() == 0
    assert (tumour_wins.array == 3).sum() == masks[3].sum()


def test_seg_masks_are_lossless(tmp_path, seg_masks):
    """The whole point: overlapping segments survive intact."""
    from .dicom_fixtures import write_seg

    masks, labels = seg_masks
    path = write_seg(tmp_path / "seg.dcm", masks, labels, spacing=(5.0, 0.9, 0.9),
                     origin_z=-450.0)

    recovered, names, geometry = nio.dicom_seg_masks(path)

    # A SEG stores only the slices some segment occupies, so the reconstructed
    # grid is that z-range rather than the source array's full extent.
    occupied = np.flatnonzero(np.any([m.any(axis=(1, 2)) for m in masks.values()], axis=0))
    first, last = int(occupied[0]), int(occupied[-1])
    assert geometry.shape == (last - first + 1, *masks[1].shape[1:])
    assert geometry.origin[0] == pytest.approx(-450.0 + first * 5.0)
    assert not geometry.array.any()

    assert names == labels
    for number, mask in masks.items():
        assert recovered[number].sum() == mask.sum(), f"segment {number} lost voxels"
        np.testing.assert_array_equal(recovered[number], mask[first : last + 1])


def test_non_seg_file_is_rejected(tmp_path, series_array):
    directory = write_ct_series(tmp_path / "series", series_array)
    one_slice = next(p for p in directory.iterdir() if p.suffix == ".dcm")
    with pytest.raises(ValueError, match="not a DICOM Segmentation"):
        nio.load_dicom_seg(one_slice)
