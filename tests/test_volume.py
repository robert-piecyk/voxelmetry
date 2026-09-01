"""Volume geometry: the invariants everything else depends on."""

import numpy as np
import pytest

from voxelmetry.volume import Volume


def test_rejects_non_3d_array():
    with pytest.raises(ValueError, match="3-D array"):
        Volume(np.zeros((10, 10)))


@pytest.mark.parametrize("bad", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0)])
def test_rejects_non_positive_spacing(bad):
    with pytest.raises(ValueError, match="strictly positive"):
        Volume(np.zeros((4, 4, 4)), spacing=bad)


def test_voxel_volume_is_product_of_spacing():
    volume = Volume(np.zeros((4, 4, 4)), spacing=(2.0, 0.5, 3.0))
    assert volume.voxel_volume_mm3 == pytest.approx(3.0)


def test_extent_is_grid_times_spacing():
    volume = Volume(np.zeros((10, 20, 30)), spacing=(2.0, 1.0, 0.5))
    assert volume.extent_mm == pytest.approx((20.0, 20.0, 15.0))


@pytest.mark.parametrize("target", [0.5, 1.0, 2.0, 3.0])
def test_resampling_preserves_physical_extent(target):
    """Resampling changes the grid, never the physical size of the scan."""
    volume = Volume(np.random.default_rng(0).random((30, 40, 50)), spacing=(3.0, 1.0, 0.8))
    resampled = volume.resample(target)
    assert resampled.extent_mm == pytest.approx(volume.extent_mm, rel=0.02)


def test_resampling_records_realised_spacing():
    """Rounding to whole voxels perturbs spacing; the volume must record the truth."""
    volume = Volume(np.zeros((7, 7, 7)), spacing=(3.0, 3.0, 3.0))
    resampled = volume.resample(2.0)
    realised_extent = [
        n * s for n, s in zip(resampled.shape, resampled.spacing, strict=True)
    ]
    assert realised_extent == pytest.approx(volume.extent_mm)


def test_resample_to_same_spacing_is_identity():
    volume = Volume(np.ones((5, 5, 5)), spacing=(1.0, 1.0, 1.0))
    assert volume.resample(1.0) is volume


def test_integer_labels_survive_resampling_unblended():
    """Nearest-neighbour is mandatory for labels: interpolation invents classes."""
    array = np.zeros((10, 10, 10), dtype=np.uint8)
    array[3:7, 3:7, 3:7] = 4
    resampled = Volume(array, spacing=(1.0, 1.0, 1.0)).resample(0.5)
    assert set(np.unique(resampled.array)) <= {0, 4}


def test_window_normalize_maps_to_unit_range():
    volume = Volume(np.linspace(-1000, 1000, 8**3).reshape(8, 8, 8))
    windowed = volume.window(-100, 300, mode="normalize")
    assert windowed.array.min() == pytest.approx(0.0)
    assert windowed.array.max() == pytest.approx(1.0)


def test_window_clip_keeps_original_units():
    volume = Volume(np.linspace(-1000, 1000, 8**3).reshape(8, 8, 8))
    clipped = volume.window(-100, 300, mode="clip")
    assert clipped.array.min() == pytest.approx(-100)
    assert clipped.array.max() == pytest.approx(300)


def test_window_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="high > low"):
        Volume(np.zeros((4, 4, 4))).window(300, 100)


def test_crop_to_mask_shifts_origin_by_the_cropped_distance():
    array = np.zeros((20, 20, 20))
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:10, 5:10, 5:10] = True
    volume = Volume(array, spacing=(2.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    cropped = volume.crop_to_mask(mask)
    assert cropped.shape == (5, 5, 5)
    assert cropped.origin == pytest.approx((10.0, 5.0, 5.0))


def test_crop_rejects_empty_mask():
    volume = Volume(np.zeros((5, 5, 5)))
    with pytest.raises(ValueError, match="empty mask"):
        volume.crop_to_mask(np.zeros((5, 5, 5), dtype=bool))
