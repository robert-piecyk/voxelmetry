"""Preprocessing must behave the same regardless of the grid it is handed."""

import numpy as np
import pytest

from nrrdvis import preprocess as npre
from nrrdvis.phantom import HU_AIR
from nrrdvis.volume import Volume


def test_named_windows_resolve_to_bounds():
    low, high = npre.window_bounds("abdomen")
    assert (low, high) == (-160.0, 240.0)


def test_explicit_window_is_level_and_width():
    assert npre.window_bounds((60, 160)) == (-20.0, 140.0)


def test_unknown_window_names_are_rejected():
    with pytest.raises(KeyError, match="Unknown window"):
        npre.window_bounds("pancreas_soft")


def test_body_mask_excludes_surrounding_air(torso):
    image, _ = torso
    mask = npre.body_mask(image)
    assert mask.any()
    # Air outside the body must not be selected.
    assert not mask[:, 0, 0].any()
    # The body occupies a substantial but not total fraction of the field.
    assert 0.15 < mask.mean() < 0.85


def test_body_mask_keeps_the_organ_inside_it(torso):
    image, labels = torso
    mask = npre.body_mask(image)
    organ = labels.array == 1
    assert mask[organ].mean() > 0.99


def test_body_mask_of_pure_air_is_empty():
    volume = Volume(np.full((10, 10, 10), HU_AIR, dtype=np.int16))
    assert not npre.body_mask(volume).any()


def test_structuring_element_is_physical_not_voxel_sized():
    """A 10 mm ball must span 10 mm whatever the slice thickness.

    v1 wrote np.ones((15, 15)) and got a different physical closing on every
    scanner. This is the property that replaces it.
    """
    fine = npre._ball_voxels(10.0, (1.0, 1.0, 1.0))
    coarse = npre._ball_voxels(10.0, (5.0, 1.0, 1.0))
    # Same physical radius: 21 voxels across at 1 mm, 5 across at 5 mm.
    assert fine.shape[0] == 21
    assert coarse.shape[0] == 5
    assert fine.shape[1] == coarse.shape[1] == 21


def test_remove_table_blanks_everything_outside_the_body(torso):
    image, _ = torso
    mask = npre.body_mask(image)
    cleaned = npre.remove_table(image, mask)
    assert np.all(cleaned.array[~mask] == cleaned.array.min())
    np.testing.assert_array_equal(cleaned.array[mask], image.array[mask])


@pytest.mark.parametrize("method", ["gaussian", "median"])
def test_denoise_reduces_variance_without_moving_the_mean(method):
    rng = np.random.default_rng(0)
    clean = np.zeros((20, 30, 30), dtype=np.float32)
    clean[5:15, 10:20, 10:20] = 100.0
    noisy = Volume(clean + rng.normal(0, 12, clean.shape).astype(np.float32))
    result = npre.denoise(noisy, strength=1.5, method=method)
    assert result.array.std() < noisy.array.std()
    assert result.array.mean() == pytest.approx(noisy.array.mean(), abs=2.0)


def test_denoise_preserves_geometry():
    volume = Volume(np.zeros((10, 10, 10), dtype=np.float32), spacing=(2.0, 1.0, 1.0))
    assert npre.denoise(volume, 1.0).spacing == volume.spacing


def test_unknown_denoise_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown denoise method"):
        npre.denoise(Volume(np.zeros((5, 5, 5))), method="bm3d")


def test_full_run_preserves_physical_extent(torso):
    image, _ = torso
    config = npre.PreprocessConfig(isotropic_mm=1.5, window="abdomen")
    result = npre.run(image, config)
    assert result.extent_mm == pytest.approx(image.extent_mm, rel=0.02)
    assert result.spacing == pytest.approx((1.5, 1.5, 1.5), rel=0.02)


def test_windowing_is_applied_last_and_normalises(torso):
    image, _ = torso
    result = npre.run(image, npre.PreprocessConfig(isotropic_mm=None, window="abdomen"))
    assert result.array.min() >= 0.0 and result.array.max() <= 1.0


def test_config_describe_lists_the_active_steps():
    text = npre.PreprocessConfig(isotropic_mm=1.0, denoise_mm=2.0).describe()
    assert "1.0 mm isotropic" in text and "denoise" in text and "window" in text


def test_empty_config_describes_itself_as_a_noop():
    config = npre.PreprocessConfig(window=None, isotropic_mm=None, strip_table=False)
    assert config.describe() == "no-op"
