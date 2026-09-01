"""Preprocessing must behave the same regardless of the grid it is handed."""

import numpy as np
import pytest

from voxelmetry import preprocess as npre
from voxelmetry.phantom import HU_AIR
from voxelmetry.volume import Volume


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


# --- modality awareness -----------------------------------------------------
# Driven by a real TCGA-LIHC liver MR: intensities 0 to 831, where the HU
# threshold selected 87% of the field of view and the abdomen window collapsed
# everything above 240 to a single value, both without any error.


def make_mr_like(shape=(20, 40, 40), seed=0):
    """Unsigned, uncalibrated intensities of the kind an MR series carries."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 40, size=shape).astype(np.uint16)  # noise background
    array[4:16, 10:30, 10:30] = rng.integers(300, 800, size=(12, 20, 20))  # body
    return Volume(array, spacing=(3.5, 1.25, 1.25), name="mr_like")


def test_hounsfield_detection(torso):
    image, _ = torso
    assert npre.is_hounsfield(image)
    assert not npre.is_hounsfield(make_mr_like())


def test_already_normalised_data_is_not_treated_as_hounsfield():
    normalised = Volume(np.linspace(0, 1, 8**3).reshape(8, 8, 8).astype(np.float32))
    assert not npre.is_hounsfield(normalised)


def test_hu_window_refuses_non_hounsfield_data():
    """Silently destroying the dynamic range is worse than failing."""
    with pytest.raises(ValueError, match="does not look like Hounsfield"):
        npre.apply_window(make_mr_like(), "abdomen")


def test_percentile_window_preserves_dynamic_range():
    mr = make_mr_like()
    windowed = npre.apply_window(mr, "percentile")
    assert windowed.array.min() == pytest.approx(0.0)
    assert windowed.array.max() == pytest.approx(1.0)
    # A HU window would have crushed the top of the range into one value.
    assert len(np.unique(windowed.array)) > 100


def test_percentile_window_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="high > low"):
        npre.percentile_window(make_mr_like(), low=99.0, high=1.0)


def test_body_mask_falls_back_to_otsu_for_non_hounsfield_data():
    """The HU threshold is meaningless here; Otsu recovers the true body."""
    mr = make_mr_like()

    # The silent failure being avoided: every voxel is "above -320 HU", so the
    # threshold selects the entire volume and the mask means nothing.
    assert (mr.array > npre.AIR_THRESHOLD_HU).all()

    auto = npre.body_mask(mr)
    true_body_fraction = (12 * 20 * 20) / (20 * 40 * 40)
    assert auto.mean() == pytest.approx(true_body_fraction, abs=0.03)
    # And it found the bright region rather than the noise floor.
    assert auto[10, 20, 20]
    assert not auto[0, 0, 0]


def test_hounsfield_data_still_uses_the_calibrated_threshold(torso):
    """Auto-detection must not change behaviour on the CT path."""
    image, _ = torso
    np.testing.assert_array_equal(
        npre.body_mask(image), npre.body_mask(image, threshold_hu=npre.AIR_THRESHOLD_HU)
    )


def test_closing_does_not_erode_a_body_that_reaches_the_field_of_view_edge():
    """Found on a real TCIA liver CT: an 8 mm closing emptied the end slices.

    binary_closing erodes after dilating, and scipy treats everything outside
    the array as background, so a body continuing past the edge of the scan
    gets eaten there. Every abdominal CT does this at the top and bottom of
    the slab, so the failure was universal and silent.
    """
    array = np.full((24, 40, 40), HU_AIR, dtype=np.int16)
    # A column of tissue spanning the entire z extent, touching both ends.
    array[:, 12:28, 12:28] = 50
    volume = Volume(array, spacing=(2.5, 1.0, 1.0))

    without = npre.body_mask(volume, closing_mm=0.0)
    with_closing = npre.body_mask(volume, closing_mm=8.0)

    assert without[0].any() and without[-1].any()
    assert with_closing[0].any(), "closing emptied the first slice"
    assert with_closing[-1].any(), "closing emptied the last slice"
    # Closing may add voxels but must never remove the body wholesale.
    assert with_closing.sum() >= without.sum() * 0.98


@pytest.mark.parametrize("closing_mm", [0.0, 4.0, 8.0, 12.0])
def test_body_fraction_is_stable_across_closing_radii(closing_mm):
    """Closing fills gaps; it must not steadily shrink the body."""
    array = np.full((20, 40, 40), HU_AIR, dtype=np.int16)
    array[3:17, 10:30, 10:30] = 50
    volume = Volume(array, spacing=(2.5, 1.0, 1.0))
    fraction = npre.body_mask(volume, closing_mm=closing_mm).mean()
    assert fraction == pytest.approx((14 * 20 * 20) / (20 * 40 * 40), abs=0.05)
