"""Morphometry checked against closed-form geometry, not against itself."""

import numpy as np
import pytest

from nrrdvis import measure as nm
from nrrdvis.phantom import (
    analytic_sphere_area_mm2,
    analytic_sphere_volume_mm3,
    sphere_phantom,
)
from nrrdvis.volume import Volume


@pytest.mark.parametrize("radius", [10.0, 20.0, 30.0])
def test_volume_matches_analytic_sphere(radius):
    """Voxel counting must land within 1% of 4/3 pi r^3."""
    _, labels = sphere_phantom(radius_mm=radius, shape=(90, 90, 90))
    measured = nm.measure_label(labels, 1).volume_mm3
    assert measured == pytest.approx(analytic_sphere_volume_mm3(radius), rel=0.01)


@pytest.mark.parametrize("radius", [15.0, 30.0])
def test_surface_area_matches_analytic_sphere(radius):
    """The smoothing constant exists to make this hold; without it the error is +9%."""
    _, labels = sphere_phantom(radius_mm=radius, shape=(90, 90, 90))
    measured = nm.measure_label(labels, 1).surface_area_mm2
    assert measured == pytest.approx(analytic_sphere_area_mm2(radius), rel=0.02)


def test_unsmoothed_area_is_the_overestimate_we_correct_for():
    """Pin the artifact itself, so a regression in the fix is visible."""
    _, labels = sphere_phantom(radius_mm=30.0, shape=(90, 90, 90))
    mask = labels.array == 1
    raw = nm.surface_area_mm2(mask, labels.spacing, sigma=0.0)
    assert raw / analytic_sphere_area_mm2(30.0) > 1.05


def test_max_diameter_of_a_sphere_is_its_diameter():
    _, labels = sphere_phantom(radius_mm=25.0, shape=(80, 80, 80))
    assert nm.measure_label(labels, 1).max_diameter_mm == pytest.approx(50.0, abs=2.0)


def test_max_diameter_finds_oblique_extent():
    """The v1 approach scanned axial slices and would miss a diagonal structure."""
    array = np.zeros((40, 40, 40), dtype=np.uint8)
    for i in range(30):
        array[i + 4, i + 4, i + 4] = 1
    measured = nm.measure_label(Volume(array, spacing=(1.0, 1.0, 1.0)), 1).max_diameter_mm
    assert measured == pytest.approx(np.sqrt(3) * 29, rel=0.05)


def test_sphere_sphericity_is_one():
    _, labels = sphere_phantom(radius_mm=25.0, shape=(80, 80, 80))
    assert nm.measure_label(labels, 1).sphericity == pytest.approx(1.0, abs=0.03)


def test_sphericity_never_exceeds_one():
    """No shape beats a sphere on area-to-volume; values above 1 are artifacts."""
    array = np.zeros((20, 20, 20), dtype=np.uint8)
    array[9:11, 9:11, 9:11] = 1
    m = nm.measure_label(Volume(array, spacing=(5.0, 1.0, 1.0)), 1)
    assert m.sphericity <= 1.0
    assert m.resolution_limited


@pytest.mark.parametrize(("length", "width"), [(100, 6), (100, 8), (120, 10)])
def test_elongated_structure_sphericity_tracks_the_closed_form(length, width):
    """A square rod has an exact sphericity; the measurement must follow it.

    Corner rounding from the surface pre-smoothing biases the result slightly
    high, which is why the tolerance is absolute rather than tight.
    """
    array = np.zeros((length + 20, width + 14, width + 14), dtype=np.uint8)
    array[10 : 10 + length, 7 : 7 + width, 7 : 7 + width] = 1
    measured = nm.measure_label(Volume(array), 1)

    volume_mm3 = length * width * width
    surface_mm2 = 4 * width * length + 2 * width * width
    analytic = (np.pi ** (1 / 3) * (6 * volume_mm3) ** (2 / 3)) / surface_mm2

    assert not measured.resolution_limited
    assert measured.sphericity == pytest.approx(analytic, abs=0.08)
    assert measured.sphericity < 0.65  # unambiguously not a sphere


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (2.5, 1.0, 1.0), (1.0, 0.7, 0.7)])
def test_volume_is_invariant_to_voxel_grid(spacing):
    """Same physical sphere, different grids: the answer must not move.

    This is the property v1 lacked. Its measurements were taken after a
    cv2.resize to 256x256 that left spacing untouched, so the reported volume
    scaled with whatever grid the scan happened to arrive on.
    """
    _, labels = sphere_phantom(radius_mm=20.0, shape=(70, 70, 70), spacing=spacing)
    measured = nm.measure_label(labels, 1).volume_mm3
    assert measured == pytest.approx(analytic_sphere_volume_mm3(20.0), rel=0.03)


def test_millilitres_are_millimetres_cubed_over_1000():
    """v1 divided by 1e6 and called the result litres, off by 1000x."""
    array = np.ones((10, 10, 10), dtype=np.uint8)
    m = nm.measure_label(Volume(array, spacing=(1.0, 1.0, 1.0)), 1)
    assert m.volume_mm3 == pytest.approx(1000.0)
    assert m.volume_ml == pytest.approx(1.0)


def test_empty_label_returns_zeros_not_an_error():
    m = nm.measure_label(Volume(np.zeros((5, 5, 5), dtype=np.uint8)), 7)
    assert m.voxels == 0 and m.volume_ml == 0.0 and m.n_components == 0


def test_measure_all_skips_background():
    array = np.zeros((10, 10, 10), dtype=np.uint8)
    array[1:4, 1:4, 1:4] = 1
    array[6:9, 6:9, 6:9] = 2
    results = nm.measure_all(Volume(array), names={1: "a", 2: "b"})
    assert [m.label for m in results] == [1, 2]
    assert [m.name for m in results] == ["a", "b"]


def test_components_are_measured_separately_largest_first():
    """Three spheres must report three diameters, not one spanning all of them."""
    array = np.zeros((60, 60, 60), dtype=np.uint8)
    volume = Volume(array, spacing=(1.0, 1.0, 1.0))
    from nrrdvis.phantom import sphere_mask

    for centre, radius in [((15, 15, 15), 8.0), ((45, 45, 45), 5.0), ((15, 45, 30), 3.0)]:
        array[sphere_mask(array.shape, centre, radius, volume.spacing)] = 2

    components = nm.measure_components(volume, 2, "lesion")
    assert len(components) == 3
    assert [c.name for c in components] == ["lesion_1", "lesion_2", "lesion_3"]
    volumes = [c.volume_mm3 for c in components]
    assert volumes == sorted(volumes, reverse=True)
    # Each diameter reflects its own sphere, not the 60 mm span between them.
    assert components[0].max_diameter_mm == pytest.approx(16.0, abs=2.0)


def test_min_volume_filters_speckle():
    array = np.zeros((40, 40, 40), dtype=np.uint8)
    array[10:20, 10:20, 10:20] = 1
    array[35, 35, 35] = 1  # single-voxel speckle
    volume = Volume(array, spacing=(1.0, 1.0, 1.0))
    assert len(nm.measure_components(volume, 1)) == 2
    assert len(nm.measure_components(volume, 1, min_volume_mm3=10.0)) == 1


def test_lesion_burden_sums_components():
    array = np.zeros((40, 40, 40), dtype=np.uint8)
    array[5:10, 5:10, 5:10] = 1
    array[20:24, 20:24, 20:24] = 1
    components = nm.measure_components(Volume(array), 1, "lesion")
    burden = nm.lesion_burden(components, reference_volume_ml=10.0)
    assert burden["n_lesions"] == 2
    assert burden["total_volume_ml"] == pytest.approx((125 + 64) / 1000.0)
    assert burden["burden_percent"] == pytest.approx(100 * (125 + 64) / 1000.0 / 10.0)


def test_lesion_burden_of_nothing_is_zero():
    assert nm.lesion_burden([])["n_lesions"] == 0


def test_table_flags_resolution_limited_rows():
    array = np.zeros((20, 20, 20), dtype=np.uint8)
    array[9:11, 9:11, 9:11] = 1
    text = nm.to_table([nm.measure_label(Volume(array, spacing=(5.0, 1.0, 1.0)), 1)])
    assert "*" in text and "thinnest axis" in text
