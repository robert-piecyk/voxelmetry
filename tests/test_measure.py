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


@pytest.mark.parametrize("block", [16, 128, 100_000])
def test_feret_diameter_is_independent_of_block_size(block):
    """Blocking exists to bound memory; it must not change the answer.

    An earlier version subsampled large hulls, which can only ever report a
    diameter that is too small. This pins the exactness.
    """
    rng = np.random.default_rng(3)
    array = np.zeros((50, 50, 50), dtype=np.uint8)
    coords = rng.integers(2, 48, size=(400, 3))
    array[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    mask = array == 1

    reference = nm.max_diameter_mm(mask, (1.0, 1.0, 1.0), block=100_000)
    assert nm.max_diameter_mm(mask, (1.0, 1.0, 1.0), block=block) == pytest.approx(reference)


def test_feret_diameter_uses_physical_spacing():
    """Two voxels 10 apart on a 3 mm axis are 30 mm apart, not 10."""
    array = np.zeros((14, 5, 5), dtype=np.uint8)
    array[1, 2, 2] = 1
    array[11, 2, 2] = 1
    measured = nm.max_diameter_mm(array == 1, (3.0, 1.0, 1.0))
    assert measured == pytest.approx(30.0)


def test_single_voxel_has_no_diameter():
    array = np.zeros((10, 10, 10), dtype=np.uint8)
    array[5, 5, 5] = 1
    assert nm.max_diameter_mm(array == 1, (1.0, 1.0, 1.0)) == 0.0


def test_single_slice_lesion_reports_a_real_surface_area():
    """A lesion confined to one slice must not report zero area.

    At 5 mm slice thickness this is routine, not an edge case. The 0.8-voxel
    pre-smooth pulls such a structure's peak below the 0.5 isolevel and the
    isosurface comes back empty; the fallback to an unsmoothed pass is what
    keeps the number finite.
    """
    array = np.zeros((20, 40, 40), dtype=np.uint8)
    array[10, 15:25, 15:25] = 1  # one slice thick
    labelmap = Volume(array, spacing=(5.0, 0.7, 0.7))

    m = nm.measure_label(labelmap, 1, "flat_lesion")
    assert m.volume_mm3 > 0
    assert m.surface_area_mm2 > 0
    assert 0 < m.sphericity <= 1.0
    assert m.resolution_limited  # and it is still flagged as unreliable


def test_smoothing_erases_a_single_slice_structure_without_the_fallback():
    """Pin the failure the fallback exists to prevent."""
    from scipy import ndimage as ndi

    array = np.zeros((20, 40, 40), dtype=np.uint8)
    array[10, 15:25, 15:25] = 1
    padded = np.pad((array == 1).astype(np.float32), 2)
    assert ndi.gaussian_filter(padded, 0.8).max() <= 0.5


def test_component_count_is_sensitive_to_connectivity():
    """Two voxels touching only at a corner: two parts by face, one by 26-way.

    This is why n_components alone is not a fragmentation score, and why the
    cohort report leads with stray volume instead.
    """
    array = np.zeros((10, 10, 10), dtype=np.uint8)
    array[4, 4, 4] = 1
    array[5, 5, 5] = 1  # diagonal neighbour only
    volume = Volume(array)
    assert nm.measure_label(volume, 1, connectivity=1).n_components == 2
    assert nm.measure_label(volume, 1, connectivity=3).n_components == 1


def test_largest_component_fraction_reports_the_stray_share():
    array = np.zeros((30, 30, 30), dtype=np.uint8)
    array[5:15, 5:15, 5:15] = 1   # 1000 voxels
    array[25, 25, 25] = 1         # 1 stray voxel
    m = nm.measure_label(Volume(array), 1)
    assert m.n_components == 2
    assert m.largest_component_fraction == pytest.approx(1000 / 1001)
    stray = m.volume_mm3 * (1 - m.largest_component_fraction)
    assert stray == pytest.approx(1.0, abs=0.01)


def test_component_measurements_are_unchanged_by_the_bounding_box_crop():
    """Components are measured inside their own box for speed, not different results.

    Centroids in particular must stay in patient coordinates: the crop shifts
    the origin, and forgetting that shift would silently report crop-local
    positions.
    """
    from nrrdvis.phantom import sphere_mask

    spacing = (2.5, 1.0, 1.0)
    shape = (60, 80, 80)
    array = np.zeros(shape, dtype=np.uint8)
    centres = [(15, 20, 20), (40, 60, 55)]
    for centre in centres:
        array[sphere_mask(shape, centre, 6.0, spacing)] = 1
    labelmap = Volume(array, spacing=spacing, origin=(100.0, -50.0, 25.0))

    components = nm.measure_components(labelmap, 1, "blob")
    assert len(components) == 2

    for measured in components:
        # Match each result back to the sphere centre it came from.
        expected = min(
            (np.array(c) * np.array(spacing) + np.array(labelmap.origin) for c in centres),
            key=lambda e: np.linalg.norm(e - np.array(measured.centroid_mm)),
        )
        assert np.allclose(measured.centroid_mm, expected, atol=1.0)
        # And the geometry itself is the sphere's, not the crop's.
        assert measured.max_diameter_mm == pytest.approx(12.0, abs=2.0)
        assert measured.sphericity == pytest.approx(1.0, abs=0.12)


def test_cropping_matches_whole_volume_measurement():
    """A single component measured cropped equals the same label measured whole."""
    array = np.zeros((40, 40, 40), dtype=np.uint8)
    array[10:20, 12:22, 14:24] = 1
    labelmap = Volume(array, spacing=(2.0, 1.0, 1.5), origin=(5.0, 7.0, 9.0))

    whole = nm.measure_label(labelmap, 1)
    cropped = nm.measure_components(labelmap, 1)[0]

    assert cropped.volume_mm3 == pytest.approx(whole.volume_mm3)
    assert cropped.surface_area_mm2 == pytest.approx(whole.surface_area_mm2, rel=1e-6)
    assert cropped.max_diameter_mm == pytest.approx(whole.max_diameter_mm)
    assert np.allclose(cropped.centroid_mm, whole.centroid_mm)
    assert cropped.bbox_mm == pytest.approx(whole.bbox_mm)
