"""Surface extraction must not distort the geometry it represents."""

import numpy as np
import pytest

from nrrdvis import mesh as nmesh
from nrrdvis.phantom import analytic_sphere_volume_mm3, sphere_phantom


def test_extracted_mesh_encloses_the_analytic_volume(sphere):
    _, labels = sphere
    mesh = nmesh.surface_from_label(labels, 1, "sphere")
    enclosed = nmesh.mesh_volume_mm3(mesh)
    assert enclosed == pytest.approx(analytic_sphere_volume_mm3(30.0), rel=0.02)


def test_empty_mask_yields_an_empty_mesh_not_an_error():
    mesh = nmesh.extract_surface(np.zeros((10, 10, 10), dtype=bool), (1.0, 1.0, 1.0))
    assert mesh.n_faces == 0 and mesh.n_vertices == 0
    assert nmesh.mesh_volume_mm3(mesh) == 0.0


def test_vertices_are_in_millimetres_not_voxels():
    """A 20 mm sphere on 4 mm voxels spans 20 mm, not 5."""
    _, labels = sphere_phantom(radius_mm=20.0, shape=(30, 30, 30), spacing=(4.0, 4.0, 4.0))
    mesh = nmesh.surface_from_label(labels, 1)
    lo, hi = mesh.bounds
    assert (hi - lo).max() == pytest.approx(40.0, abs=6.0)


def test_vertices_are_xyz_ordered(sphere):
    """Volumes index [z, y, x]; renderers expect (x, y, z)."""
    _, labels = sphere
    array = np.zeros((60, 30, 15), dtype=np.uint8)
    array[5:55, 5:25, 5:10] = 1
    from nrrdvis.volume import Volume

    mesh = nmesh.surface_from_label(Volume(array, spacing=(1.0, 1.0, 1.0)), 1)
    lo, hi = mesh.bounds
    span = hi - lo
    # Longest extent is the z axis of the array, which must land last in xyz.
    assert np.argmax(span) == 2
    assert np.argmin(span) == 0


@pytest.mark.parametrize("target", [8000, 2000, 800])
def test_decimation_hits_the_face_budget(sphere, target):
    _, labels = sphere
    mesh = nmesh.surface_from_label(labels, 1)
    assert mesh.n_faces > target
    assert nmesh.decimate(mesh, target).n_faces <= target


def test_decimation_preserves_enclosed_volume(sphere):
    _, labels = sphere
    mesh = nmesh.surface_from_label(labels, 1)
    before = nmesh.mesh_volume_mm3(mesh)
    after = nmesh.mesh_volume_mm3(nmesh.decimate(mesh, 2000))
    assert after == pytest.approx(before, rel=0.02)


def test_decimation_below_current_count_is_a_no_op(sphere):
    _, labels = sphere
    mesh = nmesh.decimate(nmesh.surface_from_label(labels, 1), 2000)
    assert nmesh.decimate(mesh, 10_000) is mesh


def test_taubin_smoothing_does_not_shrink_the_surface(sphere):
    """Plain Laplacian smoothing would contract this measurably; Taubin must not."""
    _, labels = sphere
    mesh = nmesh.decimate(nmesh.surface_from_label(labels, 1), 4000)
    before = nmesh.mesh_volume_mm3(mesh)
    after = nmesh.mesh_volume_mm3(nmesh.smooth(mesh, iterations=20))
    assert after == pytest.approx(before, rel=0.02)


def test_smoothing_rejects_invalid_taubin_parameters(sphere):
    _, labels = sphere
    mesh = nmesh.decimate(nmesh.surface_from_label(labels, 1), 500)
    with pytest.raises(ValueError, match="mu < 0 < lambda"):
        nmesh.smooth(mesh, iterations=2, lambda_=0.5, mu=-0.4)


def test_payload_roundtrips_through_base64(sphere):
    import base64

    _, labels = sphere
    mesh = nmesh.decimate(nmesh.surface_from_label(labels, 1), 500)
    payload = mesh.to_payload()
    verts = np.frombuffer(base64.b64decode(payload["vertices"]), dtype=np.float32)
    faces = np.frombuffer(base64.b64decode(payload["faces"]), dtype=np.uint32)
    np.testing.assert_allclose(verts.reshape(-1, 3), mesh.vertices)
    np.testing.assert_array_equal(faces.reshape(-1, 3), mesh.faces)


def test_binary_payload_beats_json_numbers_on_size(sphere):
    """The reason v1's output was 18 MB: numbers serialised as decimal text."""
    import json

    _, labels = sphere
    mesh = nmesh.decimate(nmesh.surface_from_label(labels, 1), 4000)
    binary = len(json.dumps(mesh.to_payload()))
    as_numbers = len(json.dumps({
        "vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist(),
    }))
    assert binary < as_numbers / 2
