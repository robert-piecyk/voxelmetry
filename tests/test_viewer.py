"""Scene assembly and HTML output."""

import base64
import json
import re

import numpy as np

from voxelmetry.phantom import TORSO_LABEL_NAMES
from voxelmetry.viewer import html as nhtml
from voxelmetry.viewer import scene as nscene
from voxelmetry.volume import Volume


def test_scene_is_built_from_whatever_labels_are_present():
    """Nothing in the viewer knows about any particular organ."""
    array = np.zeros((30, 30, 30), dtype=np.uint8)
    array[5:15, 5:15, 5:15] = 1
    array[18:26, 18:26, 18:26] = 7  # arbitrary non-contiguous label value
    scene = nscene.scene_from_labelmap(Volume(array), names={7: "widget"})
    assert [m.name for m in scene.meshes] == ["label_1", "widget"]
    assert {m.label for m in scene.measurements} == {1, 7}


def test_structures_get_distinct_colours():
    array = np.zeros((40, 40, 40), dtype=np.uint8)
    for i, label in enumerate([1, 2, 3], start=0):
        array[5 + i * 10 : 12 + i * 10, 5:12, 5:12] = label
    scene = nscene.scene_from_labelmap(Volume(array))
    colors = [m.color for m in scene.meshes]
    assert len(set(colors)) == len(colors)


def test_split_labels_produce_one_mesh_per_component(torso):
    _, labels = torso
    scene = nscene.scene_from_labelmap(labels, names=TORSO_LABEL_NAMES, split_labels=(2,))
    lesion_meshes = [m for m in scene.meshes if m.name.startswith("lesion")]
    assert len(lesion_meshes) == 3
    assert all(m.n_faces > 0 for m in lesion_meshes)


def test_each_mesh_carries_its_own_measurements(torso):
    _, labels = torso
    scene = nscene.scene_from_labelmap(labels, names=TORSO_LABEL_NAMES)
    for mesh in scene.meshes:
        assert mesh.metadata["volume_ml"] > 0
        assert mesh.metadata["name"] == mesh.name


def test_face_budget_is_respected_per_structure(torso):
    _, labels = torso
    scene = nscene.scene_from_labelmap(labels, max_faces_per_structure=3000)
    assert all(m.n_faces <= 3000 for m in scene.meshes)


def test_empty_labelmap_gives_an_empty_scene():
    scene = nscene.scene_from_labelmap(Volume(np.zeros((10, 10, 10), dtype=np.uint8)))
    assert scene.meshes == [] and scene.measurements == []
    lo, hi = scene.bounds  # must not raise
    assert lo.shape == (3,)


def test_rendered_page_is_self_contained_apart_from_three_js(torso):
    _, labels = torso
    scene = nscene.scene_from_labelmap(labels, names=TORSO_LABEL_NAMES)
    page = nhtml.render(scene)
    external = re.findall(r'src="(http[^"]+)"', page)
    assert external == [nhtml.THREE_JS_CDN]
    assert "<style>" in page and "cdnjs" in page


def test_rendered_page_embeds_decodable_geometry(torso):
    _, labels = torso
    scene = nscene.scene_from_labelmap(labels, names=TORSO_LABEL_NAMES)
    page = nhtml.render(scene)
    blob = re.search(r"var SCENE = (\{.*?\});\n", page, re.S)
    assert blob, "scene payload not found in page"
    payload = json.loads(blob.group(1))

    assert payload["structures"], "no structures embedded"
    first = payload["structures"][0]
    verts = np.frombuffer(base64.b64decode(first["vertices"]), dtype=np.float32)
    assert verts.size == first["n_vertices"] * 3


def test_page_stays_small_enough_to_share(torso):
    """v1's single-structure Plotly export was 18 MB; a full scene must not be."""
    _, labels = torso
    scene = nscene.scene_from_labelmap(labels, names=TORSO_LABEL_NAMES, split_labels=(2,))
    assert len(nhtml.render(scene).encode()) < 2_000_000


def test_standalone_toggle_controls_the_document_skeleton(torso):
    _, labels = torso
    scene = nscene.scene_from_labelmap(labels)
    assert nhtml.render(scene, standalone=True).startswith("<!doctype html>")
    assert not nhtml.render(scene, standalone=False).startswith("<!doctype html>")


def test_title_is_escaped_into_the_page():
    scene = nscene.Scene(title='liver <study> & "notes"')
    page = nhtml.render(scene)
    assert "&lt;study&gt;" in page and "&amp;" in page


def test_write_creates_parent_directories(tmp_path, torso):
    _, labels = torso
    scene = nscene.scene_from_labelmap(labels)
    path = nhtml.write(scene, tmp_path / "deep" / "nested" / "scene.html")
    assert path.exists() and path.stat().st_size > 1000
