"""Execute the generated viewer in a headless DOM.

The Python tests check that the page contains the right bytes. They cannot
check that the JavaScript inside it works. This drives the real page through
jsdom with a stubbed three.js, so the scene assembly, base64 decoding, panel
wiring and clip-plane maths are all exercised.

Skipped when node or jsdom is unavailable, so the suite still runs on a bare
Python environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from nrrdvis.phantom import TORSO_LABEL_NAMES
from nrrdvis.viewer import html as nhtml
from nrrdvis.viewer import scene_from_labelmap

HARNESS = Path(__file__).parent / "js" / "run_viewer.js"


def _node_env() -> dict[str, str] | None:
    """Environment for running the harness, or None if node/jsdom are missing."""
    if shutil.which("node") is None or not HARNESS.exists():
        return None
    env = dict(os.environ)
    probe = subprocess.run(
        ["node", "-e", "require.resolve('jsdom'); console.log('ok')"],
        capture_output=True, text=True, env=env, cwd=HARNESS.parent,
    )
    return env if probe.returncode == 0 else None


requires_node = pytest.mark.skipif(_node_env() is None, reason="node with jsdom not available")


@requires_node
def test_generated_viewer_runs_in_a_headless_dom(tmp_path, torso):
    _, labels = torso
    scene = scene_from_labelmap(labels, names=TORSO_LABEL_NAMES, split_labels=(2,))
    page = nhtml.write(scene, tmp_path / "scene.html")

    result = subprocess.run(
        ["node", str(HARNESS), str(page)],
        capture_output=True, text=True, env=_node_env(), cwd=HARNESS.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all checks passed" in result.stdout


@requires_node
def test_viewer_handles_a_single_structure(tmp_path):
    """A one-label scene is the common case and must not depend on having many."""
    import numpy as np

    from nrrdvis.volume import Volume

    array = np.zeros((30, 30, 30), dtype=np.uint8)
    array[8:22, 8:22, 8:22] = 1
    scene = scene_from_labelmap(Volume(array), names={1: "block"})
    page = nhtml.write(scene, tmp_path / "single.html")

    result = subprocess.run(
        ["node", str(HARNESS), str(page)],
        capture_output=True, text=True, env=_node_env(), cwd=HARNESS.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_page_declares_only_the_pinned_cdn_dependency(torso):
    """Anything else loading from the network would break an offline viewer."""
    import re

    _, labels = torso
    page = nhtml.render(scene_from_labelmap(labels))
    urls = re.findall(r'(?:src|href)="(https?://[^"]+)"', page)
    assert urls == [nhtml.THREE_JS_CDN]


def test_scene_payload_is_valid_json(torso):
    import re

    _, labels = torso
    page = nhtml.render(scene_from_labelmap(labels, names=TORSO_LABEL_NAMES))
    blob = re.search(r"var SCENE = (\{.*?\});\n", page, re.S)
    payload = json.loads(blob.group(1))
    assert payload["structures"] and payload["measurements"]
