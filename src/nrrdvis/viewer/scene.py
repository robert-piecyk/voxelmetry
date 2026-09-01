"""A renderable scene: surfaces plus the measurements that describe them.

Keeping the scene separate from the HTML writer means the same description can
later be sent to a different backend without touching the assembly logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..measure import LabelMeasurement, measure_components, measure_label
from ..mesh import Mesh, decimate, smooth, surface_from_label
from ..volume import Volume

#: Colour-blind-safe palette, ordered so the first entries stay distinguishable
#: under deuteranopia and protanopia. Structures are assigned in label order.
DEFAULT_PALETTE: tuple[str, ...] = (
    "#c1553b",  # clay
    "#2f6f9f",  # steel blue
    "#c9a227",  # ochre
    "#5b8c5a",  # sage
    "#7b5aa6",  # violet
    "#c2708f",  # rose
    "#3f8f8a",  # teal
    "#8a6a4f",  # umber
)


@dataclass
class Scene:
    """A set of surfaces with optional measurements and provenance.

    Attributes:
        title: Shown as the viewer heading.
        meshes: Surfaces to render, in draw order.
        measurements: Rows for the measurements panel.
        subtitle: Short line under the title, typically the source file.
        provenance: Key/value settings recorded in the viewer's info panel, so
            a saved scene explains how it was produced.
    """

    title: str = "nrrdvis"
    meshes: list[Mesh] = field(default_factory=list)
    measurements: list[LabelMeasurement] = field(default_factory=list)
    subtitle: str = ""
    provenance: dict[str, str] = field(default_factory=dict)

    def add(self, mesh: Mesh) -> Scene:
        self.meshes.append(mesh)
        return self

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box across every mesh, as ``(min_xyz, max_xyz)``."""
        populated = [m for m in self.meshes if m.n_vertices > 0]
        if not populated:
            return np.zeros(3, np.float32), np.ones(3, np.float32)
        lows = np.vstack([m.bounds[0] for m in populated])
        highs = np.vstack([m.bounds[1] for m in populated])
        return lows.min(axis=0), highs.max(axis=0)

    def total_faces(self) -> int:
        return sum(m.n_faces for m in self.meshes)


def scene_from_labelmap(
    labelmap: Volume,
    names: dict[int, str] | None = None,
    colors: dict[int, str] | None = None,
    split_labels: tuple[int, ...] = (),
    max_faces_per_structure: int = 60_000,
    smooth_iterations: int = 8,
    ignore: tuple[int, ...] = (0,),
    title: str | None = None,
    subtitle: str = "",
    min_component_volume_mm3: float = 0.0,
) -> Scene:
    """Build a complete scene from any integer-labelled segmentation.

    Takes whatever labels are present, extracts a surface per label, measures
    each and assigns colours in order. Nothing is organ-specific.

    Args:
        labelmap: Integer-valued segmentation volume.
        names: Optional label-value to structure-name mapping.
        colors: Optional label-value to ``#rrggbb`` mapping; unspecified labels
            take the next entry from :data:`DEFAULT_PALETTE`.
        split_labels: Labels to break into connected components, each rendered
            and measured separately. Typically the lesion label.
        max_faces_per_structure: Decimation budget per surface, which is what
            keeps total output size predictable.
        smooth_iterations: Taubin smoothing passes applied after decimation.
        ignore: Label values to skip; background by default.
        title: Scene title. Defaults to the volume name.
        subtitle: Short descriptive line.
        min_component_volume_mm3: When splitting, drop components below this
            size. Useful for suppressing speckle in predicted segmentations.

    Returns:
        A populated :class:`Scene`.
    """
    names = names or {}
    colors = colors or {}
    present = [int(v) for v in np.unique(labelmap.array) if int(v) not in ignore]

    scene = Scene(
        title=title or labelmap.name,
        subtitle=subtitle,
        provenance={
            "grid": "x".join(str(n) for n in labelmap.shape),
            "spacing_mm": " x ".join(f"{s:.3g}" for s in labelmap.spacing),
            "field of view (mm)": " x ".join(f"{e:.0f}" for e in labelmap.extent_mm),
            "labels": ", ".join(str(v) for v in present) or "none",
        },
    )

    color_index = 0
    for label in present:
        base_name = names.get(label, f"label_{label}")

        if label in split_labels:
            components = measure_components(
                labelmap, label, base_name, min_volume_mm3=min_component_volume_mm3
            )
            scene.measurements.extend(components)
            color = colors.get(label) or DEFAULT_PALETTE[color_index % len(DEFAULT_PALETTE)]
            color_index += 1
            _add_component_meshes(
                scene, labelmap, label, components, color,
                max_faces_per_structure, smooth_iterations, min_component_volume_mm3,
            )
            continue

        scene.measurements.append(measure_label(labelmap, label, base_name))
        color = colors.get(label) or DEFAULT_PALETTE[color_index % len(DEFAULT_PALETTE)]
        color_index += 1

        mesh = surface_from_label(labelmap, label, name=base_name, color=color)
        mesh = decimate(mesh, max_faces_per_structure)
        mesh = smooth(mesh, smooth_iterations)
        mesh.metadata = scene.measurements[-1].as_dict()
        scene.add(mesh)

    return scene


def _add_component_meshes(
    scene: Scene,
    labelmap: Volume,
    label: int,
    components: list[LabelMeasurement],
    color: str,
    max_faces: int,
    smooth_iterations: int,
    min_volume_mm3: float,
) -> None:
    """Extract one mesh per connected component of ``label``."""
    from scipy import ndimage

    mask = labelmap.array == label
    labelled, n = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    if n == 0:
        return

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    min_voxels = min_volume_mm3 / labelmap.voxel_volume_mm3

    # Small structures need a proportionally smaller face budget, or three
    # lesions cost as much as the organ containing them.
    # Same cropping as measure_components, and for the same reason: extracting
    # a lesion surface from the full grid costs seconds per lesion. The crop
    # carries a corrected origin, so meshes still land in patient coordinates.
    margin_mm = 2.0 * max(labelmap.spacing)

    for rank, component_id in enumerate(order, start=1):
        if sizes[component_id] == 0 or sizes[component_id] < min_voxels:
            continue
        if rank > len(components):
            break
        budget = max(int(max_faces * sizes[component_id] / max(sizes.max(), 1)), 500)
        component_mask = labelled == component_id
        cropped = labelmap.with_array(component_mask.astype(np.uint8)).crop_to_mask(
            component_mask, margin_mm=margin_mm
        )
        mesh = surface_from_label(
            cropped, 1, name=f"{components[rank - 1].name}", color=color,
        )
        mesh = smooth(decimate(mesh, budget), smooth_iterations)
        mesh.metadata = components[rank - 1].as_dict()
        scene.add(mesh)
