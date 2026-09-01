"""Physical measurements for the labels in a segmentation.

Volumes are voxel counts times the true voxel volume; distances are in
millimetres derived from the volume's spacing. Nothing here depends on the
voxel grid, so the same anatomy measures the same on any resampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .volume import Volume


@dataclass(frozen=True)
class LabelMeasurement:
    """Physical measurements for one label in a segmentation."""

    label: int
    name: str
    voxels: int
    volume_mm3: float
    volume_ml: float
    centroid_mm: tuple[float, float, float]
    bbox_mm: tuple[float, float, float]
    surface_area_mm2: float
    max_diameter_mm: float
    sphericity: float
    n_components: int
    largest_component_fraction: float
    min_axis_voxels: float
    resolution_limited: bool

    def as_dict(self) -> dict:
        return asdict(self)


def measure_label(
    labelmap: Volume,
    label: int,
    name: str | None = None,
    connectivity: int = 1,
) -> LabelMeasurement:
    """Measure one label within a segmentation volume.

    Args:
        labelmap: Integer-valued :class:`Volume` holding the segmentation.
        label: The value to measure.
        name: Human-readable name for the structure.
        connectivity: 1 for face-adjacency, 3 for the full 26-neighbourhood,
            when counting connected components.

    Note:
        ``n_components`` is sensitive to ``connectivity`` and is not a
        fragmentation score on its own. A ragged boundary leaves single voxels
        touching the main body only at a corner; face-adjacency counts each as
        a component, 26-adjacency absorbs them. One MSD Task03 liver label
        gives 395 at connectivity 1 and 27 at connectivity 3, for 0.04% of its
        volume. Use ``largest_component_fraction``, or the stray volume derived
        from it.

    Returns:
        A :class:`LabelMeasurement`. Empty labels yield an all-zero record
        rather than raising, so a batch report can include absent structures.
    """
    mask = labelmap.array == label
    voxels = int(mask.sum())
    spacing = np.asarray(labelmap.spacing, dtype=float)
    display_name = name or f"label_{label}"

    if voxels == 0:
        return LabelMeasurement(
            label=label, name=display_name, voxels=0, volume_mm3=0.0, volume_ml=0.0,
            centroid_mm=(0.0, 0.0, 0.0), bbox_mm=(0.0, 0.0, 0.0), surface_area_mm2=0.0,
            max_diameter_mm=0.0, sphericity=0.0, n_components=0,
            largest_component_fraction=0.0, min_axis_voxels=0.0,
            resolution_limited=False,
        )

    volume_mm3 = voxels * labelmap.voxel_volume_mm3
    # 1 millilitre is 1000 mm3, not 1e6. v1 had this wrong.
    volume_ml = volume_mm3 / 1000.0

    centroid_vox = np.asarray(ndimage.center_of_mass(mask), dtype=float)
    centroid_mm = tuple(centroid_vox * spacing + np.asarray(labelmap.origin, dtype=float))

    bbox_mm = []
    for axis in range(3):
        present = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        bbox_mm.append(float((present[-1] - present[0] + 1) * spacing[axis]))

    structure = ndimage.generate_binary_structure(3, connectivity)
    components, n_components = ndimage.label(mask, structure=structure)
    if n_components > 1:
        sizes = np.bincount(components.ravel())[1:]
        largest_fraction = float(sizes.max() / sizes.sum())
    else:
        largest_fraction = 1.0

    surface_area = surface_area_mm2(mask, labelmap.spacing)
    diameter = max_diameter_mm(mask, labelmap.spacing)

    # How many voxels the structure spans on its thinnest axis. Below roughly
    # five, partial-volume effects and the surface pre-smoothing both bite, and
    # shape descriptors stop meaning much.
    min_axis_voxels = float(min(b / s for b, s in zip(bbox_mm, spacing, strict=True)))
    resolution_limited = min_axis_voxels < RESOLUTION_LIMIT_VOXELS

    # Sphericity: how close the shape is to a sphere of the same volume.
    # 1.0 is a perfect sphere; a long thin structure tends toward 0.
    if surface_area > 0:
        equivalent_sphere_area = np.pi ** (1 / 3) * (6 * volume_mm3) ** (2 / 3)
        sphericity = float(equivalent_sphere_area / surface_area)
        # Values above 1 are impossible; no shape beats a sphere on
        # area-to-volume. They appear only when smoothing has shaved area off a
        # structure too small to resolve, so clamp and let resolution_limited
        # carry the caveat.
        sphericity = min(sphericity, 1.0)
    else:
        sphericity = 0.0

    return LabelMeasurement(
        label=label,
        name=display_name,
        voxels=voxels,
        volume_mm3=float(volume_mm3),
        volume_ml=float(volume_ml),
        centroid_mm=tuple(float(c) for c in centroid_mm),  # type: ignore[arg-type]
        bbox_mm=tuple(bbox_mm),  # type: ignore[arg-type]
        surface_area_mm2=float(surface_area),
        max_diameter_mm=float(diameter),
        sphericity=sphericity,
        n_components=int(n_components),
        largest_component_fraction=largest_fraction,
        min_axis_voxels=min_axis_voxels,
        resolution_limited=resolution_limited,
    )


def measure_all(
    labelmap: Volume,
    names: dict[int, str] | None = None,
    ignore: tuple[int, ...] = (0,),
) -> list[LabelMeasurement]:
    """Measure every label present in a segmentation.

    Args:
        labelmap: Integer-valued segmentation volume.
        names: Optional mapping of label value to structure name.
        ignore: Label values to skip; background (0) by default.

    Returns:
        One :class:`LabelMeasurement` per label, ordered by label value.
    """
    names = names or {}
    present = [int(v) for v in np.unique(labelmap.array) if int(v) not in ignore]
    return [measure_label(labelmap, label, names.get(label)) for label in present]


#: Gaussian width in voxels, applied before isosurfacing a binary mask.
#: Marching cubes on a hard 0/1 mask traces a staircase and overestimates area
#: by ~9%. Calibrated against analytic spheres (tests/test_measure.py): 0.8
#: holds the error under 1% for radii 15-30 voxels; 0 gives +8.6% and 1.5
#: starts eroding sharp features.
SURFACE_SMOOTHING_SIGMA = 0.8

#: Below this many voxels across its thinnest axis, a structure's shape
#: descriptors (sphericity, surface area) are dominated by sampling artifacts.
#: Volume and diameter stay usable; shape does not.
RESOLUTION_LIMIT_VOXELS = 5.0


def surface_area_mm2(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    sigma: float = SURFACE_SMOOTHING_SIGMA,
) -> float:
    """Surface area of a binary mask, via a marching-cubes triangulation.

    Args:
        mask: Boolean array of the structure.
        spacing: Millimetres per voxel, ``(z, y, x)``.
        sigma: Pre-smoothing width in voxels; see
            :data:`SURFACE_SMOOTHING_SIGMA`. Pass 0 to disable.

    Returns:
        Surface area in square millimetres, or 0.0 for an empty mask.
    """
    from skimage import measure as skmeasure

    from .mesh import isosurface

    # Sigma is in voxels on every axis: the staircase being corrected is an
    # artifact of the sampling grid, not of physical distance. isosurface()
    # falls back to the unsmoothed mask for structures too thin to survive the
    # blur, so a one-slice lesion reports an approximate area rather than zero.
    verts, faces, _ = isosurface(mask, spacing, sigma)
    if not len(faces):
        return 0.0
    return float(skmeasure.mesh_surface_area(verts, faces))


def max_diameter_mm(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    block: int = 2048,
) -> float:
    """Largest straight-line distance between any two voxels in the mask.

    This is the true 3-D Feret diameter. v1 approximated it by picking the
    axial slice with the most foreground pixels and taking that slice's index
    span, which ignores oblique extent and cannot see the z direction at all.

    Both endpoints of the longest chord necessarily lie on the convex hull, so
    the hull is computed first and the exact maximum taken over its vertices.
    The pairwise distances are evaluated in blocks rather than as one matrix:
    a liver hull can carry several thousand vertices, and materialising the
    full matrix for those would cost hundreds of megabytes. Blocking keeps
    peak memory at ``block x n`` while still producing the exact answer, which
    subsampling the hull would not.

    Args:
        mask: Boolean array of the structure.
        spacing: Millimetres per voxel, ``(z, y, x)``.
        block: Rows evaluated per iteration; trades memory against call count.

    Returns:
        The diameter in millimetres, or 0.0 for an empty or single-voxel mask.
    """
    if not mask.any():
        return 0.0

    # Surface voxels only: an interior point can never be a chord endpoint.
    eroded = ndimage.binary_erosion(mask, ndimage.generate_binary_structure(3, 1))
    surface = mask & ~eroded
    points = np.argwhere(surface if surface.any() else mask).astype(np.float64)
    points *= np.asarray(spacing, dtype=float)

    if len(points) < 2:
        return 0.0

    if len(points) > 4:
        try:
            from scipy.spatial import ConvexHull

            points = points[ConvexHull(points).vertices]
        except Exception:  # noqa: BLE001 - degenerate (coplanar) point sets
            pass

    squared_max = 0.0
    for start in range(0, len(points), block):
        chunk = points[start : start + block]
        # (chunk, 1, 3) against (1, n, 3) broadcasts to one block of distances.
        deltas = chunk[:, None, :] - points[None, :, :]
        squared_max = max(squared_max, float(np.einsum("ijk,ijk->ij", deltas, deltas).max()))

    return float(np.sqrt(squared_max))


def to_table(measurements: list[LabelMeasurement]) -> str:
    """Render measurements as a fixed-width text table."""
    if not measurements:
        return "(no labels found)"
    header = (
        f"{'structure':<16}{'label':>6}{'volume (mL)':>14}{'max Ø (mm)':>13}"
        f"{'sphericity':>12}{'parts':>7}"
    )
    rows = [header, "-" * len(header)]
    flagged = False
    for m in measurements:
        mark = " *" if m.resolution_limited else ""
        flagged |= m.resolution_limited
        rows.append(
            f"{m.name[:16]:<16}{m.label:>6}{m.volume_ml:>14.2f}"
            f"{m.max_diameter_mm:>13.1f}{m.sphericity:>12.3f}{m.n_components:>5}{mark:<2}"
        )
    if flagged:
        rows.append("")
        rows.append(
            f"* spans under {RESOLUTION_LIMIT_VOXELS:.0f} voxels on its thinnest axis; "
            "treat shape descriptors as indicative only"
        )
    return "\n".join(rows)


def measure_components(
    labelmap: Volume,
    label: int,
    name: str | None = None,
    min_volume_mm3: float = 0.0,
    connectivity: int = 1,
) -> list[LabelMeasurement]:
    """Measure each connected component of a label separately.

    Aggregates mislead for multifocal structures: three lesions 15 mm apart
    report a combined max diameter spanning all of them. Burden is reported per
    lesion, as RECIST does, so the label is split first.

    Args:
        labelmap: Integer-valued segmentation volume.
        label: The label to split into components.
        name: Base name; components are suffixed ``_1``, ``_2``, ...
        min_volume_mm3: Drop components smaller than this. Useful for
            suppressing single-voxel speckle in a predicted segmentation.
        connectivity: 1 for face-adjacency, 3 for the full 26-neighbourhood.

    Returns:
        One measurement per surviving component, largest first.
    """
    mask = labelmap.array == label
    display_name = name or f"label_{label}"
    if not mask.any():
        return []

    structure = ndimage.generate_binary_structure(3, connectivity)
    components, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return []

    sizes = np.bincount(components.ravel())
    sizes[0] = 0  # background
    order = np.argsort(sizes)[::-1]
    min_voxels = min_volume_mm3 / labelmap.voxel_volume_mm3

    # Each component is measured inside its own bounding box. On the full grid
    # a 1 mL lesion in a 165-million-voxel study costs ~10 s, since marching
    # cubes, erosion and the component pass each sweep the whole array.
    # find_objects returns every bounding box in one pass.
    boxes = ndimage.find_objects(components)
    pad = [max(int(round(2.0 * max(labelmap.spacing) / s)), 1) for s in labelmap.spacing]
    origin = np.asarray(labelmap.origin, dtype=float)
    spacing = np.asarray(labelmap.spacing, dtype=float)

    results = []
    for rank, component_id in enumerate(order, start=1):
        if sizes[component_id] == 0 or sizes[component_id] < min_voxels:
            continue
        box = boxes[component_id - 1]
        if box is None:  # pragma: no cover - only if labelling skipped an id
            continue

        window = tuple(
            slice(max(axis.start - pad[a], 0), min(axis.stop + pad[a], components.shape[a]))
            for a, axis in enumerate(box)
        )
        sub = components[window] == component_id
        # The origin shifts by however much was trimmed, so the centroid stays
        # in patient coordinates rather than in crop-local ones.
        sub_origin = origin + np.array([w.start for w in window], dtype=float) * spacing

        isolated = Volume(
            np.where(sub, label, 0).astype(labelmap.array.dtype),
            spacing=labelmap.spacing,
            origin=tuple(sub_origin),
            name=f"{display_name}_{rank}",
        )
        results.append(measure_label(isolated, label, f"{display_name}_{rank}"))
    return results


def lesion_burden(
    measurements: list[LabelMeasurement],
    reference_volume_ml: float | None = None,
) -> dict[str, float]:
    """Summarise a set of lesion components into reportable figures.

    Args:
        measurements: Per-component measurements, as from
            :func:`measure_components`.
        reference_volume_ml: Volume of the containing organ, if known, used to
            express burden as a percentage of that organ.

    Returns:
        A dict with the lesion count, total and largest volume, the largest
        diameter across lesions, and the sum of diameters (the quantity RECIST
        tracks between timepoints).
    """
    if not measurements:
        return {"n_lesions": 0, "total_volume_ml": 0.0, "largest_volume_ml": 0.0,
                "largest_diameter_mm": 0.0, "sum_of_diameters_mm": 0.0}

    total = float(sum(m.volume_ml for m in measurements))
    summary = {
        "n_lesions": len(measurements),
        "total_volume_ml": total,
        "largest_volume_ml": float(max(m.volume_ml for m in measurements)),
        "largest_diameter_mm": float(max(m.max_diameter_mm for m in measurements)),
        "sum_of_diameters_mm": float(sum(m.max_diameter_mm for m in measurements)),
    }
    if reference_volume_ml:
        summary["burden_percent"] = 100.0 * total / reference_volume_ml
    return summary
