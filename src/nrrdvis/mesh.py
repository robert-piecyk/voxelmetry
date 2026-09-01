"""Label maps to triangle meshes.

A liver at 1 mm yields on the order of half a million triangles, so meshes are
decimated to a face budget and smoothed before export, and serialised as base64
binary rather than JSON number lists. A multi-structure scene lands in a few
hundred kilobytes.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .volume import Volume


@dataclass
class Mesh:
    """A triangle mesh in patient millimetre coordinates.

    Attributes:
        vertices: ``(N, 3)`` float32 positions, ordered ``(x, y, z)`` because
            that is what rendering conventions expect. Extraction handles the
            flip from the volume's ``(z, y, x)``.
        faces: ``(M, 3)`` uint32 vertex indices.
        name: Structure name shown in the viewer.
        color: ``#rrggbb`` display colour.
        metadata: Free-form extras carried into the viewer, typically the
            measurement record for this structure.
    """

    vertices: np.ndarray
    faces: np.ndarray
    name: str = "surface"
    color: str = "#c8553d"
    metadata: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as ``(min_xyz, max_xyz)``."""
        if self.n_vertices == 0:
            zeros = np.zeros(3, dtype=np.float32)
            return zeros, zeros
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def centroid(self) -> np.ndarray:
        if self.n_vertices == 0:
            return np.zeros(3, dtype=np.float32)
        return self.vertices.mean(axis=0)

    def to_payload(self) -> dict:
        """Serialise to the base64 form the HTML viewer consumes."""
        return {
            "name": self.name,
            "color": self.color,
            "vertices": base64.b64encode(
                np.ascontiguousarray(self.vertices, dtype=np.float32).tobytes()
            ).decode("ascii"),
            "faces": base64.b64encode(
                np.ascontiguousarray(self.faces, dtype=np.uint32).tobytes()
            ).decode("ascii"),
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"Mesh({self.name!r}, {self.n_vertices} verts, {self.n_faces} faces)"


def isosurface(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    sigma: float = 1.0,
    step_size: int = 1,
    level: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Triangulate a binary mask, degrading gracefully when smoothing erases it.

    Pre-smoothing removes the marching-cubes staircase, but a structure one
    voxel thick on some axis has its peak pulled below ``level`` by the same
    blur and the isosurface comes back empty. At 5 mm slice thickness a lesion
    confined to one slice is routine.

    When the smoothed pass yields nothing, this falls back to the unsmoothed
    mask. That result carries the staircase overestimate, but is preferable to
    a false zero; the returned sigma tells the caller which pass was used.

    Args:
        mask: Boolean array of the structure, indexed ``[z, y, x]``.
        spacing: Millimetres per voxel, ``(z, y, x)``.
        sigma: Preferred pre-smoothing width in voxels; 0 disables smoothing.
        step_size: Marching-cubes stride.
        level: Isolevel to trace.

    Returns:
        ``(vertices, faces, sigma_used)`` in padded voxel-millimetre
        coordinates. ``sigma_used`` is 0.0 when the fallback was taken, and
        the arrays are empty when even that finds no surface.
    """
    from skimage import measure as skmeasure

    empty = (np.zeros((0, 3), np.float64), np.zeros((0, 3), np.int64), 0.0)
    if not mask.any():
        return empty

    padded = np.pad(mask.astype(np.float32), 2, mode="constant", constant_values=0)

    for attempt in ([sigma, 0.0] if sigma > 0 else [0.0]):
        data = ndimage.gaussian_filter(padded, attempt) if attempt > 0 else padded
        # Cheaper than catching the exception, and states the condition.
        if data.max() <= level:
            continue
        try:
            verts, faces, _, _ = skmeasure.marching_cubes(
                data, level=level, spacing=spacing, step_size=step_size,
                allow_degenerate=False,
            )
        except (RuntimeError, ValueError):  # pragma: no cover - degenerate masks
            continue
        if len(faces):
            return verts, faces, attempt

    return empty


def extract_surface(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    smoothing_sigma: float = 1.0,
    step_size: int = 1,
    name: str = "surface",
    color: str = "#c8553d",
) -> Mesh:
    """Extract an isosurface from a binary mask via marching cubes.

    Args:
        mask: Boolean array, indexed ``[z, y, x]``.
        spacing: Millimetres per voxel, ``(z, y, x)``.
        origin: Patient-space offset of voxel ``[0, 0, 0]``, ``(z, y, x)``.
        smoothing_sigma: Gaussian pre-smoothing in voxels. Without it the
            surface is visibly terraced along the slice axis.
        step_size: Marching-cubes stride. 2 quarters the triangle count at the
            cost of detail, and is a cheaper first pass than decimating later.
        name: Structure name.
        color: Display colour.

    Returns:
        A :class:`Mesh` with vertices in ``(x, y, z)`` millimetres. An empty
        mask yields an empty mesh rather than raising.
    """
    verts, faces, _ = isosurface(mask, spacing, smoothing_sigma, step_size)
    if not len(faces):
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), name, color)

    # Undo the 2-voxel pad, then shift into patient coordinates.
    verts -= 2.0 * np.asarray(spacing, dtype=float)
    verts += np.asarray(origin, dtype=float)

    # Volume axes are (z, y, x); renderers want (x, y, z).
    verts = verts[:, ::-1]

    return Mesh(
        vertices=np.ascontiguousarray(verts, dtype=np.float32),
        faces=np.ascontiguousarray(faces, dtype=np.uint32),
        name=name,
        color=color,
    )


def surface_from_label(
    labelmap: Volume,
    label: int,
    name: str | None = None,
    color: str = "#c8553d",
    **kwargs,
) -> Mesh:
    """Extract the surface of one label from a segmentation volume."""
    return extract_surface(
        labelmap.array == label,
        spacing=labelmap.spacing,
        origin=labelmap.origin,
        name=name or f"label_{label}",
        color=color,
        **kwargs,
    )


def decimate(mesh: Mesh, target_faces: int) -> Mesh:
    """Reduce triangle count while preserving overall shape.

    Uses quadric edge collapse when ``trimesh`` (with its fast backend) is
    available, and falls back to vertex clustering otherwise, which is cruder
    but has no extra dependency and never fails.

    Args:
        mesh: The mesh to simplify.
        target_faces: Desired face count. Meshes already at or below this are
            returned unchanged.

    Returns:
        The simplified mesh, carrying the original name, colour and metadata.
    """
    if mesh.n_faces <= target_faces or mesh.n_faces == 0:
        return mesh

    try:
        import trimesh

        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        simplified = tm.simplify_quadric_decimation(face_count=target_faces)
        if len(simplified.faces) > 0:
            return Mesh(
                np.ascontiguousarray(simplified.vertices, dtype=np.float32),
                np.ascontiguousarray(simplified.faces, dtype=np.uint32),
                mesh.name, mesh.color, dict(mesh.metadata),
            )
    except Exception:  # noqa: BLE001 - optional dependency, any failure falls through
        pass

    # The clustering grid only approximates a face count, so step the cell
    # size up until the result is at or under target (bounded retries).
    result = mesh
    for attempt in range(6):
        result = _cluster_decimate(mesh, target_faces, growth=1.35**attempt)
        if result.n_faces <= target_faces:
            break
    return result


def _cluster_decimate(mesh: Mesh, target_faces: int, growth: float = 1.0) -> Mesh:
    """Vertex-clustering fallback: snap vertices to a grid, drop degenerates.

    Args:
        mesh: Mesh to simplify.
        target_faces: Desired face count, approximated by the grid step.
        growth: Multiplier on the grid step, used by the caller to retry
            coarser when the first estimate overshoots.
    """
    lo, hi = mesh.bounds
    diagonal = float(np.linalg.norm(hi - lo)) or 1.0
    # Face count falls roughly with the square of the grid step, so the step
    # needed for a target count scales as 1/sqrt(target). The constant is
    # approximate, hence the corrective retries below.
    cell = growth * diagonal / max(np.sqrt(target_faces), 1.0)

    quantised = np.round((mesh.vertices - lo) / cell).astype(np.int64)
    unique, inverse = np.unique(quantised, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    new_vertices = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).reshape(-1, 1)
    np.add.at(new_vertices, inverse, mesh.vertices)
    new_vertices /= np.maximum(counts, 1)

    new_faces = inverse[mesh.faces]
    keep = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    return Mesh(
        np.ascontiguousarray(new_vertices, dtype=np.float32),
        np.ascontiguousarray(new_faces[keep], dtype=np.uint32),
        mesh.name, mesh.color, dict(mesh.metadata),
    )


def smooth(mesh: Mesh, iterations: int = 10, lambda_: float = 0.5, mu: float = -0.53) -> Mesh:
    """Taubin smoothing: relax the surface without the shrinkage Laplacian causes.

    Laplacian smoothing pulls every vertex toward its neighbours and contracts
    a closed surface, so volume measured off the mesh drifts downward with each
    iteration. Taubin alternates a positive and a slightly larger negative
    step, cancelling that low-frequency shrinkage while still removing the
    high-frequency staircase.

    Args:
        mesh: The mesh to smooth.
        iterations: Number of shrink/expand pairs.
        lambda_: Shrinking step size, in ``(0, 1)``.
        mu: Expanding step size; must be negative with ``|mu| > lambda_``.

    Returns:
        The smoothed mesh.
    """
    if mesh.n_faces == 0 or iterations <= 0:
        return mesh
    if not (mu < 0 < lambda_ < -mu):
        raise ValueError(f"Taubin requires mu < 0 < lambda < -mu, got lambda={lambda_}, mu={mu}")

    n = mesh.n_vertices
    edges = np.vstack([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])

    from scipy.sparse import coo_matrix

    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0

    vertices = mesh.vertices.astype(np.float64)
    for _ in range(iterations):
        for step in (lambda_, mu):
            neighbour_mean = adjacency @ vertices / degree[:, None]
            vertices += step * (neighbour_mean - vertices)

    return Mesh(
        np.ascontiguousarray(vertices, dtype=np.float32),
        mesh.faces, mesh.name, mesh.color, dict(mesh.metadata),
    )


def mesh_volume_mm3(mesh: Mesh) -> float:
    """Enclosed volume via the divergence theorem, as a check on voxel counting.

    Agreement between this and ``voxels * voxel_volume`` is a strong signal
    that extraction, decimation and smoothing have not distorted the geometry.
    """
    if mesh.n_faces == 0:
        return 0.0
    triangles = mesh.vertices[mesh.faces].astype(np.float64)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)
