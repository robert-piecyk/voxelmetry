"""Synthetic volumes with known analytic geometry.

These exist so the measurement code can be checked against arithmetic rather
than against itself. A sphere of radius r has volume 4/3 pi r^3 and surface
area 4 pi r^2 exactly, so any drift in the voxel-counting, resampling or
marching-cubes paths shows up immediately as a percentage error.

They double as a demo dataset: the CLI and viewer run end-to-end on a phantom
with no download required.
"""

from __future__ import annotations

import numpy as np

from .volume import Volume

# Approximate Hounsfield units, so phantoms exercise realistic windowing.
HU_AIR = -1000
HU_FAT = -100
HU_SOFT_TISSUE = 50
HU_LIVER = 110
HU_LESION = 25
HU_BONE = 700


def sphere_mask(
    shape: tuple[int, int, int],
    centre: tuple[float, float, float],
    radius_mm: float,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Boolean mask of a sphere defined in millimetres, not voxels.

    Args:
        shape: Grid size, ``(z, y, x)``.
        centre: Sphere centre in voxel coordinates, ``(z, y, x)``.
        radius_mm: Radius in millimetres.
        spacing: Millimetres per voxel, ``(z, y, x)``.

    Returns:
        Boolean array of ``shape``, True inside the sphere.
    """
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    squared = sum(
        (((g - c) * s) ** 2) for g, c, s in zip(grids, centre, spacing, strict=True)
    )
    return squared <= radius_mm**2


def analytic_sphere_volume_mm3(radius_mm: float) -> float:
    """Exact volume of a sphere, for comparison against a measured one."""
    return 4.0 / 3.0 * np.pi * radius_mm**3


def analytic_sphere_area_mm2(radius_mm: float) -> float:
    """Exact surface area of a sphere."""
    return 4.0 * np.pi * radius_mm**2


def sphere_phantom(
    radius_mm: float = 30.0,
    shape: tuple[int, int, int] = (80, 80, 80),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[Volume, Volume]:
    """A single centred sphere, as an intensity volume and a label map.

    Returns:
        ``(image, labelmap)`` where the sphere carries label 1.
    """
    centre = tuple(n / 2 for n in shape)
    mask = sphere_mask(shape, centre, radius_mm, spacing)

    image = np.full(shape, HU_AIR, dtype=np.int16)
    image[mask] = HU_LIVER
    labels = mask.astype(np.uint8)

    return (
        Volume(image, spacing=spacing, name="sphere_phantom"),
        Volume(labels, spacing=spacing, name="sphere_phantom_labels"),
    )


def torso_phantom(
    shape: tuple[int, int, int] = (96, 128, 128),
    spacing: tuple[float, float, float] = (2.5, 1.5, 1.5),
    n_lesions: int = 3,
    seed: int = 0,
) -> tuple[Volume, Volume]:
    """A body cross-section holding an organ with lesions inside it.

    Structurally this mimics abdominal CT: a soft-tissue body surrounded by
    air, a bright vertebra, one large organ, and small low-attenuation lesions
    within the organ. It is what the demo pipeline runs on when no real scan is
    available.

    Args:
        shape: Grid size, ``(z, y, x)``.
        spacing: Millimetres per voxel, deliberately anisotropic by default so
            that any code assuming isotropic voxels fails loudly.
        n_lesions: How many lesions to place inside the organ.
        seed: Seed for lesion placement.

    Returns:
        ``(image, labelmap)`` with labels 1 = organ, 2 = lesion, 3 = bone.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    sz, sy, sx = spacing

    image = np.full(shape, HU_AIR, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.uint8)

    # Body: an elliptical cylinder spanning most of the field of view.
    zz, yy, xx = np.ogrid[0:nz, 0:ny, 0:nx]
    body = (((yy - ny / 2) * sy / (ny * sy * 0.40)) ** 2
            + ((xx - nx / 2) * sx / (nx * sx * 0.46)) ** 2) <= 1.0
    body = np.broadcast_to(body, shape) & (zz >= nz * 0.05) & (zz < nz * 0.95)
    image[body] = HU_FAT
    inner = ndimage_erode(body, iterations=3)
    image[inner] = HU_SOFT_TISSUE

    # Vertebra: a bright posterior column, the kind of structure that breaks
    # naive "brightest connected component is the body" heuristics.
    bone = sphere_mask(shape, (nz / 2, ny * 0.76, nx / 2), min(nx * sx, ny * sy) * 0.07, spacing)
    bone = bone | np.roll(bone, int(nz * 0.25), axis=0) | np.roll(bone, -int(nz * 0.25), axis=0)
    bone &= body
    image[bone] = HU_BONE
    labels[bone] = 3

    # Organ: a large ellipsoid offset to one side, as a liver sits.
    organ_centre = (nz / 2, ny * 0.42, nx * 0.36)
    organ = (((zz - organ_centre[0]) * sz / (nz * sz * 0.30)) ** 2
             + ((yy - organ_centre[1]) * sy / (ny * sy * 0.20)) ** 2
             + ((xx - organ_centre[2]) * sx / (nx * sx * 0.22)) ** 2) <= 1.0
    organ &= inner
    image[organ] = HU_LIVER
    labels[organ] = 1

    # Lesions: small darker spheres wholly inside the organ.
    organ_voxels = np.argwhere(organ)
    for _ in range(n_lesions):
        seed_voxel = organ_voxels[rng.integers(len(organ_voxels))]
        radius = float(rng.uniform(4.0, 9.0))
        lesion = sphere_mask(shape, tuple(seed_voxel), radius, spacing) & organ
        image[lesion] = HU_LESION
        labels[lesion] = 2

    return (
        Volume(image, spacing=spacing, name="torso_phantom"),
        Volume(labels, spacing=spacing, name="torso_phantom_labels"),
    )


TORSO_LABEL_NAMES = {1: "organ", 2: "lesion", 3: "bone"}


def ndimage_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary erosion, imported lazily to keep module import cheap."""
    from scipy import ndimage

    return ndimage.binary_erosion(mask, iterations=iterations)
