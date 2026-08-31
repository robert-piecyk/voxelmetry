"""The central abstraction: a 3-D array that never loses its physical spacing.

The v1 scripts kept voxel data in bare ``numpy`` arrays and the millimetre
spacing in a separate ``pydicom`` dataset. Every ``cv2.resize`` silently
invalidated that spacing, so any measurement taken after a resize was wrong by
whatever factor the resize applied. :class:`Volume` binds the two together and
updates spacing on every geometric operation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import ndimage

# Axis order is (z, y, x) throughout: z is the slice axis, matching how
# SimpleITK hands back a DICOM series and how marching cubes expects input.
Spacing = tuple[float, float, float]


@dataclass(frozen=True)
class Volume:
    """A 3-D voxel array plus the physical geometry needed to measure it.

    Attributes:
        array: Voxel data, indexed ``[z, y, x]``.
        spacing: Millimetres per voxel along ``(z, y, x)``.
        origin: Patient-space coordinate of voxel ``[0, 0, 0]``, in ``(z, y, x)``.
        name: Free-form label used in reports and viewer output.
    """

    array: np.ndarray
    spacing: Spacing = (1.0, 1.0, 1.0)
    origin: Spacing = (0.0, 0.0, 0.0)
    name: str = "volume"

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError(f"Volume requires a 3-D array, got shape {self.array.shape}")
        if len(self.spacing) != 3:
            raise ValueError(f"spacing must have 3 elements, got {self.spacing}")
        if any(s <= 0 for s in self.spacing):
            raise ValueError(f"spacing must be strictly positive, got {self.spacing}")

    # ----- geometry -------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.array.shape  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        """Physical volume of a single voxel."""
        return float(np.prod(self.spacing))

    @property
    def extent_mm(self) -> Spacing:
        """Physical size of the whole field of view, in ``(z, y, x)``."""
        return tuple(float(n * s) for n, s in zip(self.shape, self.spacing, strict=True))

    @property
    def is_isotropic(self) -> bool:
        return bool(np.allclose(self.spacing, self.spacing[0]))

    def with_array(self, array: np.ndarray) -> Volume:
        """Return a copy carrying new voxel data but the same geometry."""
        return replace(self, array=array)

    # ----- operations that change geometry --------------------------------

    def resample(
        self,
        new_spacing: Spacing | float = 1.0,
        order: int | None = None,
    ) -> Volume:
        """Resample onto a new voxel grid, keeping physical size constant.

        Args:
            new_spacing: Target millimetres per voxel. A scalar means isotropic.
            order: Spline interpolation order. Defaults to 0 (nearest neighbour)
                for integer arrays so label values are never blended into
                nonexistent classes, and 1 (linear) for floating-point data.

        Returns:
            A new :class:`Volume` on the requested grid.
        """
        if np.isscalar(new_spacing):
            target: Spacing = (float(new_spacing),) * 3  # type: ignore[arg-type]
        else:
            target = tuple(float(s) for s in new_spacing)  # type: ignore[assignment]
        if any(s <= 0 for s in target):
            raise ValueError(f"new_spacing must be strictly positive, got {target}")

        if order is None:
            order = 0 if np.issubdtype(self.array.dtype, np.integer) else 1

        zoom = tuple(cur / new for cur, new in zip(self.spacing, target, strict=True))
        if np.allclose(zoom, 1.0):
            return self

        resampled = ndimage.zoom(self.array, zoom, order=order, mode="nearest")
        # ndimage rounds the output shape, so the spacing that was actually
        # realised differs slightly from the request. Record the real one.
        realised = tuple(
            cur * old_n / new_n
            for cur, old_n, new_n in zip(self.spacing, self.shape, resampled.shape, strict=True)
        )
        return replace(self, array=resampled, spacing=realised)  # type: ignore[arg-type]

    def crop_to_mask(self, mask: np.ndarray, margin_mm: float = 0.0) -> Volume:
        """Crop to the bounding box of ``mask``, padded by ``margin_mm``."""
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} does not match volume {self.shape}")
        if not mask.any():
            raise ValueError("cannot crop to an empty mask")

        slices = []
        new_origin = []
        for axis, (extent, spacing_mm, origin_mm) in enumerate(
            zip(mask.shape, self.spacing, self.origin, strict=True)
        ):
            present = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
            pad = int(np.ceil(margin_mm / spacing_mm))
            lo = max(int(present[0]) - pad, 0)
            hi = min(int(present[-1]) + pad + 1, extent)
            slices.append(slice(lo, hi))
            new_origin.append(origin_mm + lo * spacing_mm)

        return replace(
            self,
            array=self.array[tuple(slices)],
            origin=tuple(new_origin),  # type: ignore[arg-type]
        )

    # ----- intensity ------------------------------------------------------

    def window(
        self, low: float, high: float, mode: Literal["clip", "normalize"] = "normalize"
    ) -> Volume:
        """Apply an intensity window, e.g. a Hounsfield-unit range.

        Args:
            low: Lower bound; values below are pinned to it.
            high: Upper bound; values above are pinned to it.
            mode: ``"clip"`` keeps the original units, ``"normalize"`` rescales
                the window to ``[0, 1]`` as float32.
        """
        if high <= low:
            raise ValueError(f"window requires high > low, got low={low}, high={high}")
        clipped = np.clip(self.array, low, high)
        if mode == "clip":
            return self.with_array(clipped)
        return self.with_array(((clipped - low) / (high - low)).astype(np.float32))

    def __repr__(self) -> str:
        spacing = "x".join(f"{s:.3g}" for s in self.spacing)
        extent = "x".join(f"{e:.0f}" for e in self.extent_mm)
        return (
            f"Volume({self.name!r}, shape={self.shape}, spacing={spacing} mm, "
            f"extent={extent} mm, dtype={self.array.dtype})"
        )
