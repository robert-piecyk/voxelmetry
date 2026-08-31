"""Volumetric preprocessing: HU windowing, body extraction, denoising.

The v1 pipeline ran slice by slice and defined every morphological kernel in
voxels: ``np.ones((15, 15))`` for closing, ``np.ones((25, 25))`` for dilation.
Those numbers were tuned against one scanner's 0.57 mm pixels, so on a series
with 0.7 mm pixels the same code closes a 30% smaller physical gap. Kernels
here are specified in millimetres and converted per-volume, which makes the
behaviour identical across scanners and is the single change that matters most
for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .volume import Volume

#: Standard CT display windows, as (level, width) in Hounsfield units.
HU_WINDOWS: dict[str, tuple[float, float]] = {
    "abdomen": (40, 400),
    "liver": (60, 160),
    "lung": (-600, 1500),
    "bone": (400, 1800),
    "brain": (40, 80),
    "mediastinum": (50, 350),
}

#: Air is about -1000 HU; anything above this is tissue, table or contrast.
AIR_THRESHOLD_HU = -320.0


def window_bounds(window: str | tuple[float, float]) -> tuple[float, float]:
    """Resolve a named window or a ``(level, width)`` pair into ``(low, high)``.

    Args:
        window: A key of :data:`HU_WINDOWS`, or an explicit ``(level, width)``.

    Returns:
        The ``(low, high)`` Hounsfield bounds.

    Raises:
        KeyError: If a named window is not recognised.
    """
    if isinstance(window, str):
        if window not in HU_WINDOWS:
            raise KeyError(f"Unknown window {window!r}; known: {sorted(HU_WINDOWS)}")
        level, width = HU_WINDOWS[window]
    else:
        level, width = window
    return level - width / 2.0, level + width / 2.0


def apply_window(volume: Volume, window: str | tuple[float, float] = "abdomen") -> Volume:
    """Apply a named or explicit HU window, normalised to ``[0, 1]``."""
    low, high = window_bounds(window)
    return volume.window(low, high, mode="normalize")


def _ball_voxels(radius_mm: float, spacing: tuple[float, float, float]) -> np.ndarray:
    """Structuring element approximating a physical ball of the given radius.

    On anisotropic data this is an ellipsoid in voxel space, which is the whole
    point: a 10 mm closing should bridge 10 mm regardless of slice thickness.
    """
    radii = [max(int(round(radius_mm / s)), 0) for s in spacing]
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    distance = sum(
        (g * s / radius_mm) ** 2 for g, s in zip(grids, spacing, strict=True)
    )
    return distance <= 1.0


def body_mask(
    volume: Volume,
    threshold_hu: float = AIR_THRESHOLD_HU,
    closing_mm: float = 8.0,
    fill_holes: bool = True,
) -> np.ndarray:
    """Segment the patient's body, discarding air, table and scanner artefacts.

    The approach is v1's: threshold, take the largest connected component, then
    close. What changed is that it now runs on the whole volume at once rather
    than per slice, so a slice where the body happens to split into two blobs
    (common at the very top and bottom of a series) no longer loses half the
    anatomy. The v1 version also thresholded with Otsu after histogram
    equalisation, which is scan-dependent; a fixed HU threshold is meaningful
    because Hounsfield units are already calibrated to physical density.

    Args:
        volume: Input CT volume in Hounsfield units.
        threshold_hu: Everything above this counts as tissue.
        closing_mm: Physical radius used to close gaps in the body outline.
        fill_holes: Whether to fill enclosed air pockets, e.g. bowel gas.

    Returns:
        A boolean mask of the body, shaped like ``volume``.
    """
    tissue = volume.array > threshold_hu
    if not tissue.any():
        return np.zeros(volume.shape, dtype=bool)

    components, n = ndimage.label(tissue)
    if n > 1:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        tissue = components == int(sizes.argmax())

    if closing_mm > 0:
        tissue = ndimage.binary_closing(tissue, structure=_ball_voxels(closing_mm, volume.spacing))

    if fill_holes:
        # Fill per slice as well as in 3D: a structure open at the top and
        # bottom of the volume is not enclosed in 3D and survives a plain
        # binary_fill_holes, but is clearly interior on each axial slice.
        tissue = ndimage.binary_fill_holes(tissue)
        for z in range(tissue.shape[0]):
            tissue[z] = ndimage.binary_fill_holes(tissue[z])

    return tissue


def remove_table(
    volume: Volume,
    body: np.ndarray | None = None,
    fill_value: float | None = None,
) -> Volume:
    """Blank everything outside the body, removing the scanner table.

    Args:
        volume: Input volume.
        body: Precomputed body mask; computed if omitted.
        fill_value: Value written outside the body. Defaults to the volume
            minimum, which is air for HU data.

    Returns:
        A copy with non-body voxels set to ``fill_value``.
    """
    if body is None:
        body = body_mask(volume)
    fill = float(volume.array.min()) if fill_value is None else fill_value
    cleaned = np.where(body, volume.array, fill).astype(volume.array.dtype)
    return volume.with_array(cleaned)


def denoise(volume: Volume, strength: float = 1.0, method: str = "gaussian") -> Volume:
    """Reduce noise while keeping spacing meaningful.

    v1 used BM3D at ``sigma_psd=30/255`` on every slice. BM3D is excellent but
    costs seconds per slice, which put a 130-slice study into the tens of
    minutes and made the pipeline impractical to re-run. These alternatives are
    seconds per volume.

    Args:
        volume: Input volume.
        strength: Filter width in millimetres.
        method: ``"gaussian"`` for speed, ``"median"`` for salt-and-pepper
            noise, or ``"bilateral"`` to preserve edges at higher cost.

    Returns:
        The denoised volume.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    sigma_voxels = [strength / s for s in volume.spacing]

    if method == "gaussian":
        out = ndimage.gaussian_filter(volume.array.astype(np.float32), sigma_voxels)
    elif method == "median":
        size = [max(int(round(2 * s)) | 1, 1) for s in sigma_voxels]
        out = ndimage.median_filter(volume.array, size=size)
    elif method == "bilateral":
        from skimage.restoration import denoise_bilateral

        data = volume.array.astype(np.float32)
        scale = np.ptp(data) or 1.0
        out = np.stack([
            denoise_bilateral(
                (slice_ - data.min()) / scale, sigma_color=0.05, sigma_spatial=max(strength, 1.0)
            ) * scale + data.min()
            for slice_ in data
        ])
    else:
        raise ValueError(f"Unknown denoise method {method!r}; use gaussian, median or bilateral")

    return volume.with_array(out.astype(volume.array.dtype, copy=False))


@dataclass(frozen=True)
class PreprocessConfig:
    """Declarative description of a preprocessing run.

    Holding this as data rather than as a sequence of inline calls means a run
    can be recorded alongside its outputs, so a result can be traced back to
    the settings that produced it.
    """

    window: str | tuple[float, float] | None = "abdomen"
    isotropic_mm: float | None = 1.0
    denoise_mm: float = 0.0
    denoise_method: str = "gaussian"
    strip_table: bool = True
    closing_mm: float = 8.0

    def describe(self) -> str:
        parts = []
        if self.isotropic_mm:
            parts.append(f"resample to {self.isotropic_mm} mm isotropic")
        if self.denoise_mm:
            parts.append(f"{self.denoise_method} denoise at {self.denoise_mm} mm")
        if self.strip_table:
            parts.append(f"body mask with {self.closing_mm} mm closing")
        if self.window:
            parts.append(f"{self.window} window")
        return "; ".join(parts) or "no-op"


def run(volume: Volume, config: PreprocessConfig | None = None) -> Volume:
    """Apply a full preprocessing configuration to a volume.

    Order matters: resampling first so that later millimetre-defined kernels
    act on a known grid, table removal before windowing so the body mask is
    computed on true Hounsfield units, and windowing last because it destroys
    the HU scale.

    Args:
        volume: Input CT volume in Hounsfield units.
        config: Settings; defaults to :class:`PreprocessConfig`.

    Returns:
        The preprocessed volume.
    """
    config = config or PreprocessConfig()

    if config.isotropic_mm:
        volume = volume.resample(config.isotropic_mm)
    if config.denoise_mm:
        volume = denoise(volume, config.denoise_mm, config.denoise_method)
    if config.strip_table:
        volume = remove_table(volume, body_mask(volume, closing_mm=config.closing_mm))
    if config.window:
        volume = apply_window(volume, config.window)

    return volume
