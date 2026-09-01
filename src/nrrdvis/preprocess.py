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


def is_hounsfield(volume: Volume) -> bool:
    """Whether the intensities look like calibrated Hounsfield units.

    CT is calibrated so air sits near -1000 and soft tissue near 0. MR,
    ultrasound and already-normalised data carry arbitrary intensities, often
    unsigned, where a Hounsfield threshold means nothing. Everything in this
    module that assumes HU checks here first, because the failure is otherwise
    silent: a real liver MR from TCGA-LIHC ranges 0 to 831, so ``> -320 HU``
    selects the entire field of view and the body mask looks like it worked.

    The test is deliberately loose -- air present well below zero, and some
    tissue above it -- so it accepts a cropped or partially windowed CT while
    rejecting anything unsigned.
    """
    return bool(volume.array.min() < -300 and volume.array.max() > 100)


def percentile_window(volume: Volume, low: float = 1.0, high: float = 99.0) -> Volume:
    """Window by intensity percentile, for data with no absolute scale.

    The right default for MR, where a fixed window is meaningless because the
    intensities depend on the sequence, the scanner and the coil.

    Args:
        volume: Input volume.
        low: Lower percentile, pinned to 0.
        high: Upper percentile, pinned to 1.

    Returns:
        The volume rescaled to ``[0, 1]`` as float32.

    Raises:
        ValueError: If ``high`` is not above ``low``.
    """
    if high <= low:
        raise ValueError(f"percentile_window requires high > low, got {low} and {high}")
    lo, hi = np.percentile(volume.array, [low, high])
    if hi <= lo:  # pragma: no cover - a constant volume
        return volume.with_array(np.zeros(volume.shape, dtype=np.float32))
    return volume.window(float(lo), float(hi), mode="normalize")


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


def apply_window(
    volume: Volume,
    window: str | tuple[float, float] = "abdomen",
) -> Volume:
    """Apply a named or explicit HU window, normalised to ``[0, 1]``.

    Args:
        volume: Input volume in Hounsfield units.
        window: A key of :data:`HU_WINDOWS`, an explicit ``(level, width)``, or
            ``"percentile"`` for data with no absolute intensity scale.

    Returns:
        The windowed volume, rescaled to ``[0, 1]``.

    Raises:
        ValueError: If a Hounsfield window is requested for data that is not in
            Hounsfield units. Applying one anyway silently destroys most of the
            dynamic range rather than failing, so this refuses instead.
    """
    if window == "percentile":
        return percentile_window(volume)

    if not is_hounsfield(volume):
        raise ValueError(
            f"{volume.name!r} does not look like Hounsfield units "
            f"(range {volume.array.min():g} to {volume.array.max():g}), so the "
            f"{window!r} window would clip away most of its range. Use "
            'window="percentile" for MR or other uncalibrated data, or pass an '
            "explicit (level, width) if you know the scale."
        )

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
    threshold_hu: float | None = None,
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
        threshold_hu: Everything above this counts as tissue. When omitted,
            a Hounsfield volume uses :data:`AIR_THRESHOLD_HU` and anything else
            falls back to an Otsu threshold computed from the data, since a
            fixed HU cut is meaningless without an absolute scale.
        closing_mm: Physical radius used to close gaps in the body outline.
        fill_holes: Whether to fill enclosed air pockets, e.g. bowel gas.

    Returns:
        A boolean mask of the body, shaped like ``volume``.
    """
    if threshold_hu is None:
        if is_hounsfield(volume):
            threshold_hu = AIR_THRESHOLD_HU
        else:
            # Otsu splits background from foreground on whatever scale the data
            # happens to use. Less precise than a calibrated HU cut, but it is
            # the difference between a mask and a mask of everything.
            from skimage.filters import threshold_otsu

            sample = volume.array[:: max(volume.shape[0] // 32, 1)]
            threshold_hu = float(threshold_otsu(sample))

    tissue = volume.array > threshold_hu
    if not tissue.any():
        return np.zeros(volume.shape, dtype=bool)

    components, n = ndimage.label(tissue)
    if n > 1:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        tissue = components == int(sizes.argmax())

    if closing_mm > 0:
        structure = _ball_voxels(closing_mm, volume.spacing)
        # Closing dilates then erodes, and scipy treats everything outside the
        # array as background during the erosion. A body that extends through
        # the edge of the field of view -- which is every abdominal CT at the
        # top and bottom of the slab, and most at the sides -- is therefore
        # eaten away at the border. Measured on a real TCIA liver CT, an 8 mm
        # closing emptied the first and last slices completely: 47% of the
        # first slice became 0%. Padding with foreground first makes the
        # erosion see tissue continuing past the edge, which is the truth.
        pad = [(n // 2 + 1, n // 2 + 1) for n in structure.shape]
        padded = np.pad(tissue, pad, mode="edge")
        padded = ndimage.binary_closing(padded, structure=structure)
        tissue = padded[tuple(slice(lo, -hi) for lo, hi in pad)]

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
    """A HU window name, an explicit (level, width), "percentile" for
    uncalibrated data such as MR, or None to leave intensities alone."""
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
