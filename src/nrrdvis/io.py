"""Reading and writing volumes in the formats clinical data actually arrives in.

Everything routes through SimpleITK, which reads NRRD, NIfTI, MetaImage and
DICOM series behind one interface. The one trap worth naming: SimpleITK reports
size and spacing in ``(x, y, z)`` order while ``GetArrayFromImage`` returns
``[z, y, x]``. Mixing the two up transposes every measurement on anisotropic
data. All conversions are funnelled through this module so the flip happens in
exactly one place.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .volume import Volume

# Extensions SimpleITK can read as a single file.
_VOLUME_SUFFIXES = {".nrrd", ".nhdr", ".nii", ".mha", ".mhd", ".vtk"}


def _from_sitk(image: sitk.Image, name: str) -> Volume:
    """Convert a SimpleITK image, reversing its (x, y, z) axis convention."""
    return Volume(
        array=sitk.GetArrayFromImage(image),
        spacing=tuple(float(s) for s in reversed(image.GetSpacing())),
        origin=tuple(float(o) for o in reversed(image.GetOrigin())),
        name=name,
    )


def _to_sitk(volume: Volume) -> sitk.Image:
    """Convert back, restoring SimpleITK's (x, y, z) axis convention."""
    image = sitk.GetImageFromArray(np.ascontiguousarray(volume.array))
    image.SetSpacing(tuple(float(s) for s in reversed(volume.spacing)))
    image.SetOrigin(tuple(float(o) for o in reversed(volume.origin)))
    return image


def load(path: str | Path, name: str | None = None) -> Volume:
    """Load a volume from a file or a directory of DICOM slices.

    Args:
        path: A volume file (``.nrrd``, ``.nii``, ``.nii.gz``, ``.mha`` ...) or a
            directory holding one DICOM series.
        name: Label for the result. Defaults to the file or directory stem.

    Returns:
        The loaded :class:`Volume`, with spacing taken from the file header.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a directory holds no readable DICOM series.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such volume: {path}")

    if path.is_dir():
        return load_dicom_series(path, name=name)

    label = name or path.name.removesuffix(".gz").removesuffix("".join(path.suffixes[-1:]))
    return _from_sitk(sitk.ReadImage(str(path)), label or path.stem)


def load_dicom_series(directory: str | Path, name: str | None = None) -> Volume:
    """Load a DICOM series as one volume, ordered by slice position.

    Slice ordering comes from ImagePositionPatient rather than filename. The v1
    code sorted filenames with a natural-sort key, which happens to work for
    ``image_0``..``image_128`` but silently produces a scrambled or mirrored
    volume for any series whose filenames do not encode acquisition order.

    Args:
        directory: Folder containing the ``.dcm`` files of a single series.
        name: Label for the result. Defaults to the directory name.

    Returns:
        The assembled :class:`Volume`.

    Raises:
        ValueError: If no DICOM series is found in ``directory``.
    """
    directory = Path(directory)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))

    if series_ids:
        # Take the longest series if a folder holds more than one.
        best = max(
            (reader.GetGDCMSeriesFileNames(str(directory), sid) for sid in series_ids),
            key=len,
        )
    else:
        # 3Dircadb1 and similar sets ship slices with no extension and no
        # series metadata GDCM will index; fall back to reading every file.
        best = sorted(
            (str(p) for p in directory.iterdir() if p.is_file()),
            key=_natural_key,
        )
        if not best:
            raise ValueError(f"No DICOM files found in {directory}")

    reader.SetFileNames(best)
    try:
        image = reader.Execute()
    except RuntimeError as exc:  # pragma: no cover - depends on corrupt input
        raise ValueError(f"Could not read a DICOM series from {directory}: {exc}") from exc

    return _from_sitk(image, name or directory.name)


def save(volume: Volume, path: str | Path, compress: bool = True) -> Path:
    """Write a volume, inferring the format from the file extension.

    Args:
        volume: The volume to write.
        path: Destination path; parent directories are created as needed.
        compress: Whether to compress. NRRD and NIfTI both support it.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(_to_sitk(volume), str(path), useCompression=compress)
    return path


def _natural_key(text: str) -> list[object]:
    """Sort key that orders ``image_2`` before ``image_10``.

    Carried over from v1, where it was the only ordering available.
    """
    import re

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def is_volume_file(path: str | Path) -> bool:
    """Whether ``path`` looks like a single-file volume this module can read."""
    path = Path(path)
    name = path.name.lower()
    return name.endswith(".nii.gz") or path.suffix.lower() in _VOLUME_SUFFIXES


def load_dicom_seg(
    path: str | Path,
    name: str | None = None,
    priority: Sequence[int] | None = None,
) -> tuple[Volume, dict[int, str]]:
    """Read a DICOM Segmentation object into a label map and its segment names.

    DICOM SEG is how segmentations travel between clinical systems, and it is
    not a volume: it is a multi-frame object where each frame carries one
    segment on one slice, present only where that segment is non-empty. The
    frames must be reassembled against their patient positions to become a
    grid, which is what this does.

    Overlap is the one place a choice has to be made. DICOM SEG permits
    segments to overlap -- a tumour inside a liver is stored in both -- while
    an integer label map cannot represent that. Segments are written in
    ``priority`` order, later ones winning, which suits the usual authoring
    order of organ first and structures inside it after.

    That flattening can badly misrepresent a heavily overlapping SEG, so it is
    never silent: if any voxel is claimed by more than one segment, a warning
    names the segments and how much each lost. A real Colorectal-Liver-
    Metastases SEG carries both "Liver" and "Liver Remnant" over largely the
    same voxels, and flattening leaves "Liver" as 767 disconnected fragments.
    Use :func:`dicom_seg_masks` when segments genuinely overlap.

    Args:
        path: The ``.dcm`` SEG file, or a directory holding exactly one.
        name: Label for the result. Defaults to the file stem.
        priority: Segment numbers in increasing precedence. Defaults to
            ascending numeric order.

    Returns:
        ``(labelmap, names)`` where ``labelmap`` holds one integer per segment
        and ``names`` maps those integers to their ``SegmentLabel``.

    Raises:
        ImportError: If pydicom is not installed.
        ValueError: If the file is not a Segmentation object, or a directory
            does not hold exactly one candidate.
    """
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "Reading DICOM SEG needs pydicom. Install with: pip install 'nrrdvis[dicom]'"
        ) from exc

    path = Path(path)
    if path.is_dir():
        candidates = [p for p in sorted(path.iterdir()) if p.suffix.lower() == ".dcm"]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one .dcm SEG file in {path}, found {len(candidates)}"
            )
        path = candidates[0]

    ds = pydicom.dcmread(path)
    if getattr(ds, "Modality", None) != "SEG":
        raise ValueError(f"{path} is not a DICOM Segmentation object (Modality={ds.Modality!r})")

    shared = ds.SharedFunctionalGroupsSequence[0]
    measures = shared.PixelMeasuresSequence[0]
    row_mm, col_mm = (float(v) for v in measures.PixelSpacing)
    slice_mm = float(
        getattr(measures, "SpacingBetweenSlices", None) or measures.SliceThickness
    )

    frames = ds.PerFrameFunctionalGroupsSequence
    positions = [
        float(frame.PlanePositionSequence[0].ImagePositionPatient[2]) for frame in frames
    ]
    segment_numbers = [
        int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber) for frame in frames
    ]

    # Only slices where some segment is present appear in the file, so the grid
    # is built from the positions actually observed.
    z_values = sorted(set(positions))
    z_index = {z: i for i, z in enumerate(z_values)}

    pixels = ds.pixel_array
    if pixels.ndim == 2:  # a single-frame SEG
        pixels = pixels[np.newaxis]

    names = {
        int(segment.SegmentNumber): str(segment.SegmentLabel)
        for segment in ds.SegmentSequence
    }

    rank = (
        {segment: i for i, segment in enumerate(priority)}
        if priority is not None
        else {segment: segment for segment in names}
    )

    shape = (len(z_values), int(ds.Rows), int(ds.Columns))
    labelmap = np.zeros(shape, dtype=np.uint8)
    claims = np.zeros(shape, dtype=np.uint8)  # how many segments cover each voxel
    lost: dict[int, int] = {}

    order = sorted(range(len(frames)), key=lambda i: rank.get(segment_numbers[i], 0))
    for frame_index in order:
        segment = segment_numbers[frame_index]
        z = z_index[positions[frame_index]]
        mask = pixels[frame_index].astype(bool)

        overwritten = labelmap[z][mask]
        for previous in np.unique(overwritten[overwritten != 0]):
            lost[int(previous)] = lost.get(int(previous), 0) + int((overwritten == previous).sum())

        claims[z][mask] += 1
        labelmap[z][mask] = segment

    if lost:
        overlapping = (claims > 1).sum()
        detail = ", ".join(
            f"{names.get(seg, seg)!r} lost {count} voxels"
            for seg, count in sorted(lost.items(), key=lambda kv: -kv[1])
        )
        warnings.warn(
            f"{path.name}: segments overlap on {overlapping} voxels and cannot all be "
            f"kept in one label map ({detail}). Use load_dicom_seg(..., priority=...) "
            "to choose which wins, or dicom_seg_masks() to keep them separate.",
            stacklevel=2,
        )

    volume = Volume(
        array=labelmap,
        spacing=(slice_mm, row_mm, col_mm),
        origin=(z_values[0], 0.0, 0.0),
        name=name or path.stem,
    )
    return volume, names


def dicom_seg_masks(path: str | Path) -> tuple[dict[int, np.ndarray], dict[int, str], Volume]:
    """Read a DICOM SEG keeping every segment as its own binary mask.

    The lossless counterpart to :func:`load_dicom_seg`. Use it when segments
    genuinely overlap and flattening would destroy one of them.

    Args:
        path: The ``.dcm`` SEG file, or a directory holding exactly one.

    Returns:
        ``(masks, names, geometry)``: a boolean mask per segment number, the
        segment names, and an all-zero :class:`~nrrdvis.volume.Volume` carrying
        the shared spacing and origin, so a mask can be turned into a Volume
        with ``geometry.with_array(mask.astype(np.uint8))``.
    """
    with warnings.catch_warnings():
        # The flattening warning is irrelevant here: nothing is being flattened.
        warnings.simplefilter("ignore")
        labelmap, names = load_dicom_seg(path)

    import pydicom

    path = Path(path)
    if path.is_dir():
        path = next(p for p in sorted(path.iterdir()) if p.suffix.lower() == ".dcm")
    ds = pydicom.dcmread(path)

    frames = ds.PerFrameFunctionalGroupsSequence
    positions = [
        float(frame.PlanePositionSequence[0].ImagePositionPatient[2]) for frame in frames
    ]
    z_index = {z: i for i, z in enumerate(sorted(set(positions)))}
    pixels = ds.pixel_array
    if pixels.ndim == 2:  # pragma: no cover - single-frame SEG
        pixels = pixels[np.newaxis]

    masks = {segment: np.zeros(labelmap.shape, dtype=bool) for segment in names}
    for frame_index, frame in enumerate(frames):
        segment = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        z = z_index[positions[frame_index]]
        masks[segment][z] |= pixels[frame_index].astype(bool)

    return masks, names, labelmap.with_array(np.zeros(labelmap.shape, dtype=np.uint8))
