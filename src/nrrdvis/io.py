"""Reading and writing volumes in the formats clinical data actually arrives in.

Everything routes through SimpleITK, which reads NRRD, NIfTI, MetaImage and
DICOM series behind one interface. The one trap worth naming: SimpleITK reports
size and spacing in ``(x, y, z)`` order while ``GetArrayFromImage`` returns
``[z, y, x]``. Mixing the two up transposes every measurement on anisotropic
data. All conversions are funnelled through this module so the flip happens in
exactly one place.
"""

from __future__ import annotations

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
