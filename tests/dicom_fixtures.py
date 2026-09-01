"""Write synthetic DICOM series, so the DICOM reader is testable without data.

Real DICOM was the one input format with no automated coverage: the MSD tasks
ship NIfTI, so ``load_dicom_series`` was only ever exercised by hand. These
helpers build a valid CT series from an array, which lets the test suite pin
the behaviour that matters most -- that slices are ordered by their physical
position and not by filename.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"


def write_ct_series(
    directory: Path,
    array: np.ndarray,
    spacing: tuple[float, float, float] = (2.5, 0.8, 0.8),
    origin_z: float = -400.0,
    filename_order: str = "sequential",
    seed: int = 0,
) -> Path:
    """Write ``array`` as a CT series of single-slice DICOM files.

    Args:
        directory: Destination; created if absent.
        array: Voxel data indexed ``[z, y, x]``, written as int16.
        spacing: Millimetres per voxel, ``(z, y, x)``.
        origin_z: Patient-space z of the first slice.
        filename_order: ``"sequential"`` names files in anatomical order;
            ``"shuffled"`` assigns the same names in a random order, so the
            filenames sort cleanly but bear no relation to position. Real
            archives do this -- a TCIA liver series named 00000001.dcm onward
            had 46 of 88 adjacent pairs out of order.
        seed: Shuffle seed.

    Returns:
        The directory written to.
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    n_slices = array.shape[0]
    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()

    order = list(range(n_slices))
    if filename_order == "shuffled":
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(n_slices))

    for slice_index in range(n_slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.ImplementationClassUID = generate_uid()

        ds.SOPClassUID = CT_IMAGE_STORAGE
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Modality = "CT"
        ds.PatientName = "Phantom^Synthetic"
        ds.PatientID = "PHANTOM"
        ds.SeriesNumber = 1
        ds.InstanceNumber = slice_index + 1

        ds.Rows, ds.Columns = int(array.shape[1]), int(array.shape[2])
        ds.PixelSpacing = [float(spacing[1]), float(spacing[2])]
        ds.SliceThickness = float(spacing[0])
        ds.SpacingBetweenSlices = float(spacing[0])
        ds.ImagePositionPatient = [0.0, 0.0, origin_z + slice_index * spacing[0]]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed, as CT is
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0
        ds.PixelData = array[slice_index].astype(np.int16).tobytes()

        # The filename encodes `order[slice_index]`, so under "shuffled" the
        # names still sort as 00000000..N but point at scattered positions.
        name = f"{order[slice_index]:08d}.dcm"
        pydicom.dcmwrite(directory / name, ds, enforce_file_format=True)

    return directory
