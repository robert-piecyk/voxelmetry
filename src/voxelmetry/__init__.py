"""voxelmetry: morphometry for CT and MR segmentations, with a 3-D viewer that fits in one HTML file."""

from .io import dicom_seg_masks, load, load_dicom_seg, load_dicom_series, save
from .measure import (
    LabelMeasurement,
    lesion_burden,
    measure_all,
    measure_components,
    measure_label,
    to_table,
)
from .mesh import Mesh, decimate, extract_surface, smooth, surface_from_label
from .preprocess import PreprocessConfig, apply_window, body_mask, denoise, remove_table
from .viewer import Scene, scene_from_labelmap
from .volume import Volume

__version__ = "2.0.0"

__all__ = [
    "LabelMeasurement", "Mesh", "PreprocessConfig", "Scene", "Volume",
    "apply_window", "body_mask", "decimate", "denoise", "extract_surface",
    "dicom_seg_masks", "lesion_burden", "load", "load_dicom_seg", "load_dicom_series",
    "measure_all",
    "measure_components", "measure_label", "remove_table", "save",
    "scene_from_labelmap", "smooth", "surface_from_label", "to_table",
]
