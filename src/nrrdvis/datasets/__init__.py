"""Dataset adapters. Each exposes cases as image/label Volume pairs."""

from .msd import MSD_TASKS, Case, MSDDataset, download_command

__all__ = ["MSD_TASKS", "Case", "MSDDataset", "download_command"]
