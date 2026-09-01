"""Adapter for Medical Segmentation Decathlon collections.

Every MSD task ships the same ``dataset.json`` describing modality, label
names and case list, so one adapter reads Task03 Liver, Task09 Spleen and the
other eight without special-casing. Label names come from the file rather than
being hard-coded, which is what lets the viewer stay organ-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..io import load
from ..volume import Volume

#: Public mirror used by MONAI; each task is a single tar.
MSD_BASE_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com"

#: Task name to archive, for the download helper and documentation.
MSD_TASKS: dict[str, str] = {
    "brain": "Task01_BrainTumour", "heart": "Task02_Heart",
    "liver": "Task03_Liver", "hippocampus": "Task04_Hippocampus",
    "prostate": "Task05_Prostate", "lung": "Task06_Lung",
    "pancreas": "Task07_Pancreas", "hepaticvessel": "Task08_HepaticVessel",
    "spleen": "Task09_Spleen", "colon": "Task10_Colon",
}


@dataclass(frozen=True)
class Case:
    """One image/label pair."""

    case_id: str
    image_path: Path
    label_path: Path | None

    def load(self) -> tuple[Volume, Volume | None]:
        """Load image and label as :class:`~voxelmetry.volume.Volume` objects."""
        image = load(self.image_path, name=self.case_id)
        labels = load(self.label_path, name=f"{self.case_id}_labels") if self.label_path else None
        return image, labels


class MSDDataset:
    """A Medical Segmentation Decathlon task on disk.

    Args:
        root: Directory containing ``dataset.json``.

    Raises:
        FileNotFoundError: If ``dataset.json`` is missing.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest = self.root / "dataset.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"No dataset.json in {self.root}. Expected an unpacked MSD task directory."
            )
        self._meta = json.loads(manifest.read_text())

    @property
    def name(self) -> str:
        return str(self._meta.get("name", self.root.name))

    @property
    def modality(self) -> str:
        modalities = self._meta.get("modality", {})
        return ", ".join(str(v) for v in modalities.values()) or "unknown"

    @property
    def label_names(self) -> dict[int, str]:
        """Label value to name, background excluded."""
        return {
            int(k): str(v)
            for k, v in self._meta.get("labels", {}).items()
            if int(k) != 0
        }

    @property
    def cases(self) -> list[Case]:
        """Every training case, sorted by id."""
        entries = []
        for item in self._meta.get("training", []):
            image = self.root / str(item["image"]).lstrip("./")
            label = self.root / str(item["label"]).lstrip("./")
            # MSD archives carry macOS resource forks (._name) that are not
            # volumes; skip anything whose real file is missing.
            if not image.exists():
                continue
            entries.append(
                Case(image.name.split(".")[0], image, label if label.exists() else None)
            )
        return sorted(entries, key=lambda c: c.case_id)

    def case(self, case_id: str) -> Case:
        """Look up one case by id.

        Raises:
            KeyError: If no case matches.
        """
        for entry in self.cases:
            if entry.case_id == case_id:
                return entry
        raise KeyError(f"No case {case_id!r} in {self.name}")

    def __len__(self) -> int:
        return len(self.cases)

    def __repr__(self) -> str:
        return f"MSDDataset({self.name!r}, {len(self)} cases, labels={self.label_names})"


def download_command(task: str, destination: str | Path) -> str:
    """Return the shell command that fetches and unpacks an MSD task.

    The archives run to tens of gigabytes, so this hands back a command to run
    deliberately rather than downloading as a side effect of an import.

    Args:
        task: A key of :data:`MSD_TASKS`, e.g. ``"liver"``.
        destination: Directory to unpack into.

    Raises:
        KeyError: If the task name is not recognised.
    """
    if task not in MSD_TASKS:
        raise KeyError(f"Unknown MSD task {task!r}; known: {sorted(MSD_TASKS)}")
    archive = MSD_TASKS[task]
    return (
        f"mkdir -p {destination} && cd {destination} && "
        f"curl -L -O {MSD_BASE_URL}/{archive}.tar && "
        f"tar xf {archive}.tar && rm {archive}.tar"
    )
