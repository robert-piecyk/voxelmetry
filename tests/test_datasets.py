"""The Medical Segmentation Decathlon adapter, against a synthetic task."""

import json

import numpy as np
import pytest

from nrrdvis import io as nio
from nrrdvis.datasets import MSD_TASKS, MSDDataset, download_command
from nrrdvis.volume import Volume


@pytest.fixture
def task(tmp_path):
    """A miniature MSD task with the layout and manifest of the real thing."""
    root = tmp_path / "Task03_Liver"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir(parents=True)

    training = []
    for index in (0, 1):
        image = np.full((8, 12, 12), -1000, dtype=np.int16)
        image[2:6, 3:9, 3:9] = 60
        labels = np.zeros((8, 12, 12), dtype=np.uint8)
        labels[2:6, 3:9, 3:9] = 1
        labels[3:5, 4:6, 4:6] = 2

        name = f"liver_{index}.nii.gz"
        nio.save(Volume(image, spacing=(5.0, 0.8, 0.8)), root / "imagesTr" / name)
        nio.save(Volume(labels, spacing=(5.0, 0.8, 0.8)), root / "labelsTr" / name)
        training.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})

    # Real MSD archives ship macOS resource forks that are not volumes.
    (root / "imagesTr" / "._liver_0.nii.gz").write_bytes(b"resource fork")
    training.append({"image": "./imagesTr/absent.nii.gz", "label": "./labelsTr/absent.nii.gz"})

    (root / "dataset.json").write_text(json.dumps({
        "name": "Liver",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "liver", "2": "cancer"},
        "training": training,
    }))
    return root


def test_reads_the_manifest(task):
    dataset = MSDDataset(task)
    assert dataset.name == "Liver"
    assert dataset.modality == "CT"
    assert dataset.label_names == {1: "liver", 2: "cancer"}


def test_background_is_excluded_from_label_names(task):
    assert 0 not in MSDDataset(task).label_names


def test_cases_skip_entries_whose_files_are_missing(task):
    dataset = MSDDataset(task)
    assert len(dataset) == 2
    assert [c.case_id for c in dataset.cases] == ["liver_0", "liver_1"]


def test_case_lookup_and_load(task):
    dataset = MSDDataset(task)
    case = dataset.case("liver_1")
    image, labels = case.load()

    assert image.shape == labels.shape == (8, 12, 12)
    assert image.spacing == pytest.approx((5.0, 0.8, 0.8))
    assert set(np.unique(labels.array)) == {0, 1, 2}
    assert labels.name.endswith("_labels")


def test_unknown_case_raises_keyerror(task):
    with pytest.raises(KeyError, match="liver_99"):
        MSDDataset(task).case("liver_99")


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset.json"):
        MSDDataset(tmp_path)


def test_repr_summarises_the_task(task):
    text = repr(MSDDataset(task))
    assert "Liver" in text and "2 cases" in text


def test_download_command_names_the_archive():
    command = download_command("liver", "./data")
    assert "Task03_Liver.tar" in command and "./data" in command


def test_download_command_rejects_unknown_tasks():
    with pytest.raises(KeyError, match="Unknown MSD task"):
        download_command("kneecap", "./data")


def test_task_table_is_complete():
    assert len(MSD_TASKS) == 10
    assert MSD_TASKS["spleen"] == "Task09_Spleen"
