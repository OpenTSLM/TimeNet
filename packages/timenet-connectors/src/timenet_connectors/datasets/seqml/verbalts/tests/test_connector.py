"""Check that captions point to the exact released windows."""

import json

import numpy as np

from timenet.dataset import Record
from timenet.types import InputModality
from timenet_connectors.datasets.seqml.verbalts.connector import VerbalTsComponent, VerbalTsConnector
from timenet_connectors.datasets.seqml.verbalts.tables import signal_names


def test_generation_tasks_keep_each_caption_and_target_window(tmp_path):
    folder = tmp_path / "synthetic_u"
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"attr_list": ["trend_types"], "attr_n_ops": [4]}), encoding="utf-8")
    for split in ("train", "valid", "test"):
        np.save(folder / f"{split}_ts.npy", np.arange(6, dtype=np.float64).reshape(1, 6, 1))
        np.save(folder / f"{split}_attrs_idx.npy", np.array([[2]], dtype=np.int64))
        np.save(folder / f"{split}_text_caps.npy", np.array([[f"A rising series in {split}."]]))

    dataset = VerbalTsConnector().convert([VerbalTsComponent("synthetic_u", folder)])
    first = list(dataset.iter_tasks())
    second = list(dataset.iter_tasks())

    assert len(dataset.records) == len(first) == 3
    assert [task.id for task in first] == [task.id for task in second]
    assert all(task.input_modalities == frozenset({InputModality.TEXT, InputModality.NO_INPUT}) for task in first)
    assert all(task.inputs == () for task in first)
    assert first[0].targets == (
        next(record for record in dataset.records if record.id == "verbalts-synthetic_u-train-00000"),
    )
    assert first[0].targets is not None
    target = first[0].targets[0]
    assert isinstance(target, Record)
    assert target.signals[0].to_numpy().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_channel_labels_keep_unverified_positions():
    assert signal_names("BlindWays", (), 72)[3] == "x03"
    assert signal_names("Weather", (), 21)[0] == "x00"
    assert signal_names("ETTm1", (6,), 1) == ("var_6",)
    assert signal_names("istanbul_traffic", (2,), 1) == ("var_2",)
