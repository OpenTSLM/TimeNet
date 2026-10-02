"""Check that captions point to the exact released windows."""

import json

import numpy as np
import pytest

from timenet.dataset import Record
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import InputModality, Split
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.seqml.verbalts.connector import VerbalTsComponent, VerbalTsConnector
from timenet_connectors.datasets.seqml.verbalts.release import CHANNELS


def _write_component(folder, meta, windows, codes, captions):
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for split in ("train", "valid", "test"):
        np.save(folder / f"{split}_ts.npy", np.asarray(windows, dtype=np.float64))
        np.save(folder / f"{split}_attrs_idx.npy", np.asarray(codes, dtype=np.int64))
        np.save(folder / f"{split}_text_caps.npy", np.asarray(captions))


def test_generation_tasks_keep_each_caption_and_target_window(tmp_path):
    folder = tmp_path / "synthetic_u"
    folder.mkdir()
    meta = {"attr_list": ["trend_types", "trend_directions", "season_cycles"], "attr_n_ops": [4, 2, 4]}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for split in ("train", "valid", "test"):
        np.save(folder / f"{split}_ts.npy", np.arange(6, dtype=np.float64).reshape(1, 6, 1))
        np.save(folder / f"{split}_attrs_idx.npy", np.array([[2, 0, 3]], dtype=np.int64))
        np.save(folder / f"{split}_text_caps.npy", np.array([[f"A rising series in {split}."]]))

    dataset = VerbalTsConnector().convert([VerbalTsComponent("synthetic_u", folder)])
    first = list(dataset.iter_tasks())
    second = list(dataset.iter_tasks())

    assert len(dataset.records) == len(first) == 3
    assert [task.id for task in first] == [task.id for task in second]
    assert [task.split for task in first] == [Split.TRAIN, Split.VALIDATION, Split.TEST]
    assert [task.id for task in dataset.get_test()] == [first[2].id]
    assert all(annotation.key != "split" for record in dataset.records for annotation in record.annotations)
    assert all(task.input_modalities == frozenset({InputModality.TEXT, InputModality.NO_INPUT}) for task in first)
    assert all(task.inputs == () and task.input_annotations == () for task in first)
    assert {annotation.key: annotation.value for annotation in dataset.annotations} == {
        "synthetic_u_trend_type_labels": ["linear", "quadratic", "exponential", "logistic"],
        "synthetic_u_trend_direction_labels": ["up", "down"],
        "synthetic_u_season_cycles_labels": ["0", "1", "2", "4"],
    }
    assert first[0].targets == (
        next(record for record in dataset.records if record.id == "verbalts-synthetic_u-train-00000"),
    )
    assert first[0].targets is not None
    target = first[0].targets[0]
    assert isinstance(target, Record)
    assert target.signals[0].to_numpy().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [signal.name for signal in target.signals] == ["variable 1"]
    assert {annotation.key: annotation.value for annotation in target.annotations} == {
        "component": "synthetic_u",
        "trend_type": "exponential",
        "trend_direction": "up",
        "season_cycles": 4,
    }


def test_single_channel_windows_are_named_by_their_variable_and_checked_against_the_caption(tmp_path):
    meta = {"attr_list": ["var_id", "trend", "season", "skewness", "kurtosis"], "attr_n_ops": [7, 2, 9, 3, 3]}
    windows = np.zeros((2, 4, 1))
    codes = [[0, 0, -1, 2, 1], [6, 1, 2, 0, 0]]
    captions = [["This sequence is HUFL.\nFlat."], ["This sequence is OT.\nFlat."]]
    _write_component(tmp_path / "ETTm1", meta, windows, codes, captions)

    dataset = VerbalTsConnector().convert([VerbalTsComponent("ETTm1", tmp_path / "ETTm1")])

    named = {record.id: [signal.name for signal in record.signals] for record in dataset.records}
    assert named["verbalts-ETTm1-train-00000"] == ["HUFL"]
    assert named["verbalts-ETTm1-test-00001"] == ["OT"]
    first = next(record for record in dataset.records if record.id == "verbalts-ETTm1-train-00000")
    assert {annotation.key: annotation.value for annotation in first.annotations} == {
        "component": "ETTm1",
        "variable": "HUFL",
        "trend": "upward",
        "season_code": -1,
        "skewness": "symmetrical",
        "kurtosis": "normal",
    }
    second = next(record for record in dataset.records if record.id == "verbalts-ETTm1-valid-00000")
    shared = [annotation.content_id for annotation in first.annotations]
    assert [annotation.content_id for annotation in second.annotations] == shared
    assert {annotation.occurrence_id for annotation in first.annotations}.isdisjoint(
        {annotation.occurrence_id for annotation in second.annotations}
    )
    vocabularies = {annotation.key: annotation.value for annotation in dataset.annotations}
    assert vocabularies["ettm1_variable_labels"] == ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    assert vocabularies["ettm1_season_code_labels"] == [str(code) for code in range(-1, 9)]
    assert vocabularies["ettm1_variable_labels"].index(first.annotations[1].value) == 0
    assert vocabularies["ettm1_season_code_labels"].index(str(first.annotations[3].value)) == 0

    dataset.derive_schema()
    with TimeFWriter(tmp_path / "built", dataset) as output:
        output.write()
    with TimeFReader(DatasetVersion.open_local(tmp_path / "built/seqml/verbalts/1.0.0")) as reader:
        back = reader.read()
    assert {annotation.key: annotation.value for annotation in back.annotations} == vocabularies
    contents = reader.annotation_table("Record").column("content_id").to_pylist()
    # Each record has six annotations. Only the component annotation is shared.
    assert len(contents) == 6 * 6 and len(set(contents)) == 1 + 5 * 2
    assert back.schema is not None
    assert "ettm1_season_code_labels" in {descriptor.key for descriptor in back.schema.annotations}

    captions[1] = ["This sequence is HULL.\nFlat."]
    _write_component(tmp_path / "ETTm1-wrong", meta, windows, codes, captions)
    with pytest.raises(TimeFFormatError, match=r"opens with 'This sequence is HULL\.'"):
        VerbalTsConnector().convert([VerbalTsComponent("ETTm1", tmp_path / "ETTm1-wrong")])


def test_augmented_windows_sit_on_their_split_timeline(tmp_path):
    meta = {"attr_list": ["var_id", "trend", "season", "skewness", "kurtosis"], "attr_n_ops": [7, 2, 9, 3, 3]}
    first = np.arange(32.0)
    # Regular slide, short final slide, repeat, and new variable.
    windows = [first, first + 30, first + 34, first + 34, np.ones(32)]
    codes = [[0, 0, 0, 2, 1]] * 4 + [[6, 0, 0, 2, 1]]
    captions = [["This sequence is HUFL."]] * 4 + [["This sequence is OT."]]
    _write_component(tmp_path / "ETTm1", meta, np.reshape(windows, (5, 32, 1)), codes, captions)

    records = VerbalTsConnector().convert([VerbalTsComponent("ETTm1", tmp_path / "ETTm1")]).records

    train = records[:5]
    assert [record.signals[0].time_axis.start_index for record in train] == [0, 30, 34, 34, 0]
    assert all(record.start_time is train[0].start_time for record in train)
    assert train[0].start_time.timestamp is None
    assert records[5].start_time is not train[0].start_time
    assert train[1].signals[0].span_us == (30 * 900_000_000, 62 * 900_000_000)

    windows[2] = np.full(32, 7.0)
    _write_component(tmp_path / "ETTm1-torn", meta, np.reshape(windows, (5, 32, 1)), codes, captions)
    with pytest.raises(TimeFFormatError, match="window 2 does not continue window 1"):
        VerbalTsConnector().convert([VerbalTsComponent("ETTm1", tmp_path / "ETTm1-torn")])


def test_fixed_channel_components_take_the_release_names(tmp_path):
    meta = {"attr_list": ["guide", "hand"], "attr_n_ops": [2, 3]}
    _write_component(tmp_path / "BlindWays", meta, np.zeros((1, 3, 72)), [[1, 2]], [["Walks."]])

    dataset = VerbalTsConnector().convert([VerbalTsComponent("BlindWays", tmp_path / "BlindWays")])

    names = [signal.name for signal in dataset.records[0].signals]
    assert names == list(CHANNELS["BlindWays"])
    assert names[:4] == ["joint00_0", "joint00_1", "joint00_2", "joint01_0"]
    assert len(CHANNELS["Weather"]) == 21
    assert CHANNELS["Weather"][11:14] == ("wv", "max. wv", "wd")

    _write_component(tmp_path / "BlindWays-narrow", meta, np.zeros((1, 3, 71)), [[1, 2]], [["Walks."]])
    with pytest.raises(TimeFFormatError, match="holds 71 channels per window, but the release names 72"):
        VerbalTsConnector().convert([VerbalTsComponent("BlindWays", tmp_path / "BlindWays-narrow")])


def test_meta_that_disagrees_with_the_codebook_is_refused(tmp_path):
    folder = tmp_path / "BlindWays"
    folder.mkdir()
    (folder / "meta.json").write_text(
        json.dumps({"attr_list": ["guide", "hand"], "attr_n_ops": [2, 4]}), encoding="utf-8"
    )
    with pytest.raises(TimeFFormatError, match="declares 4 codes for 'hand'"):
        VerbalTsConnector().convert([VerbalTsComponent("BlindWays", folder)])
