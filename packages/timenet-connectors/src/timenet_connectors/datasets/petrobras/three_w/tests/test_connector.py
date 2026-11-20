from collections import Counter
from datetime import UTC, datetime
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types import US_PER_S, ClassificationTask, Split, TemporalLocalizationTask, TimeInterval
from timenet.types.text import read_value
from timenet_connectors.datasets.petrobras.three_w import ThreeWConnector
from timenet_connectors.datasets.petrobras.three_w.connector import (
    MANIFEST,
    SOURCE_REPO,
    SOURCE_REVISION,
    InstanceFile,
    read_manifest,
)
from timenet_connectors.datasets.petrobras.three_w.variables import VARIABLES


# Synthetic instances shaped like 3W v2.0.0 files: a nanosecond `timestamp`, the 27 float64
# variables, and int16 `class` and `state` columns. The values are made up for the tests, so the
# repo ships no 3W bytes.
_START = datetime(2017, 2, 1, 1, 2, 7, tzinfo=UTC)


def _write_instance(  # ruff: ignore[too-many-arguments]
    root,
    event_code,
    name,
    labels,
    *,
    present=("P-TPT", "T-TPT"),
    constant=("P-PDG",),
    gaps=(),
    step_s=1,
):
    n = len(labels)
    seconds = np.arange(n, dtype=np.int64) * step_s
    columns = {"timestamp": pa.array((seconds + int(_START.timestamp())) * 10**9, type=pa.timestamp("ns"))}
    rng = np.random.default_rng(0)
    for variable in VARIABLES:
        if variable.column == "state":
            continue
        if variable.column in constant:
            values = [0.0] * n
        elif variable.column in present:
            values = rng.normal(1e7, 1e5, n).tolist()
        elif variable.column in gaps:
            values = [None if index % 2 else 5.0 + index for index in range(n)]
        else:
            values = [None] * n
        columns[variable.column] = pa.array(values, type=pa.float64())
    columns["class"] = pa.array(labels, type=pa.int16())
    columns["state"] = pa.array([None if label is None else 0 for label in labels], type=pa.int16())
    path = root / "dataset" / str(event_code) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return InstanceFile(path=path, event_code=event_code)


def _five_wells(tmp_path):
    return [_write_instance(tmp_path, 0, f"WELL-0000{well}_20170201010207", [0] * 6) for well in range(1, 6)]


def test_one_record_per_instance_with_only_recorded_variables(tmp_path):
    refs = [*_five_wells(tmp_path), _write_instance(tmp_path, 2, "WELL-00006_20131104004101", [0, 102, 102, 2])]
    dataset = ThreeWConnector().convert(refs)

    assert len(dataset.records) == 6
    record = dataset.records[-1]
    names = {signal.name for source in record.sources for signal in source.signals}
    # The all-empty variables become no signal; `state` has a value wherever a label does.
    assert names == {"P-TPT", "T-TPT", "P-PDG", "state"}
    assert record.metadata == {
        "instance": "WELL-00006_20131104004101",
        "well": "WELL-00006",
        "provenance": "real",
        "event_code": 2,
        "start": "2017-02-01T01:02:07",
    }
    # 3W states no time zone, so the record has no absolute start time.
    assert record.start_time.timestamp is None
    signals = {signal.name: signal for source in record.sources for signal in source.signals}
    assert signals["P-TPT"].spec.spec_type == "pressure"
    assert signals["T-TPT"].spec.spec_type == "temperature"
    assert signals["P-TPT"].metadata == {
        "description": "Subsea Xmas-tree pressure at the TPT (temperature and pressure transducer)"
    }


def test_constant_variable_is_kept_and_flagged(tmp_path):
    dataset = ThreeWConnector().convert(_five_wells(tmp_path))
    signals = {signal.name: signal for source in dataset.records[0].sources for signal in source.signals}

    assert [annotation.key for annotation in signals["P-PDG"].annotations] == ["constant_value"]
    assert not signals["P-TPT"].annotations


def test_partial_gaps_are_kept_as_nulls(tmp_path):
    refs = [*_five_wells(tmp_path), _write_instance(tmp_path, 0, "WELL-00006_20170201010207", [0] * 4, gaps=("QGL",))]
    dataset = ThreeWConnector().convert(refs)
    signals = {signal.name: signal for source in dataset.records[-1].sources for signal in source.signals}

    assert signals["QGL"].to_arrow().to_pylist() == [5.0, None, 7.0, None]
    assert signals["QGL"].null_count == 2


def test_phases_become_interval_annotations_and_localization_targets(tmp_path):
    labels = [None, 0, 0, 102, 102, 102, 2]
    refs = [*_five_wells(tmp_path), _write_instance(tmp_path, 2, "WELL-00006_20131104004101", labels)]
    dataset = ThreeWConnector().convert(refs)
    record = dataset.records[-1]

    phases = []
    for annotation in record.annotations:
        if annotation.key == "event_phase":
            assert isinstance(annotation.span, TimeInterval)
            span = annotation.span
            phases.append((read_value(annotation.value), span.start_us // US_PER_S, span.exclusive_end // US_PER_S))
    assert phases == [("normal", 1, 3), ("transient", 3, 6), ("steady", 6, 7)]

    localization = [task for task in dataset.tasks if isinstance(task, TemporalLocalizationTask)]
    assert len(localization) == 1
    targets = localization[0].targets or ()
    assert all(isinstance(span, TimeInterval) for span in targets)
    assert [(span.start_us, span.exclusive_end) for span in targets if isinstance(span, TimeInterval)] == [
        (3 * US_PER_S, 6 * US_PER_S)
    ]


def test_every_instance_gets_a_classification_task(tmp_path):
    refs = [*_five_wells(tmp_path), _write_instance(tmp_path, 4, "SIMULATED_00001", [0, 4, 4])]
    dataset = ThreeWConnector().convert(refs)
    classification = [task for task in dataset.tasks if isinstance(task, ClassificationTask)]

    assert Counter(read_value(task.targets[0]) if task.targets else None for task in classification) == {
        "Normal Operation": 5,
        "Flow Instability": 1,
    }
    # Flow instability has no transient phase, so it gets no localization task.
    assert not any(isinstance(task, TemporalLocalizationTask) for task in dataset.tasks)


def test_real_instances_split_by_well_and_others_train(tmp_path):
    refs = [
        *_five_wells(tmp_path),
        _write_instance(tmp_path, 0, "WELL-00001_20170202010207", [0] * 6),
        _write_instance(tmp_path, 1, "SIMULATED_00001", [0, 101, 1]),
        _write_instance(tmp_path, 1, "DRAWN_00001", [0, 101, 1]),
    ]
    dataset = ThreeWConnector().convert(refs)
    split_of_well: dict[str, set[Split]] = {}
    for task in dataset.tasks:
        record = task.inputs[0]
        if record.metadata["provenance"] == "real":
            assert task.split is not None
            split_of_well.setdefault(str(record.metadata["well"]), set()).add(task.split)
        else:
            assert task.split is Split.TRAIN

    assert all(len(splits) == 1 for splits in split_of_well.values())
    assert set().union(*split_of_well.values()) == {Split.TRAIN, Split.VALIDATION, Split.TEST}


def test_label_outside_the_event_is_rejected(tmp_path):
    refs = [*_five_wells(tmp_path), _write_instance(tmp_path, 2, "WELL-00006_20131104004101", [0, 105])]
    with pytest.raises(TimeNetConnectorError, match="label 105"):
        ThreeWConnector().convert(refs)


def test_irregular_timestamps_are_rejected(tmp_path):
    refs = [*_five_wells(tmp_path), _write_instance(tmp_path, 0, "WELL-00006_20170201010207", [0] * 3, step_s=2)]
    with pytest.raises(TimeNetConnectorError, match="one second apart"):
        ThreeWConnector().convert(refs)


def test_storage_round_trip(tmp_path):
    refs = [*_five_wells(tmp_path), _write_instance(tmp_path, 2, "WELL-00006_20131104004101", [0, 102, 2])]
    original = ThreeWConnector().convert(refs)
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(original)
    with TimeFReader(open_local_version(registry.root / "petrobras/three-w/1.0.0")) as reader:
        restored = reader.read()

    assert len(restored.records) == len(original.records)
    assert [task.split for task in restored.tasks] == [task.split for task in original.tasks]


def test_source_is_pinned():
    # Literal pins, so moving to another commit is a deliberate test edit.
    assert SOURCE_REPO == "petrobras/3W"
    assert SOURCE_REVISION == "c86cc7c70063646910be7292de8a54a6fb2a8399"


def test_manifest_lists_every_instance_once():
    entries = read_manifest(MANIFEST)
    paths = [path for path, _, _ in entries]

    assert len(entries) == 2228
    assert len(set(paths)) == len(paths)
    assert all(
        re.fullmatch(r"dataset/[0-9]/(WELL-\d{5}_\d{14}|SIMULATED_\d{5}|DRAWN_\d{5})\.parquet", p) for p in paths
    )
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for _, _, digest in entries)
    assert Counter(path.split("/")[1] for path in paths) == {
        "0": 594,
        "1": 128,
        "2": 38,
        "3": 106,
        "4": 343,
        "5": 450,
        "6": 221,
        "7": 46,
        "8": 95,
        "9": 207,
    }
