from collections import Counter
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import OrdinalAxis, Signal, TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types import AnswerTask, ClassificationTask, Split, Task, TextHandle
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark import CONNECTOR, TSAQAConnector
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.source import TSAQAMetadata, TSAQARow
from timenet_connectors.discovery import resolve


FIXTURES = Path(__file__).parent / "fixtures"
SPLITS = {"train": Split.TRAIN, "val": Split.VALIDATION, "test": Split.TEST}


def files(directory=FIXTURES):
    return {split.value: directory / f"{name}.parquet" for name, split in SPLITS.items()}


def published(directory=FIXTURES):
    return {
        (path.name, index): row
        for path in files(directory).values()
        for index, row in enumerate(pq.read_table(path).to_pylist())
    }


def raw_series(row):
    values = json.loads(row["raw_ts"])
    return values if isinstance(values[0], list) else [values]


def answer(task: Task) -> str:
    assert task.targets is not None
    (value,) = task.targets
    assert isinstance(value, TextHandle)
    return value.read()


def check_temporal_inputs(task):
    names = (
        ["Given series", "Candidate series"]
        if len(task.inputs[0].signals) == 2
        else ["Given series", "Time series A", "Time series B", "Time series C", "Time series D"]
    )
    for signal, name in zip(task.inputs[0].signals, names, strict=True):
        (annotation,) = [a for a in signal.annotations if a.key == "series_label"]
        assert annotation.key == "series_label"
        assert annotation.value.read() == name
    assert [a.value.read() for a in task.input_annotations if a.key == "series_label"] == names
    assert not re.search(r"\[[^\n]*\]", task.prompt.read())
    assert all(name in task.prompt.read() for name in names[1:])
    assert task.metadata["question_representation"] == "raw_candidate_references"


def check_signal_context(task, row):
    (record,) = task.inputs
    signals = record.signals
    context = [annotation for annotation in task.input_annotations if annotation.key == "meta_info"]
    descriptions = json.loads(row["meta_info"]) if row["meta_info"].startswith("[") else [row["meta_info"]]
    if "ambiguous_series_context" in task.metadata["source_issues"]:
        assert [annotation.value.read() for annotation in context] == [row["meta_info"]]
        assert len(record.annotations) == 1
        assert record.annotations[0] is context[0]
        assert all(
            a.key != "meta_info" for record in task.inputs for signal in record.signals for a in signal.annotations
        )
    else:
        assert not record.annotations
        if task.metadata["question_format"] == "ordering":
            descriptions *= len(signals)
        assert [annotation.value.read() for annotation in context] == descriptions[: len(signals)]
        for signal, annotation in zip(signals, context, strict=True):
            assert [a for a in signal.annotations if a.key == "meta_info"] == [annotation]
            assert any(a is annotation for a in signal.annotations)


def check_dataset(dataset, original):
    assert not dataset.annotations
    actual_identities = []
    for record in dataset.records:
        row_identity = record.metadata["file"], record.metadata["row"]
        row = original[row_identity]
        input_count = 2 if row["task"] == "comparison" else 1
        if row["task"] == "temporal_relationship":
            input_count = len(raw_series(row))
        for signal in record.signals:
            series_index = signal.metadata["raw_series_index"]
            actual_identities.append((*row_identity, series_index))
            values = raw_series(row)[series_index]
            actual = signal.to_numpy()
            assert actual.dtype == np.dtype("float64")
            assert actual.tobytes() == np.asarray(values, dtype=np.float64).tobytes()
            assert signal.n_values == len(values)
            assert isinstance(signal.time_axis, OrdinalAxis)
            assert signal.spec.unit_value is None
            assert signal.spec.spec_type == "tsaqa_raw_series"
            assert record.metadata["is_task_input"] == (series_index < input_count)
    assert actual_identities == [
        (*identity, index) for identity, row in original.items() for index in range(len(raw_series(row)))
    ]
    tasks = list(dataset.iter_tasks())
    assert len(tasks) == len(original)
    assert [(t.metadata["file"], t.metadata["row"]) for t in tasks] == list(original)
    for task in tasks:
        row = original[task.metadata["file"], task.metadata["row"]]
        assert task.metadata["original_question"] == row["question"]
        assert task.prompt.read().splitlines()[0] == row["question"].splitlines()[0]
        if row["task"] != "temporal_relationship":
            assert task.prompt.read() == row["question"]
        check_signal_context(task, row)
        assert task.targets[0].read() == row["answer"]
        assert task.split == SPLITS[Path(task.metadata["file"]).stem]
        expected_inputs = 2 if row["task"] == "comparison" else 1
        if row["task"] == "temporal_relationship":
            expected_inputs = len(raw_series(row))
        assert len(task.inputs) == 1
        assert len(task.inputs[0].signals) == expected_inputs
        assert [signal.metadata["raw_series_index"] for signal in task.inputs[0].signals] == list(
            range(expected_inputs)
        )
        assert all(record.metadata["is_task_input"] for record in task.inputs)
        if row["task"] == "temporal_relationship":
            check_temporal_inputs(task)
        else:
            assert task.metadata["question_representation"] == "published"
        metadata = TSAQAMetadata.model_validate(task.metadata)
        assert metadata.meta_info == row["meta_info"]
        assert metadata.domain == row["domain"]
        assert metadata.dataset == row["dataset"]
        assert metadata.to_timenet_metadata() == task.metadata
        assert task.metadata["series_representation"] == "raw_ts"
        assert task.metadata["task"] == row["task"]
        assert task.metadata["question_type"] == row["question_type"]
    assert Counter(type(t) for t in tasks) == {ClassificationTask: 9, AnswerTask: 11}


def test_all_raw_values_questions_and_targets_survive_conversion():
    check_dataset(TSAQAConnector().convert(files()), published())


def test_storage_round_trip_keeps_raw_values_and_provenance_separate(tmp_path):
    dataset = TSAQAConnector().convert(files())
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(dataset)
    with TimeFReader(open_local_version(registry.root / "tsaqa/tsaqa-benchmark/1.0.0")) as reader:
        restored = reader.read()
        check_dataset(restored, published())
        assert restored.metadata.source_revision == TSAQAConnector.SOURCE_REVISION


def test_question_format_mismatch_keeps_original_label_prompt_and_answer():
    tasks = list(TSAQAConnector().convert(files()).iter_tasks())
    for index, task_type, actual_format in [
        (12, ClassificationTask, "true_or_false"),
        (13, AnswerTask, "multiple_choices"),
    ]:
        task = tasks[index]
        assert isinstance(task, task_type)
        assert task.metadata["question_type"] == "ordering"
        assert task.metadata["question_format"] == actual_format
        assert "question_type_mismatch" in TSAQAMetadata.model_validate(task.metadata).source_issues
    # Contradictory/duplicate source labels are retained, not deduplicated or repaired.
    assert answer(tasks[17]) == "F"
    assert answer(tasks[16]) != answer(tasks[-1])


@pytest.mark.parametrize(
    ("index", "question_type"),
    [(0, "multiple_choices"), (1, "true_or_false"), (0, "ordering"), (14, "multiple_choices")],
)
def test_other_question_type_conflicts_are_not_inferred(tmp_path, index, question_type):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    rewrite(tmp_path / "train.parquet", index, question_type=question_type)
    with pytest.raises(TimeNetConnectorError, match=rf"train\.parquet: row {index + 1}:"):
        TSAQAConnector().convert(files(tmp_path))


def test_registry_registers_the_existing_tasks_and_records():
    connector = TSAQAConnector()
    registry = connector.read_tasks(files())
    source_tasks = list(registry)
    converted = [task.to_timenet_task() for task in source_tasks]
    assert all(task.to_timenet_task() is existing for task, existing in zip(source_tasks, converted, strict=True))
    dataset = TimeFDataset(metadata=connector.metadata())
    registry.register_in_timenet_dataset(dataset)
    assert all(actual is existing for actual, existing in zip(dataset.iter_tasks(), converted, strict=True))
    check_dataset(dataset, published())


def test_conversion_builds_materialized_signals(monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail("conversion created a lazy signal")

    monkeypatch.setattr(Signal, "from_loader", unexpected_read)
    dataset = TSAQAConnector().convert(files())
    check_dataset(dataset, published())


def rewrite(path, index, **fields):
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[index].update(fields)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)


def test_input_ts_is_not_read_or_used(tmp_path):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    for path in files(tmp_path).values():
        table = pq.read_table(path).drop(["input_ts"])
        pq.write_table(table, path)
    check_dataset(TSAQAConnector().convert(files(tmp_path)), published())


def test_materialized_series_preserve_values_in_any_read_order(tmp_path):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    # Cross multiple source row groups and retain each sample's distinct values.
    rows = []
    original = pq.read_table(tmp_path / "train.parquet").to_pylist()[0]
    for index in range(600):
        rows.append({**original, "raw_ts": json.dumps([float(index), index + 0.5])})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet", row_group_size=199)
    dataset = TSAQAConnector().convert(files(tmp_path))
    for index in [599, 0, 199, 198, 400, 0]:
        expected = np.asarray([float(index), index + 0.5], dtype=np.float64)
        assert dataset.records[index].signals[0].to_numpy().tobytes() == expected.tobytes()


def test_unequal_raw_lengths_and_high_precision_are_not_changed(tmp_path):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    values = [[0.12345678901234566, -0.0, 1e-30], [1e20, 9.125]]
    rewrite(tmp_path / "train.parquet", 4, raw_ts=json.dumps(values))
    task = list(TSAQAConnector().convert(files(tmp_path)).iter_tasks())[4]
    for signal, expected in zip(task.inputs[0].signals, values, strict=True):
        assert signal.to_numpy().tobytes() == np.asarray(expected, dtype=np.float64).tobytes()


@pytest.mark.parametrize("field", ["raw_ts", "question", "meta_info"])
def test_materialized_values_do_not_depend_on_original_files(tmp_path, field):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    original_rows = published()
    dataset = TSAQAConnector().convert(files(tmp_path))
    original = original_rows["train.parquet", 0][field]
    replacement = "[1.0, 2.0]" if field == "raw_ts" else "changed " + original
    rewrite(tmp_path / "train.parquet", 0, **{field: replacement})
    check_dataset(dataset, original_rows)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"raw_ts": "[]"}, "raw_ts"),
        ({"raw_ts": "[null, 1]"}, "raw_ts"),
        ({"raw_ts": "[true, 1]"}, "raw_ts"),
        ({"raw_ts": '["1", 2]'}, "raw_ts"),
        ({"raw_ts": "[NaN, 1]"}, "finite"),
        ({"raw_ts": "[Infinity, 1]"}, "finite"),
        ({"raw_ts": "[[1], []]"}, "raw_ts"),
        ({"raw_ts": "[[1], [2]]"}, "requires 1 raw series"),
        ({"task": "unknown"}, "task"),
        ({"answer": "Z"}, "answer must name"),
        ({"question": "No choices"}, "question must offer"),
    ],
)
def test_invalid_source_fails_with_file_and_row(tmp_path, fields, message):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    rewrite(tmp_path / "train.parquet", 0, **fields)
    with pytest.raises(TimeNetConnectorError, match=rf"(?s)train\.parquet: row 1:.*{message}"):
        TSAQAConnector().convert(files(tmp_path))


def test_missing_split_fails_explicitly():
    with pytest.raises(TimeFValidationError, match="requires splits"):
        TSAQAConnector().convert({"test": files()["test"]})


def test_missing_raw_values_do_not_fall_back_to_input_ts(tmp_path):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "train.parquet"
    pq.write_table(pq.read_table(path).drop(["raw_ts"]), path)
    with pytest.raises(TimeNetConnectorError, match=r"cannot load TSAQA source"):
        TSAQAConnector().convert(files(tmp_path))


def test_ordering_answer_requires_each_patch_exactly_once(tmp_path):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    rewrite(tmp_path / "train.parquet", 14, answer="A,B,B,D")
    with pytest.raises(TimeNetConnectorError, match="permutation"):
        TSAQAConnector().convert(files(tmp_path))


def test_discovery_and_release_identity():
    assert CONNECTOR is TSAQAConnector
    assert resolve("tsaqa/tsaqa-benchmark") is TSAQAConnector
    metadata = TSAQAConnector().metadata()
    assert str(metadata.dataset_version) == "1.0.0"
    assert metadata.source_revision == "883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd"


@pytest.mark.parametrize("context", ['["one description"]', '["valid", 7]', '["unfinished"'])
def test_comparison_rejects_invalid_signal_context(context):
    row = next(row for row in published().values() if row["task"] == "comparison")
    row = {name: row[name] for name in TSAQARow.model_fields}
    row["meta_info"] = context
    with pytest.raises(TimeFValidationError, match="meta_info"):
        _ = TSAQARow.model_validate(row).series_descriptions


@pytest.mark.parametrize(
    ("fixture_row", "expected_indices"),
    [
        pytest.param(4, ((0,), (1,)), id="comparison-describes-each-series"),
        pytest.param(14, ((0, 1, 2, 3, 4),), id="ordering-shares-one-description"),
        pytest.param(11, (None, None, None, None), id="continuation-keeps-unresolved-descriptions"),
    ],
)
def test_source_descriptions_preserve_text_and_series_associations(fixture_row, expected_indices):
    source = pq.read_table(FIXTURES / "train.parquet").to_pylist()[fixture_row]
    row = TSAQARow.model_validate({name: source[name] for name in TSAQARow.model_fields})
    descriptions = row.series_descriptions
    source_texts = json.loads(row.meta_info) if row.meta_info.lstrip().startswith("[") else [row.meta_info]
    assert [description.text for description in descriptions] == source_texts
    assert tuple(description.series_indices for description in descriptions) == expected_indices


def test_tasks_are_complete_before_dataset_registration():
    connector = TSAQAConnector()
    source_tasks = list(connector.read_tasks(files()))
    native_tasks = [task.to_timenet_task() for task in source_tasks]
    original_annotations = [tuple(task.input_annotations) for task in native_tasks]
    for task in native_tasks:
        (record,) = task.inputs
        owned_annotations = tuple(record.walk_annotations())
        assert all(any(selected is owned for owned in owned_annotations) for selected in task.input_annotations)
    dataset = TimeFDataset(metadata=connector.metadata())
    for task in source_tasks:
        for record in task.records:
            dataset.add_record(record)
    dataset.add_tasks(native_tasks)
    for task, original in zip(native_tasks, original_annotations, strict=True):
        assert all(current is before for current, before in zip(task.input_annotations, original, strict=True))
    check_dataset(dataset, published())
