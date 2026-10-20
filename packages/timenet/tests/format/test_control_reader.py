import pyarrow as pa
import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.control_audit import audit_control_database
import timenet.format.control_reader as control_reader_module
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    InputModality,
    Split,
    StepInterval,
    StepPoint,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    TSGenerationTask,
    ureg,
)
from timenet.types.object_kind import ObjectKind

from .test_control_writer import _dataset


def test_inferred_modalities_survive_storage_and_sql_filtering(tmp_path):
    dataset = _dataset()
    signal = Signal(
        name="image",
        spec=TimeSeriesSpec(
            spec_type="image",
            name="Image",
            unit_value=ureg.dimensionless,
            dtype="uint8",
            value_shape=(2, 2, 3),
            modality=InputModality.IMAGE,
        ),
        time_axis=OrdinalAxis(),
        data=pa.FixedSizeListArray.from_arrays(pa.array(range(12), type=pa.uint8()), 12),
    )
    image_record = Record(record_id="image", sources=(Source(name="image", signals=(signal,)),))
    dataset.add_record(record=image_record)
    dataset.add_task(
        task=AnswerTask(id="image-task", inputs=(image_record,), prompt="Describe this image.", targets=("A chart.",))
    )
    dataset.add_task(task=TSGenerationTask(id="generation", prompt="Make an ECG.", targets=(dataset.records[0],)))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        assert reader.task_table(required_modalities=[InputModality.TIME_SERIES])["task_id"].to_pylist() == ["task-1"]
        assert reader.task_table(required_modalities=[InputModality.IMAGE])["task_id"].to_pylist() == ["image-task"]
        assert reader.task_table(supported_modalities=[InputModality.TEXT, InputModality.NO_INPUT])[
            "task_id"
        ].to_pylist() == ["generation"]
        records = reader.read_records()
        tasks = {task.id: task for task in reader.read_tasks(records)}

    assert tasks["image-task"].inputs[0].signals[0].spec.modality is InputModality.IMAGE
    assert tasks["task-1"].inputs[0].signals[0].spec.modality is InputModality.TIME_SERIES
    assert tasks["image-task"].input_modalities == frozenset({InputModality.IMAGE, InputModality.TEXT})
    tasks["image-task"].input_modalities = None
    assert tasks["image-task"].resolved_input_modalities == frozenset({InputModality.IMAGE, InputModality.TEXT})


def test_shifted_regular_axis_round_trip(tmp_path):
    dataset = _dataset()
    source = dataset.records[0].sources[0].sources[0]
    axis = RegularAxis.from_rate_hz(44100, offset_us=250_003).at_index(1026)
    source.signals = tuple(
        Signal.from_loader(
            id=signal.id,
            name=signal.name,
            spec=signal.spec,
            time_axis=axis,
            n_values=signal.n_values,
            loader=signal.loader,
        )
        for signal in source.signals
    )
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)
    with DuckDBControlReader(path) as reader:
        (record,) = reader.read_records()
    restored = record.signals[0].time_axis
    assert restored == axis
    assert restored is record.signals[1].time_axis
    assert record.signals[0].span_us == (273268, 273313)


def test_control_reader_hydrates_recursive_hierarchy_and_annotations(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    DuckDBControlWriter(path).write_hierarchy(dataset)
    values = {
        "lead-i": pa.array([1.0, 2.0], type=pa.float32()),
        "lead-ii": pa.array([3.0, 4.0], type=pa.float32()),
    }

    with DuckDBControlReader(path, value_loader=lambda signal_id, _spec: values[signal_id]) as reader:
        (record,) = reader.read_records()
        (task,) = reader.read_tasks((record,))

    assert record.id == "record-1"
    assert record.sources[0].id == "monitor"
    assert record.sources[0].sources[0].id == "ecg"
    lead_i, lead_ii = record.signals
    assert lead_i.time_axis is lead_ii.time_axis
    assert lead_i.spec is lead_ii.spec
    assert lead_i.metadata == lead_ii.metadata == {}
    assert lead_i.metadata is not lead_ii.metadata
    assert lead_i.to_arrow().to_pylist() == [1.0, 2.0]
    assert record.annotations[0].content_id == "sex-male"
    assert record.annotations[0].occurrence_id is not None
    assert task.inputs[0] is record
    assert task.prompt == "Alive?"
    assert task.targets == ("Yes",)
    assert task.annotations[0].name == "task_kind"


def test_reader_and_audit_reject_ambiguous_stored_task_time_span(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    second = dataset.add_record(record=Record(record_id="other", sources=(Source(name="other"),)))
    dataset.tasks[0].inputs = (*dataset.tasks[0].inputs, second)
    DuckDBControlWriter(path).write_hierarchy(dataset)
    with connect_control(path) as connection:
        connection.execute(
            "UPDATE tasks SET scope_type = 'time_interval', scope_start = 0, scope_end = 2000 WHERE task_id = 'task-1'"
        )
        with pytest.raises(TimeFFormatError, match="Record"):
            audit_control_database(connection, require_chunks=False)
    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match="Record"):
        reader.read_tasks(reader.read_records())


def test_unknown_unit_round_trips_distinct_from_dimensionless(tmp_path):
    dataset = _dataset()
    source = dataset.records[0].sources[0].sources[0]
    unknown = TimeSeriesSpec(spec_type="unknown", name="Unknown", unit_value=None)
    old_signals = source.signals
    source.signals = tuple(
        Signal.from_loader(
            id=signal.id,
            name=signal.name,
            spec=spec,
            time_axis=signal.time_axis,
            n_values=signal.n_values,
            loader=signal.loader,
        )
        for signal, spec in zip(
            old_signals,
            (unknown, unknown.model_copy(update={"spec_type": "ratio", "unit_value": ureg.dimensionless})),
            strict=True,
        )
    )
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (record,) = reader.read_records()

    assert record.signals[0].spec.unit_value is None
    assert record.signals[1].spec.unit_value == ureg.dimensionless


def test_input_modalities_round_trip_as_typed_task_data(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    dataset.tasks[0].input_modalities = frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (task,) = reader.read_tasks(reader.read_records())
        rows = reader.task_table().to_pylist()

    assert task.input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    assert rows[0]["input_modalities"] == ["text", "time_series"]


def test_modality_filters_select_tasks_and_targets_before_hydration(tmp_path, monkeypatch):
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 2)
    dataset = _dataset()
    record = dataset.records[0]
    dataset.add_tasks(
        tasks=(
            AnswerTask(
                id="task-image",
                inputs=(record,),
                prompt="Read chart",
                targets=("image answer",),
                input_modalities=frozenset({InputModality.TEXT, InputModality.IMAGE}),
            ),
            AnswerTask(
                id="task-mixed",
                inputs=(record,),
                prompt="Read chart and series",
                targets=("mixed answer",),
                input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES, InputModality.IMAGE}),
            ),
            AnswerTask(
                id="task-unknown",
                targets=("unknown",),
            ),
        )
    )
    dataset.add_tasks(
        tasks=(
            AnswerTask(
                id=f"task-series-{index}",
                inputs=(record,),
                targets=("series answer",),
            )
            for index in range(4)
        )
    )
    expected = ["task-1", "task-series-0", "task-series-1", "task-series-2", "task-series-3"]
    required = frozenset({InputModality.TIME_SERIES})
    supported = frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        tasks = list(reader.iter_tasks(required_modalities=required, supported_modalities=supported))
        task_rows = reader.task_table(required_modalities=required, supported_modalities=supported).to_pylist()
        target_rows = reader.target_table(required_modalities=required, supported_modalities=supported).to_pylist()

    assert [task.id for task in tasks] == expected
    assert [row["task_id"] for row in task_rows] == expected
    assert [row["task_id"] for row in target_rows] == expected
    assert [
        task.id for task in dataset.iter_tasks(required_modalities=required, supported_modalities=supported)
    ] == expected


def test_control_reader_hydrates_each_target_storage_type(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    record = dataset.records[0]
    signal = record.signals[0]
    spans = (
        TimePoint.seconds(0.001, time_series_ids=(signal.id,)),
        TimeInterval.seconds(0, 0.002, time_series_ids=(signal.id,)),
        StepPoint(start=1, time_series_id=signal.id),
        StepInterval(start=0, stop=2, time_series_id=signal.id),
    )
    dataset.tasks[0].targets = ("Yes", 7, 0.5, True, record, signal, *spans)
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        (task,) = reader.read_tasks(records)

    assert task.targets is not None
    text, integer, floating, boolean, target_record, target_signal, *target_spans = task.targets
    assert (text, integer, floating, boolean) == ("Yes", 7, 0.5, True)
    assert target_record is records[0]
    assert target_signal is records[0].signals[0]
    assert tuple(target_spans) == spans


def test_control_reader_preserves_ordered_integer_relationships(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    record = dataset.records[0]
    second = record.annotate(Annotation(id="age-65", key="patient_age", value=65, unit="year"))
    dataset.tasks[0].input_annotations = (second, record.annotations[0])
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        (task,) = reader.read_tasks(records)

    assert task.inputs == records
    assert tuple(annotation.content_id for annotation in task.input_annotations) == ("age-65", "sex-male")


def test_text_targets_do_not_build_a_signal_index(tmp_path, monkeypatch):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        monkeypatch.setattr(
            Record,
            "signals",
            property(lambda _record: pytest.fail("text targets walked the Signal hierarchy")),
        )
        (task,) = reader.read_tasks(records)

    assert task.targets == ("Yes",)


def test_selected_record_hydration_does_not_read_unrelated_hierarchies(tmp_path):
    dataset = make_dataset()
    selected_record, corrupt_record = dataset.records[:2]
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)
    with connect_control(path) as connection:
        connection.execute(
            """UPDATE axes SET axis_type = 'corrupt'
               WHERE axis_key IN (
                   SELECT signals.axis_key
                   FROM signals
                   JOIN sources USING (source_key)
                   JOIN records USING (record_key)
                   WHERE records.record_id = ?
               )""",
            [corrupt_record.id],
        )
    with DuckDBControlReader(path) as reader:
        (selected,) = reader.read_records((selected_record.id,))
        assert selected.id == selected_record.id
        with pytest.raises(TimeFFormatError, match="unknown type"):
            reader.read_records((corrupt_record.id,))


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "DELETE FROM annotation_contents WHERE content_id = 'sex-male'",
            "missing content",
        ),
        (
            "UPDATE sources SET parent_source_key = 999 WHERE source_id = 'ecg'",
            "missing parent",
        ),
        (
            "UPDATE signals SET source_key = 999 WHERE signal_id = 'lead-i'",
            "missing source",
        ),
        (
            "UPDATE annotation_occurrences SET object_key = 999 "
            "WHERE content_key IN ("
            "SELECT content_key FROM annotation_contents WHERE content_id = 'sex-male')",
            "missing Record",
        ),
        (
            "UPDATE annotation_occurrences SET object_type = 'Unknown' "
            "WHERE content_key IN ("
            "SELECT content_key FROM annotation_contents WHERE content_id = 'sex-male')",
            "unknown object type",
        ),
    ],
)
def test_control_reader_rejects_corrupt_relationships(tmp_path, statement, message):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())
    with connect_control(path) as connection:
        connection.execute(statement)

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match=message):
        reader.read_records()


def test_iter_tasks_for_one_record_yields_its_tasks_and_shares_record_objects(tmp_path):
    dataset = make_dataset()
    expected_answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (expected_record,) = expected_answer.inputs
    (expected_parent,) = expected_answer.from_tasks
    expected_ids = sorted(task.id for task in dataset.tasks if expected_record in task.inputs)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        record0 = next(record for record in records if record.id == expected_record.id)
        tasks = list(reader.iter_tasks((record0,)))

    assert [task.id for task in tasks] == expected_ids
    assert all(task.inputs[0] is record0 for task in tasks)
    answer = next(task for task in tasks if task.id == expected_answer.id)
    parent = next(task for task in tasks if task.id == expected_parent.id)
    assert answer.from_tasks == (parent,)
    assert answer.from_tasks[0] is parent


@pytest.mark.parametrize("materialize", [False, True])
def test_iter_tasks_hydrates_a_parent_that_sorts_after_its_dependant(tmp_path, monkeypatch, materialize):
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 1)
    dataset = make_dataset()
    answer_task = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (parent_task,) = answer_task.from_tasks
    (expected_record,) = answer_task.inputs
    assert answer_task.id < parent_task.id, "the fixture must load the dependant before its parent"
    expected_ids = {task.id for task in dataset.tasks if expected_record in task.inputs}
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        record = next(record for record in reader.read_records() if record.id == expected_record.id)
        tasks = reader.read_tasks((record,)) if materialize else list(reader.iter_tasks((record,)))

    by_id = {task.id: task for task in tasks}
    assert set(by_id) == expected_ids
    assert by_id[answer_task.id].from_tasks[0] is by_id[parent_task.id]
    assert tuple(sorted(by_id)) == tuple(task.id for task in tasks)


def test_read_tasks_rejects_records_the_database_does_not_store(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFValidationError, match="no such record"):
        reader.read_tasks((Record(record_id="ghost"),))


def test_typed_scope_and_payload_columns_round_trip(tmp_path):
    dataset = make_dataset()
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        restored = {task.id: task for task in reader.read_tasks(reader.read_records())}

    for task in dataset.tasks:
        back = restored[task.id]
        assert type(back) is type(task)
        assert back.scope == task.scope
        for field in ("target_schema", "unit", "target_name", "mode"):
            if hasattr(task, field):
                assert getattr(back, field) == getattr(task, field), field


@pytest.mark.parametrize(
    ("value", "kind"),
    [("W", "text"), (64, "integer"), (0.5, "float"), (True, "boolean"), (["yes", "no"], "text_list"), (None, None)],
)
def test_annotation_values_round_trip_in_their_own_typed_column(tmp_path, value, kind):
    dataset = _dataset()
    record = dataset.records[0]
    annotation = Annotation(
        key="probe", value=value, id="probe-content", span=None if value is not None else TimePoint(start_us=0)
    )
    record.annotate(annotation)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (restored,) = reader.read_records((record.record_id,))
    back = next(item for item in restored.annotations if item.id == "probe-content")
    stored = (
        connect_control(path, read_only=True)
        .execute("SELECT value_kind FROM annotation_contents WHERE content_id = 'probe-content'")
        .fetchone()
    )

    assert back.value == value
    assert type(back.value) is type(value)
    assert stored == (kind,)


def test_arrow_task_views_match_the_hydrated_objects(tmp_path):
    dataset = make_dataset()
    answer_task = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (classification_task,) = answer_task.from_tasks
    (record,) = answer_task.inputs
    age_annotation = next(annotation for annotation in record.annotations if annotation.key == "age")
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        tasks = {task.id: task for task in reader.read_tasks(reader.read_records())}
        table = reader.task_table().to_pylist()
        targets = reader.target_table().to_pylist()
        subset = reader.task_table(record_ids=(record.id,)).to_pylist()
        annotations = reader.annotation_table(object_type=ObjectKind.RECORD).to_pylist()

    assert [row["task_id"] for row in table] == sorted(tasks)
    answer = next(row for row in table if row["task_id"] == answer_task.id)
    assert answer["task_type"] == str(answer_task.task_type)
    assert answer["input_record_ids"] == [item.id for item in answer_task.inputs]
    assert answer["from_task_ids"] == [task.id for task in answer_task.from_tasks]
    assert answer["scope_type"] is None
    assert {row["task_id"] for row in subset} == {
        task.id for task in tasks.values() if any(item.id == record.id for item in task.inputs)
    }
    assert classification_task.targets is not None
    assert [(row["target_kind"], row["text_value"]) for row in targets if row["task_id"] == classification_task.id] == [
        ("text", target) for target in classification_task.targets
    ]
    age = next(row for row in annotations if row["content_id"] == age_annotation.id and row["object_id"] == record.id)
    assert (age["value_kind"], age["integer_value"], age["unit"]) == (
        "integer",
        age_annotation.value,
        age_annotation.unit,
    )


def test_split_round_trips_and_filters_before_hydration(tmp_path, monkeypatch):
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 2)
    dataset = _dataset()
    record = dataset.records[0]
    dataset.add_tasks(
        tasks=(
            *(
                AnswerTask(id=f"task-train-{index}", inputs=(record,), targets=("train answer",), split=Split.TRAIN)
                for index in range(3)
            ),
            AnswerTask(id="task-test", inputs=(record,), targets=("test answer",), split=Split.TEST),
        )
    )
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        train = list(reader.iter_tasks(split="train"))
        test_rows = reader.task_table(split=Split.TEST).to_pylist()
        train_targets = reader.target_table(split=Split.TRAIN).to_pylist()
        every = reader.task_table().to_pylist()
        with pytest.raises(TimeFValidationError, match="unknown split 'holdout'"):
            reader.task_table(split="holdout")

    assert [task.id for task in train] == ["task-train-0", "task-train-1", "task-train-2"]
    assert all(task.split is Split.TRAIN for task in train)
    assert [row["task_id"] for row in test_rows] == ["task-test"]
    assert [row["task_id"] for row in train_targets] == ["task-train-0", "task-train-1", "task-train-2"]
    assert {row["task_id"]: row["split"] for row in every} == {
        "task-1": None,
        "task-test": "test",
        "task-train-0": "train",
        "task-train-1": "train",
        "task-train-2": "train",
    }
