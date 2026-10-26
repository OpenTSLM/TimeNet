from dataclasses import replace
from fractions import Fraction

import pyarrow as pa
import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.control_audit import audit_control_database
import timenet.format.control_reader as control_reader_module
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.format.stored_values import PackedValueWriter
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
from timenet.types.text import read_value

from .test_control_writer import LEAD_I_ID, LEAD_II_ID, RECORD_ID, SPEC, TASK_ID, _dataset


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
    image_record = Record(sources=(Source(name="image", signals=(signal,)),))
    dataset.add_record(record=image_record)
    dataset.add_task(task=AnswerTask(inputs=(image_record,), prompt="Describe this image.", targets=("A chart.",)))
    dataset.add_task(task=TSGenerationTask(prompt="Make an ECG.", targets=(dataset.records[0],)))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        assert reader.read_task_table(required_modalities=[InputModality.TIME_SERIES])["task_id"].to_pylist() == [
            TASK_ID
        ]
        assert reader.read_task_table(required_modalities=[InputModality.IMAGE])["task_id"].to_pylist() == [2]
        assert reader.read_task_table(supported_modalities=[InputModality.TEXT, InputModality.NO_INPUT])[
            "task_id"
        ].to_pylist() == [3]
        records = reader.read_records()
        tasks = {task.id: task for task in reader.read_tasks(records)}

    assert tasks[2].inputs[0].signals[0].spec.modality is InputModality.IMAGE
    assert tasks[TASK_ID].inputs[0].signals[0].spec.modality is InputModality.TIME_SERIES
    assert tasks[2].input_modalities == frozenset({InputModality.IMAGE, InputModality.TEXT})
    tasks[2].input_modalities = None
    assert tasks[2].resolved_input_modalities == frozenset({InputModality.IMAGE, InputModality.TEXT})


def test_annotations_keep_their_order_when_a_task_refers_to_a_later_one(tmp_path):
    dataset = _dataset()
    (record,) = dataset.records
    (task,) = dataset.tasks
    record.annotate(Annotation(key="age", value=70))
    task.input_annotations = (record.annotations[1],)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (stored_record,) = reader.read_records()
        (stored_task,) = reader.read_tasks((stored_record,))

    assert [annotation.key for annotation in stored_record.annotations] == ["patient_sex", "age"]
    assert stored_task.input_annotations[0] is stored_record.annotations[1]


def test_shifted_regular_axis_round_trip(tmp_path):
    template = _dataset()
    axis = replace(RegularAxis.from_rate_hz(44100, offset_us=250_003).at_index(1026))
    signals = tuple(
        Signal.from_loader(
            name=signal.name,
            spec=signal.spec,
            time_axis=axis,
            n_values=signal.n_values,
            loader=signal.to_arrow,
        )
        for signal in template.records[0].signals
    )
    dataset = TimeFDataset(metadata=template.metadata)
    dataset.add_record(record=Record(sources=(Source(name="ECG", signals=signals),)))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)
    with DuckDBControlReader(path) as reader:
        (record,) = reader.read_records()
    restored = record.signals[0].time_axis
    assert restored == axis
    assert restored is record.signals[1].time_axis
    assert record.signals[0].span_us == (273268, 273313)


def test_control_reader_hydrates_recursive_hierarchy_and_annotations(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)
    values = {
        LEAD_I_ID: pa.array([1.0, 2.0], type=pa.float32()),
        LEAD_II_ID: pa.array([3.0, 4.0], type=pa.float32()),
    }

    with DuckDBControlReader(path, value_loader=lambda signal_id, _spec: values[signal_id]) as reader:
        (record,) = reader.read_records()
        (task,) = reader.read_tasks((record,))

    assert record.id == RECORD_ID
    assert record.sources[0].name == "Monitor"
    assert record.sources[0].sources[0].name == "ECG"
    lead_i, lead_ii = record.signals
    assert lead_i.time_axis is lead_ii.time_axis
    assert lead_i.spec is lead_ii.spec
    assert lead_i.metadata == lead_ii.metadata == {}
    assert lead_i.metadata is not lead_ii.metadata
    assert lead_i.to_arrow().to_pylist() == [1.0, 2.0]
    assert record.annotations == (Annotation(key="patient_sex", value="male"),)
    assert task.inputs[0] is record
    assert task.prompt == "Alive?"
    assert task.targets == ("Yes",)
    assert task.annotations[0].name == "task_kind"


def test_reader_and_audit_reject_ambiguous_stored_task_time_span(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    second = dataset.add_record(record=Record(sources=(Source(name="other"),)))
    dataset.tasks[0].inputs = (*dataset.tasks[0].inputs, second)
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)
    with connect_control(path) as connection:
        connection.execute(
            "UPDATE tasks SET scope_type = 'time_interval', scope_start = 0, scope_end = 2000 WHERE task_id = ?",
            [TASK_ID],
        )
        with pytest.raises(TimeFFormatError, match="Record"):
            audit_control_database(connection, require_chunks=False)
    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match="Record"):
        reader.read_tasks(reader.read_records())


def test_unknown_unit_round_trips_distinct_from_dimensionless(tmp_path):
    template = _dataset()
    unknown = TimeSeriesSpec(spec_type="unknown", name="Unknown", unit_value=None)
    signals = tuple(
        Signal.from_loader(
            name=signal.name,
            spec=spec,
            time_axis=signal.time_axis,
            n_values=signal.n_values,
            loader=signal.to_arrow,
        )
        for signal, spec in zip(
            template.records[0].signals,
            (unknown, unknown.model_copy(update={"spec_type": "ratio", "unit_value": ureg.dimensionless})),
            strict=True,
        )
    )
    dataset = TimeFDataset(metadata=template.metadata)
    dataset.add_record(record=Record(sources=(Source(name="ECG", signals=signals),)))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (record,) = reader.read_records()

    assert record.signals[0].spec.unit_value is None
    assert record.signals[1].spec.unit_value == ureg.dimensionless


def test_input_modalities_round_trip_as_typed_task_data(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    dataset.tasks[0].input_modalities = frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (task,) = reader.read_tasks(reader.read_records())
        rows = reader.read_task_table().to_pylist()

    assert task.input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    assert rows[0]["input_modalities"] == ["text", "time_series"]


def test_modality_filters_select_tasks_and_targets_before_hydration(tmp_path, monkeypatch):
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 2)
    dataset = _dataset()
    record = dataset.records[0]
    dataset.add_tasks(
        tasks=(
            AnswerTask(
                inputs=(record,),
                prompt="Read chart",
                targets=("image answer",),
                input_modalities=frozenset({InputModality.TEXT, InputModality.IMAGE}),
            ),
            AnswerTask(
                inputs=(record,),
                prompt="Read chart and series",
                targets=("mixed answer",),
                input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES, InputModality.IMAGE}),
            ),
            AnswerTask(
                targets=("unknown",),
            ),
        )
    )
    dataset.add_tasks(
        tasks=(
            AnswerTask(
                inputs=(record,),
                targets=("series answer",),
            )
            for index in range(4)
        )
    )
    expected = [TASK_ID, 5, 6, 7, 8]
    required = frozenset({InputModality.TIME_SERIES})
    supported = frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        tasks = list(reader.iter_tasks(required_modalities=required, supported_modalities=supported))
        task_rows = reader.read_task_table(required_modalities=required, supported_modalities=supported).to_pylist()
        target_rows = reader.read_target_table(required_modalities=required, supported_modalities=supported).to_pylist()

    assert [task.id for task in tasks] == expected
    assert [row["task_id"] for row in task_rows] == expected
    assert [row["task_id"] for row in target_rows] == expected
    assert [
        task.id for task in dataset.iter_tasks(required_modalities=required, supported_modalities=supported)
    ] == expected


def _spans_over(signal):
    return (
        TimePoint.seconds(0.001, time_series=(signal,)),
        TimeInterval.seconds(0, 0.002, time_series=(signal,)),
        StepPoint(start=1, time_series=signal),
        StepInterval(start=0, stop=2, time_series=signal),
    )


def test_control_reader_hydrates_each_target_storage_type(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    record = dataset.records[0]
    signal = record.signals[0]
    dataset.tasks[0].targets = ("Yes", 7, 0.5, True, record, signal, *_spans_over(signal))
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        (task,) = reader.read_tasks(records)
        rows = reader.read_target_table().to_pylist()

    assert task.targets is not None
    text, integer, floating, boolean, target_record, target_signal, *target_spans = task.targets
    assert (text, integer, floating, boolean) == ("Yes", 7, 0.5, True)
    assert target_record is records[0]
    assert target_signal is records[0].signals[0]
    assert tuple(target_spans) == _spans_over(target_signal)
    assert [(row["target_kind"], row["record_id"], row["signal_id"], row["span_signal_ids"]) for row in rows[4:]] == [
        ("record", record.id, None, None),
        ("signal", None, signal.id, None),
        ("time_point", None, None, [signal.id]),
        ("time_interval", None, None, [signal.id]),
        ("step_point", None, None, [signal.id]),
        ("step_interval", None, None, [signal.id]),
    ]


def test_control_reader_preserves_ordered_integer_relationships(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    record = dataset.records[0]
    second = record.annotate(Annotation(key="patient_age", value=65, unit="year"))
    dataset.tasks[0].input_annotations = (second, record.annotations[0])
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        (task,) = reader.read_tasks(records)

    assert task.inputs == records
    assert tuple(annotation.key for annotation in task.input_annotations) == ("patient_age", "patient_sex")


def test_text_targets_do_not_build_a_signal_index(tmp_path, monkeypatch):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(_dataset())

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
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)
    with connect_control(path) as connection:
        connection.execute(
            """UPDATE signals SET modality = 'corrupt'
               WHERE source_id IN (SELECT source_id FROM sources WHERE record_id = ?)""",
            [corrupt_record.id],
        )
    with DuckDBControlReader(path) as reader:
        (selected,) = reader.read_records((selected_record.id,))
        assert selected.id == selected_record.id
        with pytest.raises(TimeFFormatError, match="invalid modality"):
            reader.read_records((corrupt_record.id,))


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "DELETE FROM annotation_contents WHERE content_id = 1",
            "missing content",
        ),
        (
            "UPDATE sources SET parent_source_id = 999 WHERE source_id = 2",
            "missing parent",
        ),
        (
            "UPDATE signals SET source_id = 999 WHERE signal_id = 1",
            "missing source",
        ),
    ],
)
def test_control_reader_rejects_corrupt_relationships(tmp_path, statement, message):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(_dataset())
    with connect_control(path) as connection:
        connection.execute(statement)

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match=message):
        reader.read_records()


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("UPDATE annotation_occurrences SET object_id = 999 WHERE content_id = 1", "missing Record 999"),
        (
            "UPDATE annotation_occurrences SET object_type = 'Signal', object_id = 999 WHERE content_id = 1",
            "missing Signal",
        ),
        (
            "UPDATE annotation_occurrences SET object_type = 'Unknown' WHERE content_id = 1",
            "unknown object type",
        ),
        (
            "UPDATE annotation_occurrences SET object_type = 'Dataset' WHERE content_id = 1",
            "on Dataset has object id",
        ),
        (
            "UPDATE annotation_occurrences SET object_id = NULL WHERE content_id = 1",
            "on Record has object id None",
        ),
        ("UPDATE annotation_occurrences SET object_id = 999 WHERE object_type = 'Task'", "missing Task 999"),
    ],
)
def test_full_reads_reject_occurrences_with_an_invalid_owner(tmp_path, statement, message):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(_dataset())
    with connect_control(path) as connection:
        connection.execute(statement)

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match=message):
        reader.read_tasks(cached_records=reader.read_records())


def test_full_task_read_rejects_an_unknown_owner_type_without_task_annotations(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(_dataset())
    with connect_control(path) as connection:
        connection.execute("DELETE FROM annotation_occurrences WHERE object_type = 'Task'")
        connection.execute("UPDATE annotation_occurrences SET object_type = 'Unknown' WHERE content_id = 1")

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match="unknown object type"):
        reader.read_tasks()


def test_annotations_stay_with_their_owner_when_ids_collide_across_kinds(tmp_path):
    signal = Signal.from_loader(
        name="I",
        spec=SPEC,
        time_axis=RegularAxis(period_us=Fraction(2_000)),
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    source = Source(name="ECG", signals=(signal,))
    record = Record(sources=(source,))
    record.annotate(Annotation(key="owner", value="record"))
    source.annotate(Annotation(key="owner", value="source"))
    signal.annotate(Annotation(key="owner", value="signal"))
    task = AnswerTask(inputs=(record,), prompt="Alive?", targets=("Yes",))
    task.annotate(Annotation(key="owner", value="task"))
    dataset = TimeFDataset(metadata=_dataset().metadata)
    dataset.annotate(Annotation(key="owner", value="dataset"))
    dataset.add_record(record=record)
    dataset.add_task(task=task)
    # Each kind counts from 1, so the record, the source, the signal and the task share id 1.
    shared_id = record.id
    assert signal.id == task.id == shared_id
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
    with DuckDBControlReader(path) as reader:
        for records in (reader.read_records(), reader.read_records((shared_id,))):
            (back,) = records
            (back_source,) = back.sources
            (back_signal,) = back_source.signals
            assert [item.value for item in back.annotations] == ["record"]
            assert [item.value for item in back_source.annotations] == ["source"]
            assert [item.value for item in back_signal.annotations] == ["signal"]
            for tasks in (reader.read_tasks(cached_records=records), reader.read_tasks(records)):
                assert [item.value for item in tasks[0].annotations] == ["task"]
        assert [item.value for item in reader.read_dataset_annotations(dataset.metadata.dataset_id)] == ["dataset"]
        assert reader.read_dataset_annotations("org/other") == ()


def test_iter_tasks_for_one_record_yields_its_tasks_and_shares_record_objects(tmp_path):
    dataset = make_dataset()
    expected_answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (expected_record,) = expected_answer.inputs
    (expected_parent,) = expected_answer.from_tasks
    expected_ids = sorted(task.id for task in dataset.tasks if expected_record in task.inputs)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

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
    dataset = _dataset()
    record = dataset.records[0]
    parent = AnswerTask(inputs=(record,), targets=("parent",))
    child = AnswerTask(inputs=(record,), targets=("child",), from_tasks=(parent,))
    dataset.add_tasks(tasks=(child, parent))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (record,) = reader.read_records()
        tasks = reader.read_tasks((record,)) if materialize else list(reader.iter_tasks((record,)))

    by_id = {task.id: task for task in tasks}
    assert set(by_id) == {TASK_ID, child.id, parent.id}
    assert child.id < parent.id
    assert by_id[child.id].from_tasks[0] is by_id[parent.id]
    assert tuple(sorted(by_id)) == tuple(task.id for task in tasks)


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("UPDATE task_targets SET span_signal_ids = [999] WHERE target_kind = 'step_point'", "missing signal 999"),
        ("UPDATE tasks SET scope_signal_ids = [999] WHERE task_id = 4", "missing signal 999"),
        ("UPDATE task_record_refs SET record_id = 999", "missing record"),
        ("UPDATE task_targets SET record_id = 999 WHERE target_kind = 'record'", "missing record"),
        ("UPDATE task_targets SET signal_id = 999 WHERE target_kind = 'signal'", "missing signal 999"),
        ("UPDATE task_dependencies SET parent_task_id = 999", "missing task 999"),
        ("UPDATE task_record_refs SET field = 'outputs'", r"task_record_refs has unknown fields \['outputs'\]"),
    ],
)
def test_read_tasks_rejects_corrupt_task_references(tmp_path, statement, message):
    dataset = _dataset()
    record = dataset.records[0]
    step = StepPoint(start=1, time_series=record.signals[1])
    dataset.tasks[0].targets = (step, record, record.signals[0])
    steps = Signal(name="steps", spec=SPEC, time_axis=OrdinalAxis(), data=pa.array([1.0, 2.0], pa.float32()))
    step_record = Record(sources=(Source(name="steps", signals=(steps,)),))
    dataset.add_record(record=step_record)
    parent = AnswerTask(inputs=(record,), targets=("parent",))
    scoped = AnswerTask(inputs=(step_record,), targets=("scoped",), scope=StepPoint(start=1, time_series=steps))
    dataset.add_tasks(tasks=(parent, AnswerTask(inputs=(record,), targets=("child",), from_tasks=(parent,)), scoped))
    assert scoped.id == 4
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)
    with connect_control(path) as connection:
        connection.execute(statement)

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match=message):
        reader.read_tasks(cached_records=reader.read_records())


@pytest.mark.parametrize(
    "read",
    [
        lambda reader, records: reader.read_tasks(cached_records=records),
        lambda reader, records: list(reader.iter_tasks(records)),
    ],
    ids=["read_tasks", "iter_tasks"],
)
def test_read_tasks_rejects_duplicate_task_ids(tmp_path, read):
    dataset = _dataset()
    dataset.add_task(task=AnswerTask(inputs=dataset.tasks[0].inputs, targets=("No",)))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)
    with connect_control(path) as connection:
        connection.execute("UPDATE tasks SET task_id = ? WHERE task_id = 2", [TASK_ID])

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match="duplicate task IDs"):
        read(reader, reader.read_records())


def test_read_tasks_rejects_records_the_database_does_not_store(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(_dataset())

    other = TimeFDataset(metadata=_dataset().metadata)
    for _ in range(2):
        unstored = other.add_record(record=Record())
    with DuckDBControlReader(path) as reader, pytest.raises(TimeFValidationError, match="no such record"):
        reader.read_tasks((unstored,))


def test_typed_scope_and_payload_columns_round_trip(tmp_path):
    dataset = make_dataset()
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        restored = {task.id: task for task in reader.read_tasks(reader.read_records())}

    for task in dataset.tasks:
        back = restored[task.id]
        assert type(back) is type(task)
        assert _scope_key(back) == _scope_key(task)
        for field in ("target_schema", "unit", "target_name", "mode"):
            if hasattr(task, field):
                assert getattr(back, field) == getattr(task, field), field


def _scope_key(task):
    scope = task.scope
    if scope is None:
        return None
    return replace(scope, time_series=None), tuple(signal.id for signal in scope.time_series or ())


@pytest.mark.parametrize(
    ("value", "kind"),
    [("W", "text"), (64, "integer"), (0.5, "float"), (True, "boolean"), (["yes", "no"], "text_list"), (None, None)],
)
def test_annotation_values_round_trip_in_their_own_typed_column(tmp_path, value, kind):
    dataset = _dataset()
    record = dataset.records[0]
    annotation = Annotation(key="probe", value=value, span=None if value is not None else TimePoint(start_us=0))
    record.annotate(annotation)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (restored,) = reader.read_records((record.id,))
    back = next(item for item in restored.annotations if item.key == "probe")
    stored = (
        connect_control(path, read_only=True)
        .execute("SELECT value_kind FROM annotation_contents WHERE name = 'probe'")
        .fetchone()
    )

    assert back.value == value
    assert type(read_value(back.value)) is type(value)
    assert stored == (kind,)


def test_arrow_task_views_match_the_hydrated_objects(tmp_path):
    dataset = make_dataset()
    answer_task = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (classification_task,) = answer_task.from_tasks
    (record,) = answer_task.inputs
    age_annotation = next(annotation for annotation in record.annotations if annotation.key == "age")
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        tasks = {task.id: task for task in reader.read_tasks(reader.read_records())}
        table = reader.read_task_table().to_pylist()
        targets = reader.read_target_table().to_pylist()
        subset = reader.read_task_table(record_ids=(record.id,)).to_pylist()
        annotations = reader.read_annotation_table(object_type=ObjectKind.RECORD).to_pylist()

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
    age = next(row for row in annotations if row["key"] == age_annotation.key and row["object_id"] == record.id)
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
            *(AnswerTask(inputs=(record,), targets=("train answer",), split=Split.TRAIN) for index in range(3)),
            AnswerTask(inputs=(record,), targets=("test answer",), split=Split.TEST),
        )
    )
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        train = list(reader.iter_tasks(split="train"))
        test_rows = reader.read_task_table(split=Split.TEST).to_pylist()
        train_targets = reader.read_target_table(split=Split.TRAIN).to_pylist()
        every = reader.read_task_table().to_pylist()
        with pytest.raises(TimeFValidationError, match="unknown split 'holdout'"):
            reader.read_task_table(split="holdout")

    assert [task.id for task in train] == [2, 3, 4]
    assert all(task.split is Split.TRAIN for task in train)
    assert [row["task_id"] for row in test_rows] == [5]
    assert [row["task_id"] for row in train_targets] == [2, 3, 4]
    assert {row["task_id"]: row["split"] for row in every} == {
        TASK_ID: None,
        5: "test",
        2: "train",
        3: "train",
        4: "train",
    }


def test_annotations_read_back_in_the_order_they_were_attached(tmp_path):
    dataset = _dataset()
    record = dataset.records[0]
    keys = [f"key-{n}" for n in (7, 2, 9, 0, 5, 3, 8, 1, 6, 4)]
    for key in keys:
        record.annotate(Annotation(key=key, value=1))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (restored,) = reader.read_records((record.id,))

    assert [annotation.key for annotation in restored.annotations] == ["patient_sex", *keys]


def test_tasks_refer_to_the_annotations_of_the_records_passed_in(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(_dataset())

    with DuckDBControlReader(path) as reader:
        first = reader.read_records()
        second = reader.read_records()
        (task,) = reader.read_tasks(first)

    assert task.input_annotations[0] is first[0].annotations[0]
    assert task.input_annotations[0] is not second[0].annotations[0]


def test_tasks_find_the_annotations_of_records_from_another_reader(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    record = dataset.records[0]
    second = record.annotate(Annotation(key="patient_age", value=65, unit="year"))
    dataset.tasks[0].input_annotations = (second, record.annotations[0])
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as other:
        records = other.read_records()
    with DuckDBControlReader(path) as reader:
        (task,) = reader.read_tasks(records)

    assert task.inputs == records
    assert task.input_annotations[0] is records[0].annotations[1]
    assert task.input_annotations[1] is records[0].annotations[0]


def test_tasks_reject_a_record_from_another_reader_whose_annotation_was_replaced(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    record = dataset.records[0]
    second = record.annotate(Annotation(key="patient_age", value=65, unit="year"))
    dataset.tasks[0].input_annotations = (second,)
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as other:
        records = other.read_records()
    records[0].annotations = (records[0].annotations[0], Annotation(key="patient_age", value=70, unit="year"))
    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match="no hydrated object carries"):
        reader.read_tasks(records)


def test_full_range_int_ids_round_trip_as_python_ints(tmp_path):
    base = _dataset()
    largest = 2**63 - 1
    dataset = TimeFDataset.from_parts(
        metadata=base.metadata,
        records=base.records,
        tasks=base.tasks,
        schema=base.derive_schema(),
        last_ids={**base.last_ids, ObjectKind.RECORD: largest - 2},
    )
    generated = dataset.add_record(record=Record(sources=(Source(name="generated"),)))
    dataset.add_record(record=Record(sources=(Source(name="largest"),)))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        (task,) = reader.read_tasks(records)
        task_rows = reader.read_task_table().to_pylist()

    assert [record.id for record in records] == sorted([RECORD_ID, largest, generated.id])
    ids = [*(record.id for record in records), task.id]
    assert all(type(value) is int for value in ids)
    assert type(task_rows[0]["task_id"]) is int
    assert task_rows[0]["input_record_ids"] == [RECORD_ID]


def test_read_records_rejects_a_non_int_id(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(_dataset())

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFValidationError, match="positive int64 id"):
        reader.read_records((str(RECORD_ID),))  # ty: ignore[invalid-argument-type]


def test_annotation_table_gives_dataset_annotations_a_null_int64_object_id(tmp_path):
    dataset = _dataset()
    dataset.annotate(Annotation(key="site", value="clinic"))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent)).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (annotation,) = reader.read_dataset_annotations(dataset.metadata.dataset_id)
        table = reader.read_annotation_table()
        rows = {row["object_type"]: row for row in table.to_pylist()}

    assert annotation.value == "clinic"
    assert table.schema.field("object_id").type == pa.int64()
    assert rows["Dataset"]["object_id"] is None
    assert rows["Record"]["object_id"] == RECORD_ID
