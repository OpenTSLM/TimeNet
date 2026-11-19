from dataclasses import replace
from pathlib import Path
import shutil

import pyarrow as pa
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.control_audit import audit_control_database
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.format.storage import StorageSources
from timenet.format.stored_values import PackedValueWriter
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    Label,
    LabelSet,
    License,
    TimePoint,
    TimeSeriesSpec,
    Version,
    ureg,
)


SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt)
STAGE_LABELS = LabelSet(
    ["W", "N1", Label(name="R", description="REM sleep.")],
    annotations=(Annotation(key="name", value="stage", description="Sleep stage"),),
)
FINDINGS = LabelSet(["AF", "PVC"], multi_label=True)


def _record() -> Record:
    signal = Signal.from_loader(
        name="I",
        spec=SPEC,
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    return Record(sources=(Source(name="ECG", signals=(signal,)),))


def _dataset() -> tuple[TimeFDataset, Record]:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="org/label-sets",
            dataset_version=Version(1, 0, 0),
            name="Label sets",
            description="Classification tasks with label sets",
            license=License.MIT,
        )
    )
    return dataset, dataset.add_record(record=_record())


def _write(dataset: TimeFDataset, path):
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)


def _label_sets(tasks) -> list[LabelSet]:
    return [task.label_set for task in tasks if isinstance(task, ClassificationTask)]


def _read_tasks(path):
    with DuckDBControlReader(path) as reader:
        return reader.read_tasks(reader.read_records())


@pytest.fixture(scope="module")
def release(tmp_path_factory) -> Path:
    """Write one release with every kind of labeled task for all tests in this module.

    Returns:
        The path of the control database. If a test changes the release, it uses ``release_copy``.
    """
    dataset, record = _dataset()
    stage = record.annotate(Annotation(key="stage", value="N1"))
    dataset.add_tasks(
        tasks=[
            ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGE_LABELS),  # 1
            AnswerTask(inputs=(record,), prompt="Alive?", targets=("Yes",)),  # 2
            ClassificationTask(inputs=(record,), targets=("PVC", "AF"), label_set=FINDINGS),  # 3
            ClassificationTask(inputs=(record,), target_annotations=(stage,), label_set=STAGE_LABELS),  # 4
            # An equal set built separately is the same set.
            ClassificationTask(inputs=(record,), targets=("R",), label_set=replace(STAGE_LABELS)),  # 5
            # A multi-label target with no label says that none applies.
            ClassificationTask(inputs=(record,), targets=(), label_set=FINDINGS),  # 6
        ]
    )
    path = tmp_path_factory.mktemp("release") / "control.duckdb"
    _write(dataset, path)
    return path


@pytest.fixture
def release_copy(release, tmp_path) -> Path:
    """Copy the shared release so that a test can change its own copy.

    Returns:
        The path of the copy's control database.
    """
    shutil.copytree(release.parent, tmp_path / "release")
    return tmp_path / "release" / release.name


def test_each_label_set_is_stored_once_and_read_as_one_object(release):
    with connect_control(release, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
        assert connection.execute(
            "SELECT label_set_id, labels, multi_label, ordered FROM label_sets ORDER BY 1"
        ).fetchall() == [
            (
                1,
                [
                    {"name": "W", "description": None},
                    {"name": "N1", "description": None},
                    {"name": "R", "description": "REM sleep."},
                ],
                False,
                False,
            ),
            (2, [{"name": "AF", "description": None}, {"name": "PVC", "description": None}], True, False),
        ]

    stage_w, _, findings, stage_n1, stage_r, none_apply = _read_tasks(release)
    assert stage_w.label_set == STAGE_LABELS
    assert stage_w.label_set is stage_n1.label_set is stage_r.label_set
    assert findings.label_set == FINDINGS
    assert findings.label_set is none_apply.label_set
    assert findings.targets is not None
    assert [target.read() for target in findings.targets] == ["PVC", "AF"]
    assert none_apply.targets == ()


def test_a_filtered_read_still_loads_the_label_set_annotations(release):
    with DuckDBControlReader(release) as reader:
        tasks = reader.read_tasks_by_id([4, 5])

    first, second = _label_sets(tasks.values())
    assert first is second
    assert [(annotation.key, annotation.value) for annotation in first.annotations] == [("name", "stage")]


def test_the_batches_of_a_task_stream_share_one_label_set_object(release):
    with DuckDBControlReader(release) as reader:
        batches = list(reader.iter_task_batches(batch_size=1))

    first, fourth = _label_sets(batches[0] + batches[3])
    assert first is fourth


def _stored_sets(path):
    with connect_control(path, read_only=True) as connection:
        return connection.execute("SELECT label_set_id, labels, multi_label FROM label_sets ORDER BY 1").fetchall()


def _write_sets(tmp_path, *label_sets):
    dataset, record = _dataset()
    dataset.add_tasks(
        tasks=[
            ClassificationTask(inputs=(record,), targets=(label_set.labels[0].name,), label_set=label_set)
            for label_set in label_sets
        ]
    )
    path = tmp_path / "control.duckdb"
    _write(dataset, path)
    return path


def test_sets_are_stored_once_per_content_and_keep_the_first_label_order(tmp_path):
    reordered = LabelSet(STAGE_LABELS.labels[::-1])
    described = LabelSet([Label(name="W", description="Awake."), "N1", "R"])
    multi = LabelSet(STAGE_LABELS.labels, multi_label=True)
    path = _write_sets(tmp_path, STAGE_LABELS, reordered, described, multi)

    assert [row[0] for row in _stored_sets(path)] == [1, 2, 3]
    first, second, *_ = _read_tasks(path)
    assert first.label_set is second.label_set
    assert first.label_set.names == ("W", "N1", "R")


def test_ordered_sets_in_another_order_are_stored_as_two_rows_in_their_own_order(tmp_path):
    low_high = LabelSet(["low", "high"], ordered=True)
    high_low = LabelSet(["high", "low"], ordered=True)
    unordered = LabelSet(["low", "high"])
    path = _write_sets(tmp_path, low_high, high_low, LabelSet(["low", "high"], ordered=True), unordered)

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
        assert connection.execute("SELECT label_set_id, ordered FROM label_sets ORDER BY 1").fetchall() == [
            (1, True),
            (2, True),
            (3, False),
        ]
    first, second, third, fourth = _label_sets(_read_tasks(path))
    assert first == low_high and first.ordered
    assert first.names == ("low", "high")
    assert second.names == ("high", "low")
    assert first is third
    assert fourth == unordered and not fourth.ordered


def test_equal_sets_with_different_annotations_are_stored_once_with_both(tmp_path):
    first = LabelSet(STAGE_LABELS.labels, annotations=(Annotation(key="name", value="stage"),))
    second = LabelSet(
        STAGE_LABELS.labels,
        annotations=(Annotation(key="name", value="stage"), Annotation(key="source", value="manual")),
    )
    bare = LabelSet(STAGE_LABELS.labels)
    path = _write_sets(tmp_path, first, bare, second)

    assert [row[0] for row in _stored_sets(path)] == [1]
    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
        assert connection.execute(
            "SELECT count(*) FROM annotation_occurrences WHERE object_type = 'LabelSet'"
        ).fetchone() == (2,)
    tasks = _read_tasks(path)
    assert tasks[0].label_set is tasks[1].label_set is tasks[2].label_set
    assert [(a.key, a.value) for a in tasks[0].label_set.annotations] == [("name", "stage"), ("source", "manual")]


def test_occurrences_that_differ_only_in_occurrence_fields_are_all_stored(tmp_path):
    first = LabelSet(STAGE_LABELS.labels, annotations=(Annotation(key="name", value="stage", confidence=0.5),))
    first.annotate(Annotation(key="name", value="stage", occurrence_metadata={"k": 1}))
    second = LabelSet(
        STAGE_LABELS.labels,
        annotations=(
            Annotation(key="name", value="stage", confidence=0.5),
            Annotation(key="name", value="stage", source="rater", confidence=0.9),
        ),
    )
    path = _write_sets(tmp_path, first, second)

    (label_set, _) = _label_sets(_read_tasks(path))
    assert [(a.source, a.confidence, a.occurrence_metadata) for a in label_set.annotations] == [
        (None, 0.5, {}),
        (None, None, {"k": 1}),
        ("rater", 0.9, {}),
    ]


def test_writing_does_not_change_the_callers_label_sets(tmp_path):
    first = LabelSet(STAGE_LABELS.labels, annotations=(Annotation(key="name", value="stage"),))
    second = LabelSet(STAGE_LABELS.labels, annotations=(Annotation(key="source", value="manual"),))
    first_annotations, second_annotations = first.annotations, second.annotations

    _write_sets(tmp_path, first, second)

    assert first.annotations is first_annotations
    assert second.annotations is second_annotations


def test_equal_annotations_of_two_sets_share_one_content_row(tmp_path):
    path = _write_sets(
        tmp_path,
        LabelSet(STAGE_LABELS.labels, annotations=(Annotation(key="name", value="stage"),)),
        LabelSet(FINDINGS.labels, annotations=(Annotation(key="name", value="stage"),)),
    )

    with connect_control(path, read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM annotation_contents").fetchone() == (1,)
        assert connection.execute(
            "SELECT object_id FROM annotation_occurrences WHERE object_type = 'LabelSet' ORDER BY 1"
        ).fetchall() == [(1,), (2,)]


def test_the_writer_rejects_a_label_set_changed_after_add_task(tmp_path):
    dataset, record = _dataset()
    second = ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGE_LABELS)
    dataset.add_tasks(tasks=[ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGE_LABELS), second])
    second.label_set = LabelSet(["X"])

    with pytest.raises(TimeFValidationError, match=r"ClassificationTask 2: labels not in label set \['X'\]"):
        _write(dataset, tmp_path / "control.duckdb")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("labels", (Label(name="W"), Label(name="W")), "lists a label more than once"),
        ("multi_label", 1, "multi_label must be a bool, got int"),
        ("annotations", (Annotation(key="x", span=TimePoint.seconds(1.0)),), "has a span"),
    ],
)
def test_the_writer_checks_a_label_set_changed_after_it_was_built(tmp_path, field, value, message):
    label_set = LabelSet(["W"])
    dataset, record = _dataset()
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("W",), label_set=label_set))
    setattr(label_set, field, value)

    with pytest.raises(TimeFValidationError, match=message):
        _write(dataset, tmp_path / "control.duckdb")


def test_the_reader_rejects_an_annotation_of_a_missing_label_set(release_copy):
    with connect_control(release_copy) as connection:
        connection.execute("UPDATE annotation_occurrences SET object_id = 99 WHERE object_type = 'LabelSet'")

    with pytest.raises(TimeFFormatError, match="refers to missing LabelSet 99"):
        _read_tasks(release_copy)


def test_the_reader_rejects_a_task_that_names_a_missing_label_set(release_copy):
    with connect_control(release_copy) as connection:
        connection.execute("UPDATE tasks SET label_set_id = 99 WHERE task_id = 1")

    with pytest.raises(TimeFFormatError, match="classification task 1 refers to missing label set 99"):
        _read_tasks(release_copy)


@pytest.mark.parametrize(
    ("task", "message"),
    [
        (
            ClassificationTask(targets=("X",), label_set=STAGE_LABELS),
            "^ClassificationTask 1: labels not in label set \\['W', 'N1', 'R'\\]: \\['X'\\]",
        ),
        (ClassificationTask(targets=(1,), label_set=STAGE_LABELS), "each label must be text, not int"),
    ],
)
def test_add_task_rejects_targets_that_do_not_fit_the_label_set(task, message):
    dataset, record = _dataset()

    with pytest.raises(TimeFValidationError, match=message):
        dataset.add_task(task=replace(task, inputs=(record,)))
    assert dataset.tasks == ()


def test_add_task_checks_target_annotations_against_the_label_set():
    dataset, record = _dataset()
    awake = record.annotate(Annotation(key="stage", value="W"))
    unknown = record.annotate(Annotation(key="stage", value="X"))
    dataset.add_task(task=ClassificationTask(inputs=(record,), target_annotations=(awake,), label_set=STAGE_LABELS))

    with pytest.raises(TimeFValidationError, match="labels not in label set \\['W', 'N1', 'R'\\]: \\['X'\\]"):
        dataset.add_task(
            task=ClassificationTask(inputs=(record,), target_annotations=(unknown,), label_set=STAGE_LABELS)
        )


def test_a_stream_rejects_targets_that_do_not_fit_the_label_set():
    dataset, record = _dataset()
    dataset.set_task_stream(
        (ClassificationTask,),
        lambda: iter([ClassificationTask(inputs=(record,), targets=("X",), label_set=STAGE_LABELS)]),
    )

    with pytest.raises(TimeFValidationError, match="labels not in label set \\['W', 'N1', 'R'\\]: \\['X'\\]"):
        list(dataset.iter_streamed_tasks_validated())


@pytest.mark.parametrize(
    ("corruption", "field"),
    [
        ("UPDATE label_sets SET labels = [] WHERE label_set_id = 1", "label_sets.labels"),
        ("UPDATE label_sets SET labels = [{'name': 'W'}, {'name': 'W'}] WHERE label_set_id = 1", "label_sets.labels"),
        ("UPDATE label_sets SET labels = [{'name': 'W'}, {'name': ''}] WHERE label_set_id = 1", "label_sets.labels"),
        # A NULL name drops out of list_distinct, so the check for distinct names finds it.
        ("UPDATE label_sets SET labels = [{'name': 'W'}, {'name': NULL}] WHERE label_set_id = 1", "label_sets.labels"),
        (
            "UPDATE label_sets SET labels = [{'name': 'W', 'description': ''}] WHERE label_set_id = 1",
            "label_sets.labels",
        ),
        ("UPDATE tasks SET label_set_id = 1 WHERE task_type = 'answer'", "tasks.label_set_id"),
        ("UPDATE tasks SET label_set_id = NULL WHERE task_id = 1", "tasks.label_set_id"),
        ("UPDATE tasks SET label_set_id = 99 WHERE task_id = 1", "label_sets.label_set_id"),
        ("UPDATE task_targets SET text_value = 'BOGUS' WHERE task_id = 1", "task 1 has a label that is not"),
        (
            "INSERT INTO task_targets (task_id, position, target_kind, text_value) VALUES (1, 1, 'text', 'R')",
            "task 1 does not have exactly one label, but its label set is not multi-label",
        ),
        (
            "UPDATE task_targets SET target_kind = 'integer', text_value = NULL, integer_value = 3 WHERE task_id = 1",
            "task 1 has a label that is not text",
        ),
        ("UPDATE task_targets SET text_value = 'PVC' WHERE task_id = 3", "task 3 repeats a label"),
        (
            "UPDATE annotation_contents SET text_value = 'BOGUS' WHERE text_value = 'N1'",
            "task 4 has a label that is not",
        ),
    ],
)
def test_the_audit_rejects_a_corrupt_label_set(release_copy, corruption, field):
    with connect_control(release_copy) as connection:
        connection.execute(corruption)
        with pytest.raises(TimeFFormatError, match=field):
            audit_control_database(connection, require_chunks=False)


def test_the_reader_rejects_a_classification_task_without_a_label_set(release_copy):
    with connect_control(release_copy) as connection:
        connection.execute("UPDATE tasks SET label_set_id = NULL WHERE task_id = 1")

    with pytest.raises(TimeFFormatError, match="classification task 1 has no label set"):
        _read_tasks(release_copy)


def test_the_reader_rejects_a_label_set_on_another_task_type(release_copy):
    with connect_control(release_copy) as connection:
        connection.execute("UPDATE tasks SET label_set_id = 1 WHERE task_type = 'answer'")

    with pytest.raises(TimeFFormatError, match="answer task 2 has a label set"):
        _read_tasks(release_copy)
