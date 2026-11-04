from dataclasses import replace

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
    LabelMode,
    LabelSet,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)


SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt)
STAGES = LabelSet(name="stage", labels=("W", "N1", Label(name="R", description="REM sleep.")))
FINDINGS = LabelSet(name="findings", labels=("AF", "PVC"), mode=LabelMode.MULTI)


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


def _read_tasks(path):
    with DuckDBControlReader(path) as reader:
        return reader.read_tasks(reader.read_records())


def test_tasks_that_use_one_label_set_share_one_stored_row_and_one_read_object(tmp_path):
    dataset, record = _dataset()
    dataset.add_tasks(
        tasks=[
            ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES),
            # An equal set built separately is the same set.
            ClassificationTask(inputs=(record,), targets=("R",), label_set=STAGES.model_copy()),
            ClassificationTask(inputs=(record,), targets=("PVC", "AF"), label_set=FINDINGS),
            ClassificationTask(inputs=(record,), targets=("anything",)),
        ]
    )
    path = tmp_path / "control.duckdb"
    _write(dataset, path)

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
        assert connection.execute(
            "SELECT label_set_id, name, labels, label_descriptions, mode FROM label_sets ORDER BY label_set_id"
        ).fetchall() == [
            (1, "stage", ["W", "N1", "R"], [None, None, "REM sleep."], "single"),
            (2, "findings", ["AF", "PVC"], [None, None], "multi"),
        ]
        assert connection.execute("SELECT label_set_id FROM tasks ORDER BY task_id").fetchall() == [
            (1,),
            (1,),
            (2,),
            (None,),
        ]

    stage_w, stage_r, findings, unlabelled = _read_tasks(path)
    assert stage_w.label_set == STAGES
    assert stage_w.label_set is stage_r.label_set
    assert findings.label_set == FINDINGS
    assert findings.targets is not None
    assert [target.read() for target in findings.targets] == ["PVC", "AF"]
    assert unlabelled.label_set is None


def test_the_task_table_names_each_task_label_set(tmp_path):
    dataset, record = _dataset()
    dataset.add_tasks(
        tasks=[
            ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES),
            AnswerTask(inputs=(record,), prompt="Alive?", targets=("Yes",)),
        ]
    )
    path = tmp_path / "control.duckdb"
    _write(dataset, path)

    with DuckDBControlReader(path) as reader:
        assert reader.read_task_table()["label_set"].to_pylist() == ["stage", None]


OTHER_STAGES = LabelSet(name="stage", labels=("W", "R"))


def test_a_second_label_set_with_a_used_name_is_rejected_when_the_task_is_added():
    dataset, record = _dataset()
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES))

    with pytest.raises(TimeFValidationError, match="label set named 'stage' that differs from another set"):
        dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("W",), label_set=OTHER_STAGES))
    assert len(dataset.tasks) == 1


def test_two_label_sets_with_one_name_in_one_batch_add_no_task():
    dataset, record = _dataset()

    with pytest.raises(TimeFValidationError, match="label set named 'stage' that differs from another set"):
        dataset.add_tasks(
            tasks=[
                ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES),
                ClassificationTask(inputs=(record,), targets=("W",), label_set=OTHER_STAGES),
            ]
        )
    assert dataset.tasks == ()
    # The failed batch leaves no name behind.
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("W",), label_set=OTHER_STAGES))


def test_a_stream_rejects_two_label_sets_with_one_name():
    dataset, record = _dataset()
    dataset.set_task_stream(
        (ClassificationTask,),
        lambda: iter(
            [
                ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES),
                ClassificationTask(inputs=(record,), targets=("W",), label_set=OTHER_STAGES),
            ]
        ),
    )

    with pytest.raises(TimeFValidationError, match="label set named 'stage' that differs from another set"):
        list(dataset.iter_streamed_tasks_validated())


def test_the_writer_rejects_a_label_set_changed_after_the_task_was_added(tmp_path):
    dataset, record = _dataset()
    second = ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES)
    dataset.add_tasks(tasks=[ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES), second])
    second.label_set = OTHER_STAGES

    with pytest.raises(TimeFValidationError, match="label set named 'stage' that differs from an earlier set"):
        _write(dataset, tmp_path / "control.duckdb")


def test_an_empty_multi_label_target_survives_a_write_and_a_read(tmp_path):
    dataset, record = _dataset()
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=(), label_set=FINDINGS))
    path = tmp_path / "control.duckdb"
    _write(dataset, path)

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
    (task,) = _read_tasks(path)
    assert task.targets == ()
    assert task.label_set == FINDINGS


@pytest.mark.parametrize(
    ("task", "message"),
    [
        (
            ClassificationTask(targets=("X",), label_set=STAGES),
            "^ClassificationTask 1: labels not in label set 'stage': \\['X'\\]",
        ),
        (ClassificationTask(targets=("W", "R"), label_set=STAGES), ": 2 labels"),
        (ClassificationTask(targets=("AF", "AF"), label_set=FINDINGS), "repeated labels"),
        (ClassificationTask(targets=(1,), label_set=STAGES), "each label must be text, not int"),
    ],
)
def test_adding_a_task_checks_its_labels_against_its_label_set(task, message):
    dataset, record = _dataset()

    with pytest.raises(TimeFValidationError, match=message):
        dataset.add_task(task=replace(task, inputs=(record,)))
    assert dataset.tasks == ()


def test_target_annotations_are_checked_against_the_label_set():
    dataset, record = _dataset()
    awake = record.annotate(Annotation(key="stage", value="W"))
    unknown = record.annotate(Annotation(key="stage", value="X"))
    dataset.add_task(task=ClassificationTask(inputs=(record,), target_annotations=(awake,), label_set=STAGES))

    with pytest.raises(TimeFValidationError, match="labels not in label set 'stage': \\['X'\\]"):
        dataset.add_task(task=ClassificationTask(inputs=(record,), target_annotations=(unknown,), label_set=STAGES))


def test_a_multi_label_task_can_say_that_no_label_applies():
    dataset, record = _dataset()

    (task,) = dataset.add_tasks(tasks=[ClassificationTask(inputs=(record,), targets=(), label_set=FINDINGS)])
    assert task.targets == ()


def test_streamed_tasks_are_checked_against_their_label_set():
    dataset, record = _dataset()
    dataset.set_task_stream(
        (ClassificationTask,),
        lambda: iter([ClassificationTask(inputs=(record,), targets=("X",), label_set=STAGES)]),
    )

    with pytest.raises(TimeFValidationError, match="labels not in label set 'stage': \\['X'\\]"):
        list(dataset.iter_streamed_tasks_validated())


@pytest.mark.parametrize(
    ("corruption", "field"),
    [
        ("UPDATE label_sets SET mode = 'several'", "label_sets.mode"),
        ("UPDATE label_sets SET label_descriptions = [NULL]", "label_sets.label_descriptions"),
        ("UPDATE label_sets SET labels = ['W', 'W', 'R'] WHERE name = 'stage'", "label_sets.labels"),
        ("UPDATE label_sets SET labels = ['W', '', 'R'] WHERE name = 'stage'", "label_sets.labels"),
        ("UPDATE tasks SET label_set_id = 1 WHERE task_type = 'answer'", "tasks.label_set_id"),
        ("UPDATE tasks SET label_set_id = 9 WHERE task_type = 'classification'", "label_sets.label_set_id"),
        ("UPDATE task_targets SET text_value = 'BOGUS' WHERE task_id = 1", "task 1 has labels that do not fit"),
        ("INSERT INTO task_targets (task_id, position, target_kind, text_value) VALUES (1, 1, 'text', 'R')", "task 1"),
        (
            "UPDATE task_targets SET target_kind = 'integer', text_value = NULL, integer_value = 3 WHERE task_id = 1",
            "task 1",
        ),
        ("UPDATE task_targets SET text_value = 'PVC' WHERE task_id = 3", "task 3 has labels that do not fit"),
        (
            "UPDATE annotation_contents SET text_value = 'BOGUS' WHERE text_value = 'N1'",
            "task 4 has labels that do not fit",
        ),
    ],
)
def test_the_audit_rejects_a_corrupt_label_set(tmp_path, corruption, field):
    dataset, record = _dataset()
    stage = record.annotate(Annotation(key="stage", value="N1"))
    dataset.add_tasks(
        tasks=[
            ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES),
            AnswerTask(inputs=(record,), prompt="Alive?", targets=("Yes",)),
            ClassificationTask(inputs=(record,), targets=("AF", "PVC"), label_set=FINDINGS),
            ClassificationTask(inputs=(record,), target_annotations=(stage,), label_set=STAGES),
        ]
    )
    path = tmp_path / "control.duckdb"
    _write(dataset, path)

    with connect_control(path) as connection:
        connection.execute(corruption)
        with pytest.raises(TimeFFormatError, match=field):
            audit_control_database(connection, require_chunks=False)


def test_the_reader_rejects_a_label_set_on_another_task_type(tmp_path):
    dataset, record = _dataset()
    dataset.add_tasks(
        tasks=[
            ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES),
            AnswerTask(inputs=(record,), prompt="Alive?", targets=("Yes",)),
        ]
    )
    path = tmp_path / "control.duckdb"
    _write(dataset, path)
    with connect_control(path) as connection:
        connection.execute("UPDATE tasks SET label_set_id = 1 WHERE task_type = 'answer'")

    with pytest.raises(TimeFFormatError, match="answer task 2 has a label set"):
        _read_tasks(path)
