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
    TimeSeriesSpec,
    Version,
    ureg,
)


SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt)
STAGES = LabelSet(name="stage", labels=(Label(name="W"), Label(name="N1"), Label(name="R", description="REM sleep.")))
FINDINGS = LabelSet(name="findings", labels=(Label(name="AF"), Label(name="PVC")), multi_label=True)


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
            ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES),  # 1
            AnswerTask(inputs=(record,), prompt="Alive?", targets=("Yes",)),  # 2
            ClassificationTask(inputs=(record,), targets=("PVC", "AF"), label_set=FINDINGS),  # 3
            ClassificationTask(inputs=(record,), target_annotations=(stage,), label_set=STAGES),  # 4
            # An equal set built separately is the same set.
            ClassificationTask(inputs=(record,), targets=("R",), label_set=STAGES.model_copy()),  # 5
            # A multi-label target with no label says that none applies.
            ClassificationTask(inputs=(record,), targets=(), label_set=FINDINGS),  # 6
            ClassificationTask(inputs=(record,), targets=("anything",)),  # 7
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
        assert connection.execute("SELECT name, labels, multi_label FROM label_sets ORDER BY name").fetchall() == [
            ("findings", [{"name": "AF", "description": None}, {"name": "PVC", "description": None}], True),
            (
                "stage",
                [
                    {"name": "W", "description": None},
                    {"name": "N1", "description": None},
                    {"name": "R", "description": "REM sleep."},
                ],
                False,
            ),
        ]

    stage_w, _, findings, stage_n1, stage_r, none_apply, unlabelled = _read_tasks(release)
    assert stage_w.label_set == STAGES
    assert stage_w.label_set is stage_n1.label_set is stage_r.label_set
    assert findings.label_set == FINDINGS
    assert findings.label_set is none_apply.label_set
    assert findings.targets is not None
    assert [target.read() for target in findings.targets] == ["PVC", "AF"]
    assert none_apply.targets == ()
    assert unlabelled.label_set is None


def test_the_task_table_shows_the_label_set_name(release):
    with DuckDBControlReader(release) as reader:
        assert reader.read_task_table()["label_set"].to_pylist() == [
            "stage",
            None,
            "findings",
            "stage",
            "stage",
            "findings",
            None,
        ]


OTHER_STAGES = LabelSet(name="stage", labels=(Label(name="W"), Label(name="R")))
SAME_NAME = r"ClassificationTask [0-9]+: label set 'stage' differs from another set"


def test_add_task_rejects_a_different_set_with_a_used_name():
    dataset, record = _dataset()
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES))

    with pytest.raises(TimeFValidationError, match=SAME_NAME):
        dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("W",), label_set=OTHER_STAGES))
    assert len(dataset.tasks) == 1


def test_a_rejected_batch_leaves_no_label_set_behind():
    dataset, record = _dataset()

    with pytest.raises(TimeFValidationError, match=SAME_NAME):
        dataset.add_tasks(
            tasks=[
                ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES),
                ClassificationTask(inputs=(record,), targets=("W",), label_set=OTHER_STAGES),
            ]
        )
    assert dataset.tasks == ()
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("W",), label_set=OTHER_STAGES))


def test_a_stream_rejects_a_different_set_with_a_used_name():
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

    with pytest.raises(TimeFValidationError, match=SAME_NAME):
        list(dataset.iter_streamed_tasks_validated())


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        (OTHER_STAGES, SAME_NAME),
        (LabelSet(name="other", labels=(Label(name="X"),)), "ClassificationTask 2: labels not in label set 'other'"),
    ],
)
def test_the_writer_rejects_a_label_set_changed_after_add_task(tmp_path, changed, message):
    dataset, record = _dataset()
    second = ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES)
    dataset.add_tasks(tasks=[ClassificationTask(inputs=(record,), targets=("W",), label_set=STAGES), second])
    second.label_set = changed

    with pytest.raises(TimeFValidationError, match=message):
        _write(dataset, tmp_path / "control.duckdb")


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
def test_add_task_rejects_targets_that_do_not_fit_the_label_set(task, message):
    dataset, record = _dataset()

    with pytest.raises(TimeFValidationError, match=message):
        dataset.add_task(task=replace(task, inputs=(record,)))
    assert dataset.tasks == ()


def test_add_task_checks_target_annotations_against_the_label_set():
    dataset, record = _dataset()
    awake = record.annotate(Annotation(key="stage", value="W"))
    unknown = record.annotate(Annotation(key="stage", value="X"))
    dataset.add_task(task=ClassificationTask(inputs=(record,), target_annotations=(awake,), label_set=STAGES))

    with pytest.raises(TimeFValidationError, match="labels not in label set 'stage': \\['X'\\]"):
        dataset.add_task(task=ClassificationTask(inputs=(record,), target_annotations=(unknown,), label_set=STAGES))


def test_a_stream_rejects_targets_that_do_not_fit_the_label_set():
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
        ("UPDATE label_sets SET labels = [] WHERE name = 'stage'", "label_sets.labels"),
        ("UPDATE label_sets SET labels = [{'name': 'W'}, {'name': 'W'}] WHERE name = 'stage'", "label_sets.labels"),
        ("UPDATE label_sets SET labels = [{'name': 'W'}, {'name': ''}] WHERE name = 'stage'", "label_sets.labels"),
        ("UPDATE label_sets SET labels = [{'name': 'W', 'description': ''}] WHERE name = 'stage'", "label_sets.labels"),
        ("UPDATE tasks SET label_set = 'stage' WHERE task_type = 'answer'", "tasks.label_set"),
        ("UPDATE tasks SET label_set = 'missing' WHERE task_id = 1", "label_sets.name"),
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


def test_the_reader_rejects_a_label_set_on_another_task_type(release_copy):
    with connect_control(release_copy) as connection:
        connection.execute("UPDATE tasks SET label_set = 'stage' WHERE task_type = 'answer'")

    with pytest.raises(TimeFFormatError, match="answer task 2 has a label set"):
        _read_tasks(release_copy)
