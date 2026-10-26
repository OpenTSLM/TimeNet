from pathlib import Path

import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFNotFoundError, TimeFValidationError
from timenet.format.checksums import file_checksum
import timenet.format.control_reader as control_reader_module
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
from timenet.manifest import Manifest
from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    InputModality,
    ObjectKind,
    Split,
    Task,
    TSEditingTask,
    Version,
)
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import InputAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


_PARENT = DatasetRef(dataset_id="test/parent", version=Version(1, 0, 0))
_BASE = InputAlias("base")


def _write(root: Path, dataset: TimeFDataset | None = None) -> Path:
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.fixture
def release(tmp_path) -> Path:
    return _write(tmp_path)


@pytest.fixture
def reader(release):
    with TimeFReader(open_local_version(release)) as reader:
        yield reader


def _occurrence_id(reader: TimeFReader, record_id: str, content_id: str) -> str:
    (record,) = reader.iter_records((record_id,))
    occurrence_id = next(annotation for annotation in record.annotations if annotation.id == content_id).occurrence_id
    assert occurrence_id is not None
    return occurrence_id


def test_get_reads_each_kind_by_its_id(reader):
    original = make_dataset()
    record = reader.get(Record, "record-1")
    assert isinstance(record, Record)
    assert record.id == "record-1"
    assert [signal.id for signal in record.signals] == [signal.id for signal in original.records[1].signals]

    signal = reader.get(Signal, "ts-long-1")
    assert isinstance(signal, Signal)
    expected = next(signal for signal in original.records[1].signals if signal.id == "ts-long-1")
    assert signal.to_arrow().equals(expected.to_arrow())
    assert signal.read_steps(3, 10).equals(expected.to_arrow().slice(3, 7))

    task = reader.get(Task, "task-answer-0")
    assert isinstance(task, AnswerTask)
    assert task.prompt == "What rhythm?"
    assert task.inputs[0].id == "record-0"
    assert [parent.id for parent in task.from_tasks] == ["task-cls-0"]
    assert reader.get(AnswerTask, "task-answer-0").id == "task-answer-0"


def test_get_many_returns_one_object_per_id_in_order_and_one_graph(reader, monkeypatch):
    records = reader.get_many(Record, ["record-2", "record-0", "record-2"])
    assert [record.id for record in records] == ["record-2", "record-0", "record-2"]
    assert records[0] is records[2]

    answer, classification = reader.get_many(Task, iter(["task-answer-0", "task-cls-0"]))
    assert answer.from_tasks[0] is classification
    assert answer.inputs[0] is classification.inputs[0]

    built = []
    original = DuckDBControlReader.read_records

    def counting(self, record_ids=None, **kwargs):
        built.append(tuple(record_ids or ()))
        return original(self, record_ids, **kwargs)

    monkeypatch.setattr(DuckDBControlReader, "read_records", counting)
    shared, cosine, again = reader.get_many(Signal, ("ts-shared", "ts-cos-0", "ts-shared"))
    assert (shared.id, cosine.id) == ("ts-shared", "ts-cos-0")
    assert shared is again
    # Both Signals belong to record-0, which the call builds once.
    assert built == [("record-0",)]
    assert reader.get_many(Record, ()) == ()


def test_get_with_the_wrong_kind_raises_before_anything_loads(reader, monkeypatch):
    loads = []
    monkeypatch.setattr(TimeFReader, "_read_array", lambda *args: loads.append(args))
    with pytest.raises(TimeFNotFoundError, match="the stored object is a ClassificationTask"):
        reader.get(AnswerTask, "task-cls-0", load=LoadPlan.ALL_EAGER)
    assert not loads
    monkeypatch.undo()
    assert isinstance(reader.get(ClassificationTask, "task-cls-0"), ClassificationTask)
    with pytest.raises(TimeFNotFoundError, match="ts-shared"):
        reader.get(Record, "ts-shared")
    with pytest.raises(TimeFNotFoundError, match="no Signal 'record-0'"):
        reader.get(Signal, "record-0")
    for kind in (ObjectKind.TASK, str, TimeFDataset, "Task"):
        with pytest.raises(TimeFValidationError, match="get reads a Record, Signal, or task class") as raised:
            reader.get(kind, "task-cls-0")
        assert not isinstance(raised.value, TimeFNotFoundError)


def test_get_refuses_sources_and_annotations_even_by_a_stored_id(reader):
    # Both IDs name stored objects, so only the kind is refused.
    occurrence_id = _occurrence_id(reader, "record-0", "age-0")
    for kind, identity in ((Source, "source-record-0"), (Annotation, occurrence_id)):
        with pytest.raises(TimeFValidationError, match="read through the objects that hold them") as raised:
            reader.get(kind, identity)
        assert not isinstance(raised.value, TimeFNotFoundError)
        with pytest.raises(TimeFValidationError, match="get reads a Record, Signal, or task class"):
            reader.get_many(kind, [identity])
        # The kind is refused before the selector is resolved, so an unknown alias isn't reached.
        with pytest.raises(TimeFValidationError, match="read through the objects that hold them") as raised:
            reader.get(kind, identity, alias=InputAlias("other"))
        assert not isinstance(raised.value, TimeFNotFoundError)


@pytest.mark.parametrize(("kind", "present"), [(Record, "record-0"), (Signal, "ts-shared"), (Task, "task-cls-0")])
def test_a_missing_id_raises_before_anything_is_built_or_loaded(reader, monkeypatch, kind, present):
    calls = []

    def spy(owner, name):
        original = getattr(owner, name)

        def recording(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, recording)

    # The steps that build signals, records, annotations, and tasks, and the one that loads arrays.
    for name in ("_prepare_signals", "_assemble_records", "_read_annotations"):
        spy(DuckDBControlReader, name)
    spy(control_reader_module._TaskHydration, "_build")
    spy(TimeFReader, "_read_array")
    with pytest.raises(TimeFNotFoundError, match=r"'missing-1'.*'missing-2'") as raised:
        reader.get_many(kind, [present, "missing-1", "missing-2"], load=LoadPlan.ALL_EAGER)
    assert isinstance(raised.value, LookupError)
    assert not calls


def test_get_many_rejects_a_single_string_and_ids_that_are_not_strings(reader):
    with pytest.raises(TimeFValidationError, match="not the string 'record-0'"):
        reader.get_many(Record, "record-0")
    with pytest.raises(TimeFValidationError, match="object IDs are strings"):
        reader.get_many(Record, [0])
    with pytest.raises(TimeFValidationError, match="collection of IDs"):
        reader.get_many(Record, 0)
    with pytest.raises(TimeFValidationError, match="object IDs are strings"):
        reader.get(Record, None)


def _with_a_signal_target_outside_the_inputs() -> tuple[TimeFDataset, Signal]:
    """Return ``make_dataset()`` with ``task-edit``, whose input is record-0 and whose target is a Signal of record-1."""
    dataset = make_dataset()
    record_0, record_1, _ = dataset.records
    target = next(signal for signal in record_1.signals if signal.id == "ts-long-1")
    dataset.add_task(task=TSEditingTask(id="task-edit", inputs=(record_0,), targets=(target,)))
    return dataset, target


def _only_target(task: Task) -> Signal:
    assert task.targets is not None
    (signal,) = task.targets
    assert isinstance(signal, Signal)
    return signal


@pytest.mark.parametrize("read", ["get", "get_many"])
def test_a_task_read_by_id_builds_the_record_of_a_signal_target_outside_its_inputs(tmp_path, read):
    dataset, target = _with_a_signal_target_outside_the_inputs()
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        task = reader.get(Task, "task-edit") if read == "get" else reader.get_many(TSEditingTask, ["task-edit"])[0]
        signal = _only_target(task)
        assert signal.id == "ts-long-1"
        assert signal.to_arrow().equals(target.to_arrow())


@pytest.mark.parametrize("given", [True, False], ids=["given-records", "every-task"])
@pytest.mark.parametrize("read", ["read_tasks", "iter_tasks"])
def test_a_task_read_builds_the_record_of_a_signal_target_outside_its_inputs(tmp_path, read, given):
    dataset, target = _with_a_signal_target_outside_the_inputs()
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        inputs = tuple(reader.iter_records(("record-0",))) if given else None
        (task,) = (task for task in getattr(reader, read)(inputs=inputs) if task.id == "task-edit")
        signal = _only_target(task)
        assert signal.id == "ts-long-1"
        assert signal.to_arrow().equals(target.to_arrow())
        # A given Record stays the input; the target's Record is built beside it.
        assert inputs is None or task.inputs[0] is inputs[0]


def _with_annotations_outside_the_inputs(owner: str) -> TimeFDataset:
    """Return ``make_dataset()`` with ``task-context``, whose input is record-0 and whose annotations lie on record-1.

    It references an input annotation and has an answer annotation, both on record-1 itself, on its
    Source, or on its Signal ``ts-long-1``.
    """
    dataset = make_dataset()
    record_0, record_1, _ = dataset.records
    (source,) = record_1.sources
    signal = next(signal for signal in record_1.signals if signal.id == "ts-long-1")
    annotated = {"record": record_1, "source": source, "signal": signal}[owner]
    context = annotated.annotate(Annotation(key="context", value=owner))
    answer = annotated.annotate(Annotation(key="answer", value="yes"))
    dataset.add_task(
        task=AnswerTask(
            id="task-context", inputs=(record_0,), input_annotations=(context,), target_annotations=(answer,)
        )
    )
    return dataset


_READS = {
    "read": lambda reader, _: reader.read().tasks,
    "get": lambda reader, _: (reader.get(Task, "task-context"),),
    "get_many": lambda reader, _: reader.get_many(AnswerTask, ["task-context"]),
    "read_tasks": lambda reader, inputs: reader.read_tasks(inputs=inputs),
    "iter_tasks": lambda reader, inputs: tuple(reader.iter_tasks(inputs=inputs)),
}


@pytest.mark.parametrize("owner", ["record", "source", "signal"])
@pytest.mark.parametrize("given", [True, False], ids=["given-records", "no-records"])
@pytest.mark.parametrize("read", list(_READS))
def test_a_task_read_builds_the_records_of_the_annotations_it_references(tmp_path, read, given, owner):
    dataset = _with_annotations_outside_the_inputs(owner)
    (expected,) = (task for task in dataset.tasks if task.id == "task-context")
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        inputs = tuple(reader.iter_records(("record-0",))) if given else None
        (task,) = (task for task in _READS[read](reader, inputs) if task.id == "task-context")

        def portrait(annotations):
            return [(annotation.key, annotation.value, annotation.occurrence_id) for annotation in annotations]

        assert portrait(task.input_annotations) == portrait(expected.input_annotations)
        assert portrait(task.target_annotations) == portrait(expected.target_annotations)
        assert [record.id for record in task.inputs] == ["record-0"]
        assert inputs is None or read not in {"read_tasks", "iter_tasks"} or task.inputs[0] is inputs[0]


def test_read_tasks_and_iter_tasks_take_inputs_by_keyword_only(reader):
    records = tuple(reader.iter_records(("record-0",)))
    with pytest.raises(TypeError):
        reader.read_tasks(records)
    with pytest.raises(TypeError):
        reader.iter_tasks(records)
    tasks = reader.read_tasks(inputs=records)
    assert tasks
    assert [task.id for task in reader.iter_tasks(inputs=records)] == [task.id for task in tasks]
    assert all(task.inputs[0] is records[0] for task in tasks)


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"split": Split.TEST},
        {"required_modalities": (InputModality.TEXT,)},
        {"supported_modalities": (InputModality.TIME_SERIES,)},
        {"split": "train", "required_modalities": (InputModality.TIME_SERIES,)},
    ],
)
def test_read_tasks_filters_like_iter_tasks_and_the_task_table(tmp_path, filters):
    dataset = make_dataset()
    for index, task in enumerate(sorted(dataset.tasks, key=lambda task: task.id)):
        task.split = (Split.TRAIN, Split.TEST)[index % 2]
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        tasks = [task.id for task in reader.read_tasks(**filters)]
        assert tasks == [task.id for task in reader.iter_tasks(**filters)]
        assert tasks == reader.read_task_table(**filters)["task_id"].to_pylist()
    assert tasks == sorted(tasks)


def _declare_control_file(version_dir: Path, **changes) -> None:
    """Rewrite the manifest with the control database's current size and checksum, and ``changes``."""
    manifest_path = version_dir / "manifest.json"
    manifest = Manifest.model_validate_json(manifest_path.read_text())
    control = version_dir / "control.duckdb"
    (part,) = manifest.files.control
    part = part.model_copy(update={"size": control.stat().st_size, "checksum": file_checksum(control)})
    files = manifest.files.model_copy(update={"control": (part,)})
    manifest_path.write_text(manifest.model_copy(update={"files": files, **changes}).to_json())


@pytest.fixture
def child(release) -> Path:
    """Declare ``release`` a child of ``test/parent@1.0.0`` under the alias ``base``.

    The catalog then maps parent IDs to local objects: ``p-record`` to ``record-0``, ``p-signal``
    to ``ts-shared``, and ``p-task`` to ``task-answer-0``.
    """
    with connect_control(release / "control.duckdb") as connection:
        connection.execute("INSERT INTO imports VALUES (?, ?)", [_BASE, str(_PARENT)])
        connection.executemany(
            "INSERT INTO imported_objects (alias, object_kind, parent_id, object_id) VALUES (?, ?, ?, ?)",
            [
                [_BASE, ObjectKind.RECORD.value, "p-record", "record-0"],
                [_BASE, ObjectKind.SIGNAL.value, "p-signal", "ts-shared"],
                [_BASE, ObjectKind.TASK.value, "p-task", "task-answer-0"],
            ],
        )
    metadata = make_dataset().metadata.model_copy(update={"dataset_id": _PARENT.dataset_id})
    _declare_control_file(release, parent_cards={_BASE: ParentDatasetCard(metadata=metadata)})
    return release


def test_get_reads_an_imported_object_by_its_parent_id(child):
    with TimeFReader(open_local_version(child)) as reader:
        assert reader.get(Record, "p-record", alias=_BASE).id == "record-0"
        assert reader.get(Record, "p-record", parent=_PARENT).id == "record-0"
        (signal,) = reader.get_many(Signal, ["p-signal"], alias=_BASE)
        assert signal.id == "ts-shared"
        task, again = reader.get_many(AnswerTask, ["p-task", "p-task"], parent=_PARENT)
        assert task.id == "task-answer-0"
        assert task is again


def test_an_unknown_alias_parent_or_parent_id_raises_not_found(child):
    with TimeFReader(open_local_version(child)) as reader:
        with pytest.raises(TimeFNotFoundError, match="declares no input alias 'other'"):
            reader.get(Record, "p-record", alias=InputAlias("other"))
        with pytest.raises(TimeFNotFoundError, match=r"declares no input from test/parent@2\.0\.0"):
            reader.get(Record, "p-record", parent=_PARENT.model_copy(update={"version": Version(2, 0, 0)}))
        # Under an alias, IDs are the parent's: a local ID, or a parent ID of another kind, is absent.
        for kind, identity in ((Record, "record-0"), (Task, "p-record"), (Record, "p-missing")):
            with pytest.raises(TimeFNotFoundError, match=f"no {kind.__name__} '{identity}' imported under"):
                reader.get(kind, identity, alias=_BASE)


def test_malformed_import_selectors_raise_validation_errors(child):
    with TimeFReader(open_local_version(child)) as reader:
        with pytest.raises(TimeFValidationError, match="not both") as raised:
            reader.get(Record, "p-record", alias=_BASE, parent=_PARENT)
        assert not isinstance(raised.value, TimeFNotFoundError)
        with pytest.raises(TimeFValidationError, match="nonblank string"):
            reader.get(Record, "p-record", alias=InputAlias(" "))
        with pytest.raises(TimeFValidationError, match="exact DatasetRef"):
            reader.get(Record, "p-record", parent=str(_PARENT))  # ty: ignore[invalid-argument-type]
    # Under two aliases, one parent release does not say which imported copy to read.
    card = Manifest.model_validate_json((child / "manifest.json").read_text()).parent_cards[_BASE]
    _declare_control_file(child, parent_cards={_BASE: card, InputAlias("again"): card})
    with TimeFReader(open_local_version(child)) as reader:
        with pytest.raises(TimeFValidationError, match=r"under the aliases \['again', 'base'\]") as raised:
            reader.get(Record, "p-record", parent=_PARENT)
        assert not isinstance(raised.value, TimeFNotFoundError)
        assert reader.get(Record, "p-record", alias=_BASE).id == "record-0"


def test_a_catalog_that_names_another_parent_than_its_manifest_raises_a_format_error(child):
    with connect_control(child / "control.duckdb") as connection:
        connection.execute("UPDATE imports SET parent_ref = 'test/parent@1.0.1'")
    _declare_control_file(child)
    with (
        TimeFReader(open_local_version(child)) as reader,
        pytest.raises(TimeFFormatError, match="differs from its manifest"),
    ):
        reader.get(Record, "p-record", alias=_BASE)
