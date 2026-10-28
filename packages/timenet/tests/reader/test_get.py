from pathlib import Path
from typing import NamedTuple

import numpy as np
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
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


_PARENT = DatasetRef(dataset_id="test/parent", version=Version(1, 0, 0))
_BASE = ParentAlias("base")
_MISSING = (98, 99)
"""IDs that name no object of any kind in the release."""
_PARENT_IDS = {"record": 101, "signal": 201, "task": 301}
"""The IDs the parent gives the objects that ``child`` imports."""


class _Objects(NamedTuple):
    """The objects of ``make_dataset()`` that the tests read back, with the IDs the dataset gave them."""

    record0: Record
    record1: Record
    record2: Record
    shared: Signal
    """The first Signal of ``record0``."""
    cosine: Signal
    """The second Signal of ``record0``."""
    long: Signal
    """The long Signal of ``record1``."""
    classification: Task
    answer: Task
    """The answer task, which derives from ``classification``."""


def _write(root: Path, dataset: TimeFDataset | None = None) -> Path:
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.fixture
def objects() -> _Objects:
    # make_dataset() numbers every ID the same way on every call, so these IDs match the release.
    dataset = make_dataset()
    record0, record1, record2 = dataset.records
    shared, cosine = record0.signals
    classification, answer = dataset.tasks[:2]
    assert isinstance(classification, ClassificationTask)
    assert isinstance(answer, AnswerTask)
    return _Objects(record0, record1, record2, shared, cosine, record1.signals[1], classification, answer)


@pytest.fixture
def release(tmp_path) -> Path:
    return _write(tmp_path)


@pytest.fixture
def reader(release):
    with TimeFReader(open_local_version(release)) as reader:
        yield reader


def test_get_reads_each_kind_by_its_id(reader, objects):
    record = reader.get(Record, objects.record1.id)
    assert isinstance(record, Record)
    assert record.id == objects.record1.id
    assert [signal.id for signal in record.signals] == [signal.id for signal in objects.record1.signals]

    signal = reader.get(Signal, objects.long.id)
    assert isinstance(signal, Signal)
    assert signal.to_arrow().equals(objects.long.to_arrow())
    assert signal.read_steps(3, 10).equals(objects.long.to_arrow().slice(3, 7))

    task = reader.get(Task, objects.answer.id)
    assert isinstance(task, AnswerTask)
    assert task.prompt == "What rhythm?"
    assert task.inputs[0].id == objects.record0.id
    assert [parent.id for parent in task.from_tasks] == [objects.classification.id]
    assert reader.get(AnswerTask, objects.answer.id).id == objects.answer.id


def test_get_many_returns_one_object_per_id_in_order_and_one_graph(reader, objects, monkeypatch):
    order = (objects.record2.id, objects.record0.id, objects.record2.id)
    records = reader.get_many(Record, order)
    assert tuple(record.id for record in records) == order
    assert records[0] is records[2]

    answer, classification = reader.get_many(Task, iter([objects.answer.id, objects.classification.id]))
    assert answer.from_tasks[0] is classification
    assert answer.inputs[0] is classification.inputs[0]

    built = []
    original = DuckDBControlReader.read_records

    def counting(self, record_ids=None, **kwargs):
        built.append(tuple(record_ids or ()))
        return original(self, record_ids, **kwargs)

    monkeypatch.setattr(DuckDBControlReader, "read_records", counting)
    shared, cosine, again = reader.get_many(Signal, (objects.shared.id, objects.cosine.id, objects.shared.id))
    assert (shared.id, cosine.id) == (objects.shared.id, objects.cosine.id)
    assert shared is again
    # Both Signals belong to record0, which the call builds once.
    assert built == [(objects.record0.id,)]
    assert reader.get_many(Record, ()) == ()


def test_get_with_the_wrong_kind_raises_before_anything_loads(reader, objects, monkeypatch):
    loads = []
    monkeypatch.setattr(TimeFReader, "_read_array", lambda *args: loads.append(args))
    with pytest.raises(TimeFNotFoundError, match="the stored object is a ClassificationTask"):
        reader.get(AnswerTask, objects.classification.id, load=LoadPlan.ALL_EAGER)
    assert not loads
    monkeypatch.undo()
    assert isinstance(reader.get(ClassificationTask, objects.classification.id), ClassificationTask)
    # IDs are numbered per kind, so a Signal ID beyond the last Record ID names no Record.
    assert objects.long.id > objects.record2.id
    with pytest.raises(TimeFNotFoundError, match=str(objects.long.id)):
        reader.get(Record, objects.long.id)
    with pytest.raises(TimeFNotFoundError, match="no Signal 99"):
        reader.get(Signal, 99)
    for kind in (ObjectKind.TASK, str, TimeFDataset, "Task"):
        with pytest.raises(TimeFValidationError, match="get reads a Record, Signal, or task class") as raised:
            reader.get(kind, objects.classification.id)
        assert not isinstance(raised.value, TimeFNotFoundError)


def test_get_refuses_sources_and_annotations_even_by_a_stored_id(reader, release):
    # Sources and annotation occurrences have storage ids but no public ones; both rows exist.
    with connect_control(release / "control.duckdb", read_only=True) as connection:
        ((source_id,),) = connection.execute("SELECT min(source_id) FROM sources").fetchall()
        ((occurrence_id,),) = connection.execute("SELECT min(occurrence_id) FROM annotation_occurrences").fetchall()
    for kind, identity in ((Source, source_id), (Annotation, occurrence_id)):
        with pytest.raises(TimeFValidationError, match="read through the objects that hold them") as raised:
            reader.get(kind, identity)
        assert not isinstance(raised.value, TimeFNotFoundError)
        with pytest.raises(TimeFValidationError, match="get reads a Record, Signal, or task class"):
            reader.get_many(kind, [identity])
        # The kind is refused before the selector is resolved, so an unknown alias isn't reached.
        with pytest.raises(TimeFValidationError, match="read through the objects that hold them") as raised:
            reader.get(kind, identity, parent=ParentAlias("other"))
        assert not isinstance(raised.value, TimeFNotFoundError)


@pytest.mark.parametrize("kind", [Record, Signal, Task])
def test_a_missing_id_raises_before_anything_is_built_or_loaded(reader, objects, monkeypatch, kind):
    present = {Record: objects.record0, Signal: objects.shared, Task: objects.classification}[kind].id
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
    spy(control_reader_module._TaskBuilder, "_build")
    spy(TimeFReader, "_read_array")
    with pytest.raises(TimeFNotFoundError, match=r"98.*99") as raised:
        reader.get_many(kind, [present, *_MISSING], load=LoadPlan.ALL_EAGER)
    assert isinstance(raised.value, LookupError)
    assert not calls


def test_get_takes_only_positive_int64_ids(reader, objects):
    with pytest.raises(TimeFValidationError, match="not the string 'record-0'"):
        reader.get_many(Record, "record-0")
    with pytest.raises(TimeFValidationError, match="collection of IDs"):
        reader.get_many(Record, 0)
    # A string is no ID, even one that spells a stored ID; nor are a bool, zero, or None.
    for identity in (str(objects.record0.id), True, 0, -1, 2**63, None, 1.0):
        with pytest.raises(TimeFValidationError, match="Record id must be a positive int64 id"):
            reader.get(Record, identity)
        with pytest.raises(TimeFValidationError, match="Record id must be a positive int64 id"):
            reader.get_many(Record, [objects.record0.id, identity])
    # An integral ID from NumPy or Arrow reads as a plain int.
    record = reader.get(Record, np.int64(objects.record0.id))
    assert type(record.id) is int
    assert record.id == objects.record0.id


def _with_a_signal_target_outside_the_inputs() -> tuple[TimeFDataset, Task, Signal]:
    """Return ``make_dataset()`` with an editing task whose input is record0 and whose target is record1's long Signal."""
    dataset = make_dataset()
    record0, record1, _ = dataset.records
    target = record1.signals[1]
    task = dataset.add_task(task=TSEditingTask(inputs=(record0,), targets=(target,)))
    return dataset, task, target


def _only_target(task: Task) -> Signal:
    assert task.targets is not None
    (signal,) = task.targets
    assert isinstance(signal, Signal)
    return signal


@pytest.mark.parametrize("read", ["get", "get_many"])
def test_a_task_read_by_id_builds_the_record_of_a_signal_target_outside_its_inputs(tmp_path, read):
    dataset, expected, target = _with_a_signal_target_outside_the_inputs()
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        task = reader.get(Task, expected.id) if read == "get" else reader.get_many(TSEditingTask, [expected.id])[0]
        signal = _only_target(task)
        assert signal.id == target.id
        assert signal.to_arrow().equals(target.to_arrow())


@pytest.mark.parametrize("given", [True, False], ids=["given-records", "every-task"])
@pytest.mark.parametrize("read", ["read_tasks", "iter_tasks"])
def test_a_task_read_builds_the_record_of_a_signal_target_outside_its_inputs(tmp_path, read, given):
    dataset, expected, target = _with_a_signal_target_outside_the_inputs()
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        inputs = tuple(reader.iter_records((dataset.records[0].id,))) if given else None
        (task,) = (task for task in getattr(reader, read)(inputs=inputs) if task.id == expected.id)
        signal = _only_target(task)
        assert signal.id == target.id
        assert signal.to_arrow().equals(target.to_arrow())
        # A given Record stays the input; the target's Record is built beside it.
        assert inputs is None or task.inputs[0] is inputs[0]


def _with_annotations_outside_the_inputs(owner: str) -> tuple[TimeFDataset, Task]:
    """Return ``make_dataset()`` with an answer task whose input is record0 and whose annotations lie on record1.

    It references an input annotation and has an answer annotation, both on record1 itself, on its
    Source, or on its long Signal.
    """
    dataset = make_dataset()
    record0, record1, _ = dataset.records
    (source,) = record1.sources
    annotated = {"record": record1, "source": source, "signal": record1.signals[1]}[owner]
    context = annotated.annotate(Annotation(key="context", value=owner))
    answer = annotated.annotate(Annotation(key="answer", value="yes"))
    task = dataset.add_task(
        task=AnswerTask(inputs=(record0,), input_annotations=(context,), target_annotations=(answer,))
    )
    return dataset, task


_READS = {
    "read": lambda reader, _inputs, _task_id: reader.read().tasks,
    "get": lambda reader, _inputs, task_id: (reader.get(Task, task_id),),
    "get_many": lambda reader, _inputs, task_id: reader.get_many(AnswerTask, [task_id]),
    "read_tasks": lambda reader, inputs, _task_id: reader.read_tasks(inputs=inputs),
    "iter_tasks": lambda reader, inputs, _task_id: tuple(reader.iter_tasks(inputs=inputs)),
}


@pytest.mark.parametrize("owner", ["record", "source", "signal"])
@pytest.mark.parametrize("given", [True, False], ids=["given-records", "no-records"])
@pytest.mark.parametrize("read", list(_READS))
def test_a_task_read_builds_the_records_of_the_annotations_it_references(tmp_path, read, given, owner):
    dataset, expected = _with_annotations_outside_the_inputs(owner)
    record0 = dataset.records[0]
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        inputs = tuple(reader.iter_records((record0.id,))) if given else None
        (task,) = (task for task in _READS[read](reader, inputs, expected.id) if task.id == expected.id)
        assert task.input_annotations == expected.input_annotations
        assert task.target_annotations == expected.target_annotations
        assert [record.id for record in task.inputs] == [record0.id]
        assert inputs is None or read not in {"read_tasks", "iter_tasks"} or task.inputs[0] is inputs[0]


def test_a_task_read_looks_up_the_records_behind_its_references_once_and_only_when_unbuilt(tmp_path, monkeypatch):
    dataset, answer = _with_annotations_outside_the_inputs("signal")
    record0, record1, _ = dataset.records
    edit = dataset.add_task(task=TSEditingTask(inputs=(record0,), targets=(record1.signals[1],)))
    lookups: list[str] = []
    for name in ("_record_ids_of_signals", "_record_ids_of_occurrences"):
        original = getattr(DuckDBControlReader, name)

        def spy(self, ids, _name=name, _original=original):
            lookups.append(_name)
            return _original(self, ids)

        monkeypatch.setattr(DuckDBControlReader, name, spy)
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        # A read that builds every Record looks nothing up, nor does one given the Records.
        reader.read()
        reader.read_tasks(inputs=tuple(reader.iter_records((record0.id, record1.id))))
        assert lookups == []
        # The answer's two annotations share one lookup.
        reader.get(Task, answer.id)
        assert lookups == ["_record_ids_of_occurrences"]
        # The Signal target's Record holds the answer's annotations, so the lookup by Signal finds both.
        lookups.clear()
        reader.get_many(Task, [answer.id, edit.id])
        assert lookups == ["_record_ids_of_signals"]


def test_read_tasks_and_iter_tasks_take_inputs_by_keyword_only(reader, objects):
    records = tuple(reader.iter_records((objects.record0.id,)))
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
def child(release, objects) -> Path:
    """Declare ``release`` a child of ``test/parent@1.0.0`` under the alias ``base``.

    The catalog then maps parent IDs to local objects: 101 to ``record0``, 201 to its first Signal,
    and 301 to the answer task.
    """
    with connect_control(release / "control.duckdb") as connection:
        connection.execute("INSERT INTO imports VALUES (?, ?)", [_BASE, str(_PARENT)])
        connection.executemany(
            "INSERT INTO imported_objects (alias, object_kind, parent_id, object_id) VALUES (?, ?, ?, ?)",
            [
                [_BASE, ObjectKind.RECORD.value, _PARENT_IDS["record"], objects.record0.id],
                [_BASE, ObjectKind.SIGNAL.value, _PARENT_IDS["signal"], objects.shared.id],
                [_BASE, ObjectKind.TASK.value, _PARENT_IDS["task"], objects.answer.id],
            ],
        )
    metadata = make_dataset().metadata.model_copy(update={"dataset_id": _PARENT.dataset_id})
    _declare_control_file(release, parent_cards={_BASE: ParentDatasetCard(metadata=metadata)})
    return release


def test_get_reads_an_imported_object_by_its_parent_id(child, objects):
    with TimeFReader(open_local_version(child)) as reader:
        assert reader.get(Record, _PARENT_IDS["record"], parent=_BASE).id == objects.record0.id
        assert reader.get(Record, _PARENT_IDS["record"], parent=_PARENT).id == objects.record0.id
        (signal,) = reader.get_many(Signal, [_PARENT_IDS["signal"]], parent=_BASE)
        assert signal.id == objects.shared.id
        task, again = reader.get_many(AnswerTask, [_PARENT_IDS["task"], _PARENT_IDS["task"]], parent=_PARENT)
        assert task.id == objects.answer.id
        assert task is again


def test_an_unknown_alias_parent_or_parent_id_raises_not_found(child, objects):
    with TimeFReader(open_local_version(child)) as reader:
        with pytest.raises(TimeFNotFoundError, match="declares no parent alias 'other'"):
            reader.get(Record, _PARENT_IDS["record"], parent=ParentAlias("other"))
        later = _PARENT.model_copy(update={"version": Version(2, 0, 0)})
        with pytest.raises(TimeFNotFoundError, match=r"declares no parent test/parent@2\.0\.0"):
            reader.get(Record, _PARENT_IDS["record"], parent=later)
        # Under an alias, IDs are the parent's: a local ID, or a parent ID of another kind, is absent.
        for kind, identity in ((Record, objects.record0.id), (Task, _PARENT_IDS["record"]), (Record, 999)):
            with pytest.raises(TimeFNotFoundError, match=f"no {kind.__name__} {identity} imported under"):
                reader.get(kind, identity, parent=_BASE)


@pytest.mark.parametrize("parent", ["base", str(_PARENT), 7], ids=["alias-name", "release-name", "int"])
def test_a_parent_that_is_neither_an_alias_nor_a_release_raises_with_a_hint(child, parent):
    # A plain string could name an alias or a release, so the reader takes neither.
    hint = r"wrap an alias name as ParentAlias\(\.\.\.\) or an exact release as DatasetRef\.parse\(\.\.\.\)"
    with TimeFReader(open_local_version(child)) as reader:
        with pytest.raises(TimeFValidationError, match=hint) as raised:
            reader.get(Record, _PARENT_IDS["record"], parent=parent)
        assert not isinstance(raised.value, TimeFNotFoundError)
        with pytest.raises(TimeFValidationError, match=hint):
            reader.get_many(Record, [_PARENT_IDS["record"]], parent=parent)


def test_a_release_that_several_aliases_import_needs_an_alias(child, objects):
    card = Manifest.model_validate_json((child / "manifest.json").read_text()).parent_cards[_BASE]
    _declare_control_file(child, parent_cards={_BASE: card, ParentAlias("again"): card})
    with TimeFReader(open_local_version(child)) as reader:
        with pytest.raises(
            TimeFValidationError, match=r"under the parent aliases \['again', 'base'\]; pass parent=ParentAlias"
        ) as raised:
            reader.get(Record, _PARENT_IDS["record"], parent=_PARENT)
        assert not isinstance(raised.value, TimeFNotFoundError)
        assert reader.get(Record, _PARENT_IDS["record"], parent=_BASE).id == objects.record0.id


def test_a_read_restores_the_import_map_the_parent_cards_and_the_lineage(child, objects):
    card = Manifest.model_validate_json((child / "manifest.json").read_text()).parent_cards[_BASE]
    _declare_control_file(child, lineage=(card,))
    with TimeFReader(open_local_version(child)) as reader:
        dataset = reader.read()
    assert dataset._imports == {
        (_BASE, ObjectKind.RECORD, _PARENT_IDS["record"]): objects.record0.id,
        (_BASE, ObjectKind.SIGNAL, _PARENT_IDS["signal"]): objects.shared.id,
        (_BASE, ObjectKind.TASK, _PARENT_IDS["task"]): objects.answer.id,
    }
    assert all(
        type(parent_id) is int and type(child_id) is int for (_, _, parent_id), child_id in dataset._imports.items()
    )
    assert dataset.parent_cards == {_BASE: card}
    assert dataset.lineage == {"test/parent@1.0.0": card}


@pytest.mark.parametrize(
    "read",
    [lambda reader: reader.get(Record, _PARENT_IDS["record"], parent=_BASE), TimeFReader.read],
    ids=["get", "read"],
)
def test_a_catalog_that_names_another_parent_than_its_manifest_raises_a_format_error(child, read):
    with connect_control(child / "control.duckdb") as connection:
        connection.execute("UPDATE imports SET parent_ref = 'test/parent@1.0.1'")
    _declare_control_file(child)
    with (
        TimeFReader(open_local_version(child)) as reader,
        pytest.raises(TimeFFormatError, match=r"differs? from its manifest"),
    ):
        read(reader)


def test_a_catalog_without_a_parent_alias_of_its_manifest_raises_a_format_error(child, objects):
    card = Manifest.model_validate_json((child / "manifest.json").read_text()).parent_cards[_BASE]
    _declare_control_file(child, parent_cards={_BASE: card, ParentAlias("again"): card})
    with TimeFReader(open_local_version(child)) as reader:
        with pytest.raises(TimeFFormatError, match=r"differ from its manifest's parent cards"):
            reader.read()
        with pytest.raises(TimeFFormatError, match="names no parent release for parent alias 'again'"):
            reader.get(Record, _PARENT_IDS["record"], parent=ParentAlias("again"))
        assert reader.get(Record, _PARENT_IDS["record"], parent=_BASE).id == objects.record0.id


def test_a_read_of_a_catalog_with_a_corrupt_import_row_raises_a_format_error(child):
    with connect_control(child / "control.duckdb") as connection:
        connection.execute(
            "UPDATE imported_objects SET object_kind = 'Bogus' WHERE parent_id = ?", [_PARENT_IDS["record"]]
        )
    _declare_control_file(child)
    with (
        TimeFReader(open_local_version(child)) as reader,
        pytest.raises(TimeFFormatError, match="corrupt or inconsistent TimeF artifact"),
    ):
        reader.read()


def test_a_catalog_that_lists_a_parent_alias_twice_raises_a_format_error(child):
    # The schema does not enforce the unique keys, so a corrupt catalog can hold two rows for one alias.
    with connect_control(child / "control.duckdb") as connection:
        connection.execute("INSERT INTO imports VALUES (?, ?)", [_BASE, "test/parent@1.0.1"])
    _declare_control_file(child)
    with (
        TimeFReader(open_local_version(child)) as reader,
        pytest.raises(TimeFFormatError, match="the catalog lists parent alias 'base' more than once"),
    ):
        reader.get(Record, _PARENT_IDS["record"], parent=_BASE)


def test_a_catalog_that_lists_a_parent_id_twice_raises_a_format_error(child, objects):
    with connect_control(child / "control.duckdb") as connection:
        connection.execute(
            "INSERT INTO imported_objects (alias, object_kind, parent_id, object_id) VALUES (?, ?, ?, ?)",
            [_BASE, ObjectKind.RECORD.value, _PARENT_IDS["record"], objects.record1.id],
        )
    _declare_control_file(child)
    with TimeFReader(open_local_version(child)) as reader:
        with pytest.raises(
            TimeFFormatError, match=r"the catalog lists Record 101 more than once under parent alias 'base'"
        ):
            reader.get_many(Record, [_PARENT_IDS["record"]], parent=_BASE)
        # The other imported objects keep one row each, so they still read.
        assert reader.get(Signal, _PARENT_IDS["signal"], parent=_BASE).id == objects.shared.id
