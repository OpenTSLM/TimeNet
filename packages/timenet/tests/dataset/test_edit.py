"""Copy-on-write edits: remove_records and remove_signals validate-and-cascade, and the edit_version round trip."""

import re

import duckdb
import numpy as np
import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.dataset.edit import edit_version, remove_records, remove_signals
from timenet.errors import TimeFEditError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    License,
    StepInterval,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    TSCorrespondenceTask,
    TSEditingTask,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


def test_remove_records_preserves_dataset_annotations_and_their_refs():
    # A dataset annotation and a task reference to its occurrence survive an unrelated record removal.
    dataset = make_dataset()
    options = dataset.annotate(Annotation(key="answer_options", value=["yes", "no"], id="opts-shared"))
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    answer.input_annotations = (*answer.input_annotations, options)

    edited = remove_records(dataset, [dataset.records[2].id], cascade=True)

    assert [ann.id for ann in edited.annotations] == [options.id]
    kept = next(task for task in edited.tasks if task.id == answer.id)
    assert options in kept.input_annotations


def _base(tmp_path, dataset=None):
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "base", dataset) as writer:
        writer.write()
    meta = dataset.metadata
    return tmp_path / "base" / meta.dataset_id / str(meta.dataset_version)


def _answer_reads_context_via_additional_input():
    """Rewire the answer task to reach its context annotation only through an additional input record.

    Still a valid dataset — the annotation sits on a record the task is attached to — so it writes and
    reads back fine. Removing the additional input record is what strands the reference.
    """
    dataset = make_dataset()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (record,) = answer.inputs
    (shared_annotation,) = answer.input_annotations
    context_record = dataset.records[1]
    context_annotation = next(
        annotation for annotation in context_record.annotations if annotation.id == shared_annotation.id
    )
    answer.inputs = (record, context_record)
    context_record.task_ids = (*context_record.task_ids, answer.id)
    record.annotations = tuple(annotation for annotation in record.annotations if annotation.id != shared_annotation.id)
    answer.input_annotations = (context_annotation,)
    return dataset


# ---- in-memory transform ----------------------------------------------------------------------


def test_remove_leaf_record():
    dataset = make_dataset()
    dataset.derive_schema()
    removed_record = dataset.records[1]
    retained_records = (dataset.records[0], dataset.records[2])
    expected_task_ids = {task.id for task in dataset.tasks}
    assert not removed_record.task_ids, "the removed record must have no tasks"
    edited = remove_records(dataset, [removed_record.id])
    assert {record.id for record in edited.records} == {record.id for record in retained_records}
    assert {task.id for task in edited.tasks} == expected_task_ids


def test_the_sources_of_an_edit_are_copies_with_metadata_of_their_own():
    dataset = make_dataset()
    kept = dataset.records[0]
    (source,) = kept.sources
    source.metadata["calibration"] = {"gain": 1}
    edited = remove_records(dataset, [dataset.records[1].id])
    (copy,) = next(record for record in edited.records if record.id == kept.id).sources
    assert copy is not source
    assert copy == source
    calibration = copy.metadata["calibration"]
    assert isinstance(calibration, dict)
    calibration["gain"] = 2
    copy.metadata["checked"] = True
    assert source.metadata == {"calibration": {"gain": 1}}


def test_remove_unknown_record_raises():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="unknown record ids"):
        remove_records(dataset, ["nope"])


def test_remove_record_with_task_rejects_without_cascade():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="cascade=True"):
        remove_records(dataset, [dataset.records[0].id])


def test_remove_record_cascades_dependent_tasks():
    dataset = make_dataset()
    dataset.derive_schema()
    removed_record = dataset.records[0]
    retained_records = dataset.records[1:]
    independent_record = dataset.records[2]
    surviving_task = next(task for task in dataset.tasks if task.inputs == (independent_record,))
    edited = remove_records(dataset, [removed_record.id], cascade=True)
    assert {record.id for record in edited.records} == {record.id for record in retained_records}
    assert {task.id for task in edited.tasks} == {surviving_task.id}


def test_shared_annotation_survives_on_remaining_record():
    dataset = make_dataset()
    dataset.derive_schema()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (removed_record,) = answer.inputs
    (shared_annotation,) = answer.input_annotations
    retained_record = dataset.records[1]
    edited = remove_records(dataset, [removed_record.id], cascade=True)
    restored_record = next(record for record in edited.records if record.id == retained_record.id)
    assert any(annotation.id == shared_annotation.id for annotation in restored_record.annotations)


def test_input_annotation_refs_are_stripped_when_unreachable():
    # Dropping the additional input record strands the task's context annotation. It is not the answer,
    # so the task survives with the reference pruned instead of being rejected.
    dataset = _answer_reads_context_via_additional_input()
    dataset.derive_schema()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    kept_record, removed_record = answer.inputs
    edited = remove_records(dataset, [removed_record.id])
    rebuilt = next(task for task in edited.tasks if task.id == answer.id)
    assert rebuilt.inputs == (kept_record,)
    assert rebuilt.input_annotations == ()


def test_input_annotation_refs_survive_a_reachable_removal():
    # The context annotation stays on the task's input, so removing another record leaves it intact.
    dataset = make_dataset()
    dataset.derive_schema()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (context_annotation,) = answer.input_annotations
    edited = remove_records(dataset, [dataset.records[1].id])
    rebuilt = next(task for task in edited.tasks if task.id == answer.id)
    assert tuple(annotation.id for annotation in rebuilt.input_annotations) == (context_annotation.id,)


def test_target_annotation_refs_are_required():
    # The localization task stores its answer as an annotation occurrence on its input record.
    # Adding another input means dropping the original no longer costs it every record, but it does cost
    # it the answer, so the edit is rejected rather than silently rewriting the ground truth.
    dataset = make_dataset()
    localize = next(task for task in dataset.tasks if isinstance(task, TemporalLocalizationTask))
    (removed_record,) = localize.inputs
    retained_record = dataset.records[2]
    target_annotation = next(annotation for annotation in removed_record.annotations if annotation.span is not None)
    localize.inputs = (removed_record, retained_record)
    localize.targets = None
    localize.target_annotations = (target_annotation,)
    retained_record.task_ids = (*retained_record.task_ids, localize.id)
    dataset.derive_schema()

    with pytest.raises(TimeFEditError, match=re.escape(localize.id)):
        remove_records(dataset, [removed_record.id])

    edited = remove_records(dataset, [removed_record.id], cascade=True)
    assert localize.id not in {task.id for task in edited.tasks}


def test_edited_version_has_no_dangling_annotation_refs(tmp_path):
    # The end-to-end guarantee: whatever a committed edit contains, every annotation a task names is
    # resolvable from the records that task is attached to. Removing the context record previously
    # wrote the dangling reference straight into the new version.
    dataset = _answer_reads_context_via_additional_input()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    context_record = answer.inputs[1]
    base = _base(tmp_path, dataset)
    out = edit_version(base, dataset_version=Version(1, 0, 1), remove_record_ids=[context_record.id])
    with TimeFReader(open_local_version(out)) as reader:
        restored = reader.read()

    by_record = {
        record.record_id: {annotation.occurrence_id for annotation in record.annotations} for record in restored.records
    }
    for task in restored.tasks:
        reachable = set().union(*(by_record[record.id] for record in task.inputs))
        named = (*task.input_annotations, *task.target_annotations)
        assert not [annotation for annotation in named if annotation.occurrence_id not in reachable]


def test_target_and_candidate_records_are_required_whatever_the_task_type():
    dataset = make_dataset()
    edited_record = dataset.records[1]
    candidate_record = dataset.records[2]
    edit_task = dataset.add_task(
        task=TSEditingTask(
            inputs=(dataset.records[0],),
            prompt="Denoise it.",
            targets=(edited_record,),
            id="task-edit-0",
        )
    )
    correspondence_task = dataset.add_task(
        task=TSCorrespondenceTask(
            inputs=(dataset.records[0],),
            prompt="Which trace matches?",
            candidate_records=(edited_record, candidate_record),
            targets=(candidate_record,),
            id="task-corr-0",
        )
    )
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match=re.escape(edit_task.id)):
        remove_records(dataset, [edited_record.id])
    with pytest.raises(TimeFEditError, match=re.escape(correspondence_task.id)):
        remove_records(dataset, [candidate_record.id])


_REF_SPEC = TimeSeriesSpec(spec_type="sine", name="Sine", unit_value=ureg.dimensionless)


def _referenced_pair():
    """Return a dataset of the Records ``a`` and ``b``, each with one Signal (``sig-a``, ``sig-b``) and annotation."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/pair", dataset_version=Version(1, 0, 0), name="pair", description="d", license=License.MIT
        )
    )
    for name in ("a", "b"):
        signal = Signal.from_values(
            np.arange(16, dtype=np.float32),
            spec=_REF_SPEC,
            name=f"sig-{name}",
            time_axis=RegularAxis.from_rate_hz(16),
            id=f"sig-{name}",
        )
        record = dataset.add_record(record=Record(record_id=name, sources=(Source(name="m", signals=(signal,)),)))
        record.annotate(Annotation(key="note", value=name))
    return dataset


def _referencing_b(kind):
    """Return ``_referenced_pair()`` with a task whose input is ``a`` and which references ``b`` by ``kind``."""
    dataset = _referenced_pair()
    a, b = dataset.records
    tasks = {
        "candidate": lambda: TSCorrespondenceTask(inputs=(a,), candidate_records=(a, b), targets=(a,)),
        "record-target": lambda: TSEditingTask(inputs=(a,), targets=(b,)),
        "signal-target": lambda: TSEditingTask(inputs=(a,), targets=(b.signals[0],)),
        "target-annotation": lambda: AnswerTask(inputs=(a,), target_annotations=b.annotations),
        "span": lambda: AnswerTask(
            inputs=(a, b), targets=("x",), scope=TimeInterval.seconds(0.0, 0.5, time_series_ids=("sig-b",))
        ),
    }
    task = dataset.add_task(task=tasks[kind]())
    return dataset, task


@pytest.mark.parametrize("kind", ["candidate", "record-target", "signal-target", "target-annotation", "span"])
def test_removing_a_record_invalidates_each_kind_of_reference_to_it(kind):
    dataset, task = _referencing_b(kind)
    with pytest.raises(TimeFEditError, match=re.escape(f"would dangle tasks ['{task.id}']")):
        remove_records(dataset, ["b"])
    edited = remove_records(dataset, ["b"], cascade=True)
    assert (edited.tasks, [record.task_ids for record in edited.records]) == ((), [()])


def test_an_input_annotation_on_another_record_stays_while_that_record_does():
    dataset = _referenced_pair()
    a, b = dataset.records
    dataset.add_record(record=Record(record_id="c"))
    dataset.add_task(task=AnswerTask(inputs=(a,), targets=("x",), input_annotations=(*a.annotations, *b.annotations)))
    # b holds one of the task's annotations, so it is one of the task's Records: removing c keeps it.
    (task,) = remove_records(dataset, ["c"]).tasks
    assert [annotation.value for annotation in task.input_annotations] == ["a", "b"]
    # Removing b strips that context and keeps the task.
    edited = remove_records(dataset, ["b"])
    (task,) = edited.tasks
    assert task.inputs == edited.records[:1]
    assert [annotation.value for annotation in task.input_annotations] == ["a"]


# ---- copy-on-write version --------------------------------------------------------------------


def test_edit_version_round_trip(tmp_path):
    dataset = make_dataset()
    removed_record = dataset.records[1]
    retained_records = (dataset.records[0], dataset.records[2])
    base = _base(tmp_path, dataset)
    out = edit_version(base, dataset_version=Version(1, 0, 1), remove_record_ids=[removed_record.id])
    # The derived version is written next to its base, where readers find the base's payload files.
    assert out == base.parent / "1.0.1"

    with TimeFReader(open_local_version(out)) as reader:
        restored = reader.read()
    assert {record.id for record in restored.records} == {record.id for record in retained_records}

    manifest = Manifest.model_validate_json((out / "manifest.json").read_text())
    assert str(manifest.metadata.dataset_version) == "1.0.1"

    with duckdb.connect(str(out / "control.duckdb"), read_only=True) as connection:
        row = connection.execute("SELECT count(*) FROM records WHERE record_id = ?", [removed_record.id]).fetchone()
        assert row is not None
        assert row[0] == 0


def test_edit_version_same_version_rejected(tmp_path):
    dataset = make_dataset()
    base = _base(tmp_path, dataset)
    with pytest.raises(TimeFEditError, match="must differ"):
        edit_version(
            base,
            dataset_version=dataset.metadata.dataset_version,
            remove_record_ids=[dataset.records[1].id],
        )


# ---- removing Signals ---------------------------------------------------------------------------


_SPEC = TimeSeriesSpec(spec_type="sine", name="Sine", unit_value=ureg.dimensionless)


def _signal(signal_id, offset_s):
    """Return 16 values at 16 Hz, covering one second from ``offset_s``."""
    axis = RegularAxis.from_rate_hz(16, offset_us=offset_s * 1_000_000)
    return Signal.from_values(np.arange(16, dtype=np.float32), spec=_SPEC, name=signal_id, time_axis=axis, id=signal_id)


def _two_leads(time_span=None):
    """Return a local dataset whose Record ``rec`` holds ``sig-a`` (0 to 1 s) below ``monitor/leads``
    and ``sig-b`` (2 to 3 s) in ``monitor``, with annotations at 0.5 s and 2.5 s at every level, and
    one at 1.5 s on the Record."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/leads",
            dataset_version=Version(1, 0, 0),
            name="leads",
            description="d",
            license=License.MIT,
        )
    )
    signal_b = _signal("sig-b", 2)
    leads = Source(id="src-leads", name="leads", signals=(_signal("sig-a", 0),))
    monitor = Source(id="src-monitor", name="monitor", signals=(signal_b,), sources=(leads,))
    record = dataset.add_record(record=Record(record_id="rec", sources=(monitor,), time_span=time_span))
    record.add_annotations(
        [
            Annotation(key="age", value=64),
            Annotation(key="early", span=TimePoint.seconds(0.5)),
            Annotation(key="late", span=TimePoint.seconds(2.5)),
        ]
    )
    # No window places gap; the Record keeps it while it holds every Signal.
    record.annotate(Annotation(key="gap", span=TimePoint.seconds(1.5)))
    monitor.annotate(Annotation(key="src-early", span=TimePoint.seconds(0.5)))
    monitor.annotate(Annotation(key="src-late", span=TimePoint.seconds(2.5)))
    leads.annotate(Annotation(key="leads-early", span=TimePoint.seconds(0.5)))
    signal_b.annotate(Annotation(key="artifact", span=TimeInterval.seconds(2.0, 2.25)))
    return dataset


def _keys(annotated):
    return sorted(annotation.key for annotation in annotated.annotations)


def _annotation(annotated, key):
    return next(annotation for annotation in annotated.annotations if annotation.key == key)


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_drops_the_annotations_only_the_removed_signals_place():
    dataset = _two_leads()
    edited = remove_signals(dataset, ["sig-b"])
    (record,) = edited.records
    (monitor,) = record.sources
    (leads,) = monitor.sources
    assert [signal.id for signal in record.signals] == ["sig-a"]
    assert (_keys(record), _keys(monitor), _keys(leads)) == (["age", "early"], ["src-early"], ["leads-early"])
    # The original keeps its Signals and annotations.
    (original,) = dataset.records
    (original_monitor,) = original.sources
    assert sorted(signal.id for signal in original.signals) == ["sig-a", "sig-b"]
    assert (_keys(original), _keys(original_monitor)) == (["age", "early", "gap", "late"], ["src-early", "src-late"])
    # Without any Signal, the Sources stay, with the annotations that need no Signal.
    emptied = remove_signals(edited, ["sig-a"])
    (record,) = emptied.records
    assert record.signals == ()
    assert [(source.name, _keys(source)) for source in record.walk_sources()] == [("monitor", []), ("leads", [])]
    assert _keys(record) == ["age"]


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_keeps_what_lies_inside_a_records_session_span():
    edited = remove_signals(_two_leads(time_span=TimeInterval.seconds(0.0, 3.0)), ["sig-b"])
    (record,) = edited.records
    # The session span places late and gap; a Source has no session span.
    assert (_keys(record), _keys(record.sources[0])) == (["age", "early", "gap", "late"], ["src-early"])


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_rejects_or_cascades_the_tasks_that_reference_what_it_removes():
    dataset = _two_leads()
    (record,) = dataset.records
    signal_b = next(signal for signal in record.signals if signal.id == "sig-b")
    context = AnswerTask(
        id="context", inputs=(record,), targets=("x",), input_annotations=(_annotation(record, "late"),)
    )
    kept = AnswerTask(
        id="kept",
        inputs=(record,),
        targets=("x",),
        input_annotations=(_annotation(record, "early"),),
        scope=TimeInterval.seconds(0.0, 0.5, time_series_ids=("sig-a",)),
    )
    edit = TSEditingTask(id="edit", inputs=(record,), targets=(signal_b, "denoised"))
    dataset.add_tasks(
        tasks=[
            edit,
            AnswerTask(id="after-edit", inputs=(record,), targets=("z",), from_tasks=(edit,)),
            AnswerTask(
                id="scoped",
                inputs=(record,),
                targets=("x",),
                scope=TimeInterval.seconds(2.0, 2.5, time_series_ids=("sig-b",)),
            ),
            TemporalLocalizationTask(
                id="located", inputs=(record,), target_annotations=(_annotation(signal_b, "artifact"),)
            ),
            context,
            AnswerTask(id="derived", inputs=(record,), targets=("y",), from_tasks=(context,)),
            kept,
        ]
    )
    # The input annotation late is context, so only the other references invalidate a task.
    with pytest.raises(
        TimeFEditError,
        match=re.escape("removing signals ['sig-b'] would dangle tasks ['edit', 'located', 'scoped']"),
    ):
        remove_signals(dataset, ["sig-b"])
    edited = remove_signals(dataset, ["sig-b"], cascade=True)
    (record,) = edited.records
    context, derived, kept = edited.tasks
    assert [task.id for task in edited.tasks] == ["context", "derived", "kept"]
    assert record.task_ids == ("context", "derived", "kept")
    assert context.input_annotations == ()
    assert derived.from_tasks == (context,)
    assert (kept.inputs, kept.input_annotations) == ((record,), (_annotation(record, "early"),))
    assert kept.input_annotations[0] is _annotation(record, "early")
    assert len(dataset.tasks) == 7
    # Removing sig-a drops only the task scoped to it.
    others = remove_signals(dataset, ["sig-a"], cascade=True)
    assert [task.id for task in others.tasks] == ["edit", "after-edit", "scoped", "located", "context", "derived"]


def test_remove_signals_rejects_a_task_whose_step_span_names_a_removed_signal():
    dataset = _referenced_pair()
    ordinal = Signal.from_values(
        np.arange(8, dtype=np.float32), spec=_SPEC, name="sig-o", time_axis=OrdinalAxis(), id="sig-o"
    )
    # The Record keeps another Signal, so only the span names what goes.
    timed = Signal.from_values(
        np.arange(16, dtype=np.float32), spec=_SPEC, name="sig-t", time_axis=RegularAxis.from_rate_hz(16), id="sig-t"
    )
    record = dataset.add_record(record=Record(record_id="steps", sources=(Source(name="m", signals=(ordinal, timed)),)))
    dataset.add_task(
        task=AnswerTask(
            id="stepped", inputs=(record,), targets=("x",), scope=StepInterval(time_series_id="sig-o", start=0, stop=4)
        )
    )
    assert [task.id for task in remove_signals(dataset, ["sig-a"]).tasks] == ["stepped"]
    with pytest.raises(TimeFEditError, match=re.escape("would dangle tasks ['stepped']")):
        remove_signals(dataset, ["sig-o"])


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_invalidates_a_task_whose_input_keeps_no_signal():
    dataset = _two_leads()
    (record,) = dataset.records
    dataset.add_task(task=AnswerTask(id="t", inputs=(record,), targets=("x",)))
    assert [task.id for task in remove_signals(dataset, ["sig-a"]).tasks] == ["t"]
    with pytest.raises(TimeFEditError, match=re.escape("would dangle tasks ['t']")):
        remove_signals(dataset, ["sig-a", "sig-b"])
    edited = remove_signals(dataset, ["sig-a", "sig-b"], cascade=True)
    (record,) = edited.records
    assert (edited.tasks, record.signals, record.task_ids) == ((), (), ())


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
@pytest.mark.parametrize(
    ("span", "time_span", "placed"),
    [
        ((2.0, 2.5), None, False),
        ((0.0, 0.5), None, True),
        ((2.0, 2.5), TimeInterval.seconds(0.0, 3.0), True),
    ],
    ids=["outside-the-kept-window", "inside-the-kept-window", "inside-the-session"],
)
def test_remove_signals_places_a_span_without_signal_ids_as_it_places_an_annotation(span, time_span, placed):
    dataset = _two_leads(time_span=time_span)
    (record,) = dataset.records
    interval = TimeInterval.seconds(*span)
    dataset.add_tasks(
        tasks=[
            AnswerTask(id="scope", inputs=(record,), targets=("x",), scope=interval),
            TemporalLocalizationTask(id="target", inputs=(record,), targets=(interval,)),
        ]
    )
    if placed:
        assert [task.id for task in remove_signals(dataset, ["sig-b"]).tasks] == ["scope", "target"]
    else:
        with pytest.raises(TimeFEditError, match=re.escape("would dangle tasks ['scope', 'target']")):
            remove_signals(dataset, ["sig-b"])


@pytest.mark.parametrize(
    "edit",
    [lambda dataset: remove_records(dataset, "a"), lambda dataset: remove_signals(dataset, "sig-a")],
    ids=["remove_records", "remove_signals"],
)
def test_edits_reject_a_string_of_ids(edit):
    with pytest.raises(TimeFEditError, match="takes a collection of IDs, not the string"):
        edit(_referenced_pair())


def test_edit_version_removes_the_signals_first_and_forwards_cascade(tmp_path):
    dataset = _referenced_pair()
    a, b = dataset.records
    dataset.add_tasks(
        tasks=[
            TSEditingTask(id="on-signal", inputs=(b,), targets=(a.signals[0], "denoised")),
            TSEditingTask(id="on-record", inputs=(a,), targets=(b,)),
        ]
    )
    base = _base(tmp_path, dataset)
    # The Signals go first; sig-b, a Signal of the removed b, is left to b's removal.
    with pytest.raises(TimeFEditError, match=re.escape("removing signals ['sig-a'] would dangle tasks")):
        edit_version(
            base, dataset_version=Version(1, 0, 1), remove_record_ids=["b"], remove_signal_ids=["sig-a", "sig-b"]
        )
    with pytest.raises(TimeFEditError, match="takes a collection of IDs, not the string"):
        edit_version(base, dataset_version=Version(1, 0, 1), remove_signal_ids="sig-a")
    # A Signal of a removed record is left to that removal, so it names no unknown Signal.
    out = edit_version(
        base,
        dataset_version=Version(1, 0, 1),
        remove_record_ids=["b"],
        remove_signal_ids=["sig-a", "sig-b"],
        cascade=True,
    )
    with TimeFReader(open_local_version(out)) as reader:
        edited = reader.read()
    assert ([record.id for record in edited.records], edited.tasks) == (["a"], ())
    assert edited.records[0].signals == ()


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_rejects_an_unknown_id():
    with pytest.raises(TimeFEditError, match=re.escape("unknown signal ids: ['nope']")):
        remove_signals(_two_leads(), ["nope"])


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_edit_version_removes_signals(tmp_path):
    base = _base(tmp_path, _two_leads())
    out = edit_version(base, dataset_version=Version(1, 0, 1), remove_signal_ids=["sig-b"])
    with TimeFReader(open_local_version(out)) as reader:
        (record,) = reader.read().records
    assert [signal.id for signal in record.signals] == ["sig-a"]
    assert (_keys(record), _keys(record.sources[0])) == (["age", "early"], ["src-early"])
