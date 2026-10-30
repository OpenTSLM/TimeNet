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
    options = dataset.annotate(Annotation(key="answer_options", value=["yes", "no"]))
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    answer.input_annotations = (*answer.input_annotations, options)

    edited = remove_records(dataset, [dataset.records[2].id], cascade=True)

    assert list(edited.annotations) == [options]
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
        annotation for annotation in context_record.annotations if annotation == shared_annotation
    )
    answer.inputs = (record, context_record)
    record.annotations = tuple(annotation for annotation in record.annotations if annotation != shared_annotation)
    answer.input_annotations = (context_annotation,)
    return dataset


# ---- in-memory transform ----------------------------------------------------------------------


def test_remove_leaf_record():
    dataset = make_dataset()
    dataset.derive_schema()
    removed_record = dataset.records[1]
    retained_records = (dataset.records[0], dataset.records[2])
    expected_task_ids = {task.id for task in dataset.tasks}
    assert not dataset.tasks_for(removed_record), "the removed record must have no tasks"
    edited = remove_records(dataset, [removed_record.id])
    assert {record.id for record in edited.records} == {record.id for record in retained_records}
    assert {task.id for task in edited.tasks} == expected_task_ids


def test_the_records_and_sources_of_an_edit_are_copies_with_metadata_of_their_own():
    dataset = make_dataset()
    kept = dataset.records[0]
    (source,) = kept.sources
    kept.metadata["site"] = {"name": "north"}
    source.metadata["calibration"] = {"gain": 1}
    edited = remove_records(dataset, [dataset.records[1].id])
    record = next(record for record in edited.records if record.id == kept.id)
    (copy,) = record.sources
    assert record is not kept
    assert copy is not source
    assert (record.metadata, record.annotations, copy.metadata, copy.annotations) == (
        kept.metadata,
        kept.annotations,
        source.metadata,
        source.annotations,
    )
    # The Signals are copies too, so the copies hold the same Signal IDs.
    assert [signal.id for signal in copy.signals] == [signal.id for signal in source.signals]
    calibration, site = copy.metadata["calibration"], record.metadata["site"]
    assert isinstance(calibration, dict)
    assert isinstance(site, dict)
    calibration["gain"] = 2
    site["name"] = "south"
    copy.metadata["checked"] = True
    assert (kept.metadata, source.metadata) == ({"site": {"name": "north"}}, {"calibration": {"gain": 1}})


def test_remove_unknown_record_raises():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="unknown record ids"):
        remove_records(dataset, [999])


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
    assert shared_annotation in restored_record.annotations


def test_input_annotation_refs_are_stripped_when_unreachable():
    # Dropping the additional input record strands the task's context annotation. It is not the answer,
    # so the task survives with the reference pruned instead of being rejected.
    dataset = _answer_reads_context_via_additional_input()
    dataset.derive_schema()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    kept_record, removed_record = answer.inputs
    edited = remove_records(dataset, [removed_record.id])
    rebuilt = next(task for task in edited.tasks if task.id == answer.id)
    assert [record.id for record in rebuilt.inputs] == [kept_record.id]
    assert rebuilt.input_annotations == ()


def test_input_annotation_refs_survive_a_reachable_removal():
    # The context annotation stays on the task's input, so removing another record leaves it intact.
    dataset = make_dataset()
    dataset.derive_schema()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (context_annotation,) = answer.input_annotations
    edited = remove_records(dataset, [dataset.records[1].id])
    rebuilt = next(task for task in edited.tasks if task.id == answer.id)
    assert rebuilt.input_annotations == (context_annotation,)


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
    dataset.derive_schema()

    with pytest.raises(TimeFEditError, match=rf"dangle tasks \[[^\]]*\b{localize.id}\b"):
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

    by_record = {record.id: {id(annotation) for annotation in record.annotations} for record in restored.records}
    for task in restored.tasks:
        reachable = set().union(*(by_record[record.id] for record in task.inputs))
        named = (*task.input_annotations, *task.target_annotations)
        assert not [annotation for annotation in named if id(annotation) not in reachable]


def test_target_and_candidate_records_are_required_whatever_the_task_type():
    dataset = make_dataset()
    edited_record = dataset.records[1]
    candidate_record = dataset.records[2]
    edit_task = dataset.add_task(
        task=TSEditingTask(
            inputs=(dataset.records[0],),
            prompt="Denoise it.",
            targets=(edited_record,),
        )
    )
    correspondence_task = dataset.add_task(
        task=TSCorrespondenceTask(
            inputs=(dataset.records[0],),
            prompt="Which trace matches?",
            candidate_records=(edited_record, candidate_record),
            targets=(candidate_record,),
        )
    )
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match=rf"dangle tasks \[[^\]]*\b{edit_task.id}\b"):
        remove_records(dataset, [edited_record.id])
    with pytest.raises(TimeFEditError, match=rf"dangle tasks \[[^\]]*\b{correspondence_task.id}\b"):
        remove_records(dataset, [candidate_record.id])


_REF_SPEC = TimeSeriesSpec(spec_type="sine", name="Sine", unit_value=ureg.dimensionless)


def _noted(*names):
    """Return a dataset of one Record per name, each with one Signal and a ``note`` annotation of its name."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/pair", dataset_version=Version(1, 0, 0), name="pair", description="d", license=License.MIT
        )
    )
    for name in names:
        signal = Signal.from_values(
            np.arange(16, dtype=np.float32), spec=_REF_SPEC, name=f"sig-{name}", time_axis=RegularAxis.from_rate_hz(16)
        )
        record = dataset.add_record(record=Record(sources=(Source(name="m", signals=(signal,)),)))
        record.annotate(Annotation(key="note", value=name))
    return dataset


def _referencing_b(kind):
    """Return ``_noted("a", "b")`` with a task whose input is ``a`` and which references ``b`` by ``kind``."""
    dataset = _noted("a", "b")
    a, b = dataset.records
    tasks = {
        "candidate": lambda: TSCorrespondenceTask(inputs=(a,), candidate_records=(a, b), targets=(a,)),
        "record-target": lambda: TSEditingTask(inputs=(a,), targets=(b,)),
        "signal-target": lambda: TSEditingTask(inputs=(a,), targets=(b.signals[0],)),
        # An answer may lie only on a Record that the task references, so b is an input too, and a keeps
        # the task alive.
        "target-annotation": lambda: AnswerTask(inputs=(a, b), target_annotations=b.annotations),
        # A span may name only the Signals of inputs, so b is an input too, and a keeps the task alive.
        "span": lambda: AnswerTask(
            inputs=(a, b), targets=("x",), scope=TimeInterval.seconds(0.0, 0.5, time_series=b.signals)
        ),
    }
    task = dataset.add_task(task=tasks[kind]())
    return dataset, task


@pytest.mark.parametrize("kind", ["candidate", "record-target", "signal-target", "target-annotation", "span"])
def test_removing_a_record_invalidates_each_kind_of_reference_to_it(kind):
    dataset, task = _referencing_b(kind)
    b = dataset.records[1]
    with pytest.raises(TimeFEditError, match=re.escape(f"would dangle tasks [{task.id}]")):
        remove_records(dataset, [b.id])
    edited = remove_records(dataset, [b.id], cascade=True)
    assert edited.tasks == ()
    assert [record.id for record in edited.records] == [dataset.records[0].id]


def test_an_input_annotation_on_the_record_of_a_signal_target_stays_while_that_record_does():
    dataset = _noted("a", "b", "c")
    a, b, c = dataset.records
    # b holds the Signal target, so the task references b, though b is not one of its inputs.
    dataset.add_task(
        task=TSEditingTask(inputs=(a,), targets=(b.signals[0],), input_annotations=(*a.annotations, *b.annotations))
    )
    (task,) = remove_records(dataset, [c.id]).tasks
    assert [annotation.value for annotation in task.input_annotations] == ["a", "b"]
    # Removing b removes the Signal target too, so the task goes.
    assert remove_records(dataset, [b.id], cascade=True).tasks == ()


def test_an_edit_finds_the_record_of_each_signal_among_its_own_records():
    dataset = _noted("a", "b", "c")
    a, b, c = dataset.records
    dataset.add_task(task=TSEditingTask(inputs=(a,), targets=(c.signals[0],), input_annotations=c.annotations))
    edited = remove_records(dataset, [b.id])
    kept_a, kept_c = edited.records
    assert {signal_id: id(record) for signal_id, record in edited._records_by_signal_id.items()} == {
        a.signals[0].id: id(kept_a),
        c.signals[0].id: id(kept_c),
    }
    (task,) = edited.tasks
    assert [id(record) for record in edited._referenced_records(task)] == [id(kept_a), id(kept_c)]
    assert task.input_annotations == c.annotations


def _without_b(edit, dataset):
    """Return ``edit`` of a ``_noted`` dataset without its Record ``b``, or without the Signal of ``b``."""
    b = dataset.records[1]
    return edit(dataset, [b.id] if edit is remove_records else [b.signals[0].id])


@pytest.mark.parametrize("edit", [remove_records, remove_signals])
def test_the_signals_of_an_edit_are_copies_with_metadata_and_annotations_of_their_own(edit):
    dataset = _noted("a", "b")
    (signal,) = dataset.records[0].signals
    signal.metadata["gain"] = {"value": 1}
    artifact = signal.annotate(Annotation(key="artifact", span=TimeInterval.seconds(0.0, 0.25)))
    (copy,) = _without_b(edit, dataset).records[0].signals
    assert copy is not signal
    assert (copy.id, copy.annotations, copy.metadata) == (signal.id, (artifact,), signal.metadata)
    assert copy.annotations[0] is artifact
    # The copy reads the stored values of the original.
    assert copy._values is signal._values
    gain = copy.metadata["gain"]
    assert isinstance(gain, dict)
    gain["value"] = 2
    copy.metadata["checked"] = True
    copy.annotate(Annotation(key="checked", value=True))
    assert (signal.metadata, signal.annotations) == ({"gain": {"value": 1}}, (artifact,))


@pytest.mark.parametrize("edit", [remove_records, remove_signals])
def test_the_tasks_of_an_edit_point_at_its_copies_of_the_signals(edit):
    dataset = _noted("a", "b")
    a = dataset.records[0]
    (signal,) = a.signals
    ordinal = Signal.from_values(np.arange(8, dtype=np.float32), spec=_REF_SPEC, name="sig-o", time_axis=OrdinalAxis())
    stepped = dataset.add_record(record=Record(sources=(Source(name="m", signals=(ordinal,)),)))
    window = TimeInterval.seconds(0.0, 0.5, time_series=(signal,))
    dataset.add_tasks(
        tasks=[
            TSEditingTask(inputs=(a,), targets=(signal, "denoised")),
            AnswerTask(inputs=(a,), targets=("x",), scope=window),
            TemporalLocalizationTask(inputs=(a,), targets=(window,)),
            AnswerTask(inputs=(stepped,), targets=("y",), scope=StepInterval(time_series=ordinal, start=0, stop=4)),
        ]
    )
    edited = _without_b(edit, dataset)
    copies = {copy.id: copy for record in edited.records for copy in record.signals}
    copy, ordinal_copy = copies[signal.id], copies[ordinal.id]
    assert copy is not signal
    assert ordinal_copy is not ordinal
    targeted, scoped, located, step_scoped = edited.tasks
    assert targeted.targets is not None
    assert located.targets is not None
    (located_window,) = located.targets
    assert isinstance(scoped.scope, TimeInterval)
    assert isinstance(located_window, TimeInterval)
    assert isinstance(step_scoped.scope, StepInterval)
    # Signals compare by identity.
    assert targeted.targets[0] is copy
    assert scoped.scope.time_series == located_window.time_series == (copy,)
    assert step_scoped.scope.time_series is ordinal_copy


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


def _signal(name, offset_s):
    """Return 16 values at 16 Hz, covering one second from ``offset_s``."""
    axis = RegularAxis.from_rate_hz(16, offset_us=offset_s * 1_000_000)
    return Signal.from_values(np.arange(16, dtype=np.float32), spec=_REF_SPEC, name=name, time_axis=axis)


def _signal_id(dataset, name):
    """Return the ID the dataset gave the Signal of this name."""
    return next(signal.id for record in dataset.records for signal in record.signals if signal.name == name)


def _two_leads(time_span=None):
    """Return a local dataset whose one Record holds ``sig-a`` (0 to 1 s) below ``monitor/leads``
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
    leads = Source(name="leads", signals=(_signal("sig-a", 0),))
    monitor = Source(name="monitor", signals=(signal_b,), sources=(leads,))
    record = dataset.add_record(record=Record(sources=(monitor,), time_span=time_span))
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
    sig_a, sig_b = _signal_id(dataset, "sig-a"), _signal_id(dataset, "sig-b")
    edited = remove_signals(dataset, [sig_b])
    (record,) = edited.records
    (monitor,) = record.sources
    (leads,) = monitor.sources
    assert [signal.id for signal in record.signals] == [sig_a]
    assert (_keys(record), _keys(monitor), _keys(leads)) == (["age", "early"], ["src-early"], ["leads-early"])
    # The original keeps its Signals and annotations.
    (original,) = dataset.records
    (original_monitor,) = original.sources
    assert sorted(signal.id for signal in original.signals) == sorted([sig_a, sig_b])
    assert (_keys(original), _keys(original_monitor)) == (["age", "early", "gap", "late"], ["src-early", "src-late"])
    # Without any Signal, the Sources stay, with the annotations that need no Signal.
    emptied = remove_signals(edited, [sig_a])
    (record,) = emptied.records
    assert record.signals == ()
    assert [(source.name, _keys(source)) for source in record.walk_sources()] == [("monitor", []), ("leads", [])]
    assert _keys(record) == ["age"]


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_keeps_what_lies_inside_a_records_session_span():
    dataset = _two_leads(time_span=TimeInterval.seconds(0.0, 3.0))
    edited = remove_signals(dataset, [_signal_id(dataset, "sig-b")])
    (record,) = edited.records
    # The session span places late and gap; a Source has no session span.
    assert (_keys(record), _keys(record.sources[0])) == (["age", "early", "gap", "late"], ["src-early"])


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_rejects_or_cascades_the_tasks_that_reference_what_it_removes():
    dataset = _two_leads()
    (record,) = dataset.records
    signal_a, signal_b = (
        next(signal for signal in record.signals if signal.name == name) for name in ("sig-a", "sig-b")
    )
    context = AnswerTask(inputs=(record,), targets=("x",), input_annotations=(_annotation(record, "late"),))
    kept = AnswerTask(
        inputs=(record,),
        targets=("x",),
        input_annotations=(_annotation(record, "early"),),
        scope=TimeInterval.seconds(0.0, 0.5, time_series=(signal_a,)),
    )
    edit = TSEditingTask(inputs=(record,), targets=(signal_b, "denoised"))
    scoped = AnswerTask(inputs=(record,), targets=("x",), scope=TimeInterval.seconds(2.0, 2.5, time_series=(signal_b,)))
    located = TemporalLocalizationTask(inputs=(record,), target_annotations=(_annotation(signal_b, "artifact"),))
    dataset.add_tasks(
        tasks=[
            edit,
            AnswerTask(inputs=(record,), targets=("z",), from_tasks=(edit,)),
            scoped,
            located,
            context,
            AnswerTask(inputs=(record,), targets=("y",), from_tasks=(context,)),
            kept,
        ]
    )
    after_edit, derived = dataset.tasks[1], dataset.tasks[5]
    # The input annotation late is context, so only the other references invalidate a task.
    with pytest.raises(
        TimeFEditError,
        match=re.escape(
            f"removing signals [{signal_b.id}] would dangle tasks {sorted([edit.id, scoped.id, located.id])}"
        ),
    ):
        remove_signals(dataset, [signal_b.id])
    edited = remove_signals(dataset, [signal_b.id], cascade=True)
    (record,) = edited.records
    assert [task.id for task in edited.tasks] == [context.id, derived.id, kept.id]
    assert edited.tasks_for(record) == edited.tasks
    # context and derived, then kept.
    assert edited.tasks[0].input_annotations == ()
    assert edited.tasks[1].from_tasks == (edited.tasks[0],)
    assert edited.tasks[2].inputs == (record,)
    assert edited.tasks[2].input_annotations[0] is _annotation(record, "early")
    assert len(dataset.tasks) == 7
    # Removing sig-a drops only the task scoped to it.
    others = remove_signals(dataset, [signal_a.id], cascade=True)
    assert [task.id for task in others.tasks] == [
        task.id for task in (edit, after_edit, scoped, located, context, derived)
    ]


def test_remove_signals_rejects_a_task_whose_step_span_names_a_removed_signal():
    dataset = _noted("a", "b")
    ordinal = Signal.from_values(np.arange(8, dtype=np.float32), spec=_REF_SPEC, name="sig-o", time_axis=OrdinalAxis())
    # The Record keeps another Signal, so only the span names what goes.
    timed = Signal.from_values(
        np.arange(16, dtype=np.float32), spec=_REF_SPEC, name="sig-t", time_axis=RegularAxis.from_rate_hz(16)
    )
    record = dataset.add_record(record=Record(sources=(Source(name="m", signals=(ordinal, timed)),)))
    stepped = dataset.add_task(
        task=AnswerTask(inputs=(record,), targets=("x",), scope=StepInterval(time_series=ordinal, start=0, stop=4))
    )
    assert [task.id for task in remove_signals(dataset, [_signal_id(dataset, "sig-a")]).tasks] == [stepped.id]
    with pytest.raises(TimeFEditError, match=re.escape(f"would dangle tasks [{stepped.id}]")):
        remove_signals(dataset, [ordinal.id])


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_invalidates_a_task_whose_input_keeps_no_signal():
    dataset = _two_leads()
    (record,) = dataset.records
    task = dataset.add_task(task=AnswerTask(inputs=(record,), targets=("x",)))
    sig_a, sig_b = _signal_id(dataset, "sig-a"), _signal_id(dataset, "sig-b")
    assert [kept.id for kept in remove_signals(dataset, [sig_a]).tasks] == [task.id]
    with pytest.raises(TimeFEditError, match=re.escape(f"would dangle tasks [{task.id}]")):
        remove_signals(dataset, [sig_a, sig_b])
    edited = remove_signals(dataset, [sig_a, sig_b], cascade=True)
    (record,) = edited.records
    assert (edited.tasks, record.signals, edited.tasks_for(record)) == ((), (), ())


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
def test_remove_signals_places_a_span_without_signals_as_it_places_an_annotation(span, time_span, placed):
    dataset = _two_leads(time_span=time_span)
    (record,) = dataset.records
    interval = TimeInterval.seconds(*span)
    tasks = dataset.add_tasks(
        tasks=[
            AnswerTask(inputs=(record,), targets=("x",), scope=interval),
            TemporalLocalizationTask(inputs=(record,), targets=(interval,)),
        ]
    )
    sig_b = _signal_id(dataset, "sig-b")
    if placed:
        assert [task.id for task in remove_signals(dataset, [sig_b]).tasks] == [task.id for task in tasks]
    else:
        with pytest.raises(TimeFEditError, match=re.escape(f"would dangle tasks {[task.id for task in tasks]}")):
            remove_signals(dataset, [sig_b])


@pytest.mark.parametrize("edit", [remove_records, remove_signals])
def test_edits_reject_a_string_of_ids(edit):
    # A string names one ID per character, so it is refused even when it spells an ID.
    with pytest.raises(TimeFEditError, match="takes a collection of IDs, not the string '1'"):
        edit(_noted("a", "b"), "1")


def test_edit_version_removes_the_signals_first_and_forwards_cascade(tmp_path):
    dataset = _noted("a", "b")
    a, b = dataset.records
    sig_a, sig_b = a.signals[0].id, b.signals[0].id
    dataset.add_tasks(
        tasks=[
            TSEditingTask(inputs=(b,), targets=(a.signals[0], "denoised")),
            TSEditingTask(inputs=(a,), targets=(b,)),
        ]
    )
    base = _base(tmp_path, dataset)
    # The Signals go first; sig-b, a Signal of the removed b, is left to b's removal.
    with pytest.raises(TimeFEditError, match=re.escape(f"removing signals [{sig_a}] would dangle tasks")):
        edit_version(base, dataset_version=Version(1, 0, 1), remove_record_ids=[b.id], remove_signal_ids=[sig_a, sig_b])
    with pytest.raises(TimeFEditError, match="takes a collection of IDs, not the string"):
        edit_version(
            base,
            dataset_version=Version(1, 0, 1),
            remove_signal_ids=str(sig_a),  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TimeFEditError, match="takes a collection of IDs, not the string"):
        edit_version(
            base,
            dataset_version=Version(1, 0, 1),
            remove_record_ids=str(b.id),  # ty: ignore[invalid-argument-type]
        )
    # A Signal of a removed record is left to that removal, so it names no unknown Signal.
    out = edit_version(
        base,
        dataset_version=Version(1, 0, 1),
        remove_record_ids=[b.id],
        remove_signal_ids=[sig_a, sig_b],
        cascade=True,
    )
    with TimeFReader(open_local_version(out)) as reader:
        edited = reader.read()
    assert ([record.id for record in edited.records], edited.tasks) == ([a.id], ())
    assert edited.records[0].signals == ()


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_remove_signals_rejects_an_unknown_id():
    with pytest.raises(TimeFEditError, match=re.escape("unknown signal ids: [999]")):
        remove_signals(_two_leads(), [999])


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_edit_version_removes_signals(tmp_path):
    dataset = _two_leads()
    base = _base(tmp_path, dataset)
    out = edit_version(base, dataset_version=Version(1, 0, 1), remove_signal_ids=[_signal_id(dataset, "sig-b")])
    with TimeFReader(open_local_version(out)) as reader:
        (record,) = reader.read().records
    assert [signal.name for signal in record.signals] == ["sig-a"]
    assert (_keys(record), _keys(record.sources[0])) == (["age", "early"], ["src-early"])
