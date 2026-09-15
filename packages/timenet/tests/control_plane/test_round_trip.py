"""The round trip that decides the swap: write a dataset, read it back, assert it is the same one.

``assert_datasets_equal`` compares the metadata, every record's fields and values, every annotation
and every task field for field. The per-field assertions below then name what the control plane has
to carry for each of the eight typed tasks, so a regression says which field was lost rather than
only that the datasets differ.
"""

from fractions import Fraction
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

from timenet.control_plane import schema as ddl
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    ForecastingTask,
    License,
    LocalizationMode,
    ScalarPredictionTask,
    StepInterval,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


_SPEC = TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=ureg.millivolt)


def _timed(time_series_id: str) -> TimeSeries:
    return TimeSeries(
        spec=_SPEC,
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(4)),
        loader=lambda: pa.array([0.0, 1.0, 2.0, 3.0], type=pa.float32()),
        source_id=f"src-{time_series_id}",
        time_series_id=time_series_id,
        n_values=4,
    )


def _ordinal(time_series_id: str) -> TimeSeries:
    return TimeSeries.from_values(
        [float(i) for i in range(6)],
        spec=TimeSeriesSpec(spec_type="steps", name="Steps", unit_value=ureg.dimensionless),
        signal="s",
        time_axis=OrdinalAxis(),
        time_series_id=time_series_id,
    )


def _every_task_type() -> TimeFDataset:
    """Build one dataset carrying every typed task and every annotation shape.

    Returns:
        The dataset. Every id is fixed, so two calls produce equal datasets.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/every-task",
            dataset_version=Version(1, 0, 0),
            name="Every task",
            description="One record per task type, and every annotation shape.",
            license=License.CC_BY_4_0,
            domains=(Domain.CARDIOLOGY,),
        )
    )
    shared = _timed("ts-shared")
    ordinal = _ordinal("ts-ordinal")

    query = dataset.add_record(time_series=(shared, _timed("ts-query")), record_id="rec-query")
    context = dataset.add_record(time_series=(shared,), record_id="rec-context")
    horizon = dataset.add_record(time_series=(_timed("ts-horizon"),), record_id="rec-horizon")
    edited = dataset.add_record(time_series=(_timed("ts-edited"),), record_id="rec-edited")
    stepped = dataset.add_record(time_series=(ordinal,), record_id="rec-stepped")

    cohort = Annotation(key="cohort", value="A", id="ann-cohort")
    query.add_annotations(
        [
            Annotation(key="age", value=64, unit="years", id="ann-age"),
            cohort,
            Annotation(key="stimulus", span=TimePoint.seconds(0.5), id="ann-point"),
            Annotation(
                key="artifact",
                value="motion",
                source="rater-A",
                span=TimeInterval.seconds(0.0, 0.25, time_series_ids=("ts-shared",)),
                id="ann-interval",
            ),
        ]
    )
    context.add_annotation(cohort)  # the same instance in two records, stored once
    options = Annotation(key="answer_options", value=["yes", "no"], id="ann-options")
    dataset.register_annotations([options])

    classification = ClassificationTask(
        target="afib", target_schema="scp5", id="task-classification", scope=TimeInterval.seconds(0.0, 0.5)
    )
    dataset.add_tasks(
        query,
        [
            classification,
            AnswerTask(
                prompt="What rhythm?",
                target="Normal.",
                rationale="Regular intervals with one peak per cycle.",
                input_annotation_ids=("ann-cohort", "ann-options"),
                from_tasks=(classification,),
                id="task-answer",
            ),
            ScalarPredictionTask(target=62.5, unit="bpm", target_name="mean_rate", id="task-scalar"),
            TemporalLocalizationTask(
                prompt="Locate the beats.",
                mode=LocalizationMode.EXHAUSTIVE,
                target=(TimePoint.seconds(0.25), TimeInterval.seconds(0.5, 0.75, time_series_ids=("ts-query",))),
                id="task-localize",
            ),
            TemporalLocalizationTask(prompt="Any P-waves?", target=(), id="task-localize-empty"),
            TSCorrespondenceTask(
                prompt="Which trace matches?",
                candidate_record_ids=("rec-context", "rec-horizon"),
                target=("rec-context",),
                id="task-correspondence",
            ),
            TSCorrespondenceTask(target_time_series_ids=("ts-query",), id="task-correspondence-series"),
            AnswerTask(target_annotation_ids=("ann-options",), id="task-answer-by-reference"),
        ],
    )
    dataset.add_task(
        horizon,
        ForecastingTask(context_record_ids=("rec-context",), target_record_id="rec-horizon", id="task-forecast-record"),
    )
    dataset.add_task(
        stepped,
        ForecastingTask(
            scope=StepInterval(time_series_id="ts-ordinal", start=0, stop=4),
            target_span=StepInterval(time_series_id="ts-ordinal", start=4, stop=6),
            id="task-forecast-steps",
        ),
    )
    dataset.add_task(
        edited,
        TSEditingTask(
            prompt="Remove the baseline wander.",
            source_record_id="rec-query",
            target_record_id="rec-edited",
            id="task-edit",
        ),
    )
    dataset.add_task(edited, TSGenerationTask(prompt="10 s of sinus.", target_record_id="rec-edited", id="task-gen"))
    return dataset


def _round_trip(tmp_path: Path, dataset: TimeFDataset, **kwargs) -> TimeFDataset:
    """Write a dataset and read it straight back."""
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    version_dir = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        return reader.read()


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_the_testing_fixture_round_trips(tmp_path, backend):
    assert_datasets_equal(make_dataset(), _round_trip(tmp_path, make_dataset(), values_backend=backend))


def test_every_typed_task_round_trips(tmp_path):
    assert_datasets_equal(_every_task_type(), _round_trip(tmp_path, _every_task_type()))


@pytest.fixture(scope="module")
def restored(tmp_path_factory):
    """The every-task dataset, written and read back once for the per-field assertions below."""
    return _round_trip(tmp_path_factory.mktemp("every-task"), _every_task_type())


def _task(restored, task_id):
    return next(task for task in restored.tasks if task.id == task_id)


def test_a_classification_keeps_its_label_schema_and_scope(restored):
    task = _task(restored, "task-classification")
    assert isinstance(task, ClassificationTask)
    assert (task.target, task.target_schema) == ("afib", "scp5")
    assert task.scope == TimeInterval.seconds(0.0, 0.5)


def test_an_answer_keeps_its_rationale_inputs_and_derivation(restored):
    task = _task(restored, "task-answer")
    assert isinstance(task, AnswerTask)
    assert task.target == "Normal."
    assert task.rationale == "Regular intervals with one peak per cycle."
    assert task.input_annotation_ids == ("ann-cohort", "ann-options")
    assert [parent.id for parent in task.from_tasks] == ["task-classification"]


def test_an_answer_stored_by_reference_keeps_its_target_ids(restored):
    task = _task(restored, "task-answer-by-reference")
    assert isinstance(task, AnswerTask)
    assert task.target is None
    assert task.target_annotation_ids == ("ann-options",)


def test_a_scalar_prediction_keeps_its_number_unit_and_name(restored):
    task = _task(restored, "task-scalar")
    assert isinstance(task, ScalarPredictionTask)
    assert task.target == pytest.approx(62.5)
    assert (task.unit, task.target_name) == ("bpm", "mean_rate")


def test_a_localization_keeps_its_mode_and_every_span_shape(restored):
    task = _task(restored, "task-localize")
    assert isinstance(task, TemporalLocalizationTask)
    assert task.mode is LocalizationMode.EXHAUSTIVE
    assert task.target == (
        TimePoint.seconds(0.25),
        TimeInterval.seconds(0.5, 0.75, time_series_ids=("ts-query",)),
    )


def test_an_empty_localization_target_stays_empty_rather_than_absent(restored):
    # () says the task looked and found nothing; None says the answer is stored by reference.
    task = _task(restored, "task-localize-empty")
    assert isinstance(task, TemporalLocalizationTask)
    assert task.target == ()
    assert task.mode is LocalizationMode.SPARSE


def test_a_forecast_by_record_keeps_its_context_and_horizon(restored):
    task = _task(restored, "task-forecast-record")
    assert isinstance(task, ForecastingTask)
    assert task.context_record_ids == ("rec-context",)
    assert task.target_record_id == "rec-horizon"
    assert task.target_span is None


def test_a_forecast_by_span_keeps_its_step_frame(restored):
    task = _task(restored, "task-forecast-steps")
    assert isinstance(task, ForecastingTask)
    assert task.target_span == StepInterval(time_series_id="ts-ordinal", start=4, stop=6)
    assert task.scope == StepInterval(time_series_id="ts-ordinal", start=0, stop=4)
    assert task.context_record_ids == ()


def test_an_edit_keeps_both_of_its_records(restored):
    task = _task(restored, "task-edit")
    assert isinstance(task, TSEditingTask)
    assert (task.source_record_id, task.target_record_id) == ("rec-query", "rec-edited")


def test_a_generation_keeps_its_target_record(restored):
    task = _task(restored, "task-gen")
    assert isinstance(task, TSGenerationTask)
    assert task.target_record_id == "rec-edited"


def test_a_correspondence_keeps_its_pool_and_answer(restored):
    task = _task(restored, "task-correspondence")
    assert isinstance(task, TSCorrespondenceTask)
    assert task.candidate_record_ids == ("rec-context", "rec-horizon")
    assert task.target == ("rec-context",)


def test_a_correspondence_answering_with_series_keeps_those_ids(restored):
    task = _task(restored, "task-correspondence-series")
    assert isinstance(task, TSCorrespondenceTask)
    assert task.target_time_series_ids == ("ts-query",)
    assert task.candidate_record_ids == ()


# ---- annotations --------------------------------------------------------------------------------


def _annotations(restored, record_id):
    record = next(r for r in restored.records if r.record_id == record_id)
    return {annotation.key: annotation for annotation in record.annotations}


def test_a_static_annotation_keeps_its_typed_value_and_unit(restored):
    age = _annotations(restored, "rec-query")["age"]
    assert age.span is None
    assert age.value == 64 and isinstance(age.value, int)
    assert age.unit == "years"


def test_a_point_annotation_comes_back_as_a_point(restored):
    stimulus = _annotations(restored, "rec-query")["stimulus"]
    assert stimulus.span == TimePoint.seconds(0.5)


def test_an_interval_annotation_keeps_its_scope_and_provenance(restored):
    artifact = _annotations(restored, "rec-query")["artifact"]
    assert artifact.span == TimeInterval.seconds(0.0, 0.25, time_series_ids=("ts-shared",))
    assert artifact.source == "rater-A"
    assert artifact.value == "motion"


def test_an_annotation_on_two_records_comes_back_on_both(restored):
    assert _annotations(restored, "rec-query")["cohort"] == _annotations(restored, "rec-context")["cohort"]


def test_a_registered_annotation_comes_back_carried_by_no_record(restored):
    assert [annotation.id for annotation in restored.registered_annotations] == ["ann-options"]
    assert "answer_options" not in _annotations(restored, "rec-query")


def test_the_annotation_payload_is_stored_once_however_many_records_carry_it(tmp_path):
    dataset = _every_task_type()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    version_dir = tmp_path / dataset.metadata.dataset_id / "1.0.0"
    connection = duckdb.connect(str(version_dir / "control.duckdb"), read_only=True)
    try:
        payloads = connection.execute("SELECT count(*) FROM annotations WHERE key = 'cohort'").fetchall()
        attachments = connection.execute(
            "SELECT count(*) FROM record_annotations a JOIN annotations n ON n.annotation_id = a.annotation_id "
            "WHERE n.key = 'cohort'"
        ).fetchall()
    finally:
        connection.close()
    assert payloads == [(1,)]
    assert attachments == [(2,)]


# ---- the commit protocol ------------------------------------------------------------------------


def test_a_failed_validation_publishes_nothing_and_leaves_no_staging_directory(tmp_path, monkeypatch):
    # The checks run inside the load transaction, so a build that does not hold together never
    # reaches COMMIT and never reaches the version directory either.
    always_fails = (*ddl.VALIDATIONS, ("planted failure", "SELECT record_id FROM records"))
    monkeypatch.setattr(ddl, "VALIDATIONS", always_fails)
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFValidationError, match="planted failure"), TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    assert not (tmp_path / "timenet/hello-world" / "1.0.0").exists()
    assert not list((tmp_path / "timenet/hello-world").glob("*.tmp-*"))


def test_a_version_on_a_non_local_filesystem_reads_through_a_local_copy(tmp_path):
    # DuckDB opens a database through its own filesystem layer, not the pyarrow one the rest of the
    # reader uses, so an object-store version is copied down once and removed again on close.
    import pyarrow.fs as pafs  # noqa: PLC0415

    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    version_dir = tmp_path / "timenet/hello-world" / "1.0.0"
    remote = DatasetVersion(
        manifest=DatasetVersion.open_local(version_dir).manifest,
        filesystem=pafs.SubTreeFileSystem(str(version_dir), pafs.LocalFileSystem()),
        root="",
    )
    with TimeFReader(remote) as reader:
        assert_datasets_equal(make_dataset(), reader.read())
        copied = reader._control_plane()._materialized
        assert copied is not None and copied.exists()
    assert not copied.exists()


def test_a_read_that_spans_several_batches_keeps_stored_order(tmp_path, monkeypatch):
    # A batch answers for many records at once, so a corpus larger than one batch has to stitch the
    # batches back together in order. Shrinking the batch makes a three-record fixture do that.
    import timenet.reader.reader as reader_module  # noqa: PLC0415

    monkeypatch.setattr(reader_module, "_RECORD_BATCH_ROWS", 1)
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    version_dir = tmp_path / "timenet/hello-world" / "1.0.0"
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert [record.record_id for record in reader.iter_records()] == ["record-0", "record-1", "record-2"]
        selected = list(reader.iter_records(record_ids=["record-2", "record-0"]))
        assert [record.record_id for record in selected] == ["record-0", "record-2"]
        assert_datasets_equal(make_dataset(), reader.read())
