from fractions import Fraction

import numpy as np
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    ForecastingTask,
    License,
    TimeInterval,
    Version,
)
from timenet_connectors.datasets.adityalab.time_mmd import tasks
from timenet_connectors.datasets.adityalab.time_mmd.domains import ENERGY, GASOLINE_PRICE
from timenet_connectors.datasets.adityalab.time_mmd.keys import AnnotationKey
from timenet_connectors.datasets.adityalab.time_mmd.timeline import US_PER_DAY


_RECORD_ID = "time-mmd-energy"
_WEEK = Fraction(7 * US_PER_DAY)


def _weekly(n: int, *, start_index: int = 0, signal: str = "us") -> TimeSeries:
    return TimeSeries.from_values(
        np.arange(n, dtype=np.float32),
        spec=GASOLINE_PRICE,
        signal=signal,
        time_axis=RegularAxis(period_us=_WEEK, start_index=start_index),
        time_series_id=f"{_RECORD_ID}-{signal}",
    )


def _scope(task) -> TimeInterval:
    assert isinstance(task.scope, TimeInterval)
    return task.scope


def _target_span(task) -> TimeInterval:
    assert isinstance(task.target_span, TimeInterval)
    return task.target_span


@pytest.mark.parametrize(("n", "origin"), [(100, 80), (1622, 1298), (15979, 12784), (5, 4), (9, 8)])
def test_the_test_split_is_the_last_fifth_as_the_protocol_computes_it(n, origin):
    assert tasks.test_origin(n) == origin
    assert tasks.test_origin(n) == n - int(n * 0.2)


def test_one_task_for_each_origin_of_the_test_split_that_fits_the_horizon():
    built = list(tasks.iter_forecasting_tasks(_RECORD_ID, _weekly(100), horizons=(10,)))
    assert [task.id for task in built] == [f"time-mmd-energy-forecast-h10-o{origin}" for origin in range(80, 91)]


def test_a_horizon_that_fits_no_origin_gives_no_task():
    assert list(tasks.iter_forecasting_tasks(_RECORD_ID, _weekly(100), horizons=(30,))) == []


def test_the_tasks_come_by_horizon_and_then_by_origin():
    built = list(tasks.iter_forecasting_tasks(_RECORD_ID, _weekly(100), horizons=(20, 10)))
    # Horizon 20 fits one origin, 80. Horizon 10 fits the eleven origins 80 to 90.
    assert [task.id for task in built[:3]] == [
        "time-mmd-energy-forecast-h20-o80",
        "time-mmd-energy-forecast-h10-o80",
        "time-mmd-energy-forecast-h10-o81",
    ]
    assert len(built) == 1 + 11


def test_the_context_is_everything_before_the_origin_and_the_target_is_the_horizon():
    task = next(tasks.iter_forecasting_tasks(_RECORD_ID, _weekly(100), horizons=(10,)))
    assert _scope(task) == TimeInterval.micros(0, 80 * 7 * US_PER_DAY)
    assert _target_span(task) == TimeInterval.micros(
        80 * 7 * US_PER_DAY, 90 * 7 * US_PER_DAY, time_series_ids=("time-mmd-energy-us",)
    )
    assert _scope(task).time_series_ids is None
    assert task.record_ids == (_RECORD_ID,)
    assert task.prompt is None and task.target is None


def test_the_origin_counts_from_the_record_zero_when_the_series_starts_later():
    task = next(tasks.iter_forecasting_tasks(_RECORD_ID, _weekly(100, start_index=3), horizons=(10,)))
    assert _target_span(task).start_us == (3 + 80) * 7 * US_PER_DAY
    # The context starts where the series starts, so a consumer can resolve it to steps.
    assert _scope(task).start_us == 3 * 7 * US_PER_DAY


def test_the_context_resolves_to_the_steps_of_the_series():
    target = _weekly(100, start_index=3)
    task = next(tasks.iter_forecasting_tasks(_RECORD_ID, target, horizons=(10,)))
    assert target.step_range(_scope(task)) == (0, 80)
    assert target.step_range(_target_span(task)) == (80, 90)


def test_a_series_with_no_cadence_gives_no_forecast():
    ordinal = TimeSeries.from_values(
        np.arange(10, dtype=np.float32), spec=GASOLINE_PRICE, signal="us", time_axis=OrdinalAxis()
    )
    with pytest.raises(TimeFFormatError, match="no regular cadence"):
        next(tasks.iter_forecasting_tasks(_RECORD_ID, ordinal, horizons=(2,)))


def _fact(row: int, start_day: int, end_day: int, key: str = AnnotationKey.REPORT_FACT) -> Annotation:
    return Annotation(
        key=key,
        value=f"Synthetic fact {row}.",
        span=TimeInterval.micros(start_day * US_PER_DAY, end_day * US_PER_DAY),
        id=f"{_RECORD_ID}-report-fact-{row}",
    )


def test_a_caption_covers_the_record_up_to_the_end_of_the_report_days():
    target = _weekly(10, start_index=2)  # values from day 14 through day 84
    built = list(tasks.iter_caption_tasks(_RECORD_ID, target, [_fact(0, 21, 26)]))
    assert [task.id for task in built] == ["time-mmd-energy-report-fact-0-caption"]
    assert _scope(built[0]) == TimeInterval.micros(14 * US_PER_DAY, 26 * US_PER_DAY)
    assert target.step_range(_scope(built[0])) == (0, 2)
    assert built[0].target_annotation_ids == ("time-mmd-energy-report-fact-0",)
    assert built[0].prompt is None and built[0].target is None


def test_a_report_with_no_value_beneath_it_gets_no_caption():
    target = _weekly(10, start_index=2)  # values from day 14 through day 84
    before = _fact(0, 0, 7)
    after = _fact(1, 84, 91)
    straddling = _fact(2, 80, 90)
    built = list(tasks.iter_caption_tasks(_RECORD_ID, target, [before, after, straddling]))
    assert [task.id for task in built] == ["time-mmd-energy-report-fact-2-caption"]


def test_only_a_report_fact_gets_a_caption():
    target = _weekly(10)
    others = [
        _fact(0, 0, 7, key=AnnotationKey.SEARCH_FACT),
        _fact(1, 0, 7, key=AnnotationKey.REPORT_PREDICTION),
        Annotation(key=AnnotationKey.TARGET_SIGNAL, value="us", id="static"),
    ]
    assert list(tasks.iter_caption_tasks(_RECORD_ID, target, others)) == []


def test_a_target_with_no_timeline_gives_no_caption():
    ordinal = TimeSeries.from_values(
        np.arange(10, dtype=np.float32), spec=GASOLINE_PRICE, signal="us", time_axis=OrdinalAxis()
    )
    with pytest.raises(TimeFFormatError, match="no timeline"):
        next(tasks.iter_caption_tasks(_RECORD_ID, ordinal, [_fact(0, 0, 7)]))


def _dataset() -> TimeFDataset:
    return TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="adityalab/time-mmd",
            dataset_version=Version(1, 0, 0),
            name="Time-MMD",
            description="A synthetic dataset for the task tests.",
            license=License.CC_BY_4_0,
        )
    )


def test_iter_tasks_walks_every_record_and_finds_its_target():
    dataset = _dataset()
    record = dataset.add_record(time_series=(_weekly(100, signal="east_coast"), _weekly(100)), record_id=_RECORD_ID)
    record.add_annotation(_fact(0, 21, 26))
    built = list(tasks.iter_tasks(dataset.records, (ENERGY,)))
    forecasts = [task for task in built if isinstance(task, ForecastingTask)]
    captions = [task for task in built if isinstance(task, AnswerTask)]
    assert len(forecasts) == 9 + 0  # horizon 12 fits origins 80..88; 24, 36 and 48 fit none
    assert {_target_span(task).time_series_ids for task in forecasts} == {("time-mmd-energy-us",)}
    assert [task.id for task in captions] == ["time-mmd-energy-report-fact-0-caption"]


def test_a_record_of_no_known_domain_raises():
    dataset = _dataset()
    dataset.add_record(time_series=(_weekly(10),), record_id="time-mmd-traffic")
    with pytest.raises(TimeFValidationError, match="belongs to no domain"):
        next(tasks.iter_tasks(dataset.records, (ENERGY,)))


def test_a_record_without_its_target_raises():
    dataset = _dataset()
    dataset.add_record(time_series=(_weekly(10, signal="east_coast"),), record_id=_RECORD_ID)
    with pytest.raises(TimeFFormatError, match="holds 0 series named 'us'"):
        next(tasks.iter_tasks(dataset.records, (ENERGY,)))
