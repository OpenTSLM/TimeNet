import pytest

from timenet.dataset import OrdinalAxis, Record, Signal, Source
from timenet.errors import TimeFFormatError
from timenet.types import US_PER_S, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.bases.seizures import RecordingSeizures, SeizureInterval, WindowLabel


def _record() -> Record:
    spec = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.microvolt)
    return Record(
        sources=[Source("eeg", signals=[Signal.from_values([0.0], spec=spec, name="c", time_axis=OrdinalAxis())])]
    )


def _seconds(start: int, end: int) -> SeizureInterval:
    return SeizureInterval.from_whole_seconds(start, end)


def test_a_window_that_only_touches_a_seizure_boundary_is_no_seizure():
    seizures = RecordingSeizures(recording_id="r", duration_us=10 * US_PER_S, seizures=(_seconds(4, 6),))
    assert seizures.label_window(2 * US_PER_S, 4 * US_PER_S) is WindowLabel.NO_SEIZURE
    assert seizures.label_window(6 * US_PER_S, 8 * US_PER_S) is WindowLabel.NO_SEIZURE
    assert seizures.label_window(5 * US_PER_S, 7 * US_PER_S) is WindowLabel.SEIZURE


def test_a_recording_with_no_seizure_gets_an_explicit_empty_target():
    seizures = RecordingSeizures(recording_id="r", duration_us=10 * US_PER_S, seizures=())
    assert seizures.create_localization_task(_record(), split=None).targets == ()


def test_windows_cover_whole_windows_only():
    seizures = RecordingSeizures(recording_id="r", duration_us=5 * US_PER_S, seizures=())
    tasks = list(seizures.iter_window_tasks(_record(), window_us=2 * US_PER_S, split=None))
    assert [task.scope for task in tasks] == [TimeInterval.seconds(0, 2), TimeInterval.seconds(2, 4)]


def test_seizures_that_overlap_are_refused():
    with pytest.raises(TimeFFormatError, match="starts before the one before it ends"):
        RecordingSeizures(recording_id="r", duration_us=10 * US_PER_S, seizures=(_seconds(1, 5), _seconds(4, 6)))


def test_a_seizure_past_the_recording_end_is_refused():
    with pytest.raises(TimeFFormatError, match="ends after the recording"):
        RecordingSeizures(recording_id="r", duration_us=10 * US_PER_S, seizures=(_seconds(8, 11),))


def test_a_seizure_that_ends_before_it_starts_is_refused():
    with pytest.raises(TimeFFormatError, match="end after it starts"):
        _seconds(6, 6)
