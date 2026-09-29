from fractions import Fraction

import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, TimePoint, TimeSeriesSpec, ureg


SPEC = TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=ureg.millivolt)


def _signal(signal_id: str, name: str) -> Signal:
    return Signal(
        id=signal_id,
        name=name,
        data=[1.0],
        spec=SPEC,
        time_axis=RegularAxis(period_us=Fraction(1_000)),
    )


def test_source_selection_attaches_shared_timed_content_only_to_selected_signals():
    lead_i = _signal("lead-i", "I")
    lead_ii = _signal("lead-ii", "II")
    temperature = _signal("temperature", "temperature")
    source = Source(id="ecg", name="ECG", signals=(lead_i, lead_ii, temperature))

    occurrences = source.select(signals=(lead_i, lead_ii)).annotate(
        Annotation(key="status", value="off", span=TimePoint.micros(500))
    )

    assert occurrences[0].content_id == occurrences[1].content_id
    assert occurrences[0].occurrence_id != occurrences[1].occurrence_id
    assert lead_i.annotations == (occurrences[0],)
    assert lead_ii.annotations == (occurrences[1],)
    assert source.annotations == temperature.annotations == ()


def test_source_selection_resolves_signal_names():
    lead_i = _signal("lead-i", "I")
    lead_ii = _signal("lead-ii", "II")
    source = Source(id="ecg", name="ECG", signals=(lead_i, lead_ii))
    annotation = Annotation(key="quality", value="reviewed")

    occurrences = source.select(signal_names=("II", "I")).annotate(annotation)

    assert [occurrence.content_id for occurrence in occurrences] == [annotation.content_id] * 2
    assert lead_ii.annotations == (occurrences[0],)
    assert lead_i.annotations == (occurrences[1],)


@pytest.mark.parametrize("owner", ["signal", "source", "record"])
def test_constructor_and_attachment_reject_time_annotations_without_a_timeline(owner):
    annotation = Annotation(key="event", span=TimePoint.micros(500))
    signal = Signal(name="ordered", data=[1.0], spec=SPEC, time_axis=OrdinalAxis())
    source = Source(name="ordered", signals=(signal,))

    def build(annotations):
        if owner == "signal":
            return Signal(name="ordered", data=[1.0], spec=SPEC, time_axis=OrdinalAxis(), annotations=annotations)
        if owner == "source":
            return Source(name="ordered", signals=(signal,), annotations=annotations)
        return Record(sources=(source,), annotations=annotations)

    with pytest.raises(TimeFValidationError, match="no timeline"):
        build((annotation,))
    with pytest.raises(TimeFValidationError, match="no timeline"):
        build(()).annotate(annotation)
