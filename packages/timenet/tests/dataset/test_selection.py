from fractions import Fraction

from timenet.dataset import RegularAxis, Signal, Source
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


def test_source_selection_scopes_one_timed_annotation_to_signal_objects():
    lead_i = _signal("lead-i", "I")
    lead_ii = _signal("lead-ii", "II")
    source = Source(id="ecg", name="ECG", signals=(lead_i, lead_ii))

    (occurrence,) = source.select(signals=(lead_i, lead_ii)).annotate(
        Annotation(key="status", value="off", span=TimePoint.seconds(5))
    )

    assert occurrence.span is not None
    assert occurrence.span.time_series_ids == ("lead-i", "lead-ii")
    assert source.annotations == (occurrence,)


def test_source_selection_resolves_signal_names():
    lead_i = _signal("lead-i", "I")
    lead_ii = _signal("lead-ii", "II")
    source = Source(id="ecg", name="ECG", signals=(lead_i, lead_ii))
    annotation = Annotation(key="quality", value="reviewed")

    occurrences = source.select(signal_names=("II", "I")).annotate(annotation)

    assert [occurrence.content_id for occurrence in occurrences] == [annotation.content_id] * 2
    assert lead_ii.annotations == (occurrences[0],)
    assert lead_i.annotations == (occurrences[1],)
