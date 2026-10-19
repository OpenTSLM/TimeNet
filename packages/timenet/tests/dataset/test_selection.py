from fractions import Fraction

import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, DatasetMetadata, License, TimePoint, TimeSeriesSpec, Version, ureg


SPEC = TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=ureg.millivolt)


def _signal(name: str) -> Signal:
    return Signal(
        name=name,
        data=[1.0],
        spec=SPEC,
        time_axis=RegularAxis(period_us=Fraction(1_000)),
    )


def _add_to_a_dataset(source: Source) -> None:
    metadata = DatasetMetadata(
        dataset_id="timenet/selection-test",
        dataset_version=Version(1, 0, 0),
        name="Selection test",
        description="test",
        license=License.CC_BY_4_0,
    )
    TimeFDataset(metadata=metadata).add_record(record=Record(sources=(source,)))


def test_source_selection_attaches_shared_timed_content_only_to_selected_signals():
    lead_i = _signal("I")
    lead_ii = _signal("II")
    temperature = _signal("temperature")
    source = Source(name="ECG", signals=(lead_i, lead_ii, temperature))

    occurrences = source.select(signals=(lead_i, lead_ii)).annotate(
        Annotation(key="status", value="off", span=TimePoint.micros(500))
    )
    _add_to_a_dataset(source)

    assert occurrences[0].id is not None
    assert occurrences[0].id == occurrences[1].id
    assert occurrences[0].occurrence_id != occurrences[1].occurrence_id
    assert lead_i.annotations == (occurrences[0],)
    assert lead_ii.annotations == (occurrences[1],)
    assert source.annotations == temperature.annotations == ()


def test_source_selection_resolves_signal_names():
    lead_i = _signal("I")
    lead_ii = _signal("II")
    source = Source(name="ECG", signals=(lead_i, lead_ii))
    annotation = Annotation(key="quality", value="reviewed")

    _add_to_a_dataset(source)

    occurrences = source.select(signal_names=("II", "I")).annotate(annotation)

    assert annotation.id is not None
    assert [occurrence.id for occurrence in occurrences] == [annotation.id] * 2
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
