from pydantic import ValidationError
import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.format.task_codec import (
    decode_input_modalities,
    decode_span,
    decode_target,
    encode_span,
    encode_task_payload,
)
from timenet.format.task_target_kind import TaskTargetKind
from timenet.format.task_target_row import TargetValueRow
from timenet.types import (
    AnswerTask,
    DatasetMetadata,
    License,
    StepInterval,
    TemporalLocalizationTask,
    TimePoint,
    TimeSeriesSpec,
    Version,
    ureg,
)


def test_task_payload_keeps_targets_out_of_configuration_columns():
    task = AnswerTask(prompt="Alive?", targets=("Yes", 1))

    assert encode_task_payload(task) == {"target_schema": None, "unit": None, "target_name": None, "mode": None}


def _signals():
    spec = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)
    timed = Signal.from_values([0.0] * 4, spec=spec, name="timed", time_axis=RegularAxis.from_rate_hz(1))
    ordinal = Signal.from_values([0.0] * 8, spec=spec, name="ordinal", time_axis=OrdinalAxis())
    metadata = DatasetMetadata(
        dataset_id="timenet/codec",
        dataset_version=Version(1, 0, 0),
        name="Codec",
        description="test",
        license=License.CC_BY_4_0,
    )
    TimeFDataset(metadata=metadata).add_record(record=Record(sources=(Source(name="s", signals=(timed, ordinal)),)))
    return timed, ordinal


def test_task_span_codec_preserves_frame_shape_and_scope():
    timed, ordinal = _signals()
    spans = (
        TimePoint.seconds(2, time_series=(timed,)),
        StepInterval(time_series=ordinal, start=3, stop=8),
    )
    task = TemporalLocalizationTask(targets=(spans[0],))

    encoded = [encode_span(span) for span in spans]
    assert [item["span_type"] for item in encoded if item] == ["time_point", "step_interval"]
    assert [item["signal_ids"] for item in encoded if item] == [(timed.id,), (ordinal.id,)]
    signals = {timed.id: timed, ordinal.id: ordinal}
    assert tuple(decode_span(*_columns(item), signals=signals) for item in encoded) == spans
    assert encode_task_payload(task)["mode"] == "sparse"


def _columns(encoded):
    return encoded["span_type"], encoded["span_start"], encoded["span_end"], encoded["signal_ids"]


def test_target_codec_keeps_boolean_and_integer_rows_distinct():
    assert TargetValueRow.from_target(True).target_kind == "boolean"
    assert TargetValueRow.from_target(1).target_kind == "integer"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"target_kind": "unknown"}, "target_kind"),
        ({"target_kind": "text"}, "requires text_value"),
        ({"target_kind": "signal"}, "requires signal_id"),
        (
            {"target_kind": "text", "text_value": "yes", "signal_id": 1},
            "does not allow signal_id",
        ),
        ({"target_kind": "step_point", "span_start": 0}, "requires exactly one span Signal"),
        (
            {"target_kind": "step_point", "span_start": 0, "span_signal_ids": (1, 2)},
            "requires exactly one span Signal",
        ),
    ],
)
def test_target_rows_reject_invalid_layouts_for_writes_and_reads(row, message):
    with pytest.raises(ValidationError, match=message):
        TargetValueRow.model_validate(row)
    with pytest.raises(TimeFFormatError, match=message):
        decode_target(row, records={}, signals={})


def test_stored_signal_target_requires_a_signal_id():
    with pytest.raises(ValidationError, match="requires signal_id"):
        TargetValueRow(target_kind=TaskTargetKind.SIGNAL)


def test_target_decoder_rejects_a_missing_record_reference():
    row = TargetValueRow(target_kind=TaskTargetKind.RECORD, record_id=99)
    with pytest.raises(TimeFFormatError, match="missing Record 99"):
        decode_target(row, records={}, signals={})


@pytest.mark.parametrize(
    ("stored", "message"),
    [
        ([], "no stored input modalities"),
        (["time_series", "smell"], "unknown stored input modality"),
        (["text", "text"], "repeated stored input modalities"),
    ],
)
def test_invalid_stored_input_modalities_are_format_errors(stored, message):
    # Each call re-validates: the decoder cache never stores a failure.
    for _ in range(2):
        with pytest.raises(TimeFFormatError, match=message):
            decode_input_modalities(stored)
