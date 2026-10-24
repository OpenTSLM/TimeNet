from pydantic import ValidationError
import pytest

from timenet.dataset import OrdinalAxis, RegularAxis, Signal
from timenet.errors import TimeFFormatError
from timenet.format.task_codec import (
    decode_span,
    decode_target,
    encode_span,
    encode_task_payload,
)
from timenet.format.task_target_kind import TaskTargetKind
from timenet.format.task_target_row import TargetValueRow, TaskTargetRow
from timenet.types import (
    AnswerTask,
    StepInterval,
    TemporalLocalizationTask,
    TimePoint,
    TimeSeriesSpec,
    ureg,
)


def test_task_payload_keeps_targets_out_of_configuration_columns():
    task = AnswerTask(prompt="Alive?", targets=("Yes", 1))

    assert encode_task_payload(task) == {"target_schema": None, "unit": None, "target_name": None, "mode": None}


def _signals():
    spec = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)
    timed = Signal.from_values([0.0] * 4, spec=spec, name="timed", time_axis=RegularAxis.from_rate_hz(1))
    ordinal = Signal.from_values([0.0] * 8, spec=spec, name="ordinal", time_axis=OrdinalAxis())
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
            {"target_kind": "text", "text_value": "yes", "signal_id": "ecg"},
            "does not allow signal_id",
        ),
        ({"target_kind": "step_point", "span_start": 0}, "requires exactly one span Signal"),
        (
            {"target_kind": "step_point", "span_start": 0, "span_signal_ids": ("ecg", "eeg")},
            "requires exactly one span Signal",
        ),
    ],
)
def test_target_rows_reject_invalid_layouts_for_writes_and_reads(row, message):
    with pytest.raises(ValidationError, match=message):
        TargetValueRow.model_validate(row)
    with pytest.raises(TimeFFormatError, match=message):
        decode_target(row, records={}, signals={})


def test_stored_signal_target_requires_a_signal_key():
    with pytest.raises(ValidationError, match="requires signal_key"):
        TaskTargetRow(task_key=1, position=0, target_kind=TaskTargetKind.SIGNAL)


def test_target_decoder_rejects_a_missing_record_reference():
    row = TargetValueRow(target_kind=TaskTargetKind.RECORD, record_id="missing")
    with pytest.raises(TimeFFormatError, match="missing Record 'missing'"):
        decode_target(row, records={}, signals={})
