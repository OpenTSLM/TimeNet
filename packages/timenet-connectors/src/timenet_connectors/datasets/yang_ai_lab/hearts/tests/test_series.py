"""Check that related HEARTS frames keep their source timing."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.yang_ai_lab.hearts.series import records_for


def test_related_frames_share_clock_and_keep_gap_and_start_offset():
    reference = pd.DataFrame(
        {
            "Timestamp": [
                datetime(2022, 11, 14, 9, 14),
                datetime(2022, 11, 14, 9, 15),
                datetime(2022, 11, 14, 9, 17),
            ],
            "Libre GL": [90.0, 91.0, 93.0],
        }
    )
    window = pd.DataFrame(
        {
            "Timestamp": [datetime(2022, 11, 19, 17, 46), datetime(2022, 11, 19, 17, 47)],
            "Libre GL": [100.0, 101.0],
        }
    )
    (record,) = records_for("cgmacros", Path("case.pkl"), {"reference_cgm_df": reference, "window_df": window}, "case")
    source = record.sources[0]
    frames = {child.name: child for child in source.sources}
    first = frames["reference_cgm_df"].signals[0]
    second = frames["window_df"].signals[0]

    assert isinstance(first.time_axis, IrregularAxis)
    assert first.time_axis.last_us == 180_000_000
    assert isinstance(second.time_axis, RegularAxis)
    window_span = second.span_us
    assert window_span is not None
    assert window_span[0] == 5 * 86_400_000_000 + 8 * 3_600_000_000 + 32 * 60_000_000
    assert record.start_time.timestamp is None


def test_independent_audio_clips_do_not_acquire_alignment():
    records = records_for(
        "vctk",
        Path("case.pkl"),
        {"reference_audio": np.array([0.1, 0.2]), "test_audio": np.array([0.3, 0.4])},
        "case",
    )
    assert len(records) == 2
    assert records[0].start_time is not records[1].start_time
    assert all(record.start_time.timestamp is None for record in records)
    assert [record.signals[0].name for record in records] == ["reference_audio", "test_audio"]


def test_missing_timestamp_is_not_converted_to_a_relative_zero():
    frame = pd.DataFrame({"Timestamp": [pd.NaT], "Libre GL": [90.0]})
    with pytest.raises(TimeFFormatError, match="missing timestamps"):
        records_for("cgmacros", Path("case.pkl"), {"window_df": frame}, "case")
