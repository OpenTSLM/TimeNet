"""Check that related HEARTS frames keep their source timing."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from timenet.dataset import Record
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet_connectors.datasets.yang_ai_lab.hearts.series import source_for


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
    source = source_for("cgmacros", Path("case.pkl"), {"reference_df": reference, "window_df": window}, "case")
    record = Record(sources=(source,))
    frames = {child.name: child for child in source.sources}
    first = frames["reference_df"].signals[0]
    second = frames["window_df"].signals[0]

    assert frames["reference_df"].start_time is frames["window_df"].start_time
    assert frames["reference_df"].start_time.timestamp is None
    assert isinstance(first.time_axis, IrregularAxis)
    assert first.time_axis.last_us == 180_000_000
    assert isinstance(second.time_axis, RegularAxis)
    window_span = second.span_us
    assert window_span is not None
    assert window_span[0] == 5 * 86_400_000_000 + 8 * 3_600_000_000 + 32 * 60_000_000
    assert record.start_time is None
