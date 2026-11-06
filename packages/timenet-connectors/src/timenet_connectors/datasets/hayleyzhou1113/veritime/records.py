"""Build records from checked source descriptors."""

from functools import partial

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import OrdinalAxis, RegularAxis, TimeAxis
from timenet.json import JsonValue
from timenet_connectors.datasets.hayleyzhou1113.veritime.source import Row, load_series
from timenet_connectors.datasets.hayleyzhou1113.veritime.vocabulary import KINDS


def build_record(rows: list[Row]) -> Record:
    """Build the Record of one series, whose signal rereads the first of the rows that publish it.

    Returns:
        The Record, whose values stay in the source file until read.
    """
    first = rows[0]
    kind = KINDS[first.series_key.kind]
    axis: TimeAxis = OrdinalAxis() if kind.period_us is None else RegularAxis(period_us=kind.period_us)
    signal = Signal.from_loader(
        name=kind.signal,
        spec=kind.spec,
        time_axis=axis,
        n_values=first.n_values,
        loader=partial(load_series, first),
    )
    # Scenario rows can publish one series from either split.
    metadata: dict[str, JsonValue] = {
        "kind": first.series_key.kind.value,
        "rows": [{"part": row.location.part.value, "row": row.location.location.line - 1} for row in rows],
    }
    return Record(sources=(Source(name=kind.spec.name, signals=(signal,)),), metadata=metadata)
