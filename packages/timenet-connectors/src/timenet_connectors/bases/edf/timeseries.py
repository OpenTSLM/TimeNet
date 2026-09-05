"""Turn the signals of one recording into :class:`~timenet.dataset.TimeSeries`.

The caller passes the signal-to-spec table and the loader factory, thus this module holds
nothing of this dataset and reads no record. Every other attribute comes from the header.
"""

from collections.abc import Callable, Mapping
from fractions import Fraction

import pyarrow as pa

from timenet.dataset import TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import TimeSeriesSpec
from timenet_connectors.bases.edf import reader


def build(
    record_id: str,
    file: reader.EdfFile,
    specs: Mapping[str, TimeSeriesSpec],
    *,
    loader: Callable[[reader.EdfFile, int], Callable[[], pa.Array]],
) -> tuple[TimeSeries, ...]:
    """Give one time series for each signal that the header of a recording names.

    Args:
        record_id: The id of the record these signals belong to, from ``connector.py``.
            ``source_id`` and each ``time_series_id`` are built from it.
        file: Its open PSG file, from :func:`reader.open_edf`.
        specs: The table of the release, from a signal name to its spec.
        loader: Builds the lazy loader of one signal, from the open file and a signal index.

    Returns:
        One time series for each signal, in the order of the header.

    Raises:
        TimeFFormatError: If the header names a signal that the spec table does not hold.
    """
    header = file.header
    series = []
    for index, signal in enumerate(header.signals):
        spec = specs.get(signal)
        if spec is None:
            raise TimeFFormatError(f"{file.path}: the header names a signal this release does not record: {signal!r}")

        samples_per_record = header.samples_per_record[index]
        series.append(
            TimeSeries(
                spec=spec,
                signal=signal,
                # From the header of this file, thus the signals at 1 Hz and the one
                # recording that writes records of 60 s need no special case.
                time_axis=RegularAxis.from_rate_hz(Fraction(samples_per_record) / header.record_duration),
                loader=loader(file, index),
                source_id=record_id,
                time_series_id=f"{record_id}-{signal}",
                n_values=header.num_records * samples_per_record,
            )
        )
    return tuple(series)
