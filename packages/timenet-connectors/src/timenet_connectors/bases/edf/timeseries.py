"""Turn the signals of one recording into :class:`~timenet.dataset.Signal`.

The caller passes the signal-to-spec table and the loader factory, thus this module holds
nothing of this dataset and reads no sample. Every other attribute comes from the header.
"""

from collections.abc import Callable, Mapping
from fractions import Fraction

import pyarrow as pa

from timenet.dataset import RegularAxis, Signal
from timenet.errors import TimeFFormatError
from timenet.types import TimeSeriesSpec
from timenet_connectors.bases.edf import reader


def build(
    file: reader.EdfFile,
    specs: Mapping[str, TimeSeriesSpec],
    *,
    loader: Callable[[reader.EdfFile, int], Callable[[], pa.Array]],
    names: Mapping[int, str] | None = None,
) -> tuple[Signal, ...]:
    """Give one time series for each signal that the header of a recording names.

    Args:
        file: Its open PSG file, from :func:`reader.open_edf`.
        specs: The table of the release, from a signal name to its spec.
        loader: Builds the lazy loader of one signal, from the open file and a signal index.
        names: The signals to build, as a header position mapped to the signal name, when a
            release writes channels that hold no measurement or a label more than once. The
            spec still comes from the header label. ``None`` builds every signal under its
            header label.

    Returns:
        One time series for each signal it builds, in the order of the header.

    Raises:
        TimeFFormatError: If the header names a signal that the spec table does not hold.
    """
    header = file.header
    series = []
    if names is None:
        names = dict(enumerate(header.signals))

    for index, name in names.items():
        signal = header.signals[index]
        spec = specs.get(signal)
        if spec is None:
            raise TimeFFormatError(f"{file.path}: the header names a signal this release does not record: {signal!r}")

        samples_per_record = header.samples_per_record[index]
        series.append(
            Signal.from_loader(
                spec=spec,
                name=name,
                # From the header of this file, thus the signals at 1 Hz and the one
                # recording that writes records of 60 s need no special case.
                time_axis=RegularAxis.from_rate_hz(Fraction(samples_per_record) / header.record_duration),
                loader=loader(file, index),
                n_values=header.num_records * samples_per_record,
            )
        )
    return tuple(series)
