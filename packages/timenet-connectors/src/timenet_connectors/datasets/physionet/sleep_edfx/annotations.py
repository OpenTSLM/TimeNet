"""Turn one scoring into :class:`~timenet.types.Annotation` objects on its recording's timeline.

A technician scored every 30 s epoch of a recording. The file stores a run of equal epochs as
one entry, not one row for each epoch. This module turns each entry into one annotation.

The caller reads the entries and passes them in. This module reads no file.
``connector.py`` measures the overrun and reports it. It needs that measure for the session
span of the record as well.
"""

from timenet.dataset import Signal
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, TimeInterval
from timenet_connectors.bases.edf import reader
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey


# The scoring followed the 1968 Rechtschaffen and Kales manual, with Fpz-Cz and Pz-Oz in place
# of its EEG derivations. R&K scores from EEG, EOG and EMG together, thus these four signals.
_signal_names_for_annotations = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")


def _signals_of_annotation(recording_id: str, series: tuple[Signal, ...]) -> tuple[Signal, ...]:
    """Give the four scoring signals, in the order they are named.

    Args:
        recording_id: The recording these signals belong to, for the error message.
        series: Its time series.

    Returns:
        One signal for each scoring signal name, in that order.

    Raises:
        TimeFFormatError: If the recording does not hold one of the scoring signals.
    """
    by_signal = {one.name: one for one in series}
    missing = [signal for signal in _signal_names_for_annotations if signal not in by_signal]
    if missing:
        raise TimeFFormatError(f"{recording_id}: does not hold the signals {missing}")

    return tuple(by_signal[signal] for signal in _signal_names_for_annotations)


def measure_overrun_microseconds(entries: tuple[reader.EdfAnnotation, ...], end_microseconds: int) -> int:
    """Give how far a scoring reaches past the last recorded data point.

    A scoring and the signals it annotates are two files, so nothing makes them agree. This
    release pads a scoring to a full day whatever time the recording stopped, thus its last
    entry often ends after the signals do. This measures that and changes nothing.

    Args:
        entries: The scoring, as the file states it.
        end_microseconds: Where the recorded signals stop, from
            :func:`reader.compute_signal_end_microseconds`.

    Returns:
        The overrun in microseconds, or 0 when the scoring fits.
    """
    if not entries:
        return 0

    last = max(entry.onset_microseconds + entry.duration_microseconds for entry in entries)
    return max(0, last - end_microseconds)


def build(entries: tuple[reader.EdfAnnotation, ...]) -> list[Annotation]:
    """Give one annotation for each entry of a scoring, in file order.

    Every entry becomes an annotation, whatever its label, with the onset and the duration the
    file states. An entry that reaches past the last recorded data point warns and is kept.

    Args:
        entries: The scoring, as :func:`reader.read_annotations` gives it.

    Returns:
        One annotation for each entry.
    """
    return [
        Annotation(
            key=AnnotationKey.SLEEP_STAGE,
            value=entry.label,
            span=TimeInterval.micros(
                entry.onset_microseconds,
                entry.onset_microseconds + entry.duration_microseconds,
            ),
        )
        for entry in entries
    ]


def attach(recording_id: str, series: tuple[Signal, ...], stages: list[Annotation]) -> None:
    """Attach each scoring entry to its four scoring Signals with shared content."""
    for signal in _signals_of_annotation(recording_id, series):
        for stage in stages:
            signal.annotate(stage)
