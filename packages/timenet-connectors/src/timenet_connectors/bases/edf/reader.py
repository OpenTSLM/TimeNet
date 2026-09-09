"""Read the EDF container: the header of a file, and one data record from it.

EDF holds an ASCII header and then data records of 16-bit integers. One record holds the
samples of every signal for one stretch of time, one signal after the other. Signals of
different rates therefore hold a different count of samples in the same record.

This module parses no bytes. ``edfio`` does that. A connector that reads EDF declares
``edfio`` in its ``requirements.txt``.

:class:`EdfHeader` and :class:`EdfFile` are the boundary: ``edfio`` types stay inside this
file, thus a later change of library does not reach the modules that call it.
"""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple
import warnings

import edfio
import numpy as np
import pyarrow as pa

from timenet.errors import TimeFFormatError
from timenet.types import US_PER_S


# edfio repairs a short file instead of refusing it: it warns, lowers the record count of the
# header it gives back, and reads on. A caller of a lazy loader never sees that warning, thus
# this module matches the text of it. edfio writes one of these two for a file that ends early.
_SHORT_FILE_WARNINGS = ("but file contains", "Data was truncated")


class EdfHeader(NamedTuple):
    """The header of one EDF file. Each tuple below holds one entry for each signal."""

    # A local wall clock with no timezone, so it is not the Unix anchor Record.start_time wants.
    start_time: datetime
    # EDF calls this the "local patient identification". A release can anonymize it, and a
    # connector then takes the subject id from elsewhere, such as the filename.
    patient_id: str
    num_records: int
    record_duration: Fraction
    signals: tuple[str, ...]
    units: tuple[str, ...]
    # Free text such as "HP:16Hz Rectification LP:0.7Hz". Two signals of one modality can hold
    # different quantities, and only this field says so.
    prefiltering: tuple[str, ...]
    samples_per_record: tuple[int, ...]


class EdfFile(NamedTuple):
    """An open EDF file: its header, and the ``edfio.Edf`` handle behind it."""

    path: Path
    header: EdfHeader
    handle: Any


class EdfAnnotation(NamedTuple):
    """One annotation of an EDF+ file: a labelled stretch of the recording timeline."""

    onset_microseconds: int
    duration_microseconds: int
    label: str


def open_edf(path: Path) -> EdfFile:
    """Open an EDF file, read its header, and map its records.

    Reading is lazy: a caller that wants the header alone reads no signal bytes. Open a file
    one time and pass the result to :func:`read_signal`, so its header is parsed once.

    Args:
        path: The ``.edf`` file.

    Returns:
        The open file, with its header.

    Raises:
        TimeFFormatError: If the file is not readable as EDF, if it is shorter than the
            records that its own header states, if it is discontinuous, or if it states no
            start date.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            edf = edfio.read_edf(path, lazy_load_data=True)
        # edfio has no single exception type for a bad file, so catch broadly and re-raise.
        except Exception as exc:
            raise TimeFFormatError(f"{path}: cannot read this file as EDF") from exc

    _raise_on_a_short_file(path, caught)
    return EdfFile(path=path, header=_extract_header(edf, path), handle=edf)


def _raise_on_a_short_file(path: Path, caught: list[warnings.WarningMessage]) -> None:
    """Refuse a file that ends before its header says, and pass every other warning on.

    Reading held the warnings of ``edfio`` back, thus this raises on the one that reports a
    short file and raises every other one again. A release states a header field twice and
    the two can disagree, and a caller wants to hear about that.

    Args:
        path: The file that was read.
        caught: The warnings that reading it raised.

    Raises:
        TimeFFormatError: If ``edfio`` found fewer data records than the header states.
    """
    for entry in caught:
        text = str(entry.message)
        if any(warning in text for warning in _SHORT_FILE_WARNINGS):
            raise TimeFFormatError(f"{path}: ends before its own header says. {text}")

        warnings.warn_explicit(entry.message, entry.category, entry.filename, entry.lineno)


def _extract_header(edf: Any, path: Path) -> EdfHeader:
    """Read the header fields out of an open file.

    Args:
        edf: The ``edfio.Edf``.
        path: Its path, for the error message.

    Returns:
        The fields of its header.

    Raises:
        TimeFFormatError: If the file is a discontinuous EDF+D, or if its start date is
            anonymized.
    """
    # An EDF+D file writes its records on a broken timeline. A gap between two records holds
    # no samples, and no header field states how long it is, thus every span this module
    # measures would be wrong. Refuse such a file rather than time it wrongly.
    if edf.reserved.startswith("EDF+D"):
        raise TimeFFormatError(f"{path}: is a discontinuous EDF+D file, which this reader does not read")

    try:
        start_time = edf.startdatetime
    # A release can replace the start date with an X, and edfio then raises instead of guessing.
    except edfio.AnonymizedDateError as exc:
        raise TimeFFormatError(f"{path}: states no start date, because its header is anonymized") from exc

    signals = edf.signals
    samples_per_record = tuple(int(signal.samples_per_data_record) for signal in signals)
    return EdfHeader(
        start_time=start_time,
        patient_id=edf.local_patient_identification,
        num_records=int(edf.num_data_records),
        record_duration=Fraction(str(edf.data_record_duration)),
        signals=tuple(signal.label for signal in signals),
        units=tuple(signal.physical_dimension for signal in signals),
        prefiltering=tuple(signal.prefiltering for signal in signals),
        samples_per_record=samples_per_record,
    )


def convert_digital_to_physical(
    digital: np.ndarray,
    *,
    digital_min: float,
    digital_max: float,
    physical_min: float,
    physical_max: float,
) -> np.ndarray:
    """Convert stored integers into the physical unit that the header names.

    EDF stores no volts. It stores counts, and the header of each file states which range of
    physical values those counts cover::

        gain = (physical_max - physical_min) / (digital_max - digital_min)
        physical = (digital - digital_min) * gain + physical_min

    The four values must come from the header of the file that holds these counts.

    Args:
        digital: The stored counts.
        digital_min: The smallest count that the signal writes.
        digital_max: The largest count that the signal writes.
        physical_min: The value that ``digital_min`` means.
        physical_max: The value that ``digital_max`` means.

    Returns:
        The values in the physical unit of the signal, as float32.

    Raises:
        TimeFFormatError: If the digital range is empty, because it then gives no scale.
    """
    span = digital_max - digital_min
    if span == 0:
        raise TimeFFormatError(f"an EDF signal with a digital range of {digital_min} to {digital_max} has no scale")

    gain = (physical_max - physical_min) / span
    return (np.asarray(digital, dtype="float32") - digital_min) * gain + physical_min


def _convert_signal(signal: Any, digital: np.ndarray) -> np.ndarray:
    """Convert counts of one signal with the physical range its own header states.

    Returns:
        The values in the physical unit of the signal.
    """
    return convert_digital_to_physical(
        digital,
        digital_min=signal.digital_min,
        digital_max=signal.digital_max,
        physical_min=signal.physical_min,
        physical_max=signal.physical_max,
    )


def build_signal_loader(file: EdfFile, index: int) -> Callable[[], pa.Array]:
    """Build the lazy loader of one signal.

    The loader holds the open file, thus the signals of one recording share one open file and
    one memory map. A build keeps every file it opened open until the writer has called the
    loaders.

    Args:
        file: The open file, from :func:`open_edf`.
        index: The signal, in the order of the header.

    Returns:
        A loader that takes no argument and gives the signal in physical units.
    """

    def load() -> pa.Array:
        return pa.array(read_signal(file, index))

    return load


def read_signal(file: EdfFile, index: int) -> np.ndarray:
    """Read one whole signal, from its first sample to its last.

    Reading one signal does not read the others.

    Args:
        file: The open file, from :func:`open_edf`.
        index: The signal, counted from zero, in the order of the header.

    Returns:
        Every sample of that signal, in physical units.

    Raises:
        TimeFFormatError: If the file holds no signal with this index.
    """
    if not 0 <= index < len(file.header.signals):
        raise TimeFFormatError(
            f"{file.path}: holds {len(file.header.signals)} signals, thus signal {index} does not exist"
        )

    signal = file.handle.signals[index]
    return _convert_signal(signal, signal.digital)


def read_record(file: EdfFile, index: int) -> tuple[np.ndarray, ...]:
    """Read one data record, and give one array for each signal.

    The arrays have different lengths when the signals have different rates.

    Reading one record reads one record. It does not read the signals it slices, thus a
    caller that walks a recording record by record never holds the whole recording.

    Args:
        file: The open file, from :func:`open_edf`.
        index: The record to read, counted from zero.

    Returns:
        One array for each signal, in the order of the header, in physical units.

    Raises:
        TimeFFormatError: If the file holds no record with this index.
    """
    if not 0 <= index < file.header.num_records:
        raise TimeFFormatError(
            f"{file.path}: holds {file.header.num_records} records, thus record {index} does not exist"
        )

    # A record is named by the stretch of seconds it covers. ``signal.digital`` would read the
    # whole signal and hold it, so ask for those seconds and let the map give the rest back.
    start_second = float(index * file.header.record_duration)
    stop_second = float((index + 1) * file.header.record_duration)
    return tuple(
        _convert_signal(signal, signal.get_digital_slice(start_second, stop_second)) for signal in file.handle.signals
    )


def compute_signal_end_microseconds(header: EdfHeader) -> int:
    """Give where the recorded signals stop, in microseconds from the first sample.

    The timeline starts at zero, so this is both the length of the signals and the exclusive
    end of the timeline. An annotation is measured against this bound.

    Every record of the file follows the one before it with no gap, because :func:`open_edf`
    refuses a discontinuous file. The records therefore multiply out to the whole timeline.

    Args:
        header: The header of a signal file.

    Returns:
        ``num_records * record_duration`` as whole microseconds.

    Raises:
        TimeFFormatError: If that product is not a whole number of microseconds.
    """
    total = header.num_records * header.record_duration * US_PER_S
    if total.denominator != 1:
        raise TimeFFormatError(f"an EDF duration of {total} us is not a whole number of microseconds")
    return int(total)


def read_annotations(file: EdfFile) -> tuple[EdfAnnotation, ...]:
    """Read every annotation of an EDF+ file, in file order.

    An EDF+ file states an onset in seconds as text. Parsing that text with
    :class:`~decimal.Decimal` keeps the onset the file states; a float would move it.

    Each data record opens with an empty-text annotation that only timestamps the record.
    ``if annotation.text`` drops those.

    Args:
        file: The open file that holds the annotations, from :func:`open_edf`. An EDF+ file
            carries its own annotations, thus a caller that already holds the recording reads
            them from it. A scoring that lives in a file of its own is opened first.

    Returns:
        One :class:`EdfAnnotation` for each labelled entry, in microseconds.

    Raises:
        TimeFFormatError: If an onset or a duration is not a whole number of microseconds.
    """  # noqa: DOC502 (raised by _seconds_to_microseconds, not directly here)
    return tuple(
        EdfAnnotation(
            onset_microseconds=_seconds_to_microseconds(annotation.onset),
            duration_microseconds=_seconds_to_microseconds(annotation.duration or 0),
            label=annotation.text,
        )
        for annotation in file.handle.annotations
        if annotation.text
    )


def _seconds_to_microseconds(seconds: float) -> int:
    """Convert seconds, as ``edfio`` states them, into whole microseconds.

    Returns:
        The value in microseconds.

    Raises:
        TimeFFormatError: If the value is not a whole number of microseconds.
    """
    total = Decimal(str(seconds)) * US_PER_S
    if total != total.to_integral_value():
        raise TimeFFormatError(f"an EDF+ annotation time of {seconds} s is not a whole number of microseconds")
    return int(total)
