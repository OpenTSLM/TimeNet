"""Read the files of the release: the EDF container, and the subject tables beside it.

EDF holds an ASCII header and then data records of 16-bit integers. One record holds the
samples of every signal for one stretch of time, one signal after the other. Signals of
different rates therefore hold a different count of samples in the same record.

A subject table is a legacy ``.xls`` workbook. :func:`read_table_rows` gives its rows and reads
no meaning into one. It is the only function of the connector that opens a workbook, thus every
function that decodes a row takes rows and never a path.

This module parses no bytes. ``edfio`` and ``xlrd`` do that. This connector declares both in its
``requirements.txt``.

:class:`EdfHeader` and :class:`EdfFile` are the boundary: ``edfio`` types stay inside this
file, thus a later change of library does not reach the modules that call it.
"""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import xlrd

from timenet.errors import TimeFFormatError
from timenet.types import US_PER_S


_BYTES_PER_SAMPLE = 2  # EDF writes every sample as a little-endian signed 16-bit integer.


class EdfHeader(NamedTuple):
    """The header of one EDF file. Each tuple below holds one entry for each channel."""

    # A local wall clock with no timezone, so it is not the Unix anchor Sample.start_time wants.
    start_time: datetime
    # EDF calls this the "local patient identification". It is anonymous in this release, thus
    # the subject id of a sample comes from the filename instead.
    patient_id: str
    num_records: int
    record_duration: Fraction
    channels: tuple[str, ...]
    units: tuple[str, ...]
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
    one time and pass the result to :func:`read_channel` and :func:`read_record`, so its
    header is parsed once.

    Args:
        path: The ``.edf`` file.

    Returns:
        The open file, with its header.

    Raises:
        TimeFFormatError: If the file is not readable as EDF, or if it is shorter than the
            records that its own header states.
    """
    import edfio  # noqa: PLC0415

    try:
        edf = edfio.read_edf(path, lazy_load_data=True)
    # edfio has no single exception type for a bad file, so catch broadly and re-raise.
    except Exception as exc:
        raise TimeFFormatError(f"{path}: cannot read this file as EDF") from exc

    return EdfFile(path=path, header=_extract_header(edf, path), handle=edf)


def _extract_header(edf: Any, path: Path) -> EdfHeader:
    """Read the header fields out of an open file.

    Args:
        edf: The ``edfio.Edf``.
        path: Its path, for the error message.

    Returns:
        The fields of its header.

    Raises:
        TimeFFormatError: If the file is shorter than the records that its own header states.
    """
    signals = edf.signals
    samples_per_record = tuple(int(signal.samples_per_data_record) for signal in signals)
    # edfio only warns on a truncated file and repairs the record count, so raise instead.
    num_sample_bytes = edf.num_data_records * sum(samples_per_record) * _BYTES_PER_SAMPLE
    expected_num_bytes = edf.bytes_in_header_record + num_sample_bytes
    actual_num_bytes = path.stat().st_size
    if actual_num_bytes < expected_num_bytes:
        raise TimeFFormatError(
            f"{path}: the header states {edf.num_data_records} records, which need "
            f"{expected_num_bytes} bytes, but the file holds {actual_num_bytes}"
        )

    return EdfHeader(
        start_time=edf.startdatetime,
        patient_id=edf.local_patient_identification,
        num_records=int(edf.num_data_records),
        record_duration=Fraction(str(edf.data_record_duration)),
        channels=tuple(signal.label for signal in signals),
        units=tuple(signal.physical_dimension for signal in signals),
        samples_per_record=samples_per_record,
    )


def read_header(path: Path) -> EdfHeader:
    """Read the header of an EDF file, and no sample of it.

    Returns:
        The fields of its header.
    """
    return open_edf(path).header


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


def build_channel_loader(file: EdfFile, index: int) -> Callable[[], pa.Array]:
    """Build the lazy loader of one channel.

    The loader holds the open file, thus the channels of one recording share one open file and
    one memory map. A build keeps every file it opened open until the writer has called the
    loaders.

    Args:
        file: The open file, from :func:`open_edf`.
        index: The channel, in the order of the header.

    Returns:
        A loader that takes no argument and gives the channel in physical units.
    """

    def load() -> pa.Array:
        return pa.array(read_channel(file, index))

    return load


def read_channel(file: EdfFile, index: int) -> np.ndarray:
    """Read one whole channel, from its first sample to its last.

    Reading one channel does not read the others.

    Args:
        file: The open file, from :func:`open_edf`.
        index: The channel, counted from zero, in the order of the header.

    Returns:
        Every sample of that channel, in physical units.

    Raises:
        TimeFFormatError: If the file holds no channel with this index.
    """
    if not 0 <= index < len(file.header.channels):
        raise TimeFFormatError(
            f"{file.path}: holds {len(file.header.channels)} channels, thus channel {index} does not exist"
        )

    signal = file.handle.signals[index]
    return _convert_signal(signal, signal.digital)


def read_record(file: EdfFile, index: int) -> tuple[np.ndarray, ...]:
    """Read one data record, and give one array for each signal.

    The arrays have different lengths when the signals have different rates.

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

    return tuple(
        _convert_signal(signal, signal.digital[index * count : (index + 1) * count])
        for signal, count in zip(file.handle.signals, file.header.samples_per_record, strict=True)
    )


def compute_signal_end_microseconds(header: EdfHeader) -> int:
    """Give where the recorded signals stop, in microseconds from the first sample.

    The timeline starts at zero, so this is both the length of the signals and the exclusive
    end of the timeline. An annotation is measured against this bound.

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


def read_annotations(path: Path) -> tuple[EdfAnnotation, ...]:
    """Read every annotation of an EDF+ file, in file order.

    An EDF+ file states an onset in seconds as text. Parsing that text with
    :class:`~decimal.Decimal` keeps the onset the file states; a float would move it.

    Each data record opens with an empty-text annotation that only timestamps the record.
    ``if annotation.text`` drops those.

    Args:
        path: The ``*-Hypnogram.edf`` file.

    Returns:
        One :class:`EdfAnnotation` for each labelled entry, in microseconds.

    Raises:
        TimeFFormatError: If the file is not readable as EDF, or if an onset or a duration is
            not a whole number of microseconds.
    """  # noqa: DOC502 (raised by open_edf and _seconds_to_microseconds, not directly here)
    return tuple(
        EdfAnnotation(
            onset_microseconds=_seconds_to_microseconds(annotation.onset),
            duration_microseconds=_seconds_to_microseconds(annotation.duration or 0),
            label=annotation.text,
        )
        for annotation in open_edf(path).handle.annotations
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


def measure_overrun_microseconds(annotations: tuple[EdfAnnotation, ...], end_microseconds: int) -> int:
    """Give how far a set of annotations reaches past the last recorded sample.

    An EDF+ file and the signals it annotates are two files, so nothing makes them agree. A
    scoring often ends after its signals stop, because its last entry pads the file to a full
    day whatever time the recording stopped. This measures that and changes nothing.

    Args:
        annotations: The annotations, as the file states them.
        end_microseconds: Where the recorded signals stop, from :func:`compute_signal_end_microseconds`.

    Returns:
        The overrun in microseconds, or 0 when they fit.
    """
    if not annotations:
        return 0
    last = max(annotation.onset_microseconds + annotation.duration_microseconds for annotation in annotations)
    return max(0, last - end_microseconds)


def read_table_rows(path: Path) -> list[tuple[object, ...]]:
    """Open a subject table and give every row of its data sheet, header rows included.

    This gives the cells as the workbook states them and decodes nothing.
    :mod:`~timenet_connectors.datasets.physionet.sleep_edfx.tables` states what a column means.

    Both workbooks carry three sheets, and only the first holds data. This takes the first sheet
    and searches for no sheet by name.

    Args:
        path: The ``.xls`` workbook.

    Returns:
        One tuple of cell values for each row of the first sheet, in sheet order.

    Raises:
        TimeFFormatError: If the file does not open as a workbook, or holds no data sheet.
    """
    try:
        # These are legacy BIFF8 workbooks that Excel wrote. openpyxl reads only the ZIP-based
        # .xlsx format and cannot open them at all.
        book = xlrd.open_workbook(path)
    # xlrd has no single exception type for a bad file, so catch broadly and re-raise.
    except Exception as exc:
        raise TimeFFormatError(f"{path}: cannot read this file as a workbook") from exc

    if book.nsheets == 0:
        raise TimeFFormatError(f"{path}: holds no data sheet")

    sheet = book.sheet_by_index(0)
    return [tuple(sheet.row_values(index)) for index in range(sheet.nrows)]
