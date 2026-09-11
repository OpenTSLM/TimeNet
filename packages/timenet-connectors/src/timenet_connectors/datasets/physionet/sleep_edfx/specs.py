"""What each signal of the release measures, and what its own header declares about it.

Each study gets its own table. Three signals are the same in both, and each study holds signals
the other does not. Both name a submental EMG, but the cassette one is not the same measurement
as the telemetry one.

TimeF takes the unit of a signal from these tables and not from the EDF header, because the
header states no dimension on some files and states something that is not a unit on others.
"""

from pathlib import Path

from timenet.errors import TimeFFormatError
from timenet.types import DataSource, TimeSeriesSpec, ureg
from timenet_connectors.bases.edf import reader


_SOURCE = DataSource(data_source_type="physionet", name="Sleep-EDF Expanded", provider="PhysioNet")

_EEG = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.microvolt, data_source=_SOURCE)
_EOG = TimeSeriesSpec(spec_type="eog", name="EOG", unit_value=ureg.microvolt, data_source=_SOURCE)
_EMG = TimeSeriesSpec(spec_type="emg", name="EMG", unit_value=ureg.microvolt, data_source=_SOURCE)

# The cassette recorder rectified its submental EMG and low-passed the result at 0.7 Hz, so a
# cassette value is an amplitude and not a potential. The telemetry recorder did neither.
_EMG_ENVELOPE = TimeSeriesSpec(
    spec_type="emg_envelope", name="EMG envelope", unit_value=ureg.microvolt, data_source=_SOURCE
)

_RESPIRATION = TimeSeriesSpec(
    spec_type="respiration", name="Oro-nasal airflow", unit_value=ureg.dimensionless, data_source=_SOURCE
)
_MARKER = TimeSeriesSpec(spec_type="marker", name="Event marker", unit_value=ureg.dimensionless, data_source=_SOURCE)
_TEMPERATURE = TimeSeriesSpec(
    spec_type="temperature", name="Rectal temperature", unit_value=ureg.degC, data_source=_SOURCE
)

_SHARED = {
    "EEG Fpz-Cz": _EEG,
    "EEG Pz-Oz": _EEG,
    "EOG horizontal": _EOG,
}

CASSETTE_SPECS = _SHARED | {
    "EMG submental": _EMG_ENVELOPE,
    "Resp oro-nasal": _RESPIRATION,
    "Temp rectal": _TEMPERATURE,
    "Event marker": _MARKER,
}

TELEMETRY_SPECS = _SHARED | {
    "EMG submental": _EMG,
    "Marker": _MARKER,
}

# The physical dimension each signal writes in its own EDF header, over all 197 PSG files of
# the release. A signal name declares the same set in both studies, so one table covers both.
HEADER_DIMENSIONS = {
    "EEG Fpz-Cz": frozenset({"uV"}),
    "EEG Pz-Oz": frozenset({"uV"}),
    "EOG horizontal": frozenset({"uV"}),
    "EMG submental": frozenset({"uV"}),
    "Resp oro-nasal": frozenset({""}),
    "Temp rectal": frozenset({"DegC", ""}),
    "Event marker": frozenset({""}),
    "Marker": frozenset({"ID+M-E"}),
}


def check_header_dimensions(path: Path, header: reader.EdfHeader) -> None:
    """Refuse a file whose header declares a physical dimension the release does not declare.

    A signal the tables do not name is left to the caller, which refuses it with the name of
    the signal rather than with the name of its dimension.

    Args:
        path: The PSG file, for the error message.
        header: Its header, from :func:`reader.open_edf`.

    Raises:
        TimeFFormatError: If a signal declares a dimension the release does not declare for it.
    """
    for signal, dimension in zip(header.signals, header.units, strict=True):
        declared = HEADER_DIMENSIONS.get(signal)
        if declared is not None and dimension not in declared:
            raise TimeFFormatError(
                f"{path}: signal {signal!r} declares the physical dimension {dimension!r}, and this "
                f"release declares {sorted(declared)} for it"
            )
