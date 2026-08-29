"""What each channel of the release measures.

One table covers both studies. They share four of the eight channel names, and the rest are
unique to one study.
"""

from timenet.types import DataSource, TimeSeriesSpec, ureg


_SOURCE = DataSource(data_source_type="physionet", name="Sleep-EDF Expanded", provider="PhysioNet")

_EEG = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.microvolt, data_source=_SOURCE)
_EOG = TimeSeriesSpec(spec_type="eog", name="EOG", unit_value=ureg.microvolt, data_source=_SOURCE)
_EMG = TimeSeriesSpec(spec_type="emg", name="EMG", unit_value=ureg.microvolt, data_source=_SOURCE)
_RESPIRATION = TimeSeriesSpec(
    spec_type="respiration", name="Oro-nasal airflow", unit_value=ureg.dimensionless, data_source=_SOURCE
)
_MARKER = TimeSeriesSpec(spec_type="marker", name="Event marker", unit_value=ureg.dimensionless, data_source=_SOURCE)
_TEMPERATURE = TimeSeriesSpec(
    spec_type="temperature", name="Rectal temperature", unit_value=ureg.degC, data_source=_SOURCE
)

SPECS = {
    "EEG Fpz-Cz": _EEG,
    "EEG Pz-Oz": _EEG,
    "EOG horizontal": _EOG,
    "EMG submental": _EMG,
    "Resp oro-nasal": _RESPIRATION,
    "Temp rectal": _TEMPERATURE,
    "Event marker": _MARKER,  # for cassette study
    "Marker": _MARKER,  # for telemetry study
}
