"""What each signal of the release measures.

One table covers both studies. They share four of the eight signal names, and the rest are
unique to one study.
"""

from timenet.types import TimeSeriesSpec, ureg


_EEG = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.microvolt)
_EOG = TimeSeriesSpec(spec_type="eog", name="EOG", unit_value=ureg.microvolt)
_EMG = TimeSeriesSpec(spec_type="emg", name="EMG", unit_value=ureg.microvolt)
_RESPIRATION = TimeSeriesSpec(spec_type="respiration", name="Oro-nasal airflow", unit_value=ureg.dimensionless)
_MARKER = TimeSeriesSpec(spec_type="marker", name="Event marker", unit_value=ureg.dimensionless)
_TEMPERATURE = TimeSeriesSpec(spec_type="temperature", name="Rectal temperature", unit_value=ureg.degC)

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
