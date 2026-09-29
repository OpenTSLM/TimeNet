"""One :class:`~timenet.types.TimeSeriesSpec` per kind of signal the evaluation folders ship.

This module reads no file. It holds values and nothing else.

Seven of the eleven folders ship values somebody already normalised, by three methods that do not
agree, and those get a dimensionless spec. Four do not, and those keep the unit their measurements
are in. The README says which folder is which and how it was measured.

A dimensionless spec here means "this release states no unit", not "these numbers have none".
"""

from timenet.types import TimeSeriesSpec, ureg


# The folders store their values as double, so every spec here says so. The default is float32,
# and casting a double down to it would change the numbers the release states.
_DTYPE = "float64"

ACCELERATION_G = TimeSeriesSpec(
    spec_type="slip_eval_acceleration_g",
    name="SLIP-eval Acceleration",
    unit_value=ureg.standard_gravity,
    dtype=_DTYPE,
)
"""Accelerometer values still in g. ``wisdm``'s resultant magnitude has a median of 1.0004 and
``uci_har``'s total-acceleration triple 1.0232, which is gravity."""

ACCELERATION_MS2 = TimeSeriesSpec(
    spec_type="slip_eval_acceleration_ms2",
    name="SLIP-eval Acceleration Deviation",
    unit_value=ureg.meter / ureg.second**2,
    dtype=_DTYPE,
)
"""``AsphaltObstacles``: metres per second squared, with each window's own mean removed. Removing a
mean keeps the unit and loses the zero point, so a value is a deviation and not a magnitude."""

EEG_UV = TimeSeriesSpec(
    spec_type="slip_eval_eeg",
    name="SLIP-eval EEG",
    unit_value=ureg.microvolt,
    dtype=_DTYPE,
)
"""``sleepEDF``: raw microvolts, ranging -211 to 209 with per-channel standard deviations of
21.17 and 12.94. Nothing was rescaled."""

NORMALISED = TimeSeriesSpec(
    spec_type="slip_eval_normalised",
    name="SLIP-eval Normalised Signal",
    unit_value=ureg.dimensionless,
    dtype=_DTYPE,
)
"""Every signal of a folder whose values somebody rescaled. The method differs by folder and none of
them published the constants, so no unit can be recovered."""

BY_FOLDER: dict[str, TimeSeriesSpec] = {
    "AsphaltObstacles": ACCELERATION_MS2,
    "Beijing_AQI": NORMALISED,
    "PPG_CVA": NORMALISED,
    "PPG_DM": NORMALISED,
    "PPG_HTN": NORMALISED,
    "ptbxl": NORMALISED,
    "sleepEDF": EEG_UV,
    "studentlife": NORMALISED,
    "uci_har": ACCELERATION_G,
    "wesad": NORMALISED,
    "wisdm": ACCELERATION_G,
}
"""The spec every signal of a folder uses.

``uci_har`` was measured rather than assumed: the card calls it "Accelerometer + Gyroscope", and all
six of its signals are acceleration in g. Nothing in this release is a rate of turn."""
