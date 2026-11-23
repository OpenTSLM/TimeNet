"""What each column of a HAI data file measures, from the "HAI data points" table of the manual.

The manual is ``hai_dataset_technical_details.pdf`` at the pinned commit. Its table on pages 12 to
15 gives each point's name, unit and description. Each point gets one spec, named after the
quantity that the manual describes, with the manual's unit. Two points that the manual describes
in the same words and units share one spec (``P1_FT01`` and ``P1_FT03``, ``P2_VTR03`` and
``P2_VTR04``).

A column the manual names in brackets under an older name, such as ``P3_LT01`` for ``P3_LIT01``,
gets the spec of that point. The manual also states that the 23.05 column of DCS block 1001.15
duplicates ``P1_B2016``. The manual prints ``-`` as the unit of some points; their unit stays
unset, as does the unit of every column that the manual does not list. ``Boolean`` points are
dimensionless.
"""

from dataclasses import dataclass
import re

from timenet.types import TimeSeriesSpec, ureg


_UNITS: dict[str, str | None] = {
    "bar": "bar",
    "mm": "millimeter",
    "l/h": "liter / hour",
    "mmH2O": "mmH2O",
    "degC": "degC",
    "%": "percent",
    "Boolean": "dimensionless",
    # The manual prints the quantity, "Voltage", for the 24 V supply of the DCS.
    "V": "volt",
    "RPM": "revolutions_per_minute",
    "um": "micrometer",
    "rad/s": "radian / second",
    "mHz": "millihertz",
    "Hz": "hertz",
    "MW": "megawatt",
    "-": None,
}
"""The pint unit of each unit the manual prints. ``-`` means the manual states none."""


@dataclass(frozen=True)
class Point:
    """One row of the manual's table of HAI data points."""

    name: str
    unit: str
    """The unit as the manual prints it: a key of the unit table, ``-`` for none."""
    description: str

    @property
    def spec(self) -> TimeSeriesSpec:
        """The spec of the point: its quantity, as the manual describes it, and its unit."""
        unit = _UNITS[self.unit]
        return TimeSeriesSpec(
            spec_type=_spec_type(self.description),
            name=self.description,
            unit_value=None if unit is None else ureg.Unit(unit),
            dtype="float64",
        )


def _spec_type(description: str) -> str:
    """Return the snake-case spec type of a description, such as ``heat_exchanger_outlet_pressure``.

    Returns:
        The spec type.
    """
    return re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")


# The table as printed, in its order, with the manual's wording. "degC" stands for the manual's
# degree sign and "um" for its micrometre sign.
MANUAL_POINTS = (
    Point("P1_B2004", "bar", "Heat-exchanger outlet pressure setpoint"),
    Point("P1_B2016", "bar", "Pressure demand for thermal power output control"),
    Point("P1_B3004", "mm", "Water level setpoint (return water tank)"),
    Point("P1_B3005", "l/h", "Discharge flowrate setpoint (return water tank)"),
    Point("P1_B4002", "degC", "Heat-exchanger outlet temperature setpoint"),
    Point("P1_B4005", "%", "Temperature PID control output"),
    Point("P1_B400B", "l/h", "Water outflow rate setpoint (heating water tank)"),
    Point("P1_B4022", "degC", "Temperature demand for thermal power output control"),
    Point("P1_FCV01D", "%", "Position command for the FCV01 valve"),
    Point("P1_FCV01Z", "%", "Current position of the FCV01 valve"),
    Point("P1_FCV02D", "%", "Position command for the FCV02 valve"),
    Point("P1_FCV02Z", "%", "Current position of the FCV02 valve"),
    Point("P1_FCV03D", "%", "Position command for the FCV03 valve"),
    Point("P1_FCV03Z", "%", "Current position of the FCV03 valve"),
    Point("P1_FT01", "mmH2O", "Measured flowrate of the return water tank"),
    Point("P1_FT01Z", "l/h", "Water inflow rate converted from P1_FT01"),
    Point("P1_FT02", "mmH2O", "Measured flowrate of heating water tank"),
    Point("P1_FT02Z", "l/h", "Water outflow rate conversion from P1_FT02"),
    Point("P1_FT03", "mmH2O", "Measured flowrate of the return water tank"),
    Point("P1_FT03Z", "l/h", "Water outflow rate converted from P1_FT03"),
    Point("P1_LCV01D", "%", "Position command for the LCV01 valve"),
    Point("P1_LCV01Z", "%", "Current position of the LCV01 valve"),
    Point("P1_LIT01", "mm", "Water level of the return water tank"),
    Point("P1_PCV01D", "%", "Position command for the PCV01 valve"),
    Point("P1_PCV01Z", "%", "Current position of the PCV01 valve"),
    Point("P1_PCV02D", "%", "Position command for the PCV2 valve"),
    Point("P1_PCV02Z", "%", "Current position of the PCV02 valve"),
    Point("P1_PIT01", "bar", "Heat-exchanger outlet pressure"),
    Point("P1_PIT01_HH", "bar", "Highest outlet pressure of the heat-exchanger"),
    Point("P1_PIT02", "bar", "Water supply pressure of the heating water pump"),
    Point("P1_PP01AD", "Boolean", "Start command of the main water pump PP01A"),
    Point("P1_PP01AR", "Boolean", "Running state of the main water pump PP01A"),
    Point("P1_PP01BD", "Boolean", "Start command of the main water pump PP01B"),
    Point("P1_PP01BR", "Boolean", "Running state of the main water pump PP01B"),
    Point("P1_PP02D", "Boolean", "Start command of the heating water pump PP02"),
    Point("P1_PP02R", "Boolean", "Running state of the heating water pump PP02"),
    Point("P1_PP04", "%", "Control out of the cooler pump"),
    Point("P1_PP04SP", "degC", "Cooler temperature setpoint"),
    Point("P1_SOL01D", "Boolean", "Open command of the main water tank supply valve"),
    Point("P1_SOL03D", "Boolean", "Open command of the main water tank drain valve"),
    Point("P1_STSP", "Boolean", "Start/stop command of the boiler DCS"),
    Point("P1_TIT01", "degC", "Heat-exchanger outlet temperature"),
    Point("P1_TIT02", "degC", "Temperature of the heating water tank"),
    Point("P1_TIT03", "degC", "Temperature of the main water tank"),
    Point("P2_24Vdc", "V", "DCS 24V Input Voltage"),
    Point("P2_ATSW_Lamp", "Boolean", "Lamp of the Auto SW"),
    Point("P2_AutoGo", "Boolean", "Auto start button"),
    Point("P2_AutoSD", "RPM", "Auto speed demand"),
    Point("P2_Emerg", "Boolean", "Emergency button"),
    Point("P2_MASW", "Boolean", "Manual(1)/Auto(0) SW"),
    Point("P2_MASW_Lamp", "Boolean", "Lamp of Manual SW"),
    Point("P2_ManualGO", "Boolean", "Manual start button"),
    Point("P2_ManualSD", "RPM", "Manual speed demand"),
    Point("P2_OnOff", "Boolean", "On/off switch of the turbine DCS"),
    Point("P2_RTR", "RPM", "RPM trip rate"),
    Point("P2_SCO", "-", "Control output value of the speed controller"),
    Point("P2_SCST", "RPM", "Speed change proportional to frequency change of the STM"),
    Point("P2_SIT01", "RPM", "Current turbine RPM measured by speed probe"),
    Point("P2_TripEx", "Boolean", "Trip emergency exit button"),
    Point("P2_VIBTR01", "um", "Shaft-vibration-related Y-axis displacement near the 1st mass wheel"),
    Point("P2_VIBTR02", "um", "Shaft-vibration-related X-axis displacement near the 1st mass wheel"),
    Point("P2_VIBTR03", "um", "Shaft-vibration-related Y-axis displacement near the 2nd mass wheel"),
    Point("P2_VIBTR04", "um", "Shaft-vibration-related X-axis displacement near the 2nd mass wheel"),
    Point("P2_VT01", "rad/s", "Phase lag signal of the key phasor probe"),
    Point("P2_VTR01", "um", "Preset vibration limit for the sensor P2_VIBTR01"),
    Point("P2_VTR02", "um", "Preset vibration limit for the sensor P2_VIBTR02"),
    Point("P2_VTR03", "um", "Preset vibration limit for the sensor P2_VIBTR03"),
    Point("P2_VTR04", "um", "Preset vibration limit for the sensor P2_VIBTR03"),
    Point("P3_FIT01", "-", "Flow rate of water flowing into the upper water tank"),
    Point("P3_LCP01D", "-", "Speed command for the pump LCP01"),
    Point("P3_LCV01D", "-", "Position command for the valve LCV01"),
    Point("P3_LH01", "%", "High water level set-point"),
    Point("P3_LIT01", "%", "Water level of the upper water tank"),
    Point("P3_LL01", "%", "Low water level set-point"),
    Point("P3_PIT01", "-", "Pressure of water flowing into the upper water tank"),
    Point("P4_HT_FD", "mHz", "Frequency deviation of HTM"),
    Point("P4_HT_LD", "MW", "Electrical load demand of HTM"),
    Point("P4_HT_PO", "MW", "Output power of HTM"),
    Point("P4_HT_PS", "MW", "Scheduled power demand of HTM"),
    Point("P4_LD", "MW", "Total electrical load demand"),
    Point("P4_ST_FD", "Hz", "Frequency deviation of STM"),
    Point("P4_ST_GOV", "-", "Gate opening rate of STM"),
    Point("P4_ST_LD", "MW", "Electrical load demand of STM"),
    Point("P4_ST_PO", "MW", "Output power of STM"),
    Point("P4_ST_PS", "MW", "Scheduled power demand of STM"),
    Point("P4_ST_PT01", "-", "Digital value of steam pressure of STM"),
    Point("P4_ST_TT01", "-", "Digital value of steam temperature of STM"),
)
"""The 87 rows of the manual's table of HAI data points."""

_BY_NAME = {point.name: point for point in MANUAL_POINTS}

OTHER_NAMES = {
    # The manual gives these older names in brackets in the row of the point.
    "P2_Auto": "P2_AutoGo",
    "P2_SD01": "P2_AutoSD",
    "P2_Emgy": "P2_Emerg",
    "P2_On": "P2_OnOff",
    "P2_VYT02": "P2_VIBTR01",
    "P2_VXT02": "P2_VIBTR02",
    "P2_VYT03": "P2_VIBTR03",
    "P2_VXT03": "P2_VIBTR04",
    "P3_LT01": "P3_LIT01",
    # The manual prints P2_AutoGo; the data files spell it P2_AutoGO.
    "P2_AutoGO": "P2_AutoGo",
    # The manual's table of HAIEnd points lists DCS block 1001.15 as a duplicate of P1_B2016.
    "x1001_15_ASSIGN_OUT": "P1_B2016",
}
"""Columns that the manual ties to a point of another name."""

ATTACK_FLAG = TimeSeriesSpec(
    spec_type="hai_attack_flag", name="HAI attack flag", unit_value=ureg.dimensionless, dtype="int8"
)
"""HAI's own label: 1 while an attack runs, 0 otherwise."""


def manual_point(column: str) -> Point | None:
    """Return the manual's point of a column, under its own name or a name the manual ties to it.

    Returns:
        The point, or ``None`` if the manual does not list the column.
    """
    return _BY_NAME.get(OTHER_NAMES.get(column, column))


def spec_of(column: str) -> TimeSeriesSpec:
    """Return the spec of a data column.

    A column of the manual gets the spec of its point. Any other column gets a spec of its own, with
    the unit unset.

    Returns:
        The spec.
    """
    point = manual_point(column)
    if point is not None:
        return point.spec
    block = re.fullmatch(r"x(\d{4})_(\d{2})_\w+", column)
    if block is not None:
        # The manual lists the 23.05 DCS block outputs among the HAIEnd points, with "-" as unit.
        number = f"{block[1]}.{block[2]}"
        return TimeSeriesSpec(
            spec_type=f"dcs_block_{block[1]}_{block[2]}_output",
            name=f"Output of DCS block {number}",
            unit_value=None,
            dtype="float64",
        )
    return TimeSeriesSpec(
        spec_type=f"hai_{column.lower()}", name=f"{column} (not in the HAI manual)", unit_value=None, dtype="float64"
    )
