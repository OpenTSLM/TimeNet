"""The 3W process variables, grouped by where they are measured on the well.

Names, descriptions and units come from ``[PARQUET_FILE_PROPERTIES]`` in ``dataset/dataset.ini``.
"""

from dataclasses import dataclass

from timenet.types import MissingReason, MissingValues, TimeSeriesSpec, ureg


# A real well rarely has every sensor, and a sensor can stop reporting for a while. The release
# leaves such a timestep empty. In v2.0.0, a real instance records a median of 11 of the 27
# variables, and four variables (PT-P, P-JUS-BS, QBS and P-MON-SDV-P) are empty in every instance.
_MISSING = MissingValues(
    reason=MissingReason.NOT_RECORDED,
    description="The well's system recorded no value for this timestep.",
)

PRESSURE = TimeSeriesSpec(
    spec_type="pressure", name="Pressure", unit_value=ureg.pascal, dtype="float64", missing=_MISSING
)
TEMPERATURE = TimeSeriesSpec(
    spec_type="temperature", name="Temperature", unit_value=ureg.degC, dtype="float64", missing=_MISSING
)
FLOW_RATE = TimeSeriesSpec(
    spec_type="flow_rate", name="Flow rate", unit_value=ureg.meter**3 / ureg.second, dtype="float64", missing=_MISSING
)
OPENING = TimeSeriesSpec(
    spec_type="valve_opening", name="Choke opening", unit_value=ureg.percent, dtype="float64", missing=_MISSING
)
# The release states a valve as 0 (closed), 0.5 (in transition) or 1 (open).
VALVE_STATE = TimeSeriesSpec(
    spec_type="valve_state", name="Valve state", unit_value=ureg.dimensionless, dtype="float64", missing=_MISSING
)
# The release keeps the well's operational status as integer codes 0-8 and does not state their
# meaning in the dataset files, so the codes are kept as they are.
WELL_STATUS = TimeSeriesSpec(
    spec_type="well_status", name="Well status code", unit_value=ureg.dimensionless, dtype="float64", missing=_MISSING
)


@dataclass(frozen=True)
class Variable:
    """One process variable of 3W."""

    column: str
    """The column name in the 3W Parquet files, which is also the signal name."""
    source: str
    """The measuring point, used as the TimeNet source name."""
    spec: TimeSeriesSpec
    """The kind of quantity the variable measures."""
    description: str
    """The variable's description in ``dataset.ini``, kept in the signal's metadata."""


DOWNHOLE = "Downhole gauge (PDG)"
XMAS_TREE = "Subsea Xmas tree"
PRODUCTION_CHOKE = "Production line and choke (PCK)"
GAS_LIFT = "Gas lift line"
SERVICE_LINE = "Service line"
VALVES = "Valve states"
WELL = "Well status"

VARIABLES: tuple[Variable, ...] = (
    Variable("P-PDG", DOWNHOLE, PRESSURE, "Downhole pressure at the PDG (permanent downhole gauge)"),
    Variable("T-PDG", DOWNHOLE, TEMPERATURE, "Downhole temperature at the PDG (permanent downhole gauge)"),
    Variable(
        "P-TPT", XMAS_TREE, PRESSURE, "Subsea Xmas-tree pressure at the TPT (temperature and pressure transducer)"
    ),
    Variable(
        "T-TPT", XMAS_TREE, TEMPERATURE, "Subsea Xmas-tree temperature at the TPT (temperature and pressure transducer)"
    ),
    Variable("PT-P", XMAS_TREE, PRESSURE, "Subsea Xmas-tree pressure downstream of the PWV in the production line"),
    Variable("P-ANULAR", XMAS_TREE, PRESSURE, "Pressure in the well annulus"),
    Variable("P-MON-CKP", PRODUCTION_CHOKE, PRESSURE, "Upstream pressure of the PCK (production choke)"),
    Variable("P-JUS-CKP", PRODUCTION_CHOKE, PRESSURE, "Downstream pressure of the PCK (production choke)"),
    Variable("T-MON-CKP", PRODUCTION_CHOKE, TEMPERATURE, "Upstream temperature of the PCK (production choke)"),
    Variable("T-JUS-CKP", PRODUCTION_CHOKE, TEMPERATURE, "Downstream temperature of the PCK (production choke)"),
    Variable("ABER-CKP", PRODUCTION_CHOKE, OPENING, "Opening of the PCK (production choke)"),
    Variable("P-MON-SDV-P", PRODUCTION_CHOKE, PRESSURE, "Upstream pressure of the production SDV (shutdown valve)"),
    Variable("P-MON-CKGL", GAS_LIFT, PRESSURE, "Upstream pressure of the GLCK (gas lift choke)"),
    Variable("P-JUS-CKGL", GAS_LIFT, PRESSURE, "Downstream pressure of the GLCK (gas lift choke)"),
    Variable("ABER-CKGL", GAS_LIFT, OPENING, "Opening of the GLCK (gas lift choke)"),
    Variable("QGL", GAS_LIFT, FLOW_RATE, "Gas lift flow rate"),
    Variable("P-JUS-BS", SERVICE_LINE, PRESSURE, "Downstream pressure of the SP (service pump)"),
    Variable("QBS", SERVICE_LINE, FLOW_RATE, "Flow rate at the SP (service pump)"),
    Variable("ESTADO-DHSV", VALVES, VALVE_STATE, "State of the DHSV (downhole safety valve)"),
    Variable("ESTADO-M1", VALVES, VALVE_STATE, "State of the PMV (production master valve)"),
    Variable("ESTADO-M2", VALVES, VALVE_STATE, "State of the AMV (annulus master valve)"),
    Variable("ESTADO-PXO", VALVES, VALVE_STATE, "State of the PXO (pig-crossover) valve"),
    Variable("ESTADO-SDV-GL", VALVES, VALVE_STATE, "State of the gas lift SDV (shutdown valve)"),
    Variable("ESTADO-SDV-P", VALVES, VALVE_STATE, "State of the production SDV (shutdown valve)"),
    Variable("ESTADO-W1", VALVES, VALVE_STATE, "State of the PWV (production wing valve)"),
    Variable("ESTADO-W2", VALVES, VALVE_STATE, "State of the AWV (annulus wing valve)"),
    Variable("ESTADO-XO", VALVES, VALVE_STATE, "State of the XO (crossover) valve"),
    Variable("state", WELL, WELL_STATUS, "Well operational status"),
)
"""Every variable, in measuring-point order."""

SOURCE_ORDER: tuple[str, ...] = (DOWNHOLE, XMAS_TREE, PRODUCTION_CHOKE, GAS_LIFT, SERVICE_LINE, VALVES, WELL)
