"""The 3W process variables, grouped by where they are measured on the well.

Names, descriptions and units come from ``[PARQUET_FILE_PROPERTIES]`` in ``dataset/dataset.ini``.
Every variable can have gaps in a real well, so every spec declares missing values. The source
leaves a gap empty and the connector keeps it empty.
"""

from dataclasses import dataclass

from pint import Unit

from timenet.types import MissingReason, MissingValues, TimeSeriesSpec, ureg


_MISSING = MissingValues(
    reason=MissingReason.NOT_RECORDED,
    description="3W leaves a timestep empty when the well's system recorded no value for it.",
)


@dataclass(frozen=True)
class Variable:
    """One process variable of 3W."""

    column: str
    """The column name in the 3W Parquet files."""
    source: str
    """The measuring point the variable belongs to, used as the TimeNet source name."""
    spec: TimeSeriesSpec
    """The spec of the signal built from the column."""


def _variable(column: str, source: str, description: str, unit: Unit) -> Variable:
    """Build one variable with a float64 spec that allows missing values.

    Args:
        column: The 3W column name.
        source: The measuring point.
        description: The description from ``dataset.ini``.
        unit: The physical unit.

    Returns:
        The variable.
    """
    spec = TimeSeriesSpec(
        spec_type=f"3w_{column.lower().replace('-', '_')}",
        name=description,
        unit_value=unit,
        dtype="float64",
        missing=_MISSING,
    )
    return Variable(column=column, source=source, spec=spec)


DOWNHOLE = "Downhole gauge (PDG)"
XMAS_TREE = "Subsea Xmas tree"
PRODUCTION_CHOKE = "Production line and choke (PCK)"
GAS_LIFT = "Gas lift line"
SERVICE_LINE = "Service line"
VALVES = "Valve states"
WELL = "Well status"

_PA = ureg.pascal
_DEG_C = ureg.degC
_PERCENT = ureg.percent
_FLOW = ureg.meter**3 / ureg.second
_STATE = ureg.dimensionless

VARIABLES: tuple[Variable, ...] = (
    _variable("P-PDG", DOWNHOLE, "Downhole pressure at the PDG (permanent downhole gauge)", _PA),
    _variable("T-PDG", DOWNHOLE, "Downhole temperature at the PDG (permanent downhole gauge)", _DEG_C),
    _variable("P-TPT", XMAS_TREE, "Subsea Xmas-tree pressure at the TPT (temperature and pressure transducer)", _PA),
    _variable(
        "T-TPT", XMAS_TREE, "Subsea Xmas-tree temperature at the TPT (temperature and pressure transducer)", _DEG_C
    ),
    _variable("PT-P", XMAS_TREE, "Subsea Xmas-tree pressure downstream of the PWV in the production line", _PA),
    _variable("P-ANULAR", XMAS_TREE, "Pressure in the well annulus", _PA),
    _variable("P-MON-CKP", PRODUCTION_CHOKE, "Upstream pressure of the PCK (production choke)", _PA),
    _variable("P-JUS-CKP", PRODUCTION_CHOKE, "Downstream pressure of the PCK (production choke)", _PA),
    _variable("T-MON-CKP", PRODUCTION_CHOKE, "Upstream temperature of the PCK (production choke)", _DEG_C),
    _variable("T-JUS-CKP", PRODUCTION_CHOKE, "Downstream temperature of the PCK (production choke)", _DEG_C),
    _variable("ABER-CKP", PRODUCTION_CHOKE, "Opening of the PCK (production choke)", _PERCENT),
    _variable("P-MON-SDV-P", PRODUCTION_CHOKE, "Upstream pressure of the production SDV (shutdown valve)", _PA),
    _variable("P-MON-CKGL", GAS_LIFT, "Upstream pressure of the GLCK (gas lift choke)", _PA),
    _variable("P-JUS-CKGL", GAS_LIFT, "Downstream pressure of the GLCK (gas lift choke)", _PA),
    _variable("ABER-CKGL", GAS_LIFT, "Opening of the GLCK (gas lift choke)", _PERCENT),
    _variable("QGL", GAS_LIFT, "Gas lift flow rate", _FLOW),
    _variable("P-JUS-BS", SERVICE_LINE, "Downstream pressure of the SP (service pump)", _PA),
    _variable("QBS", SERVICE_LINE, "Flow rate at the SP (service pump)", _FLOW),
    _variable("ESTADO-DHSV", VALVES, "State of the DHSV (downhole safety valve)", _STATE),
    _variable("ESTADO-M1", VALVES, "State of the PMV (production master valve)", _STATE),
    _variable("ESTADO-M2", VALVES, "State of the AMV (annulus master valve)", _STATE),
    _variable("ESTADO-PXO", VALVES, "State of the PXO (pig-crossover) valve", _STATE),
    _variable("ESTADO-SDV-GL", VALVES, "State of the gas lift SDV (shutdown valve)", _STATE),
    _variable("ESTADO-SDV-P", VALVES, "State of the production SDV (shutdown valve)", _STATE),
    _variable("ESTADO-W1", VALVES, "State of the PWV (production wing valve)", _STATE),
    _variable("ESTADO-W2", VALVES, "State of the AWV (annulus wing valve)", _STATE),
    _variable("ESTADO-XO", VALVES, "State of the XO (crossover) valve", _STATE),
    _variable("state", WELL, "Well operational status code", _STATE),
)
"""Every variable, in measuring-point order. ``state`` keeps the integer codes of the release."""

SOURCE_ORDER: tuple[str, ...] = (DOWNHOLE, XMAS_TREE, PRODUCTION_CHOKE, GAS_LIFT, SERVICE_LINE, VALVES, WELL)
