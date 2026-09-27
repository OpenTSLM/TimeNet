"""The five specs and the six time axes of the VerbalTS release. Values only, no I/O.

The paper describes normalization of the four real-world components in its dataset appendix.
The synthetic values are generated, and no release artifact gives an instrument unit.
These converted values are dimensionless. The paper does not establish array-channel order.
Source: https://proceedings.mlr.press/v267/gu25a.html (dataset appendix, pages 18-20).

No component ships a timestamp, so an axis states a cadence and nothing else. The real-world
cadences come from the paper that released the data, not from the artifact. The paper does not state
the BlindWays frame rate, so it uses an ordinal axis. The two synthetic sets have no clock either.
ETTm1's original CSV identifies its 15-minute sampling level:
https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTm1.csv
"""

from fractions import Fraction

from timenet.dataset.axis import OrdinalAxis, RegularAxis, TimeAxis
from timenet.types import TimeSeriesSpec, ureg


_SYNTHETIC = TimeSeriesSpec(
    spec_type="synthetic",
    name="Synthetic generated variable",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_WEATHER = TimeSeriesSpec(
    spec_type="weather",
    name="Jena weather variable (z-scored)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_POSE = TimeSeriesSpec(
    spec_type="pose",
    name="Body-joint coordinate (z-scored)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_POWER = TimeSeriesSpec(
    spec_type="power",
    name="Electricity transformer load variable (z-scored)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_TRAFFIC = TimeSeriesSpec(
    spec_type="traffic",
    name="Istanbul traffic index variable (z-scored)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)

SPEC_BY_COMPONENT: dict[str, TimeSeriesSpec] = {
    "synthetic_u": _SYNTHETIC,
    "synthetic_m": _SYNTHETIC,
    "Weather": _WEATHER,
    "BlindWays": _POSE,
    "ETTm1": _POWER,
    "istanbul_traffic": _TRAFFIC,
}
"""The spec every series of a component carries. One spec covers both synthetic sets, which differ
only in their signal count, so the release declares five spec types and not six."""

AXIS_BY_COMPONENT: dict[str, TimeAxis] = {
    "synthetic_u": OrdinalAxis(),
    "synthetic_m": OrdinalAxis(),
    "Weather": RegularAxis.from_rate_hz(Fraction(1, 600)),
    "BlindWays": OrdinalAxis(),
    "ETTm1": RegularAxis.from_rate_hz(Fraction(1, 900)),
    "istanbul_traffic": RegularAxis.from_rate_hz(Fraction(1, 600)),
}
"""Component cadences from the paper's dataset appendix and the original ETTm1 release."""
