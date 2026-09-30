"""The five specs and the six time axes of the VerbalTS release. Values only, no I/O.

The paper states that VerbalTS z-scored each of the four real-world components per variable against
that component's own train split, and it published neither the mean nor the standard deviation
vector. The two synthetic sets are generated, so they carry no instrument reading either. No spec
can state a physical unit, every spec is dimensionless, and the physical signal names are labels.

No component ships a timestamp, so an axis states a cadence and nothing else. The real-world
cadences come from the paper that released the data, not from the artifact. The paper does not state
the BlindWays frame rate, and the README records that axis as open. The two synthetic sets have no
clock, so they take an ordinal axis.
"""

from fractions import Fraction

from timenet.dataset.axis import OrdinalAxis, RegularAxis, TimeAxis
from timenet.types import DataSource, TimeSeriesSpec, ureg


_SOURCE = DataSource(data_source_type="google-drive", name="VerbalTS", provider="SeqML")

_SYNTHETIC = TimeSeriesSpec(
    spec_type="synthetic",
    name="Synthetic generated variable (z-scored)",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
    dtype="float64",
)
_WEATHER = TimeSeriesSpec(
    spec_type="weather",
    name="Jena weather variable (z-scored)",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
    dtype="float64",
)
_POSE = TimeSeriesSpec(
    spec_type="pose",
    name="Body-joint coordinate (z-scored)",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
    dtype="float64",
)
_POWER = TimeSeriesSpec(
    spec_type="power",
    name="Electricity transformer load variable (z-scored)",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
    dtype="float64",
)
_TRAFFIC = TimeSeriesSpec(
    spec_type="traffic",
    name="Istanbul traffic index variable (z-scored)",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
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
    "BlindWays": RegularAxis.from_rate_hz(60),
    "ETTm1": RegularAxis.from_rate_hz(Fraction(1, 900)),
    "istanbul_traffic": RegularAxis.from_rate_hz(Fraction(1, 600)),
}
"""The time axis every series of a component carries. The README states where each cadence comes
from and what the evidence for it is."""
