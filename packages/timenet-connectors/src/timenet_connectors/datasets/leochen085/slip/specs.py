"""The one :class:`~timenet.types.TimeSeriesSpec` every SLIP series uses.

The corpus states no unit for anything: no column carries one, and ``meta.csv`` gives a rate per
corpus and no quantity. So every series is dimensionless, and one spec covers the release.
"""

from timenet.types import TimeSeriesSpec, ureg


SERIES = TimeSeriesSpec(
    spec_type="slip_series",
    name="SLIP Series",
    unit_value=ureg.dimensionless,
)
"""Every series in the corpus. Dimensionless because the release names no unit, not because the
values have none."""
