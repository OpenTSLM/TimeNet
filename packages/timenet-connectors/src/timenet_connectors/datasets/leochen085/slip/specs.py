"""The one :class:`~timenet.types.TimeSeriesSpec` every SLIP series uses.

The corpus states no unit for anything. No column carries one, and ``meta.csv`` gives a rate per
corpus and no quantity. Every series is therefore dimensionless, and one spec covers the release.
"""

from timenet.types import DataSource, TimeSeriesSpec, ureg


_SOURCE = DataSource(data_source_type="huggingface", name="SlipDataset", provider="LeoChen085")

SERIES = TimeSeriesSpec(
    spec_type="slip_series",
    name="SLIP Series",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
)
"""Every series in the corpus. The unit is dimensionless because the release names no unit."""
