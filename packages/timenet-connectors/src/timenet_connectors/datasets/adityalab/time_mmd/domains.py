"""What each domain of the release holds, as data that one reader walks.

The release is one directory for each domain, and every domain ships the same three files: a
numerical CSV, and a ``*_report.csv`` and a ``*_search.csv`` of text. The columns of the
numerical file differ from domain to domain, and so does the cadence. A :class:`DomainShape`
states those differences, so the reader in
:mod:`~timenet_connectors.datasets.adityalab.time_mmd.numerical` is written for no domain in
particular. A third domain is one more shape here and no new branch there.

This connector covers two of the nine domains. ``energy`` is weekly and ``environment`` is
daily, so one schema has to place both on a timeline. It does, because a record measures
everything in microseconds from one midnight
(:mod:`~timenet_connectors.datasets.adityalab.time_mmd.timeline`).

Every numerical file names its default forecasting target ``OT``, and the release states what
that column holds in a figure of its README and nowhere in the file. The identity of each
``OT`` below was checked against the agency that publishes it. The README of this connector
states how.
"""

from dataclasses import dataclass
from pathlib import Path

from timenet.types import DataSource, TimeSeriesSpec, ureg
from timenet_connectors.datasets.adityalab.time_mmd.keys import AnnotationKey, TextKind


@dataclass(frozen=True)
class SignalColumn:
    """One column of a numerical file that becomes a signal of the record."""

    column: str
    """The header, as the release writes it."""
    signal: str
    """The name the signal takes in TimeF."""
    spec: TimeSeriesSpec
    """What the column measures, and the dtype its values take."""


@dataclass(frozen=True)
class ConstantColumn:
    """A column that states one fact about the whole domain, repeated on every row.

    It becomes one annotation with no span. A build stops when the rows disagree, because the
    record then has no one value to state.
    """

    column: str
    """The header, as the release writes it."""
    key: AnnotationKey
    """The key of the annotation the column becomes."""
    description: str
    """What the annotation states, for the schema."""
    numeric: bool = False
    """Whether the value is a whole number, which the annotation then carries as an ``int``."""


@dataclass(frozen=True)
class DomainShape:
    """How to read one domain of the release."""

    name: str
    """The lowercase name this connector uses for the domain, and the leaf of its record id."""
    directory: str
    """The name of the domain's directory in the release, which its three files also carry."""
    period_days: int
    """The cadence of the numerical file, in days."""
    signals: tuple[SignalColumn, ...]
    """The columns that become signals, in the order the record lists them."""
    target: str
    """The signal that holds ``OT``, the column the release names as the default target."""
    horizons: tuple[int, ...]
    """The forecasting horizons the Time-MMD paper evaluates at this cadence, in values."""
    constants: tuple[ConstantColumn, ...] = ()
    """The columns that state one fact for the whole domain."""

    def numerical_path(self, root: Path) -> Path:
        """Give the numerical file of the domain, under the root of a fetched release.

        Args:
            root: The directory that holds the ``numerical`` and ``textual`` trees.

        Returns:
            The path of the file.
        """
        return root / "numerical" / self.directory / f"{self.directory}.csv"

    def textual_path(self, root: Path, kind: TextKind) -> Path:
        """Give one textual file of the domain, under the root of a fetched release.

        Args:
            root: The directory that holds the ``numerical`` and ``textual`` trees.
            kind: Which of the two textual files.

        Returns:
            The path of the file.
        """
        return root / "textual" / self.directory / f"{self.directory}_{kind}.csv"

    @property
    def relative_paths(self) -> tuple[str, ...]:
        """The three files of the domain, relative to the root of the release, with forward slashes."""
        root = Path()
        return (
            self.numerical_path(root).as_posix(),
            self.textual_path(root, TextKind.REPORT).as_posix(),
            self.textual_path(root, TextKind.SEARCH).as_posix(),
        )


_ENERGY_SOURCE = DataSource(
    data_source_type="time-mmd",
    name="Time-MMD energy domain: EIA weekly retail gasoline prices",
    provider="U.S. Energy Information Administration",
)

# pint defines no currency, and a unit the SDK's registry cannot parse would break every read
# of the manifest, because the reader rebuilds a spec's unit from its name. The name states the
# unit in words instead. Defining a currency in the shared registry is a change to the core
# package, and this connector does not make it.
GASOLINE_PRICE = TimeSeriesSpec(
    spec_type="retail_gasoline_price",
    name="Retail gasoline price, all grades, all formulations, in US dollars per gallon",
    unit_value=ureg.dimensionless,
    data_source=_ENERGY_SOURCE,
)


def _eia_column(region: str) -> str:
    # The release keeps the EIA headers, the two spaces before the unit included.
    return f"Weekly {region} All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)"


# The weekly US retail gasoline price, which the release names OT, and the eight regional
# prices EIA publishes beside it. OT is the national series of the same EIA table: its values
# match "U.S. All Grades All Formulations Retail Gasoline Prices" on every week checked.
ENERGY = DomainShape(
    name="energy",
    directory="Energy",
    period_days=7,
    signals=(
        SignalColumn("OT", "us", GASOLINE_PRICE),
        SignalColumn(_eia_column("East Coast"), "east_coast", GASOLINE_PRICE),
        SignalColumn(_eia_column("New England (PADD 1A)"), "new_england", GASOLINE_PRICE),
        SignalColumn(_eia_column("Central Atlantic (PADD 1B)"), "central_atlantic", GASOLINE_PRICE),
        SignalColumn(_eia_column("Lower Atlantic (PADD 1C)"), "lower_atlantic", GASOLINE_PRICE),
        SignalColumn(_eia_column("Midwest"), "midwest", GASOLINE_PRICE),
        SignalColumn(_eia_column("Gulf Coast"), "gulf_coast", GASOLINE_PRICE),
        SignalColumn(_eia_column("Rocky Mountain"), "rocky_mountain", GASOLINE_PRICE),
        SignalColumn(_eia_column("West Coast"), "west_coast", GASOLINE_PRICE),
    ),
    target="us",
    horizons=(12, 24, 36, 48),
)


_ENVIRONMENT_SOURCE = DataSource(
    data_source_type="time-mmd",
    name="Time-MMD environment domain: EPA AirData daily AQI by core-based statistical area",
    provider="U.S. Environmental Protection Agency",
)

# The six categories of the EPA index, from the cleanest air up. The release holds five of them:
# no day of the New York area reached "Hazardous".
AQI_CATEGORIES = (
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
)

# The pollutants an AQI is computed from, as EPA's files spell them. The release holds five of
# them: no day of the New York area was defined by SO2.
AQI_POLLUTANTS = ("CO", "NO2", "Ozone", "PM10", "PM2.5", "SO2")

AQI = TimeSeriesSpec(
    spec_type="air_quality_index",
    name="Daily Air Quality Index",
    unit_value=ureg.dimensionless,
    dtype="int16",
    data_source=_ENVIRONMENT_SOURCE,
)
AQI_CATEGORY = TimeSeriesSpec(
    spec_type="aqi_category",
    name="Category of the day's AQI, as EPA labels the ranges of the index",
    unit_value=ureg.dimensionless,
    dtype="enum",
    categories=AQI_CATEGORIES,
    data_source=_ENVIRONMENT_SOURCE,
)
AQI_DEFINING_PARAMETER = TimeSeriesSpec(
    spec_type="aqi_defining_parameter",
    name="Pollutant whose index was the day's AQI",
    unit_value=ureg.dimensionless,
    dtype="enum",
    categories=AQI_POLLUTANTS,
    data_source=_ENVIRONMENT_SOURCE,
)
AQI_DEFINING_SITE = TimeSeriesSpec(
    spec_type="aqi_defining_site",
    name="Monitoring site whose reading was the day's AQI, as its AQS site id",
    unit_value=ureg.dimensionless,
    dtype="str",
    data_source=_ENVIRONMENT_SOURCE,
)
AQI_SITES_REPORTING = TimeSeriesSpec(
    spec_type="aqi_sites_reporting",
    name="Monitoring sites that reported that day",
    unit_value=ureg.dimensionless,
    dtype="int16",
    data_source=_ENVIRONMENT_SOURCE,
)

# The daily AQI of one core-based statistical area, which the release names OT, with the four
# columns EPA states beside it. The columns are those of EPA's "daily AQI by CBSA" files, and
# the two CBSA columns hold one value on every row, so they describe the record and not a day.
ENVIRONMENT = DomainShape(
    name="environment",
    directory="Environment",
    period_days=1,
    signals=(
        SignalColumn("OT", "aqi", AQI),
        SignalColumn("Category", "category", AQI_CATEGORY),
        SignalColumn("Defining Parameter", "defining_parameter", AQI_DEFINING_PARAMETER),
        SignalColumn("Defining Site", "defining_site", AQI_DEFINING_SITE),
        SignalColumn("Number of Sites Reporting", "sites_reporting", AQI_SITES_REPORTING),
    ),
    target="aqi",
    horizons=(48, 96, 192, 336),
    constants=(
        ConstantColumn(
            "CBSA",
            AnnotationKey.CBSA,
            "The core-based statistical area the AQI is computed for, as EPA names it.",
        ),
        ConstantColumn(
            "CBSA Code",
            AnnotationKey.CBSA_CODE,
            "The code of the core-based statistical area the AQI is computed for.",
            numeric=True,
        ),
    ),
)

# The domains this connector builds, in the order their records are added.
DOMAINS = (ENERGY, ENVIRONMENT)
