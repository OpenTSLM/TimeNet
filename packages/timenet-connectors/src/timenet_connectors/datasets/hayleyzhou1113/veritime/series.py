"""What a published series measures, by task, and the rows that publish the same series.

ChatTS asks several questions about one series, so 2,520 scenario rows hold only 1,259 distinct series.
Rows that publish equal values in the same kind share one :class:`Series`, and so one Record. No two
knowledge rows publish the same series.

The knowledge series are what the authors' questions paste in, and they are not always what the
question describes. In the test file, which keeps each original, an ECG series is every second
sample of 1,000 at 300 Hz, so 150 Hz, and an RCW series every eighth sample of 4,000 at 2 kHz, so
250 Hz; the questions still state 300 Hz, and 2 kHz and 4,000 points. The axes give the rate of the
values as published. CTU states its 2-minute cadence and is not decimated. EMG states no rate.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum, unique
from fractions import Fraction

from timenet.dataset.axis import OrdinalAxis, RegularAxis, TimeAxis
from timenet.errors import TimeFValidationError
from timenet.types import TimeSeriesSpec, ureg
from timenet_connectors.datasets.hayleyzhou1113.veritime.release import Family, SourceTask
from timenet_connectors.datasets.hayleyzhou1113.veritime.rows import PublishedSeries, Row


_MICROSECONDS_PER_SECOND = 1_000_000


@unique
class SeriesKind(StrEnum):
    """What a series measures: a ChatTS series, or one knowledge task's signal."""

    SCENARIO = "scenario"
    CTU = "ctu"
    ECG = "ecg"
    EMG = "emg"
    RCW = "rcw"

    @classmethod
    def of(cls, task: SourceTask) -> "SeriesKind":
        """Return the kind of series a task's rows publish.

        Returns:
            ``SCENARIO`` for the three scenario tasks, or the knowledge task's own kind.
        """
        return cls.SCENARIO if task.family is Family.SCENARIO else cls(task.slug)

    @property
    def family(self) -> Family:
        """Return the family whose rows publish series of the kind."""
        return Family.SCENARIO if self is SeriesKind.SCENARIO else Family.KNOWLEDGE

    @property
    def spec(self) -> TimeSeriesSpec:
        """Return the spec of the kind's signals."""
        return _SPECS[self]

    @property
    def signal_name(self) -> str:
        """Return the name of the kind's one signal."""
        return _SIGNAL_NAMES[self]

    @property
    def n_values(self) -> int | None:
        """Return the length every series of the kind has, or None when lengths vary."""
        return _LENGTHS.get(self)

    def time_axis(self) -> TimeAxis:
        """Return a new axis for a signal of the kind: the rate of its values as published, if known.

        Returns:
            A regular axis at that rate, or an ordinal axis where the release gives none.
        """
        period = _PERIODS_US.get(self)
        return OrdinalAxis() if period is None else RegularAxis(period_us=period)


_SPECS = {
    # The ChatTS series are synthetic, with no unit or cadence.
    SeriesKind.SCENARIO: TimeSeriesSpec(
        spec_type="chatts_series", name="ChatTS series", unit_value=None, dtype="float64"
    ),
    SeriesKind.CTU: TimeSeriesSpec(
        spec_type="electricity_consumption", name="Household electricity consumption", unit_value=None, dtype="float64"
    ),
    SeriesKind.ECG: TimeSeriesSpec(spec_type="ecg", name="Single-lead ECG", unit_value=None, dtype="float64"),
    # The EMG question states the amplitude in mV; the others state no unit.
    SeriesKind.EMG: TimeSeriesSpec(spec_type="emg", name="Needle EMG", unit_value=ureg.millivolt, dtype="float64"),
    SeriesKind.RCW: TimeSeriesSpec(
        spec_type="hydrophone_audio", name="Hydrophone audio", unit_value=None, dtype="float64"
    ),
}
_SIGNAL_NAMES = {
    SeriesKind.SCENARIO: "series",
    SeriesKind.CTU: "consumption",
    SeriesKind.ECG: "ecg",
    SeriesKind.EMG: "emg",
    SeriesKind.RCW: "audio",
}
_LENGTHS = {SeriesKind.CTU: 720, SeriesKind.ECG: 500, SeriesKind.EMG: 600, SeriesKind.RCW: 500}
_PERIODS_US = {
    SeriesKind.CTU: Fraction(120 * _MICROSECONDS_PER_SECOND),
    SeriesKind.ECG: Fraction(_MICROSECONDS_PER_SECOND, 300 // 2),
    SeriesKind.RCW: Fraction(_MICROSECONDS_PER_SECOND, 2000 // 8),
}


@dataclass(eq=False)
class Series:
    """One distinct published series, and every row that publishes it, in file order."""

    kind: SeriesKind
    published: PublishedSeries
    rows: list[Row] = field(default_factory=list)

    @property
    def value_source(self) -> Row:
        """Return the row a signal rereads the values from: the first that publishes them."""
        return self.rows[0]


def group_series(rows: Iterable[Row]) -> list[Series]:
    """Group rows by the series they publish.

    Returns:
        Each distinct series of each kind, in the order its first row appears.

    Raises:
        TimeFValidationError: If a knowledge series does not have its kind's length.
    """
    by_key: dict[tuple[SeriesKind, str], Series] = {}
    for row in rows:
        kind = SeriesKind.of(row.task)
        if kind.n_values is not None and row.series.n_values != kind.n_values:
            raise TimeFValidationError(
                f"{row.ref.context}: a {kind} series has {kind.n_values} values, not {row.series.n_values}"
            )
        series = by_key.setdefault((kind, row.series.digest), Series(kind, row.series))
        series.rows.append(row)
    return list(by_key.values())
