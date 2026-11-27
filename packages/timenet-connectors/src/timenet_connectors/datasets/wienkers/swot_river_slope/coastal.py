"""QC, astronomical phase and issue definitions for the signed SWOT benchmark."""

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import cos, floor, isfinite, radians
from statistics import median

from timenet.errors import TimeFValidationError


DAY = timedelta(days=1)
REPEAT = DAY * 20.86455  # NASA SWOT science orbit, User Handbook §4.1.
SCIENCE_START = datetime(2023, 7, 21, tzinfo=UTC)
J2000 = datetime(2000, 1, 1, 12, tzinfo=UTC)
SCALED = 10_000
WSE_LIMITS = (-450, 8000)
WSE_U_LIMITS = (1e-6, 0.5)
SLOPE_LIMIT = 0.05
SLOPE_U_LIMITS = (1e-10, 5e-5)
BASELINE_VISITS = 4
RELEASED_YEARS = frozenset({2023, 2024})
COASTAL_ENDPOINT = "coastal_t2_daily_2_to_21d_v2"
SEALED_REGIONS = frozenset({3, 4, 5})
FIELDS = ("reach_id", "time_str", "wse", "wse_u", "slope", "slope_u", "reach_q")


@dataclass(frozen=True, slots=True)
class Pass:
    """One Hydrocron reach observation in the upstream units."""

    reach_id: int
    time: datetime
    wse: float
    wse_u: float
    slope: float
    slope_u: float
    reach_q: int

    @property
    def fall(self) -> float:
        """Positive-downstream fall in metres per 10 kilometres."""
        return self.slope * SCALED

    @property
    def fall_u(self) -> float:
        """One standard uncertainty in metres per 10 kilometres."""
        return self.slope_u * SCALED

    @property
    def z(self) -> float:
        """Signed fall divided by its reported standard uncertainty."""
        return self.fall / self.fall_u


def read_pass(row: dict[str, str], reach_id: int) -> Pass | None:
    """Reject off-split rows before decoding any measured values, then apply Hydrocron QC."""
    if reach_id // 10**10 in SEALED_REGIONS or int(row["reach_id"]) != reach_id:
        raise TimeFValidationError("Unexpected or sealed SWORD reach.")
    try:
        at = datetime.fromisoformat(row["time_str"])
    except ValueError:
        return None
    if at.year not in RELEASED_YEARS or at.utcoffset() != timedelta(0):
        raise TimeFValidationError("Hydrocron timestamp outside the released 2023-2024 UTC period.")
    try:
        wse, wse_u, slope, slope_u, quality = (float(row[k]) for k in FIELDS[2:])
    except ValueError:
        return None
    if not (
        all(isfinite(value) for value in (wse, wse_u, slope, slope_u, quality))
        and quality in {0, 1}
        and WSE_LIMITS[0] <= wse <= WSE_LIMITS[1]
        and WSE_U_LIMITS[0] <= wse_u <= WSE_U_LIMITS[1]
        and -SLOPE_LIMIT <= slope <= SLOPE_LIMIT
        and SLOPE_U_LIMITS[0] <= slope_u <= SLOPE_U_LIMITS[1]
    ):
        return None
    return Pass(reach_id, at, wse, wse_u, slope, slope_u, int(quality))


def daily_visits(passes: Sequence[Pass]) -> list[Pass]:
    """Choose the minimum slope uncertainty per UTC day, then the earliest acquisition."""
    days: dict[datetime, Pass] = {}
    for p in sorted(passes, key=lambda p: (p.slope_u, p.time, p.wse_u)):
        days.setdefault(p.time.replace(hour=0, minute=0, second=0, microsecond=0), p)
    return sorted(days.values(), key=lambda p: p.time)


def astronomical_phase(at: datetime, longitude: float) -> dict[str, float]:
    """Return mean equilibrium phases and equal-weight astronomical indices.

    The angles use mean lunar and solar longitudes and the Greenwich sidereal angle at J2000.
    Spherical harmonic order supplies the longitude correction. No local admittance, propagation
    lag, nodal correction or amplitudes are fitted. The indices are dimensionless.

    Returns:
        Constituent angles in degrees and dimensionless astronomical indices.
    """
    days = (at - J2000) / DAY
    sidereal = 280.46061837 + 360.98564736629 * days + longitude
    moon = 218.3164477 + 13.17639648 * days
    sun = 280.46646 + 0.98564736 * days
    phases = {
        "M2": (2 * (sidereal - moon)) % 360,
        "S2": (2 * (sidereal - sun)) % 360,
        "K1": (sidereal - 90) % 360,
        "O1": (sidereal - 2 * moon + 90) % 360,
    }
    semidiurnal = (cos(radians(phases["M2"])) + cos(radians(phases["S2"]))) / 2
    return {
        **phases,
        "spring_index": (1 + cos(radians(phases["M2"] - phases["S2"]))) / 2,
        "semidiurnal_index": semidiurnal,
        "stage_index": sum(cos(radians(value)) for value in phases.values()) / 4,
    }


@dataclass(frozen=True, slots=True)
class Issue:
    """A UTC issue and its pre-issue evidence and conditional verification."""

    at: datetime
    history: tuple[Pass, ...]
    baseline: float
    reference: Pass
    minimum: Pass | None
    scheduled: datetime | None
    verified: Pass

    @property
    def reversal(self) -> bool:
        """True when verification fall is strictly negative."""
        return self.verified.fall < 0

    @property
    def reversal_sig(self) -> bool:
        """True when verification fall plus one standard uncertainty is negative."""
        return self.verified.fall + self.verified.fall_u < 0

    @property
    def steepening(self) -> bool:
        """True when verification exceeds the common baseline by more than 0.20 m/10 km."""
        return self.verified.fall > self.baseline + 0.20


def planned_overpass(issue: datetime, prior: Sequence[Pass]) -> datetime | None:
    """Estimate the first nominal eligible overpass using only pre-issue timestamps.

    A previous science-orbit visit is propagated by integer repeat cycles. This clocks past
    tracks forward and cannot discover tracks that have yet to return a QC-valid acquisition.

    Returns:
        The first estimated eligible UTC instant, if available.
    """
    earliest = issue + 2 * DAY
    candidates = []
    for p in prior:
        if not SCIENCE_START <= p.time < issue:
            continue
        repeats = floor((earliest - p.time) / REPEAT)
        nominal = p.time + max(1, repeats) * REPEAT
        if nominal < earliest:
            nominal += REPEAT
        if nominal <= issue + 21 * DAY:
            candidates.append(nominal)
    return min(candidates, default=None)


def coastal_issues(passes: Sequence[Pass], year: int) -> Iterator[Issue]:
    """Yield the fixed three-day coastal issue grid for a 2023 or 2024 reach."""
    if year not in RELEASED_YEARS:
        raise TimeFValidationError("Only 2023-2024 coastal issues may be constructed.")
    visits = daily_visits(passes)
    times = [p.time for p in visits]
    nonnegative = [p for p in visits if p.fall >= 0]
    nonnegative_times = [p.time for p in nonnegative]
    start = datetime(year, 1, 1, tzinfo=UTC)
    stop = datetime(year + 1, 1, 1, tzinfo=UTC) - 21 * DAY
    for step in range(0, (stop - start).days, 3):
        issue = start + step * DAY
        end = bisect_left(times, issue)
        end_nonneg = bisect_left(nonnegative_times, issue)
        verify = bisect_left(times, issue + 2 * DAY)
        if (
            end < BASELINE_VISITS
            or end_nonneg < BASELINE_VISITS
            or verify == len(times)
            or times[verify] > issue + 21 * DAY
        ):
            continue
        baseline = median(p.fall for p in nonnegative[end_nonneg - BASELINE_VISITS : end_nonneg])
        window = visits[bisect_left(times, issue - 21 * DAY) : end]
        # Include the latest four nonnegative visits, all recent visits, and the latest four signed visits.
        first = min(
            end - BASELINE_VISITS,
            bisect_left(times, nonnegative_times[end_nonneg - BASELINE_VISITS]),
            end - len(window),
        )
        history = tuple(visits[first:end])
        yield Issue(
            issue,
            history,
            baseline,
            nonnegative[end_nonneg - BASELINE_VISITS],
            min(window, key=lambda p: p.fall) if window else None,
            planned_overpass(issue, visits[max(0, end - 48) : end]),
            visits[verify],
        )
