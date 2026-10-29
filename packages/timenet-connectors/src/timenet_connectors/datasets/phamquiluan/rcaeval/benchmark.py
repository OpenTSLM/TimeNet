"""What the authors' benchmark fixes: the faults, the root cause indicator of each, and the window a method sees.

These come from the paper (section 3.2 and 4.2) and the authors' ``main.py``, not from the zips. ``tasks``
applies them.
"""

from enum import StrEnum, unique

from timenet.errors import TimeFFormatError
from timenet.types import US_PER_S, TimeInterval
from timenet_connectors.datasets.phamquiluan.rcaeval.release import Suite


@unique
class Fault(StrEnum):
    """An injected fault, by the code the case directories use."""

    CPU = "cpu"
    MEM = "mem"
    DISK = "disk"
    SOCKET = "socket"
    DELAY = "delay"
    LOSS = "loss"
    F1 = "f1"
    F2 = "f2"
    F3 = "f3"
    F4 = "f4"
    F5 = "f5"

    @property
    def description(self) -> str:
        """Return the fault as the paper's section 3.2 names it, such as ``CPU hog``."""
        return _FAULT_DESCRIPTIONS[self]

    def indicator_metric(self, suite: Suite) -> str | None:
        """Return the metric ``main.py`` scores as the fault's root cause indicator, or ``None``.

        ``main.py`` scores a ranked ``<service>_<metric>`` against ``<root cause service>_<metric>``:
        ``cpu``, ``mem`` and ``socket`` for those faults, ``latency`` for ``delay`` and ``loss``, and for
        ``disk`` ``diskio``, except in RE1, which has no disk metric and is scored on ``latency``. It
        scores no indicator for the code-level faults ``f1`` to ``f5``.

        The paper names a different indicator for ``loss``: the failed requests or error codes. The
        connector follows the code, which produced the paper's scores.

        Returns:
            The metric, or ``None`` for a code-level fault.
        """
        if self is Fault.DISK:
            return "latency" if suite.benchmark == "RE1" else "diskio"
        return _INDICATOR_METRICS.get(self)


_FAULT_DESCRIPTIONS = {
    Fault.CPU: "CPU hog",
    Fault.MEM: "Memory leak",
    Fault.DISK: "Disk stress",
    Fault.SOCKET: "Socket stress",
    Fault.DELAY: "Network delay",
    Fault.LOSS: "Network packet loss",
    Fault.F1: "Incorrect parameter values",
    Fault.F2: "Missing parameters",
    Fault.F3: "Missing function call",
    Fault.F4: "Incorrect return values",
    Fault.F5: "Missing exception handlers",
}

_INDICATOR_METRICS = {
    Fault.CPU: "cpu",
    Fault.MEM: "mem",
    Fault.SOCKET: "socket",
    Fault.DELAY: "latency",
    Fault.LOSS: "latency",
}

AUTHORS_LATENCY_RENAME = ("_latency-90", "_latency")
"""``main.py`` renames each ``<service>_latency-90`` metric to ``<service>_latency`` and drops ``_latency-50``,
so the ``latency`` it scores is the 90th percentile."""


def indicator_column(columns: list[str], service: str, metric: str, *, context: str) -> str:
    """Return the metrics column that is a case's root cause indicator, as ``main.py`` names it after its rename.

    Args:
        columns: The case's metric column names, as published.
        service: The root cause service.
        metric: The indicator metric, from :meth:`Fault.indicator_metric`.
        context: The case, for the error message.

    Returns:
        The one column that ``main.py`` reads as ``<service>_<metric>``.

    Raises:
        TimeFFormatError: If no column, or more than one, is that indicator.
    """
    wanted = f"{service}_{metric}"
    old, new = AUTHORS_LATENCY_RENAME
    found = [column for column in columns if column.replace(old, new) == wanted]
    if len(found) != 1:
        raise TimeFFormatError(f"{context}: expected one metric named {wanted!r} after main.py's rename, got {found}")
    return found[0]


LENGTH_MINUTES = 20
"""``main.py``'s ``--length``, the default the paper's scores use."""
ROWS_PER_SIDE = LENGTH_MINUTES * 60 // 2
"""``main.py`` gives a method the last ``ROWS_PER_SIDE`` metric rows before the inject time and the first
``ROWS_PER_SIDE`` from it. The metrics are one row per second, so that is 600 s on each side. Its
multi-source methods take 40 rows of the 15 s derived series on each side, the same 600 s."""
METRICS_PERIOD_S = 1
"""The metrics' cadence: every case's rows are one second apart, which the loader checks."""
SIDE_US = ROWS_PER_SIDE * METRICS_PERIOD_S * US_PER_S


def window(inject_us: int, metrics_first_us: int, metrics_end_us: int) -> TimeInterval:
    """Return the window ``main.py`` gives a method: up to 600 s before the inject time, and up to 600 s from it.

    The metrics are gap-free at one row per second, so ``tail(600)`` of the rows before the inject time
    and ``head(600)`` of the rest cover exactly this interval, cut at the metrics' first and last rows.

    Args:
        inject_us: The inject time, as an offset on the Record's timeline.
        metrics_first_us: The first metric row's offset.
        metrics_end_us: One period past the last metric row's offset.

    Returns:
        The interval, on the Record's timeline.
    """
    return TimeInterval.micros(max(metrics_first_us, inject_us - SIDE_US), min(metrics_end_us, inject_us + SIDE_US))
