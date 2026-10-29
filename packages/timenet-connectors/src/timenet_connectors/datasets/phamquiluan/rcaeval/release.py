"""The RCAEval release as published: its Zenodo record, nine zips, their files, and their pinned inventory."""

from collections.abc import Mapping
from enum import StrEnum, unique
from pathlib import Path
from typing import NamedTuple

from timenet.errors import TimeFValidationError


RECORD_URL = "https://zenodo.org/records/14590730"
"""The Zenodo record of the RCAEval datasets, the files the authors' ``main.py`` downloads."""
RECORD_DOI = "10.5281/zenodo.14590730"


@unique
class System(StrEnum):
    """One of the three microservice systems, by the code the case names use."""

    ONLINE_BOUTIQUE = "ob"
    SOCK_SHOP = "ss"
    TRAIN_TICKET = "tt"

    @property
    def display_name(self) -> str:
        """Return the system's name, as the paper writes it, such as ``Online Boutique``."""
        return _SYSTEM_NAMES[self]


_SYSTEM_NAMES = {
    System.ONLINE_BOUTIQUE: "Online Boutique",
    System.SOCK_SHOP: "Sock Shop",
    System.TRAIN_TICKET: "Train Ticket",
}


@unique
class Suite(StrEnum):
    """One of the nine datasets: a benchmark suite (RE1 to RE3) on one system, and one zip of the record."""

    RE1_OB = "RE1-OB"
    RE1_SS = "RE1-SS"
    RE1_TT = "RE1-TT"
    RE2_OB = "RE2-OB"
    RE2_SS = "RE2-SS"
    RE2_TT = "RE2-TT"
    RE3_OB = "RE3-OB"
    RE3_SS = "RE3-SS"
    RE3_TT = "RE3-TT"

    @property
    def benchmark(self) -> str:
        """Return the benchmark suite, ``RE1``, ``RE2`` or ``RE3``."""
        return self.value.split("-")[0]

    @property
    def system(self) -> System:
        """Return the system the suite's cases ran on."""
        return System(self.value.split("-")[1].lower())

    @property
    def archive_filename(self) -> str:
        """Return the zip's name in the record, such as ``RE2-OB.zip``, whose one top-level directory is the suite."""
        return f"{self.value}.zip"

    @property
    def archive_url(self) -> str:
        """Return the zip's download URL."""
        return f"{RECORD_URL}/files/{self.archive_filename}?download=1"

    @property
    def metrics_file(self) -> str:
        """Return the name of a case's metrics file.

        RE2 and RE3 cases hold ``simple_metrics.csv``, which ``main.py`` reads. RE1-SS and RE1-TT cases
        hold ``simple_data.csv`` next to a raw ``data.csv``; ``simple_data.csv`` has the metrics the paper
        counts, named as in every other suite. RE1-OB cases hold only ``data.csv``, already named so.
        """
        if self is Suite.RE1_OB:
            return "data.csv"
        if self.benchmark == "RE1":
            return "simple_data.csv"
        return "simple_metrics.csv"


class PinnedArchive(NamedTuple):
    """One zip of the record, as pinned: its digest and what its case directories hold."""

    sha256: str
    cases: int
    logs: int
    """Cases with a ``logs.csv``."""
    traces: int
    """Cases with a ``traces.csv`` that holds rows."""
    header_only_traces: int
    """Cases whose ``traces.csv`` holds only its header: Sock Shop is not traced, so they hold no trace."""
    derived: int
    """Cases with the authors' derived series: ``logts.csv``, and ``tracets_err.csv`` and ``tracets_lat.csv`` where traced."""
    root_cause_files: int
    """Cases with a ``root_cause.txt``."""


class FoundArchive(NamedTuple):
    """What one zip's case directories were found to hold, in the fields of :class:`PinnedArchive` but the digest."""

    cases: int
    logs: int
    traces: int
    header_only_traces: int
    derived: int
    root_cause_files: int

    def check(self, path: Path, pinned: PinnedArchive) -> None:
        """Compare these counts with the pinned zip's.

        Raises:
            TimeFValidationError: If the zip isn't the pinned one.
        """
        if tuple(self) != pinned[1:]:
            raise TimeFValidationError(f"{path}: holds {self}, the pinned zip holds {pinned}")


PINNED_ARCHIVES: Mapping[Suite, PinnedArchive] = {
    Suite.RE1_OB: PinnedArchive(
        sha256="4a709297e0a829f0f2ee8a7792a6d74da32d663c600565b7fffc860963b840c4",
        cases=125,
        logs=0,
        traces=0,
        header_only_traces=0,
        derived=0,
        root_cause_files=0,
    ),
    Suite.RE1_SS: PinnedArchive(
        sha256="b4424b0b3863b7397712caa0f305ef59964b03784dfcb23e23e0a95a2e746f99",
        cases=125,
        logs=0,
        traces=0,
        header_only_traces=0,
        derived=0,
        root_cause_files=0,
    ),
    Suite.RE1_TT: PinnedArchive(
        sha256="2b33b7ab07198e0d69f229e697bfcef794a656e8db73a1d732142effde17c595",
        cases=125,
        logs=0,
        traces=0,
        header_only_traces=0,
        derived=0,
        root_cause_files=0,
    ),
    Suite.RE2_OB: PinnedArchive(
        sha256="0605a36cdcad8a6ae0107f2357c9c91ecee2c4ab5d72579bffea0372d9747513",
        cases=90,
        logs=90,
        traces=90,
        header_only_traces=0,
        derived=90,
        root_cause_files=0,
    ),
    Suite.RE2_SS: PinnedArchive(
        sha256="7aff9a3a0df7e2febbce4f75f0b7ba332da943aacadbffe6d5113a588ef6e295",
        cases=90,
        logs=90,
        traces=0,
        header_only_traces=0,
        derived=90,
        root_cause_files=0,
    ),
    Suite.RE2_TT: PinnedArchive(
        sha256="6706311d2c00d9f5a335f73f6a11f4ad4417522abed7e7ee8c88b8e898088805",
        cases=90,
        logs=90,
        traces=90,
        header_only_traces=0,
        derived=90,
        root_cause_files=0,
    ),
    Suite.RE3_OB: PinnedArchive(
        sha256="4885c87665ee4f1c80734b0e8f0411f91f6dae5751b7506ee90a29f12d952ee7",
        cases=30,
        logs=30,
        traces=30,
        header_only_traces=0,
        derived=0,
        root_cause_files=0,
    ),
    Suite.RE3_SS: PinnedArchive(
        sha256="6c760fe241e7c4ec11c280a55388b2361ddb31e074c1b97893b11c8c334def08",
        cases=30,
        logs=30,
        traces=0,
        header_only_traces=13,
        derived=0,
        root_cause_files=8,
    ),
    Suite.RE3_TT: PinnedArchive(
        sha256="10b4f14de766af5b7dafedb851439605bbc6c62f7ca1d8a7bcd53f6d974263fe",
        cases=30,
        logs=30,
        traces=30,
        header_only_traces=0,
        derived=0,
        root_cause_files=0,
    ),
}

NOT_CASES: Mapping[str, str] = {
    "RE2-OB/checkoutservice_cpu/multi-source-data": (
        "a copy of checkoutservice_cpu/1 for the authors' multi-source demo, with the same inject time, logs and "
        "traces; it has no simple_metrics.csv, so main.py skips it too"
    ),
}
"""Directories that hold an ``inject_time.txt`` but are not one of the 735 cases, with the reason."""

FAULT_DIRECTORY_SUFFIXES: Mapping[str, str] = {
    "RE3-TT/ts-route-service_f3_1": (
        "a second directory of ts-route-service f3 cases, whose repetitions 1 to 3 are three more cases"
    ),
}
"""Fault directories named ``<service>_<fault>_<n>`` rather than ``<service>_<fault>``, with the reason."""


class UntimedRows(NamedTuple):
    """A metrics file's rows without a time under one clock, which the Signals leave out."""

    rows: int
    values: int
    """The values in those rows, in the clock's columns."""


PINNED_UNTIMED_ROWS: Mapping[tuple[str, str], UntimedRows] = {
    ("RE1-OB/adservice_cpu/5", "time"): UntimedRows(8, 272),
    ("RE1-OB/checkoutservice_cpu/4", "time"): UntimedRows(262, 4033),
    # The rows after the error table's last time hold no value.
    ("RE1-OB/checkoutservice_cpu/4", "time.1"): UntimedRows(148, 0),
    # The error table holds 63 rows, timed 582 s after the first row of the main table.
    ("RE1-OB/productcatalogservice_cpu/3", "time.1"): UntimedRows(4138, 0),
    ("RE1-OB/productcatalogservice_cpu/4", "time"): UntimedRows(14, 476),
    ("RE1-SS/catalogue_disk/3", "time"): UntimedRows(74, 1628),
    ("RE1-SS/user_mem/1", "time"): UntimedRows(114, 2203),
    ("RE1-TT/ts-order-service_mem/3", "time"): UntimedRows(113, 9209),
    ("RE2-OB/checkoutservice_cpu/2", "time"): UntimedRows(256, 5716),
    ("RE2-OB/checkoutservice_mem/2", "time"): UntimedRows(258, 5585),
}
"""The metric rows without a time, by case and clock column: rows at the end of the file whose ``time`` is
empty but whose metrics hold values. ``main.py`` forward-fills ``time`` into them; the connector leaves them
out, since nothing in the file says when they were measured."""
