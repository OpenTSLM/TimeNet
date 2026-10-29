"""The failure cases of one zip: who is the root cause, which fault, when it was injected, and which files there are.

A case is a directory ``<suite>/<service>_<fault>/<repetition>/``, so its path names its ground truth.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import NamedTuple
import zipfile

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.phamquiluan.rcaeval.benchmark import Fault
from timenet_connectors.datasets.phamquiluan.rcaeval.release import FAULT_DIRECTORY_SUFFIXES, NOT_CASES, Suite


INJECT_TIME_FILE = "inject_time.txt"
ROOT_CAUSE_FILE = "root_cause.txt"
_FAULT_DIRECTORY = re.compile(r"(?P<service>[a-z0-9-]+)_(?P<fault>[a-z0-9]+)(?:_(?P<suffix>[0-9]+))?")
_CASE_DEPTH = 3
"""A case directory's depth in its zip: ``<suite>/<service>_<fault>/<repetition>``."""
_REPETITION = re.compile(r"[0-9]+")
_INJECT_TIME = re.compile(r"[0-9]+")
_STRAY_FILES = frozenset({".DS_Store"})
"""Files the zips hold outside case directories that carry no data: macOS folder metadata."""
_STRAY_ARCHIVES = frozenset({"RE2-OB/checkoutservice_cpu/multi-source-data.zip"})
"""The zipped copy of the multi-source demo directory in :data:`~release.NOT_CASES`."""


class ZipMember(NamedTuple):
    """One file inside one zip, hashable so a reader can cache what it parsed."""

    archive: Path
    name: str


@dataclass(frozen=True)
class Case:
    """One failure case: its ground truth, its inject time, and the files of its directory."""

    suite: Suite
    archive: Path
    fault_directory: str
    """The case's parent directory as published, such as ``checkoutservice_cpu``."""
    repetition: str
    """The case's directory as published, such as ``1``."""
    service: str
    """The root cause service: the ground truth of the coarse-grained task."""
    fault: Fault
    inject_s: int
    """The fault injection time, in Unix seconds, from ``inject_time.txt``."""
    files: frozenset[str]
    root_cause_log: str | None
    """The text of ``root_cause.txt``, which 8 RE3-SS cases hold."""

    @property
    def path(self) -> str:
        """Return the case's directory inside its zip, such as ``RE2-OB/checkoutservice_cpu/1``."""
        return f"{self.suite}/{self.fault_directory}/{self.repetition}"

    def member(self, filename: str) -> ZipMember:
        """Return one file of the case's directory."""
        return ZipMember(self.archive, f"{self.path}/{filename}")


def list_cases(archive: Path, suite: Suite) -> tuple[Case, ...]:
    """Read one zip's directory listing and return its cases, in path order.

    Args:
        archive: The suite's zip.
        suite: The suite the zip holds.

    Returns:
        Every case of the zip, without the directories of :data:`~release.NOT_CASES`.

    Raises:
        TimeFFormatError: If an entry is outside the expected layout, a directory name doesn't parse, or
            an inject time isn't a whole number of seconds.
    """
    with zipfile.ZipFile(archive) as opened:
        names = [info.filename for info in opened.infolist() if not info.is_dir()]
        directories: defaultdict[str, set[str]] = defaultdict(set)
        for name in names:
            parent, _, filename = name.rpartition("/")
            if filename in _STRAY_FILES or name in _STRAY_ARCHIVES:
                continue
            if len(parent.split("/")) != _CASE_DEPTH or not parent.startswith(f"{suite}/"):
                raise TimeFFormatError(f"{archive}: {name!r} is not a file of a case directory")
            directories[parent].add(filename)
        cases = []
        for path in sorted(directories, key=_path_order):
            if path in NOT_CASES:
                continue
            files = frozenset(directories[path])
            _, fault_directory, repetition = path.split("/")
            service, fault = _parse_fault_directory(suite, fault_directory, archive)
            if not _REPETITION.fullmatch(repetition):
                raise TimeFFormatError(f"{archive}: {path!r} doesn't end in a repetition number")
            for required in (INJECT_TIME_FILE, suite.metrics_file):
                if required not in files:
                    raise TimeFFormatError(f"{archive}: {path!r} has no {required}")
            inject = opened.read(f"{path}/{INJECT_TIME_FILE}").decode().strip()
            if not _INJECT_TIME.fullmatch(inject):
                raise TimeFFormatError(f"{archive}: {path}/{INJECT_TIME_FILE} holds {inject!r}, not Unix seconds")
            root_cause = opened.read(f"{path}/{ROOT_CAUSE_FILE}").decode().strip() if ROOT_CAUSE_FILE in files else None
            cases.append(
                Case(
                    suite=suite,
                    archive=archive,
                    fault_directory=fault_directory,
                    repetition=repetition,
                    service=service,
                    fault=fault,
                    inject_s=int(inject),
                    files=files,
                    root_cause_log=root_cause,
                )
            )
    missing = sorted(path for path in NOT_CASES if path.startswith(f"{suite}/") and path not in directories)
    if missing:
        raise TimeFFormatError(f"{archive}: has no {missing}, which the pinned zip holds")
    return tuple(cases)


def _path_order(path: str) -> tuple[str, int]:
    fault_directory, _, repetition = path.rpartition("/")
    return fault_directory, int(repetition) if repetition.isdigit() else -1


def _parse_fault_directory(suite: Suite, fault_directory: str, archive: Path) -> tuple[str, Fault]:
    match = _FAULT_DIRECTORY.fullmatch(fault_directory)
    if match is None or match["fault"] not in set(Fault):
        raise TimeFFormatError(f"{archive}: {fault_directory!r} is not <service>_<fault>")
    if match["suffix"] is not None and f"{suite}/{fault_directory}" not in FAULT_DIRECTORY_SUFFIXES:
        raise TimeFFormatError(f"{archive}: {fault_directory!r} has a suffix the pinned zip doesn't have")
    return match["service"], Fault(match["fault"])
