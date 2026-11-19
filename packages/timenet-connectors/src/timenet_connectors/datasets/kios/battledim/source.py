"""Read the BattLeDIM release: the SCADA and leak tables, the leak list and the network topology.

The release holds one semicolon-separated CSV per year and sensor type, with a decimal comma and a
naive ``Timestamp`` column on a 5 minute grid. ``dataset_configuration.yaml`` is the configuration
that generated the data. It lists every leak and the sensor of each type. ``L-TOWN.inp`` is the
EPANET model of the network. :class:`Release` reads them all once and checks that they agree.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import yaml

from timenet.errors import TimeNetConnectorError


STEP = timedelta(minutes=5)
YEARS = (2018, 2019)
# The SCADA file of each sensor type, and the key that lists its sensors in the configuration.
SCADA_FILES = {
    "pressure": ("SCADA_Pressures", "pressure_sensors"),
    "flow": ("SCADA_Flows", "flow_sensors"),
    "tank_level": ("SCADA_Levels", "level_sensors"),
    "amr_demand": ("SCADA_Demands", "amrs"),
}
LEAK_TYPES = ("abrupt", "incipient")
_TIME_FORMAT = "%Y-%m-%d %H:%M"
_LEAK_FIELDS = 6  # linkID, startTime, endTime, leakDiameter (m), leakType, peakTime
_REPAIR_FIELDS = 2  # linkID, endTime
_LINK_SECTIONS = frozenset({"[PIPES]", "[PUMPS]", "[VALVES]"})


def _parse_time(text: str, where: str) -> datetime:
    """Read a naive release time as UTC; the release states no time zone.

    Returns:
        The time, in UTC.

    Raises:
        TimeNetConnectorError: If the text is not a ``YYYY-MM-DD HH:MM`` time on the 5 minute grid.
    """
    try:
        moment = datetime.strptime(text.strip(), _TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError as error:
        raise TimeNetConnectorError(f"{where}: {text!r} is not a {_TIME_FORMAT} time") from error
    if moment.minute % 5:
        raise TimeNetConnectorError(f"{where}: {text!r} is not on the 5 minute grid")
    return moment


@dataclass(frozen=True)
class Table:
    """One CSV of the release: its first timestamp and one float64 column per sensor, in file order."""

    start: datetime
    columns: dict[str, np.ndarray]

    @property
    def n_rows(self) -> int:
        """The number of 5 minute steps."""
        return len(next(iter(self.columns.values())))


def read_table(path: Path) -> Table:
    """Parse one semicolon-separated, decimal-comma CSV and check its time grid and values.

    Returns:
        The table, with every value column as float64.

    Raises:
        TimeNetConnectorError: If the file has no value column, a cell that is not a timestamp or a
            number, a step that is not 5 minutes, a missing value, or a value that is not finite.
    """
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(";")
    if header[0] != "Timestamp" or len(header) == 1:
        raise TimeNetConnectorError(f"{path.name}: expected a Timestamp column and sensor columns, got {header}")
    names = header[1:]
    if len(set(names)) != len(names):
        raise TimeNetConnectorError(f"{path.name}: repeats a column name")
    try:
        table = pacsv.read_csv(
            path,
            parse_options=pacsv.ParseOptions(delimiter=";"),
            convert_options=pacsv.ConvertOptions(
                decimal_point=",",
                column_types={"Timestamp": pa.timestamp("s"), **dict.fromkeys(names, pa.float64())},
                timestamp_parsers=["%Y-%m-%d %H:%M:%S"],
            ),
        )
    except pa.ArrowInvalid as error:
        raise TimeNetConnectorError(f"{path.name}: cannot be read as timestamps and numbers: {error}") from error
    times = table.column("Timestamp").to_numpy().astype("datetime64[s]").astype(np.int64)
    if len(times) == 0:
        raise TimeNetConnectorError(f"{path.name}: holds no rows")
    steps = np.diff(times)
    if (steps != STEP.total_seconds()).any():
        bad = int(np.flatnonzero(steps != STEP.total_seconds())[0])
        raise TimeNetConnectorError(f"{path.name}: row {bad + 2} is not 5 minutes after the row before it")
    columns = {}
    for name in names:
        column = table.column(name)
        if column.null_count:
            raise TimeNetConnectorError(f"{path.name}: column {name} has {column.null_count} missing values")
        values = column.to_numpy()
        if not np.isfinite(values).all():
            raise TimeNetConnectorError(f"{path.name}: column {name} holds a value that is not finite")
        columns[name] = values
    return Table(start=datetime.fromtimestamp(int(times[0]), tz=UTC), columns=columns)


@dataclass(frozen=True)
class Leak:
    """One leak of the configuration, with its outflow from the leak tables.

    ``end`` is the last 5 minute step with the leak, as the configuration and the leak tables write it.
    A leak that is never repaired ends at the last step of the recording.
    """

    pipe: str
    start: datetime
    end: datetime
    diameter_m: float
    leak_type: str
    peak: datetime
    repaired: bool


def _parse_leak(line: object, where: str) -> Leak:
    """Read one ``linkID, startTime, endTime, leakDiameter (m), leakType, peakTime`` line.

    Returns:
        The leak, not yet marked as repaired.

    Raises:
        TimeNetConnectorError: If the line does not hold six fields of the stated kinds.
    """
    fields = [field.strip() for field in str(line).split(",")]
    if len(fields) != _LEAK_FIELDS:
        raise TimeNetConnectorError(f"{where}: leak line {line!r} does not hold six fields")
    pipe, start, end, diameter, leak_type, peak = fields
    if leak_type not in LEAK_TYPES:
        raise TimeNetConnectorError(f"{where}: leak {pipe} has type {leak_type!r}, not one of {LEAK_TYPES}")
    try:
        diameter_m = float(diameter)
    except ValueError as error:
        raise TimeNetConnectorError(f"{where}: leak {pipe} has diameter {diameter!r}") from error
    leak = Leak(
        pipe=pipe,
        start=_parse_time(start, where),
        end=_parse_time(end, where),
        diameter_m=diameter_m,
        leak_type=leak_type,
        peak=_parse_time(peak, where),
        repaired=False,
    )
    if not leak.start <= leak.peak <= leak.end:
        raise TimeNetConnectorError(f"{where}: leak {pipe} peaks outside its start and end")
    return leak


class Network:
    """The links of ``L-TOWN.inp`` as an undirected graph, to find the pressure sensor nearest a pipe."""

    def __init__(self, path: Path) -> None:
        """Read the ``[PIPES]``, ``[PUMPS]`` and ``[VALVES]`` sections: a link id and its two nodes.

        Raises:
            TimeNetConnectorError: If a link line names fewer than two nodes or a link id repeats.
        """
        self.links: dict[str, tuple[str, str]] = {}
        self.pipes: set[str] = set()
        section = ""
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
            text = line.split(";", 1)[0].strip()
            if not text:
                continue
            if text.startswith("["):
                section = text.upper()
                continue
            if section not in _LINK_SECTIONS:
                continue
            link, *nodes = text.split()
            if len(nodes) < len(("node1", "node2")):
                raise TimeNetConnectorError(f"{path.name}: link line {line!r} names fewer than two nodes")
            if link in self.links:
                raise TimeNetConnectorError(f"{path.name}: link {link} appears twice")
            self.links[link] = (nodes[0], nodes[1])
            if section == "[PIPES]":
                self.pipes.add(link)
        self._neighbours: dict[str, set[str]] = {}
        for a, b in self.links.values():
            self._neighbours.setdefault(a, set()).add(b)
            self._neighbours.setdefault(b, set()).add(a)

    @property
    def nodes(self) -> set[str]:
        """Every node that a link touches."""
        return set(self._neighbours)

    def nearest(self, pipe: str, sensors: list[str]) -> tuple[str, int]:
        """Find the sensor node with the fewest links to either end of ``pipe``.

        Ties go to the sensor listed first, so the answer does not depend on set order.

        Returns:
            The sensor and its number of links from the nearer end of the pipe.

        Raises:
            TimeNetConnectorError: If no sensor is connected to the pipe.
        """
        hops = dict.fromkeys(self.links[pipe], 0)
        queue = deque(hops)
        while queue:
            node = queue.popleft()
            for neighbour in self._neighbours[node]:
                if neighbour not in hops:
                    hops[neighbour] = hops[node] + 1
                    queue.append(neighbour)
        reachable = [sensor for sensor in sensors if sensor in hops]
        if not reachable:
            raise TimeNetConnectorError(f"no pressure sensor is connected to pipe {pipe}")
        best = min(reachable, key=lambda sensor: (hops[sensor], sensors.index(sensor)))
        return best, hops[best]


class Release:
    """The parsed release: the SCADA tables of both years, the leaks and the network.

    Reading checks that the two years join on one 5 minute grid with the same sensors, that the
    configuration, the leak tables and the repair report agree, and that every sensor and leak pipe
    is in the network model.
    """

    def __init__(self, root: Path) -> None:
        """Read and check the release in ``root``.

        Raises:
            TimeNetConnectorError: If a file is missing or the files disagree.
        """
        self.root = root
        config = self._yaml("dataset_configuration.yaml")
        if not isinstance(config, dict):
            raise TimeNetConnectorError("dataset_configuration.yaml is not a mapping")
        self.sensors: dict[str, list[str]] = {}
        for kind, (_, key) in SCADA_FILES.items():
            listed = config.get(key)
            if not isinstance(listed, list) or not listed:
                raise TimeNetConnectorError(f"dataset_configuration.yaml lists no {key}")
            self.sensors[kind] = [str(sensor) for sensor in listed]
            if len(set(self.sensors[kind])) != len(self.sensors[kind]):
                raise TimeNetConnectorError(f"dataset_configuration.yaml lists a {key} entry twice")
        self.scada = {kind: self._join(name, self.sensors[kind]) for kind, (name, _) in SCADA_FILES.items()}
        self.start = self.scada["pressure"][0]
        self.n_steps = len(next(iter(self.scada["pressure"][1].values())))
        for kind, (start, columns) in self.scada.items():
            if start != self.start or len(next(iter(columns.values()))) != self.n_steps:
                raise TimeNetConnectorError(f"the {kind} tables do not cover the pressure tables' time range")
        times = config.get("times") or {}
        if _parse_time(str(times.get("StartTime")), "times") != self.start or (
            _parse_time(str(times.get("EndTime")), "times") != self.end
        ):
            raise TimeNetConnectorError("dataset_configuration.yaml times differ from the SCADA tables")
        self.leak_outflow = self._join_leaks()
        self.leaks = self._read_leaks(config)
        self.network = Network(self._path("L-TOWN.inp"))
        self._check_network()

    @property
    def end(self) -> datetime:
        """The last 5 minute step of the recording."""
        return self.start + STEP * (self.n_steps - 1)

    def step_of(self, moment: datetime) -> int:
        """The index of ``moment`` on the recording's 5 minute grid."""
        return int((moment - self.start) / STEP)

    def micros(self, moment: datetime) -> int:
        """The offset of ``moment`` from the start of the recording, in whole microseconds."""
        return (moment - self.start) // timedelta(microseconds=1)

    def _path(self, name: str) -> Path:
        path = self.root / name
        if not path.is_file():
            raise TimeNetConnectorError(f"the release in {self.root} is missing {name}")
        return path

    def _yaml(self, name: str) -> object:
        try:
            return yaml.safe_load(self._path(name).read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise TimeNetConnectorError(f"{name} is not valid YAML") from error

    def _join(self, name: str, sensors: list[str]) -> tuple[datetime, dict[str, np.ndarray]]:
        """Join the 2018 and 2019 tables of one sensor type into one recording.

        Returns:
            The first timestamp, and each sensor's values over both years.

        Raises:
            TimeNetConnectorError: If the years name other sensors than the configuration lists, or
                the second year does not start 5 minutes after the first ends.
        """
        tables = [read_table(self._path(f"{year}_{name}.csv")) for year in YEARS]
        for year, table in zip(YEARS, tables, strict=True):
            if set(table.columns) != set(sensors):
                raise TimeNetConnectorError(
                    f"{year}_{name}.csv columns differ from the sensors that dataset_configuration.yaml lists"
                )
        first, second = tables
        if second.start != first.start + STEP * first.n_rows:
            raise TimeNetConnectorError(f"{YEARS[1]}_{name}.csv does not continue {YEARS[0]}_{name}.csv")
        return first.start, {
            sensor: np.concatenate([first.columns[sensor], second.columns[sensor]]) for sensor in sensors
        }

    def _join_leaks(self) -> dict[str, np.ndarray]:
        """Join the two leak tables into one outflow series per pipe, zero where a year does not list it.

        Returns:
            Each leaking pipe's outflow over both years, in m3/h.

        Raises:
            TimeNetConnectorError: If a leak table does not cover its year of the SCADA tables.
        """
        tables = [read_table(self._path(f"{year}_Leakages.csv")) for year in YEARS]
        offset = 0
        outflow: dict[str, np.ndarray] = {}
        for year, table in zip(YEARS, tables, strict=True):
            if table.start != self.start + STEP * offset:
                raise TimeNetConnectorError(f"{year}_Leakages.csv does not line up with the SCADA tables")
            for pipe, values in table.columns.items():
                series = outflow.setdefault(pipe, np.zeros(self.n_steps))
                series[offset : offset + table.n_rows] = values
            offset += table.n_rows
        if offset != self.n_steps:
            raise TimeNetConnectorError("the leak tables do not cover the SCADA tables")
        return outflow

    def _read_leaks(self, config: dict) -> list[Leak]:
        """Read the configured leaks and check them against the leak tables and the repair report.

        Returns:
            The leaks in order of their start, each marked as repaired or not.

        Raises:
            TimeNetConnectorError: If a configured leak is outside the recording or its leak table,
                a leak table has a pipe the configuration lacks, or a repair time disagrees.
        """
        # The list opens with a comment line that names the fields, which YAML reads as a null item.
        lines = [line for line in (config.get("leakages") or []) if line is not None]
        leaks = [_parse_leak(line, "dataset_configuration.yaml") for line in lines]
        if len({leak.pipe for leak in leaks}) != len(leaks):
            raise TimeNetConnectorError("dataset_configuration.yaml lists a pipe in two leaks")
        if {leak.pipe for leak in leaks} != set(self.leak_outflow):
            raise TimeNetConnectorError("the leak tables and dataset_configuration.yaml name different pipes")
        # A report of comment lines only, with no repair, reads as an empty YAML document.
        report = self._yaml("2018_Fixed_Leakages_Report.txt") or []
        if not isinstance(report, list):
            raise TimeNetConnectorError("2018_Fixed_Leakages_Report.txt is not a list of repairs")
        repairs = {}
        for line in report:
            fields = [field.strip() for field in str(line).split(",")]
            if len(fields) != _REPAIR_FIELDS:
                raise TimeNetConnectorError(f"2018_Fixed_Leakages_Report.txt: line {line!r} is not 'linkID, endTime'")
            if fields[0] in repairs:
                raise TimeNetConnectorError(f"2018_Fixed_Leakages_Report.txt repairs {fields[0]} twice")
            repairs[fields[0]] = _parse_time(fields[1], "2018_Fixed_Leakages_Report.txt")
        checked = []
        for leak in leaks:
            if not self.start <= leak.start <= leak.end <= self.end:
                raise TimeNetConnectorError(f"leak {leak.pipe} is outside the recording")
            if leak.pipe in repairs and repairs[leak.pipe] != leak.end:
                raise TimeNetConnectorError(f"leak {leak.pipe}: the repair report and the configuration disagree")
            # The report covers the historical year: every leak that ends within 2018 must be in it.
            if leak.end.year == YEARS[0] and leak.pipe not in repairs:
                raise TimeNetConnectorError(f"leak {leak.pipe} ends in {YEARS[0]} but the repair report lacks it")
            outflow = self.leak_outflow[leak.pipe]
            active = np.flatnonzero(outflow > 0)
            if len(active) == 0 or active[0] < self.step_of(leak.start) or active[-1] > self.step_of(leak.end) + 1:
                raise TimeNetConnectorError(f"leak {leak.pipe}: its outflow lies outside its configured start and end")
            checked.append(replace(leak, repaired=leak.end < self.end))
        if set(repairs) - {leak.pipe for leak in checked if leak.repaired}:
            raise TimeNetConnectorError("2018_Fixed_Leakages_Report.txt repairs a leak the configuration keeps open")
        return sorted(checked, key=lambda leak: (leak.start, leak.pipe))

    def _check_network(self) -> None:
        """Check that every sensor and leak pipe is in the network model.

        Raises:
            TimeNetConnectorError: If one is missing.
        """
        nodes = self.network.nodes
        for kind in ("pressure", "tank_level", "amr_demand"):
            missing = [sensor for sensor in self.sensors[kind] if sensor not in nodes]
            if missing:
                raise TimeNetConnectorError(f"L-TOWN.inp has no node for the {kind} sensors {missing}")
        missing = [sensor for sensor in self.sensors["flow"] if sensor not in self.network.links]
        if missing:
            raise TimeNetConnectorError(f"L-TOWN.inp has no link for the flow sensors {missing}")
        missing = [leak.pipe for leak in self.leaks if leak.pipe not in self.network.pipes]
        if missing:
            raise TimeNetConnectorError(f"L-TOWN.inp has no pipe for the leaks {missing}")
