"""Exercise full-day reuse, heterogeneous telemetry, graph columns, and canonical answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from openpyxl import Workbook  # ty: ignore[unresolved-import] - connector dependency
import pytest

from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import Split, TimeInterval
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.microsoft.openrca import connector as connector_module
from timenet_connectors.datasets.microsoft.openrca.connector import OpenRcaConnector
from timenet_connectors.datasets.microsoft.openrca.preparation import OpenRcaSource


_TZ = ZoneInfo("Asia/Shanghai")
_HOUR_US = 3_600_000_000
_DAY_US = 24 * _HOUR_US
_LEADING_US = 372_000_000


def _epoch(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_TZ).timestamp())


def _csv_rows(rows: Sequence[Mapping[str, object]]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _task_files(prefix: str, roots: Sequence[Mapping[str, object]], tasks: list[str]) -> dict[str, str]:
    queries = [
        {"task_index": task, "instruction": instruction, "scoring_points": "SECRET TARGET TEXT"}
        for task, instruction in zip(tasks, _instructions(roots), strict=True)
    ]
    return {
        f"{prefix}/query.csv": _csv_rows(queries),
        f"{prefix}/record.csv": _csv_rows(roots),
    }


def _instructions(roots: Sequence[Mapping[str, object]]) -> list[str]:
    output = []
    for root in roots:
        moment = datetime.fromtimestamp(cast("int", root["timestamp"]), _TZ)
        start = moment.replace(minute=moment.minute - moment.minute % 30, second=0)
        end = (
            start.replace(minute=start.minute + 30)
            if start.minute == 0
            else start.replace(hour=start.hour + 1, minute=0)
        )
        output.append(f"Find the failure between {start:%Y-%m-%d %H:%M:%S} and {end:%Y-%m-%d %H:%M:%S}.")
    return output


def _bank_files() -> dict[str, str]:
    day = "2021_03_04"
    at_midnight, at_incident = _epoch("2021-03-04 00:00:00"), _epoch("2021-03-04 09:05:00")
    roots = [
        {
            "level": "container",
            "component": "Tomcat01",
            "timestamp": at_incident,
            "datetime": "2021-03-04 09:05:00",
            "reason": "high CPU usage",
        },
        {
            "level": "container",
            "component": "Mysql01",
            "timestamp": at_incident + 60,
            "datetime": "2021-03-04 09:06:00",
            "reason": "high memory usage",
        },
    ]
    files = _task_files("Bank", roots, ["task_7", "task_3"])
    base = f"Bank/telemetry/{day}"
    files.update(
        {
            f"{base}/metric/metric_app.csv": _csv_rows(
                [
                    {"timestamp": at_incident, "rr": 20, "sr": 95, "cnt": 3, "mrt": 70, "tc": "ServiceA"},
                    {"timestamp": at_midnight, "rr": 10, "sr": 100, "cnt": 2, "mrt": 50, "tc": "ServiceA"},
                ]
            ),
            f"{base}/metric/metric_container.csv": _csv_rows(
                [
                    {
                        "timestamp": at_midnight,
                        "cmdb_id": "Tomcat01",
                        "kpi_name": "cpu",
                        "value": 10.0,
                    },
                    {
                        "timestamp": at_incident,
                        "cmdb_id": "Tomcat01",
                        "kpi_name": "cpu",
                        "value": 99.0,
                    },
                ]
            ),
            f"{base}/trace/trace_span.csv": _csv_rows(
                [
                    {
                        "timestamp": at_midnight - _LEADING_US // 1_000_000,
                        "cmdb_id": "Tomcat01",
                        "parent_id": "None",
                        "span_id": "span-root",
                        "trace_id": "trace-a",
                        "duration": 12,
                    },
                    {
                        "timestamp": at_incident,
                        "cmdb_id": "Mysql01",
                        "parent_id": "span-root",
                        "span_id": "span-child",
                        "trace_id": "trace-a",
                        "duration": 4,
                    },
                ]
            ),
            f"{base}/log/log_service.csv": _csv_rows(
                [
                    {
                        "log_id": "log-1",
                        "timestamp": at_incident,
                        "cmdb_id": "Tomcat01",
                        "log_name": "application",
                        "value": "request started",
                    }
                ]
            ),
        }
    )
    return files


def _market_files() -> dict[str, str]:
    output: dict[str, str] = {}
    for cloudbed, minute, component in (("cloudbed-1", 5, "node-1"), ("cloudbed-2", 35, "node-2")):
        day = "2022_03_20"
        timestamp = _epoch(f"2022-03-20 10:{minute:02d}:00")
        root = {
            "level": "node",
            "component": component,
            "timestamp": timestamp,
            "datetime": f"2022-03-20 10:{minute:02d}:00",
            "reason": "node CPU load",
        }
        prefix = f"Market/{cloudbed}"
        output.update(_task_files(prefix, [root], ["task_5"]))
        base = f"{prefix}/telemetry/{day}"
        long_metric = _csv_rows([{"timestamp": timestamp, "cmdb_id": component, "kpi_name": "cpu", "value": 88.0}])
        for metric in ("container", "mesh", "node", "runtime"):
            output[f"{base}/metric/metric_{metric}.csv"] = long_metric
        output[f"{base}/metric/metric_service.csv"] = _csv_rows(
            [{"service": "frontend-grpc", "timestamp": timestamp, "rr": 1, "sr": 90, "mrt": 3, "count": 4}]
        )
        output[f"{base}/trace/trace_span.csv"] = _csv_rows(
            [
                {
                    "timestamp": timestamp * 1000,
                    "cmdb_id": "frontend-0",
                    "span_id": "span-1",
                    "trace_id": "trace-1",
                    "duration": 3,
                    "type": "rpc",
                    "status_code": "0",
                    "operation_name": "Checkout",
                    "parent_span": "",
                }
            ]
        )
        log = _csv_rows(
            [
                {
                    "log_id": "log-1",
                    "timestamp": timestamp,
                    "cmdb_id": "frontend-0",
                    "log_name": "application",
                    "value": "severity: info",
                }
            ]
        )
        output[f"{base}/log/log_proxy.csv"] = log
        output[f"{base}/log/log_service.csv"] = log
    return output


def _telecom_files() -> dict[str, str]:
    day = "2020_05_29"
    timestamp = _epoch("2020-05-29 03:05:00")
    root = {
        "level": "node",
        "component": "os_001",
        "timestamp": timestamp,
        "datetime": "2020-05-29 03:05:00",
        "reason": "CPU fault",
    }
    output = _task_files("Telecom", [root], ["task_1"])
    base = f"Telecom/telemetry/{day}"
    output[f"{base}/metric/metric_app.csv"] = _csv_rows(
        [
            {
                "serviceName": "osb_001",
                "startTime": timestamp * 1000,
                "avg_time": 0.3,
                "num": 1,
                "succee_num": 1,
                "succee_rate": 1.0,
            }
        ]
    )
    long_metric = _csv_rows(
        [
            {
                "itemid": "999999996381330",
                "name": "cpu",
                "bomc_id": "ZJ-001",
                "timestamp": timestamp * 1000,
                "value": 97,
                "cmdb_id": "os_001",
            }
        ]
    )
    for metric in ("container", "middleware", "node", "service"):
        output[f"{base}/metric/metric_{metric}.csv"] = long_metric
    output[f"{base}/trace/trace_span.csv"] = _csv_rows(
        [
            {
                "callType": "OSB",
                "startTime": timestamp * 1000,
                "elapsedTime": 5,
                "success": "True",
                "traceId": "trace-1",
                "id": "span-1",
                "pid": "None",
                "cmdb_id": "os_001",
                "dsName": "",
                "serviceName": "osb_001",
            }
        ]
    )
    return output


def _write_files(root: Path, files: Mapping[str, str]) -> None:
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")


def _fixture_source(root: Path, *, omit_market_trace: bool = False) -> OpenRcaSource:
    _write_files(root, _bank_files())
    market_files = _market_files()
    if omit_market_trace:
        del market_files["Market/cloudbed-2/telemetry/2022_03_20/trace/trace_span.csv"]
    _write_files(root, market_files)
    _write_files(root, _telecom_files())
    workbook_path = root / "AppTelecomDeploymentList.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "deployment"
    sheet.append(["type", "host", "name"])
    sheet.append(["docker", "os_001", "docker_001"])
    workbook.save(workbook_path)
    return OpenRcaSource(cache_dir=root, root=root, revision="fixture")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("openrca")
    return OpenRcaConnector().convert([_fixture_source(root)])


def test_download_fetches_metadata_then_only_task_bearing_days(tmp_path, monkeypatch):
    fixture = _fixture_source(tmp_path / "source")
    calls = []

    def fake_snapshot(repo, revision, cache_dir, patterns):
        assert repo == "tracer-cloud/opensre"
        assert revision == "07714872ea2cec77c13f9dec17a688e9df9621d1"
        assert cache_dir == tmp_path / "cache"
        calls.append(patterns)
        return fixture.root

    monkeypatch.setattr(connector_module, "hub_snapshot", fake_snapshot)
    source = OpenRcaConnector().download(tmp_path / "cache")[0]

    assert source.root == fixture.root
    assert len(calls) == 2
    assert "Bank/query.csv" in calls[0]
    assert "Bank/telemetry/2021_03_04/trace/trace_span.csv" in calls[1]
    assert not any("query_alerts" in path for path in calls[1])


def _facts(annotations):
    return {annotation.key: annotation for annotation in annotations}


def test_tasks_share_full_day_records_and_use_incident_annotations(converted):
    assert len(converted.records) == 4
    assert len(converted.tasks) == 5
    assert {task.split for task in converted.tasks} == {Split.TEST}
    first, second = converted.tasks[:2]
    assert first.inputs[0] is second.inputs[0]
    assert first.scope is None and second.scope is None
    incident = _facts(first.input_annotations)["incident_window"]
    assert incident.span == TimeInterval.micros(
        9 * _HOUR_US + _LEADING_US,
        9 * _HOUR_US + _HOUR_US // 2 + _LEADING_US,
    )
    assert incident.metadata["upstream_bounds"] == "inclusive"
    assert "SECRET TARGET TEXT" not in repr(first)

    bank = first.inputs[0]
    assert bank.start_time.timestamp == _epoch("2021-03-04 00:00:00") * 1_000_000 - _LEADING_US
    assert bank.time_span == TimeInterval.micros(0, _DAY_US + _LEADING_US)
    metric_app = next(source for source in bank.walk_sources() if source.name == "metric_app.csv")
    service = next(source for source in metric_app.sources if source.name == "tc=ServiceA")
    rr = next(signal for signal in service.walk_signals() if signal.name == "rr")
    assert rr.to_arrow().to_pylist() == [10.0, 20.0]
    assert rr.time_offsets_us().tolist() == [
        _LEADING_US,
        9 * _HOUR_US + 5 * 60_000_000 + _LEADING_US,
    ]
    telemetry_schema = json.loads(cast("str", _facts(first.input_annotations)["telemetry_schema"].value))
    assert _facts(first.input_annotations)["telemetry_schema"].id == f"{bank.record_id}-telemetry-schema"
    trace_schema = next(item for item in telemetry_schema if item["file"] == "trace/trace_span.csv")
    assert trace_schema["source_timestamp_unit"] == "seconds_or_milliseconds"
    assert trace_schema["timestamp_factor_to_us"] == 1_000_000

    metric_container = next(source for source in bank.walk_sources() if source.name == "metric_container.csv")
    container_part = next(source for source in metric_container.walk_sources() if source.name.startswith("part "))
    assert {signal.name for signal in container_part.signals} == {"cmdb_id", "kpi_name", "value"}
    assert len({signal.time_axis.axis_id for signal in container_part.signals}) == 1


def test_targets_are_canonical_and_multi_root_answers_are_chronological(converted):
    first, second = converted.tasks[:2]
    target = json.loads(first.targets[0])
    assert list(target) == ["1", "2"]
    assert target["1"] == {
        "root cause component": "Tomcat01",
        "root cause occurrence datetime": "2021-03-04 09:05:00",
        "root cause reason": "high CPU usage",
    }
    assert json.loads(second.targets[0]) == {
        "1": {"root cause component": "Tomcat01"},
        "2": {"root cause component": "Mysql01"},
    }
    assert first.metadata["time_tolerance_seconds"] == 60
    assert "time_tolerance_seconds" not in second.metadata


def test_trace_columns_share_an_axis_and_preserve_nullable_graph_edges(converted):
    bank = converted.tasks[0].inputs[0]
    trace_table = next(source for source in bank.walk_sources() if source.name == "trace_span.csv")
    part = next(source for source in trace_table.walk_sources() if source.name.startswith("part "))
    assert len({signal.time_axis.axis_id for signal in part.signals}) == 1
    columns = {signal.name: signal.to_arrow().to_pylist() for signal in part.signals}
    assert columns["trace_id"] == ["trace-a", "trace-a"]
    assert columns["span_id"] == ["span-root", "span-child"]
    assert columns["parent_id"] == [None, "span-root"]
    assert part.signals[0].time_offsets_us().tolist()[0] == 0


def test_telecom_tasks_receive_deployment_topology(converted):
    telecom = next(task for task in converted.tasks if task.metadata["system"] == "telecom")
    facts = _facts(telecom.input_annotations)
    assert facts["deployment_topology"].value == [
        "deployment: type | host | name",
        "deployment: docker | os_001 | docker_001",
    ]
    assert facts["timezone"].value == "Asia/Shanghai"


def test_fixture_round_trips_records_tasks_and_shared_axes(converted, tmp_path):
    converted.derive_schema()
    with TimeFWriter(tmp_path / "registry", converted) as writer:
        writer.write()
    version = DatasetVersion.open_local(tmp_path / "registry/microsoft/openrca/1.0.0")
    with TimeFReader(version) as reader:
        assert len(tuple(reader.iter_records())) == 4
        tasks = tuple(reader.iter_tasks())
        assert len(tasks) == 5
        assert tasks[0].scope is None
        assert _facts(tasks[0].input_annotations)["incident_window"].span == TimeInterval.micros(
            9 * _HOUR_US + _LEADING_US,
            9 * _HOUR_US + _HOUR_US // 2 + _LEADING_US,
        )


def test_missing_required_source_file_fails_clearly(tmp_path):
    with pytest.raises(TimeFFormatError, match=r"trace/trace_span.csv"):
        OpenRcaConnector().convert([_fixture_source(tmp_path, omit_market_trace=True)])
