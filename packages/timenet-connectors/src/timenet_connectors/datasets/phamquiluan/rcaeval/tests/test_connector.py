from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar
import zipfile

import pytest

from timenet.dataset import Signal
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types.text import read_value
from timenet_connectors.datasets.phamquiluan.rcaeval import RCAEvalConnector
from timenet_connectors.datasets.phamquiluan.rcaeval.benchmark import Fault, indicator_column, window
from timenet_connectors.datasets.phamquiluan.rcaeval.loader import RCAEvalDatasetLoader
from timenet_connectors.datasets.phamquiluan.rcaeval.records import record_metadata
from timenet_connectors.datasets.phamquiluan.rcaeval.release import PinnedArchive, Suite, UntimedRows
from timenet_connectors.datasets.phamquiluan.rcaeval.tasks import RCAEvalIndicatorTask, RCAEvalServiceTask


_S = 1_000_000
_T0 = 1_700_000_000
_TRACE_HEADER = "time,traceID,spanID,serviceName,methodName,operationName,startTimeMillis,startTime,duration,statusCode,parentSpanID"

# RE1-OB: one file with two tables side by side, each with its own clock. The main table's last two rows
# have no time: one holds two values, the other none. The error table's clock is written as floats, as
# pandas writes a column with an empty field, and times the first of those rows.
_RE1_OB_TWO_CLOCKS = (
    "time,adservice_cpu,adservice_latency-90,frontend_mem,time,frontend_error\n"
    + "".join(
        f"{_T0 + second},{second}.5,0.{second},{100 + second},{_T0 + second}.0,{second % 2}\n" for second in range(10)
    )
    + f",9.9,9.8,,{_T0 + 10}.0,1\n,,,,,\n"
)
# RE1-OB: one table, with one missing value.
_RE1_OB_ONE_CLOCK = "time,adservice_cpu,adservice_latency-50,adservice_latency-90\n" + "".join(
    f"{_T0 + 20 + second},{second},{'' if second == 4 else second / 10},{second / 100}\n" for second in range(10)
)
# RE2-OB: metrics from T0+100, logs from 0.876543211 s before them, spans out of order, and derived series
# from 5 s before the metrics.
_RE2_OB_METRICS = "time,checkoutservice_cpu,checkoutservice_diskio\n" + "".join(
    f"{_T0 + 100 + second},0.{second},{75387 + second}\n" for second in range(10)
)
_RE2_OB_LOGS = (
    "time,timestamp,container_name,message,level,req_path,error\n"
    f"21:24,{(_T0 + 99) * 1000 * _S + 123456789},checkoutservice,first line,info,,\n"
    f'21:24,{(_T0 + 101) * 1000 * _S + 5},checkoutservice,"Exception\n\tat Checkout.java:12",error,/cart,\n'
    f"21:25,{(_T0 + 102) * 1000 * _S},frontend,NA,,,\n"
    f"21:25,{(_T0 + 102) * 1000 * _S},frontend,,,,\n"
)
_RE2_OB_TRACES = (
    f"{_TRACE_HEADER}\n"
    f"21:24,t1,s1,frontend,Get,GET /cart,{(_T0 + 101) * 1000},{(_T0 + 101) * _S + 500},1200,0.0,\n"
    f"21:24,t1,s2,checkoutservice,,PlaceOrder,{(_T0 + 100) * 1000},{(_T0 + 100) * _S + 250},300,,s1\n"
    f"21:24,t2,s3,frontend,Get,GET /,{(_T0 + 101) * 1000},{(_T0 + 101) * _S + 500},80,14.0,\n"
)
_RE2_OB_LOGTS = f"time,checkoutservice_1,frontend_2\n{_T0 + 95},3,4\n{_T0 + 110},5,6\n"
_RE2_OB_TRACETS = f"time,frontend_Get,checkoutservice_PlaceOrder\n{_T0 + 95},0.0,\n{_T0 + 110},1.0,2.0\n"
_RE3_METRICS = "time,carts_cpu,front-end_mem\n" + "".join(f"{_T0 + 200 + second},1.0,2\n" for second in range(10))
_RE3_LOGS = (
    "time,timestamp,container_name,message,pod_name,node_name\n"
    f"03:20,{(_T0 + 201) * 1000 * _S},carts,POST /cart 500,carts-1,node-1\n"
)
_ROOT_CAUSE = "03:20,1700000201000000000,carts,POST /cart 500,carts-1,node-1"


def _write_zip(path: Path, files: Mapping[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def _release(tmp_path: Path) -> list[Path]:
    """Write four fixture zips in the layout of the record."""
    return [
        _write_zip(
            tmp_path / "RE1-OB.zip",
            {
                "RE1-OB/.DS_Store": "",
                "RE1-OB/adservice_cpu/1/inject_time.txt": f"{_T0 + 5}\n",
                "RE1-OB/adservice_cpu/1/data.csv": _RE1_OB_TWO_CLOCKS,
                "RE1-OB/adservice_delay/1/inject_time.txt": f"{_T0 + 25}",
                "RE1-OB/adservice_delay/1/data.csv": _RE1_OB_ONE_CLOCK,
            },
        ),
        _write_zip(
            tmp_path / "RE2-OB.zip",
            {
                "RE2-OB/checkoutservice_disk/1/inject_time.txt": f"{_T0 + 105}",
                "RE2-OB/checkoutservice_disk/1/simple_metrics.csv": _RE2_OB_METRICS,
                "RE2-OB/checkoutservice_disk/1/metrics.csv": "time,not_read\n",
                "RE2-OB/checkoutservice_disk/1/logs.csv": _RE2_OB_LOGS,
                "RE2-OB/checkoutservice_disk/1/traces.csv": _RE2_OB_TRACES,
                "RE2-OB/checkoutservice_disk/1/logts.csv": _RE2_OB_LOGTS,
                "RE2-OB/checkoutservice_disk/1/tracets_err.csv": _RE2_OB_TRACETS,
                "RE2-OB/checkoutservice_disk/1/tracets_lat.csv": _RE2_OB_TRACETS,
                "RE2-OB/checkoutservice_cpu/multi-source-data/inject_time.txt": f"{_T0 + 105}",
                "RE2-OB/checkoutservice_cpu/multi-source-data.zip": "",
            },
        ),
        _write_zip(
            tmp_path / "RE3-SS.zip",
            {
                "RE3-SS/carts_f1/1/inject_time.txt": f"{_T0 + 205}",
                "RE3-SS/carts_f1/1/simple_metrics.csv": _RE3_METRICS,
                "RE3-SS/carts_f1/1/logs.csv": _RE3_LOGS,
                "RE3-SS/carts_f1/1/traces.csv": f"{_TRACE_HEADER}\n",
                "RE3-SS/carts_f1/1/root_cause.txt": f"{_ROOT_CAUSE}\n",
            },
        ),
        _write_zip(
            tmp_path / "RE3-TT.zip",
            {
                "RE3-TT/ts-route-service_f3_1/2/inject_time.txt": f"{_T0 + 205}",
                "RE3-TT/ts-route-service_f3_1/2/simple_metrics.csv": _RE3_METRICS.replace("carts", "ts-route-service"),
            },
        ),
    ]


_NO_HASH = "0" * 64
_FIXTURE_ARCHIVES = {
    Suite.RE1_OB: PinnedArchive(
        _NO_HASH, cases=2, logs=0, traces=0, header_only_traces=0, derived=0, root_cause_files=0
    ),
    Suite.RE2_OB: PinnedArchive(
        _NO_HASH, cases=1, logs=1, traces=1, header_only_traces=0, derived=1, root_cause_files=0
    ),
    Suite.RE3_SS: PinnedArchive(
        _NO_HASH, cases=1, logs=1, traces=0, header_only_traces=1, derived=0, root_cause_files=1
    ),
    Suite.RE3_TT: PinnedArchive(
        _NO_HASH, cases=1, logs=0, traces=0, header_only_traces=0, derived=0, root_cause_files=0
    ),
}
_FIXTURE_UNTIMED = {
    ("RE1-OB/adservice_cpu/1", "time"): UntimedRows(rows=2, values=2),
    ("RE1-OB/adservice_cpu/1", "time.1"): UntimedRows(rows=1, values=0),
}


class _FixtureConnector(RCAEvalConnector):
    """The connector, pinned to the fixtures' inventory instead of the record's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_ARCHIVES: ClassVar[Mapping[Suite, PinnedArchive]] = _FIXTURE_ARCHIVES
    SOURCE_UNTIMED_ROWS: ClassVar[Mapping[tuple[str, str], UntimedRows]] = _FIXTURE_UNTIMED


def _axis(signal: Signal) -> tuple:
    axis = signal.time_axis
    if isinstance(axis, RegularAxis):
        return ("regular", int(axis.period_us), axis.offset_us)
    assert isinstance(axis, IrregularAxis)
    return ("irregular", axis.first_us, axis.last_us)


def _output(dataset) -> dict:
    """Everything the dataset holds but values, as plain data."""
    records = []
    for record in dataset.records:
        records.append(
            {
                "start": record.start_time.timestamp,
                "metadata": record_metadata(record).model_dump(mode="json"),
                "annotations": [
                    (a.key, read_value(a.value), None if a.span is None else a.span.start_us)
                    for a in record.annotations
                ],
                "sources": {
                    source.name: (
                        dict(source.metadata),
                        {s.name: (s.spec.spec_type, s.spec.dtype, s.n_values, _axis(s)) for s in source.signals},
                    )
                    for source in record.sources
                },
            }
        )
    tasks = [
        (
            task.metadata["case"],
            task.targets[0].name if isinstance(task.targets[0], Signal) else read_value(task.targets[0]),
            task.metadata.get("authors_indicator"),
            (task.scope.start_us, task.scope.end_us),
            tuple(a.key for a in task.input_annotations),
            tuple((a.key, read_value(a.value)) for a in task.annotations),
            task.split,
            task.prompt,
        )
        for task in dataset.iter_tasks()
    ]
    return {
        "annotations": [(a.key, read_value(a.value)) for a in dataset.annotations],
        "records": records,
        "tasks": tasks,
    }


_METRIC = "rcaeval_metric_{}"
_EXPECTED = {
    "annotations": [("source_url", "https://zenodo.org/records/14590730"), ("source_doi", "10.5281/zenodo.14590730")],
    "records": [
        {
            "start": _T0 * _S,
            "metadata": {"subject_id": "Online Boutique", "suite": "RE1-OB"},
            "annotations": [("inject_time", _T0 + 5, 5 * _S), ("system", "Online Boutique", None)],
            "sources": {
                "metrics": (
                    {
                        "file": "data.csv",
                        "column_order": ["adservice_cpu", "adservice_latency-90", "frontend_mem", "frontend_error"],
                        "clocks": {
                            "time": ["adservice_cpu", "adservice_latency-90", "frontend_mem"],
                            "time.1": ["frontend_error"],
                        },
                    },
                    {
                        "adservice_cpu": (_METRIC.format("cpu"), "float64", 10, ("regular", _S, 0)),
                        "adservice_latency-90": (_METRIC.format("latency_90"), "float64", 10, ("regular", _S, 0)),
                        "frontend_mem": (_METRIC.format("mem"), "float64", 10, ("regular", _S, 0)),
                        "frontend_error": (_METRIC.format("error"), "float64", 11, ("regular", _S, 0)),
                    },
                )
            },
        },
        {
            "start": (_T0 + 20) * _S,
            "metadata": {"subject_id": "Online Boutique", "suite": "RE1-OB"},
            "annotations": [("inject_time", _T0 + 25, 5 * _S), ("system", "Online Boutique", None)],
            "sources": {
                "metrics": (
                    {
                        "file": "data.csv",
                        "column_order": ["adservice_cpu", "adservice_latency-50", "adservice_latency-90"],
                    },
                    {
                        "adservice_cpu": (_METRIC.format("cpu"), "float64", 10, ("regular", _S, 0)),
                        "adservice_latency-50": (_METRIC.format("latency_50"), "float64", 10, ("regular", _S, 0)),
                        "adservice_latency-90": (_METRIC.format("latency_90"), "float64", 10, ("regular", _S, 0)),
                    },
                )
            },
        },
        {
            # The derived series start first, 5 s before the metrics.
            "start": (_T0 + 95) * _S,
            "metadata": {"subject_id": "Online Boutique", "suite": "RE2-OB"},
            "annotations": [("inject_time", _T0 + 105, 10 * _S), ("system", "Online Boutique", None)],
            "sources": {
                "metrics": (
                    {"file": "simple_metrics.csv", "column_order": ["checkoutservice_cpu", "checkoutservice_diskio"]},
                    {
                        "checkoutservice_cpu": (_METRIC.format("cpu"), "float64", 10, ("regular", _S, 5 * _S)),
                        "checkoutservice_diskio": (_METRIC.format("diskio"), "float64", 10, ("regular", _S, 5 * _S)),
                    },
                ),
                "logs": (
                    {
                        "file": "logs.csv",
                        "column_order": [
                            "time",
                            "timestamp",
                            "container_name",
                            "message",
                            "level",
                            "req_path",
                            "error",
                        ],
                    },
                    {
                        name: (f"rcaeval_logs_{name}", "str", 4, ("irregular", 4 * _S + 123456, 7 * _S))
                        for name in ("time", "container_name", "message", "level", "req_path", "error")
                    }
                    | {
                        "timestamp_sub_us_ns": (
                            "rcaeval_logs_timestamp_sub_us_ns",
                            "uint16",
                            4,
                            ("irregular", 4 * _S + 123456, 7 * _S),
                        )
                    },
                ),
                "traces": (
                    {"file": "traces.csv", "column_order": _TRACE_HEADER.split(",")},
                    {
                        name: (
                            f"rcaeval_traces_{name}",
                            "float64" if name in {"startTimeMillis", "duration", "statusCode"} else "str",
                            3,
                            ("irregular", 5 * _S + 250, 6 * _S + 500),
                        )
                        for name in _TRACE_HEADER.split(",")
                        if name != "startTime"
                    }
                    | {
                        "source_row": (
                            "rcaeval_traces_source_row",
                            "uint32",
                            3,
                            ("irregular", 5 * _S + 250, 6 * _S + 500),
                        )
                    },
                ),
                **{
                    name: (
                        {
                            "file": f"{name}.csv",
                            "column_order": columns,
                        },
                        dict.fromkeys(columns, (f"rcaeval_{name}", "float64", 2, ("regular", 15 * _S, 0))),
                    )
                    for name, columns in (
                        ("logts", ["checkoutservice_1", "frontend_2"]),
                        ("tracets_err", ["frontend_Get", "checkoutservice_PlaceOrder"]),
                        ("tracets_lat", ["frontend_Get", "checkoutservice_PlaceOrder"]),
                    )
                },
            },
        },
        {
            "start": (_T0 + 200) * _S,
            "metadata": {"subject_id": "Sock Shop", "suite": "RE3-SS"},
            "annotations": [("inject_time", _T0 + 205, 5 * _S), ("system", "Sock Shop", None)],
            "sources": {
                "metrics": (
                    {"file": "simple_metrics.csv", "column_order": ["carts_cpu", "front-end_mem"]},
                    {
                        "carts_cpu": (_METRIC.format("cpu"), "float64", 10, ("regular", _S, 0)),
                        "front-end_mem": (_METRIC.format("mem"), "float64", 10, ("regular", _S, 0)),
                    },
                ),
                "logs": (
                    {
                        "file": "logs.csv",
                        "column_order": ["time", "timestamp", "container_name", "message", "pod_name", "node_name"],
                    },
                    {
                        name: (f"rcaeval_logs_{name}", "str", 1, ("irregular", _S, _S))
                        for name in ("time", "container_name", "message", "pod_name", "node_name")
                    }
                    | {"timestamp_sub_us_ns": ("rcaeval_logs_timestamp_sub_us_ns", "uint16", 1, ("irregular", _S, _S))},
                ),
            },
        },
        {
            "start": (_T0 + 200) * _S,
            "metadata": {"subject_id": "Train Ticket", "suite": "RE3-TT"},
            "annotations": [("inject_time", _T0 + 205, 5 * _S), ("system", "Train Ticket", None)],
            "sources": {
                "metrics": (
                    {"file": "simple_metrics.csv", "column_order": ["ts-route-service_cpu", "front-end_mem"]},
                    {
                        "ts-route-service_cpu": (_METRIC.format("cpu"), "float64", 10, ("regular", _S, 0)),
                        "front-end_mem": (_METRIC.format("mem"), "float64", 10, ("regular", _S, 0)),
                    },
                ),
            },
        },
    ],
    # Every window is cut at the metrics: the fixtures are shorter than 600 s on each side.
    "tasks": [
        (
            "RE1-OB/adservice_cpu/1",
            "adservice",
            None,
            (0, 10 * _S),
            ("inject_time", "system"),
            (("fault", "cpu"),),
            None,
            None,
        ),
        (
            "RE1-OB/adservice_cpu/1",
            "adservice_cpu",
            "adservice_cpu",
            (0, 10 * _S),
            ("inject_time", "system"),
            (("fault", "cpu"),),
            None,
            None,
        ),
        (
            "RE1-OB/adservice_delay/1",
            "adservice",
            None,
            (0, 10 * _S),
            ("inject_time", "system"),
            (("fault", "delay"),),
            None,
            None,
        ),
        (
            "RE1-OB/adservice_delay/1",
            "adservice_latency-90",
            "adservice_latency",
            (0, 10 * _S),
            ("inject_time", "system"),
            (("fault", "delay"),),
            None,
            None,
        ),
        (
            "RE2-OB/checkoutservice_disk/1",
            "checkoutservice",
            None,
            (5 * _S, 15 * _S),
            ("inject_time", "system"),
            (("fault", "disk"),),
            None,
            None,
        ),
        (
            "RE2-OB/checkoutservice_disk/1",
            "checkoutservice_diskio",
            "checkoutservice_diskio",
            (5 * _S, 15 * _S),
            ("inject_time", "system"),
            (("fault", "disk"),),
            None,
            None,
        ),
        (
            "RE3-SS/carts_f1/1",
            "carts",
            None,
            (0, 10 * _S),
            ("inject_time", "system"),
            (("fault", "f1"), ("root_cause_log", _ROOT_CAUSE)),
            None,
            None,
        ),
        (
            "RE3-TT/ts-route-service_f3_1/2",
            "ts-route-service",
            None,
            (0, 10 * _S),
            ("inject_time", "system"),
            (("fault", "f3"),),
            None,
            None,
        ),
    ],
}


def _signal(dataset, record: int, source: str, name: str) -> Signal:
    (match,) = (
        s for src in dataset.records[record].sources if src.name == source for s in src.signals if s.name == name
    )
    return match


def test_converts_the_fixture(tmp_path):
    dataset = _FixtureConnector().convert(_release(tmp_path))

    assert _output(dataset) == _EXPECTED


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _FixtureConnector().convert(_release(tmp_path))
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("phamquiluan/rcaeval", "1.0.0")) as reader:
        restored = reader.read()

    assert _output(restored) == _EXPECTED
    for record, source, name in [(0, "metrics", "frontend_error"), (2, "logs", "message"), (2, "traces", "source_row")]:
        assert _signal(restored, record, source, name).to_arrow().to_pylist() == (
            _signal(original, record, source, name).to_arrow().to_pylist()
        )


def test_metrics_keep_their_values_and_leave_untimed_rows_out(tmp_path):
    dataset = _FixtureConnector().convert(_release(tmp_path))

    # The main table's ten timed rows; the two rows without a time are left out.
    assert _signal(dataset, 0, "metrics", "adservice_cpu").to_arrow().to_pylist() == [s + 0.5 for s in range(10)]
    # The error table has its own clock, which times one row more.
    assert _signal(dataset, 0, "metrics", "frontend_error").to_arrow().to_pylist() == [*(s % 2 for s in range(10)), 1]
    # An empty field stays missing.
    assert _signal(dataset, 1, "metrics", "adservice_latency-50").to_arrow().to_pylist()[3:6] == [0.3, None, 0.5]
    # Integers become float64 exactly.
    assert _signal(dataset, 2, "metrics", "checkoutservice_diskio").to_arrow().to_pylist()[:2] == [75387.0, 75388.0]


def test_logs_keep_their_text_and_nanoseconds(tmp_path):
    dataset = _FixtureConnector().convert(_release(tmp_path))
    message = _signal(dataset, 2, "logs", "message")

    # A quoted newline is part of the message, and only an empty field is missing.
    assert message.to_arrow().to_pylist() == ["first line", "Exception\n\tat Checkout.java:12", "NA", None]
    assert list(message.time_offsets_us()) == [4 * _S + 123456, 6 * _S, 7 * _S, 7 * _S]
    assert _signal(dataset, 2, "logs", "timestamp_sub_us_ns").to_arrow().to_pylist() == [789, 5, 0, 0]


def test_spans_are_sorted_by_start_and_keep_their_file_row(tmp_path):
    dataset = _FixtureConnector().convert(_release(tmp_path))

    # The second span starts first; the other two start together and keep their file order.
    assert list(_signal(dataset, 2, "traces", "spanID").time_offsets_us()) == [5 * _S + 250, 6 * _S + 500, 6 * _S + 500]
    assert _signal(dataset, 2, "traces", "spanID").to_arrow().to_pylist() == ["s2", "s1", "s3"]
    assert _signal(dataset, 2, "traces", "source_row").to_arrow().to_pylist() == [1, 0, 2]
    assert _signal(dataset, 2, "traces", "statusCode").to_arrow().to_pylist() == [None, 0.0, 14.0]
    assert _signal(dataset, 2, "traces", "methodName").to_arrow().to_pylist() == [None, "Get", "Get"]


def test_indicator_targets_are_the_records_own_signals(tmp_path):
    dataset = _FixtureConnector().convert(_release(tmp_path))
    record_signals = {id(s) for record in dataset.records for s in record.walk_signals()}

    targets = [(task.inputs[0], (task.targets or (None,))[0]) for task in dataset.iter_tasks()]
    indicators = [(record, target) for record, target in targets if isinstance(target, Signal)]
    assert len(indicators) == 3
    assert all(id(target) in record_signals for _, target in indicators)
    assert all(any(target is s for s in record.walk_signals()) for record, target in indicators)


def test_tasks_of_selects_a_level(tmp_path):
    loader = RCAEvalDatasetLoader.parse_from(_release(tmp_path), archives=_FIXTURE_ARCHIVES, untimed=_FIXTURE_UNTIMED)

    assert [task.case.path for task in loader.tasks_of(RCAEvalIndicatorTask)] == [
        "RE1-OB/adservice_cpu/1",
        "RE1-OB/adservice_delay/1",
        "RE2-OB/checkoutservice_disk/1",
    ]
    assert len(list(loader.tasks_of(RCAEvalServiceTask))) == 5
    assert all(task.split is None for task in loader.tasks())


@pytest.mark.parametrize(
    ("fault", "suite", "metric"),
    [
        (Fault.CPU, Suite.RE1_OB, "cpu"),
        (Fault.MEM, Suite.RE2_TT, "mem"),
        (Fault.SOCKET, Suite.RE2_SS, "socket"),
        (Fault.DELAY, Suite.RE2_OB, "latency"),
        (Fault.LOSS, Suite.RE1_TT, "latency"),
        (Fault.DISK, Suite.RE1_SS, "latency"),
        (Fault.DISK, Suite.RE2_OB, "diskio"),
        (Fault.F5, Suite.RE3_OB, None),
    ],
)
def test_indicator_metric_follows_main_py(fault, suite, metric):
    assert fault.indicator_metric(suite) == metric


def test_indicator_column_applies_the_authors_rename():
    columns = ["svc_latency-50", "svc_latency-90", "svc_cpu"]

    assert indicator_column(columns, "svc", "latency", context="case") == "svc_latency-90"
    with pytest.raises(TimeFFormatError, match="expected one metric"):
        indicator_column([*columns, "svc_latency"], "svc", "latency", context="case")
    with pytest.raises(TimeFFormatError, match="expected one metric"):
        indicator_column(columns, "svc", "socket", context="case")


def test_window_is_600_s_each_side_cut_at_the_metrics():
    assert (window(1000 * _S, 0, 2000 * _S).start_us, window(1000 * _S, 0, 2000 * _S).end_us) == (400 * _S, 1600 * _S)
    assert (window(100 * _S, 50 * _S, 200 * _S).start_us, window(100 * _S, 50 * _S, 200 * _S).end_us) == (
        50 * _S,
        200 * _S,
    )


def test_inventory_mismatch_fails(tmp_path):
    archives = {**_FIXTURE_ARCHIVES, Suite.RE1_OB: _FIXTURE_ARCHIVES[Suite.RE1_OB]._replace(cases=3)}

    with pytest.raises(TimeFValidationError, match=r"RE1-OB\.zip: holds"):
        RCAEvalDatasetLoader.parse_from(_release(tmp_path), archives=archives, untimed=_FIXTURE_UNTIMED)


def test_untimed_rows_must_be_pinned(tmp_path):
    with pytest.raises(TimeFValidationError, match="metric rows without a time"):
        RCAEvalDatasetLoader.parse_from(_release(tmp_path), archives=_FIXTURE_ARCHIVES, untimed={})


def test_each_suite_loads_once(tmp_path):
    refs = _release(tmp_path)

    with pytest.raises(TimeFValidationError, match="at most once"):
        RCAEvalDatasetLoader.parse_from([*refs, refs[0]], archives=_FIXTURE_ARCHIVES, untimed=_FIXTURE_UNTIMED)
    with pytest.raises(TimeFValidationError, match="at most once"):
        RCAEvalDatasetLoader.parse_from([tmp_path / "other.zip"], archives=_FIXTURE_ARCHIVES)


def _rewrite(tmp_path: Path, archive: str, member: str, content: str) -> list[Path]:
    """Write the fixture with one file replaced, or added if it is new."""
    refs = _release(tmp_path)
    path = tmp_path / archive
    with zipfile.ZipFile(path) as opened:
        files = {name: opened.read(name).decode() for name in opened.namelist()}
    _write_zip(path, files | {member: content})
    return refs


_CPU = "RE1-OB/adservice_cpu/1/"
_DISK = "RE2-OB/checkoutservice_disk/1/"


@pytest.mark.parametrize(
    ("archive", "member", "content", "message"),
    [
        ("RE1-OB.zip", _CPU + "inject_time.txt", "1700000005.5", "not Unix seconds"),
        ("RE1-OB.zip", _CPU + "inject_time.txt", f"{_T0 + 50}", "not inside the metrics"),
        ("RE1-OB.zip", _CPU + "data.csv", "time,adservice_cpu\n1,1\n3,1\n", "with no gap"),
        ("RE1-OB.zip", _CPU + "data.csv", "time,adservice_cpu\n1.5,1\n", "whole Unix seconds"),
        ("RE1-OB.zip", _CPU + "data.csv", "cpu,time\n1,1\n", "first column is not 'time'"),
        ("RE1-OB.zip", _CPU + "data.csv", "time,adservice_bogus\n1,1\n", "not <service>_<metric>"),
        ("RE1-OB.zip", _CPU + "data.csv", "time,adservice_cpu\n1,inf\n", "infinite or NaN"),
        ("RE1-OB.zip", _CPU + "data.csv", f"time,adservice_cpu\n1,{2**53 + 1}\n", "exactly"),
        ("RE1-OB.zip", _CPU + "data.csv", "time,adservice_cpu\n1,high\n", "not numbers"),
        ("RE1-OB.zip", "RE1-OB/adservice_cpu/extra.csv", "", "not a file of a case directory"),
        ("RE1-OB.zip", "RE1-OB/adservice_cpu_2/1/inject_time.txt", "1", "a suffix"),
        ("RE1-OB.zip", "RE1-OB/adservice_hang/1/inject_time.txt", "1", "not <service>_<fault>"),
        ("RE1-OB.zip", "RE1-OB/adservice_mem/x/inject_time.txt", "1", "repetition number"),
        ("RE1-OB.zip", "RE1-OB/adservice_mem/1/inject_time.txt", "1", "has no data.csv"),
        (
            "RE2-OB.zip",
            _DISK + "logs.csv",
            "time,timestamp,message\n0,2,b\n0,1,a\n",
            "not in timestamp order",
        ),
        ("RE2-OB.zip", _DISK + "logs.csv", "time,timestamp,message\n0,,a\n", "have no timestamp"),
        ("RE2-OB.zip", _DISK + "traces.csv", "time,spanID\n0,s\n", "not the trace columns"),
        (
            "RE2-OB.zip",
            _DISK + "traces.csv",
            f"{_TRACE_HEADER}\n0,t,s,a,m,o,1,2000,1,0.0,\n",
            "not startTime in milliseconds",
        ),
        ("RE2-OB.zip", _DISK + "logts.csv", "time,a_1\n0,1\n10,1\n", "every 15 s"),
    ],
)
def test_malformed_release_fails_with_its_path(tmp_path, archive, member, content, message):
    refs = _rewrite(tmp_path, archive, member, content)

    with pytest.raises(TimeFFormatError, match=message):
        RCAEvalDatasetLoader.parse_from(refs, archives=_FIXTURE_ARCHIVES, untimed=_FIXTURE_UNTIMED)


def test_a_missing_pinned_non_case_fails(tmp_path):
    refs = _release(tmp_path)
    path = tmp_path / "RE2-OB.zip"
    with zipfile.ZipFile(path) as opened:
        files = {name: opened.read(name).decode() for name in opened.namelist() if "multi-source-data/" not in name}
    _write_zip(path, files)

    with pytest.raises(TimeFFormatError, match="which the pinned zip holds"):
        RCAEvalDatasetLoader.parse_from(refs, archives=_FIXTURE_ARCHIVES, untimed=_FIXTURE_UNTIMED)


def test_signals_check_their_length_when_reread(tmp_path):
    refs = _release(tmp_path)
    dataset = _FixtureConnector().convert(refs)
    with zipfile.ZipFile(refs[0]) as opened:
        files = {name: opened.read(name).decode() for name in opened.namelist()}
    _write_zip(refs[0], files | {_CPU + "data.csv": _RE1_OB_TWO_CLOCKS.replace(f"{_T0 + 9},9.5", ",9.5")})

    with pytest.raises(TimeFFormatError, match="has 9 values, expected 10"):
        _signal(dataset, 0, "metrics", "adservice_cpu").to_arrow()
