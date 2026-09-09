"""The ARFBench connector: anomaly question answering over Datadog observability metrics.

Source repo ``Datadog/ARFBench`` on the HuggingFace Hub, read at a pinned commit. It ships one QA
table, ``arfbench-qa.csv``, and 748 Parquet files under ``arfbench-ts-data/``. A Parquet file is one
metric at one sampling interval, named ``{incident}_{index}_{interval_seconds}.parquet``, in long
form: one row per (epoch, tag group) observation. The filename is the only correct join key to the QA
table's ``query_group``; the file's own ``query_name`` column names a different series on most files.

Each of the 750 questions becomes one record that carries the one or two metrics the question cites,
one signal per tag group, plus an :class:`~timenet.types.AnswerTask` holding the question and its
answer. The release publishes each metric at up to six intervals and no interval covers every metric,
so a record uses the finest interval published for every series it cites. That keeps all 750
questions and opens 205 of the 748 files, which is also the scope this connector downloads.

A row whose value is null is carried as a missing timestep, and a signal that skips a step of its
file's grid carries its own time offsets. The README beside this module records what the release
states inconsistently and what this connector did about it.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass
from fractions import Fraction
import functools
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, DataSource, TimeSeriesSpec, ureg


REPO = "Datadog/ARFBench"
REVISION = "1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a"
"""The pinned commit. A branch name would let the same code read different bytes later."""

QA_CSV = "arfbench-qa.csv"
TS_DIR = "arfbench-ts-data"

ANCHOR_US = 1_741_305_600_000_000
"""A derived origin: 2025-03-07T00:00:00Z, the UTC day boundary the corpus starts in. The release
states no zero of its own. Every record shares this one, so a metric that two questions cite has one
set of time offsets and is stored once."""

US_PER_S = 1_000_000
_EMPTY_SIGNAL = "value"
"""Stands in for the empty tag label the release writes for a metric with no grouping."""

_ID_PREFIX = "arfbench"
"""The one prefix every id this connector states is built from."""

_SOURCE = DataSource(data_source_type="huggingface", name="ARFBench", provider="Datadog")
_METRIC = TimeSeriesSpec(
    spec_type="metric",
    name="Observability metric",
    unit_value=ureg.dimensionless,
    dtype="float64",
    data_source=_SOURCE,
    nullable=True,
)


@dataclass(frozen=True)
class ARFBenchSource:
    """The fetched sources: two paths and the interval listing, never any rows."""

    qa_csv: Path
    """The QA table."""
    ts_dir: Path
    """The directory holding the downloaded series files."""
    published: Mapping[str, tuple[int, ...]]
    """The sampling intervals, in seconds, each series publishes at the pinned revision."""


def _parse_series_file(path: str) -> tuple[str, int] | None:
    """Split a series file path into its series id and its sampling interval.

    Args:
        path: A path inside the repo, such as ``arfbench-ts-data/35928_0_10.parquet``.

    Returns:
        The ``(series id, interval in seconds)`` pair, or ``None`` if the path is not a series file.
    """
    file = PurePosixPath(path)
    if file.parent.name != TS_DIR or file.suffix != ".parquet":
        return None
    incident, _, rest = file.stem.partition("_")
    index, _, interval = rest.partition("_")
    if not incident or not index or not interval.isdigit():
        return None
    return f"{incident}_{index}", int(interval)


def _published_intervals(paths: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Collect the intervals each series publishes.

    Args:
        paths: Every path in the repo.

    Returns:
        A mapping of series id to its sorted intervals in seconds.
    """
    published: dict[str, set[int]] = {}
    for path in paths:
        parsed = _parse_series_file(path)
        if parsed is not None:
            published.setdefault(parsed[0], set()).add(parsed[1])
    return {series_id: tuple(sorted(intervals)) for series_id, intervals in published.items()}


def _interval_for(cited: Sequence[str], published: Mapping[str, tuple[int, ...]]) -> int:
    """Choose a question's sampling interval: the finest one every series it cites publishes.

    The release publishes a metric at up to six intervals and not every metric publishes all six, so
    no single interval answers every question. This rule keeps all 750.

    Args:
        cited: The series ids the question cites.
        published: The intervals each series publishes.

    Returns:
        The interval in seconds.

    Raises:
        TimeFFormatError: If the cited series publish no interval in common.
    """
    shared = set(published.get(cited[0], ()))
    for series_id in cited[1:]:
        shared &= set(published.get(series_id, ()))
    if not shared:
        raise TimeFFormatError(
            f"the series {list(cited)} publish no sampling interval in common at revision {REVISION}, "
            f"so the question citing them has no series to carry"
        )
    return min(shared)


def _cited_series(row: Mapping[str, str]) -> tuple[str, ...]:
    """Return the series ids one QA row cites.

    Args:
        row: A row of the QA table.

    Returns:
        The cited series ids, in the order the row lists them.
    """
    return tuple(part.strip() for part in row["query_group"].split(","))


def _options_id(options: Sequence[str]) -> str:
    """Build the id of the annotation holding one candidate-answer list.

    The id comes from the value, so the questions that offer the same options reference one shared
    annotation instead of each carrying a copy.

    Args:
        options: The candidate answers.

    Returns:
        The annotation id.
    """
    digest = hashlib.sha1(json.dumps(list(options)).encode("utf-8")).hexdigest()  # noqa: S324 (an id, not security)
    return f"{_ID_PREFIX}-options-{digest[:12]}"


def _iter_qa_rows(qa_csv: Path) -> Iterator[dict[str, str]]:
    """Stream the QA table, so its rows never sit in a list.

    Args:
        qa_csv: The QA table.

    Yields:
        Each row as a dict. The ``question`` field spans two physical lines.
    """
    with qa_csv.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


@functools.lru_cache(maxsize=2)
def _read_file(path: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode one series file into per-signal time offsets, values and a missing mask.

    A row whose value is null keeps its place and is marked missing, so the stored series holds every
    row the release publishes. The rows are sorted by (tag group, epoch): row order in the release is
    arbitrary, and the pandas index the files carry does not restore it. Time offsets are measured
    from :data:`ANCHOR_US`, the zero every record shares.

    Two files stay decoded at once. The writer sorts the series it writes by ``source_id``, which is
    the file, so it asks for one file's signals in a run and this cache answers them from one decode.

    Args:
        path: The series file.

    Returns:
        A mapping of tag group to its ``(time offsets in microseconds, values, missing mask)`` triple.

    Raises:
        TimeFFormatError: If the file holds no row at all, if a row carries a null tag label, or if
            the file holds both an empty tag label and one spelled like :data:`_EMPTY_SIGNAL`.
    """
    table = pq.read_table(path, columns=["epoch", "group", "value"], memory_map=True)
    column = table.column("value").combine_chunks()
    values = column.to_numpy(zero_copy_only=False)
    missing = np.asarray(column.is_null())
    offsets = table.column("epoch").to_numpy(zero_copy_only=False).astype("datetime64[us]").astype("int64") - ANCHOR_US
    labels = table.column("group").combine_chunks().dictionary_encode()
    names = [str(name) for name in labels.dictionary.to_pylist()]
    codes = labels.indices.fill_null(-1).to_numpy(zero_copy_only=False)

    if codes.size == 0:
        raise TimeFFormatError(f"the series file {path} holds no row at revision {REVISION}, so it carries no signal")
    if bool((codes < 0).any()):
        raise TimeFFormatError(
            f"the series file {path} holds a value under a null tag label at revision {REVISION}; "
            f"the release writes an empty label for a metric with no grouping, so a null is a shape "
            f"this connector has no signal name for"
        )
    if "" in names and _EMPTY_SIGNAL in names:
        raise TimeFFormatError(
            f"the series file {path} holds both an empty tag label and one spelled {_EMPTY_SIGNAL!r} "
            f"at revision {REVISION}; the empty label is stored under that name, so the two tag "
            f"groups would collide on one signal and one of them would be lost"
        )
    order = np.lexsort((offsets, codes))
    values, offsets, codes, missing = values[order], offsets[order], codes[order], missing[order]

    starts = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1, [len(codes)]))
    return {
        names[codes[start]] or _EMPTY_SIGNAL: (
            offsets[start:stop].copy(),
            values[start:stop].copy(),
            missing[start:stop].copy(),
        )
        for start, stop in itertools.pairwise(starts)
    }


def _values_loader(path: str, signal: str) -> Callable[[], pa.Array]:
    """Build the lazy values loader of one signal.

    Args:
        path: The series file.
        signal: The tag group, as :func:`_read_file` keys it.

    Returns:
        A callable returning the signal's values, null where the release publishes a null.
    """

    def load() -> pa.Array:
        _, values, missing = _read_file(path)[signal]
        return pa.array(values, mask=missing, type=pa.float64())

    return load


def _offsets_loader(path: str, signal: str) -> Callable[[], pa.Array]:
    """Build the lazy time-offsets loader of one gapped signal.

    Args:
        path: The series file.
        signal: The tag group, as :func:`_read_file` keys it.

    Returns:
        A callable returning the signal's time offsets in microseconds.
    """

    def load() -> pa.Array:
        return pa.array(_read_file(path)[signal][0], type=pa.int64())

    return load


def _signals_for(path: str, series_id: str, interval_s: int) -> tuple[TimeSeries, ...]:
    """Build one metric's signals, with lazy loaders and an axis per signal.

    A signal that holds every step of its file's grid gets a regular axis. A signal that skips a step
    of the grid carries its own time offsets instead.

    Args:
        path: The series file.
        series_id: The metric's id, such as ``35928_0``.
        interval_s: The file's sampling interval in seconds.

    Returns:
        The signals, ordered by tag group.
    """
    step_us = interval_s * US_PER_S
    signals = _read_file(path)
    built = []
    for index, name in enumerate(sorted(signals)):
        offsets = signals[name][0]
        regular = int(offsets[0]) % step_us == 0 and bool(np.all(np.diff(offsets) == step_us))
        built.append(
            TimeSeries(
                spec=_METRIC,
                signal=f"{series_id}/{name}",
                time_axis=(
                    RegularAxis.from_rate_hz(Fraction(1, interval_s)).at_index(int(offsets[0]) // step_us)
                    if regular
                    else IrregularAxis.spanning(offsets)
                ),
                loader=_values_loader(path, name),
                time_offsets_loader=None if regular else _offsets_loader(path, name),
                source_id=f"{series_id}@{interval_s}s",
                time_series_id=f"{_ID_PREFIX}-{series_id}-{interval_s}-{index}",
                n_values=len(offsets),
            )
        )
    return tuple(built)


def _record_annotations(row: Mapping[str, str], cited: Sequence[str], interval_s: int) -> list[Annotation]:
    """Build one question's record annotations.

    The two interpolation flags ride only on the rows that set them, because this connector
    interpolates nothing either way.

    Args:
        row: The QA row.
        cited: The series ids the row cites.
        interval_s: The chosen sampling interval in seconds.

    Returns:
        The annotations to attach.
    """
    incidents = list(dict.fromkeys(series_id.partition("_")[0] for series_id in cited))
    annotations = [
        Annotation(key="task_category", value=row["task_category"]),
        Annotation(key="difficulty", value=row["difficulty"]),
        Annotation(key="query_group", value=row["query_group"]),
        Annotation(key="interval_s", value=interval_s, unit="second"),
        Annotation(key="incident_ids", value=incidents),
    ]
    annotations.extend(
        Annotation(key=flag, value=True) for flag in ("interpolate_1", "interpolate_2") if row[flag] == "1"
    )
    return annotations


class ARFBenchConnector(BaseConnector[ARFBenchSource]):
    """Connector for ARFBench (Hub repo ``Datadog/ARFBench``)."""

    def download(self, cache_dir: Path) -> list[ARFBenchSource]:  # noqa: PLR6301 (BaseConnector override)
        """Fetch the QA table and the series files the questions resolve to.

        Only the 205 series files the questions cite are fetched, so the cache on disk is exactly what
        ``convert`` reads. Picking those 205 needs the ``query_group`` column, so this method reads
        that one column of the QA table it has just fetched.

        Args:
            cache_dir: The directory that holds the downloaded files.

        Returns:
            A single-element list holding the :class:`ARFBenchSource` handle.

        Raises:
            ImportError: If ``huggingface_hub``, declared in this connector's ``requirements.txt``,
                is not installed.
            TimeNetDownloadError: If the pinned revision holds no QA table.
        """
        # discovery.available() imports every connector module to read its CONNECTOR, and
        # huggingface_hub is declared in this connector's requirements.txt rather than by the
        # package. A module-level import would break dataset listing for every connector in an
        # environment without it.
        try:
            from huggingface_hub import list_repo_files, snapshot_download  # noqa: PLC0415 (see the comment above)
        except ImportError as exc:
            raise ImportError(
                f"reading {REPO!r} needs huggingface_hub, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc

        # The listing comes before any fetch: the interval rule is decidable only once every
        # interval each metric publishes is known, and the file names are what state that.
        published = _published_intervals(list_repo_files(REPO, repo_type="dataset", revision=REVISION))
        root = Path(
            snapshot_download(
                REPO, repo_type="dataset", revision=REVISION, cache_dir=str(cache_dir), allow_patterns=[QA_CSV]
            )
        )
        qa_csv = root / QA_CSV
        if not qa_csv.is_file():
            raise TimeNetDownloadError(f"{REPO!r} at revision {REVISION} holds no {QA_CSV} under {root}")

        wanted: set[str] = set()
        for row in _iter_qa_rows(qa_csv):
            cited = _cited_series(row)
            interval_s = _interval_for(cited, published)
            wanted.update(f"{TS_DIR}/{series_id}_{interval_s}.parquet" for series_id in cited)
        snapshot_download(
            REPO, repo_type="dataset", revision=REVISION, cache_dir=str(cache_dir), allow_patterns=sorted(wanted)
        )
        return [ARFBenchSource(qa_csv=qa_csv, ts_dir=root / TS_DIR, published=published)]

    def convert(self, raw_refs: list[ARFBenchSource]) -> TimeFDataset:
        """Build one record per question, sharing the signals of a metric that several cite.

        A first pass registers one annotation per distinct candidate-answer list, so the tasks
        reference shared lists instead of each carrying its own. The second pass builds the records. A
        metric is read once per interval and its signals are reused, so a metric several questions
        cite is stored once.

        Args:
            raw_refs: The single-element list from :meth:`download`.

        Returns:
            The dataset: one record and one task per question.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        option_sets = {tuple(json.loads(row["options_str"])) for row in _iter_qa_rows(source.qa_csv)}
        dataset.register_annotations(
            Annotation(key="answer_options", value=list(options), id=_options_id(options))
            for options in sorted(option_sets)
        )

        signals: dict[tuple[str, int], tuple[TimeSeries, ...]] = {}
        for index, row in enumerate(_iter_qa_rows(source.qa_csv)):
            cited = _cited_series(row)
            interval_s = _interval_for(cited, source.published)
            time_series: tuple[TimeSeries, ...] = ()
            for series_id in cited:
                key = (series_id, interval_s)
                if key not in signals:
                    path = source.ts_dir / f"{series_id}_{interval_s}.parquet"
                    signals[key] = _signals_for(str(path), series_id, interval_s)
                time_series += signals[key]
            record_id = f"{_ID_PREFIX}-{index:03d}"
            record = dataset.add_record(time_series=time_series, record_id=record_id, start_time=ANCHOR_US)
            record.add_annotations(_record_annotations(row, cited, interval_s))
            dataset.add_task(
                record,
                AnswerTask(
                    prompt=row["question"],
                    target=row["correct_answer"],
                    input_annotation_ids=(_options_id(json.loads(row["options_str"])),),
                ),
            )
        return dataset


CONNECTOR = ARFBenchConnector
