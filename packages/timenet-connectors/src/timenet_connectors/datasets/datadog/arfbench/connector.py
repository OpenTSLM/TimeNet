"""ARFBench questions over every source resolution and its released chart images."""

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
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, InputModality, TimeSeriesSpec, ureg
from timenet_connectors.sources.hub import hub_files, hub_snapshot
from timenet_connectors.sources.images import image_signal


REPO = "Datadog/ARFBench"
REVISION = "1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a"
"""The pinned commit. A branch name would let the same code read different bytes later."""

QA_CSV = "arfbench-qa.csv"
TS_DIR = "arfbench-ts-data"
IMAGE_DIR = "arfbench-images"

ANCHOR_US = 1_741_305_600_000_000
"""A derived origin: 2025-03-07T00:00:00Z, the UTC day boundary the corpus starts in. The release
states no zero of its own. Every record shares this one, so the two records a question cites agree
about what a time offset means and a reader can compare them offset for offset."""

US_PER_S = 1_000_000
_EMPTY_SIGNAL = "value"
"""Stands in for the empty tag label the release writes for a metric with no grouping."""

_ID_PREFIX = "arfbench"
"""The one prefix every id this connector states is built from."""

_FLAGS = ("interpolate_1", "interpolate_2")
"""The QA table's two interpolation flags, carried on the tasks that set them."""

_METRIC = TimeSeriesSpec(
    spec_type="metric",
    name="Observability metric",
    unit_value=ureg.dimensionless,
    dtype="float64",
    nullable=True,
)


@dataclass(frozen=True)
class ARFBenchSource:
    """The fetched sources: two paths and the interval listing, never any rows."""

    qa_csv: Path
    """The QA table."""
    ts_dir: Path
    """The directory holding the downloaded series files."""
    image_dir: Path
    """The directory holding the question charts."""
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


def _intervals_for(cited: Sequence[str], published: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
    """Return every sampling interval shared by the series a question cites.

    The release publishes a metric at up to six intervals. A question gets one task per
    common interval, with the same source records shared by other questions.

    Args:
        cited: The series ids the question cites.
        published: The intervals each series publishes.

    Returns:
        The intervals in seconds, from finest to coarsest.

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
    return tuple(sorted(shared))


def _cited_series(row: Mapping[str, str]) -> tuple[str, ...]:
    """Return the series ids one QA row cites.

    Args:
        row: A row of the QA table.

    Returns:
        The cited series ids, in the order the row lists them.
    """
    return tuple(part.strip() for part in row["query_group"].split(","))


def _images_for(cited: Sequence[str], image_dir: Path) -> tuple[str, ...]:
    """Resolve the source chart files in the native visual input order.

    Returns:
        The combined chart when present, followed by each cited metric chart.

    Raises:
        TimeFFormatError: If a cited metric has no chart.
    """
    names = []
    combined = f"{'-'.join(cited)}.png"
    if len(cited) > 1 and (image_dir / combined).is_file():
        names.append(combined)
    for series_id in cited:
        name = f"{series_id}.png"
        if not (image_dir / name).is_file():
            raise TimeFFormatError(f"ARFBench chart {name!r} is missing from {image_dir}")
        names.append(name)
    return tuple(names)


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


def _value_id(key: str, value: str) -> str:
    """Build the content id of a task annotation whose value repeats across questions.

    Every task that states the same ``(key, value)`` shares one annotation content and carries its
    own occurrence, so a value like ``Tier 1`` is stored once.

    Args:
        key: The annotation key.
        value: The annotation value.

    Returns:
        The annotation id.
    """
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()  # noqa: S324 (an id, not security)
    return f"{_ID_PREFIX}-{key}-{digest[:12]}"


def _record_id(series_id: str, interval_s: int) -> str:
    """Build the record id of one series file.

    Args:
        series_id: The metric's id, such as ``35928_0``.
        interval_s: The file's sampling interval in seconds.

    Returns:
        The record id.
    """
    return f"{_ID_PREFIX}-{series_id}-{interval_s}"


def _task_id(index: int) -> str:
    """Build the task id of one question, from its row number in the QA table.

    Args:
        index: The question's row number.

    Returns:
        The task id.
    """
    return f"{_ID_PREFIX}-{index:03d}"


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

    Two files stay decoded at once. The writer groups the series it writes by ``source_id``, which is
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


def _signals_for(path: str, series_id: str, interval_s: int) -> tuple[Signal, ...]:
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
            Signal.from_loader(
                spec=_METRIC,
                name=name,
                time_axis=(
                    RegularAxis.from_rate_hz(Fraction(1, interval_s)).at_index(int(offsets[0]) // step_us)
                    if regular
                    else IrregularAxis.spanning(offsets)
                ),
                loader=_values_loader(path, name),
                time_offsets_loader=None if regular else _offsets_loader(path, name),
                source_id=f"{series_id}@{interval_s}s",
                id=f"{_ID_PREFIX}-{series_id}-{interval_s}-{index}",
                n_values=len(offsets),
            )
        )
    return tuple(built)


def _record_for(path: Path, series_id: str, interval_s: int) -> Record:
    """Build the record of one series file: one source for the metric, one signal per tag group.

    Args:
        path: The series file.
        series_id: The metric's id, such as ``35928_0``.
        interval_s: The file's sampling interval in seconds.

    Returns:
        The record, anchored at :data:`ANCHOR_US` and annotated with the metric, its incident and
        its interval.
    """
    record_id = _record_id(series_id, interval_s)
    record = Record(
        record_id=record_id,
        sources=(
            Source(
                id=f"{record_id}-source",
                name=f"metric {series_id} at {interval_s} s",
                signals=_signals_for(str(path), series_id, interval_s),
            ),
        ),
        start_time=ANCHOR_US,
    )
    record.add_annotations(
        [
            Annotation(key="metric_id", value=series_id),
            Annotation(key="incident_id", value=series_id.partition("_")[0]),
            Annotation(key="interval_s", value=interval_s, unit="second"),
        ]
    )
    return record


def _task_annotations(row: Mapping[str, str]) -> list[Annotation]:
    """Build one question's task annotations.

    The two interpolation flags ride only on the rows that set them, because this connector
    interpolates nothing either way.

    Args:
        row: The QA row.

    Returns:
        The annotations to attach to the task.
    """
    annotations = [
        Annotation(key=key, value=row[key], id=_value_id(key, row[key])) for key in ("task_category", "difficulty")
    ]
    annotations.extend(
        Annotation(key=flag, value=True, id=f"{_ID_PREFIX}-{flag}") for flag in _FLAGS if row[flag] == "1"
    )
    return annotations


class ARFBenchConnector(BaseConnector[ARFBenchSource]):
    """Connector for ARFBench (Hub repo ``Datadog/ARFBench``)."""

    values_backend = "zarr"

    def download(self, cache_dir: Path) -> list[ARFBenchSource]:  # noqa: PLR6301 (BaseConnector override)
        """Fetch every numerical resolution and each chart at the pinned revision.

        Args:
            cache_dir: The directory that holds the downloaded files.

        Returns:
            A single-element list holding the :class:`ARFBenchSource` handle.

        Raises:
            TimeNetDownloadError: If the pinned revision holds no QA table.
        """
        published = _published_intervals(hub_files(REPO, REVISION))
        root = hub_snapshot(REPO, REVISION, cache_dir, (QA_CSV, f"{TS_DIR}/*.parquet", f"{IMAGE_DIR}/*.png"))
        qa_csv = root / QA_CSV
        if not qa_csv.is_file():
            raise TimeNetDownloadError(f"{REPO!r} at revision {REVISION} holds no {QA_CSV} under {root}")
        return [ARFBenchSource(qa_csv=qa_csv, ts_dir=root / TS_DIR, image_dir=root / IMAGE_DIR, published=published)]

    def convert(self, raw_refs: list[ARFBenchSource]) -> TimeFDataset:
        """Build shared records for every cited resolution and chart, then stream tasks.

        A first pass collects every cited ``(metric, interval)`` and image. Each becomes one
        shared record, so tasks can reference them without duplicating values.

        The tasks are not built here. They stream from the QA table through :meth:`_iter_tasks`, so
        no record carries the id of its tasks.

        Args:
            raw_refs: The single-element list from :meth:`download`.

        Returns:
        The dataset with numerical tasks at every common resolution and visual tasks.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        option_sets: set[tuple[str, ...]] = set()
        cited_files: set[tuple[str, int]] = set()
        image_names: set[str] = set()
        for row in _iter_qa_rows(source.qa_csv):
            option_sets.add(tuple(json.loads(row["options_str"])))
            cited = _cited_series(row)
            for interval_s in _intervals_for(cited, source.published):
                cited_files.update((series_id, interval_s) for series_id in cited)
            image_names.update(_images_for(cited, source.image_dir))

        options = {
            _options_id(candidates): dataset.annotate(
                Annotation(key="answer_options", value=list(candidates), id=_options_id(candidates))
            )
            for candidates in sorted(option_sets)
        }
        records = {
            key: dataset.add_record(record=_record_for(source.ts_dir / f"{key[0]}_{key[1]}.parquet", *key))
            for key in sorted(cited_files)
        }
        images = {}
        for name in sorted(image_names):
            identifier = f"{_ID_PREFIX}-image-{Path(name).stem}"
            images[name] = dataset.add_record(
                record=Record(
                    record_id=identifier,
                    sources=(
                        Source(
                            id=f"{identifier}-source",
                            name="ARFBench chart",
                            signals=(
                                image_signal(source.image_dir / name, signal_id=f"{identifier}-rgb", name="chart"),
                            ),
                        ),
                    ),
                )
            )
        dataset.set_task_stream([AnswerTask], lambda: self._iter_tasks(source, records, images, options))
        return dataset

    @staticmethod
    def _iter_tasks(
        source: ARFBenchSource,
        records: Mapping[tuple[str, int], Record],
        images: Mapping[str, Record],
        options: Mapping[str, Annotation],
    ) -> Iterator[AnswerTask]:
        """Yield numerical and chart-reading tasks for each QA row.

        The QA table is read again here rather than held, so the stream answers the same rows every
        time the writer asks for them.

        Args:
            source: The download handle naming the QA table.
            records: The records :meth:`convert` built, keyed by ``(metric, interval)``.
            options: The dataset's ``answer_options`` occurrences, keyed by content id.

        Yields:
            Each question as an answer task, naming its records and its candidate-answer annotation.
        """
        for index, row in enumerate(_iter_qa_rows(source.qa_csv)):
            cited = _cited_series(row)
            candidates = options[_options_id(json.loads(row["options_str"]))]
            for interval_s in _intervals_for(cited, source.published):
                task = AnswerTask(
                    id=f"{_task_id(index)}-series-{interval_s}s",
                    inputs=tuple(records[series_id, interval_s] for series_id in cited),
                    prompt=row["question"],
                    targets=(row["correct_answer"],),
                    input_annotations=(candidates,),
                    input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
                    metadata={"question_id": index, "resolution_seconds": interval_s},
                )
                for annotation in _task_annotations(row):
                    task.annotate(annotation)
                yield task
            chart = AnswerTask(
                id=f"{_task_id(index)}-image",
                inputs=tuple(images[name] for name in _images_for(cited, source.image_dir)),
                prompt=row["question"],
                targets=(row["correct_answer"],),
                input_annotations=(candidates,),
                input_modalities=frozenset({InputModality.TEXT, InputModality.IMAGE}),
                metadata={"question_id": index},
            )
            for annotation in _task_annotations(row):
                chart.annotate(annotation)
            yield chart


CONNECTOR = ARFBenchConnector
