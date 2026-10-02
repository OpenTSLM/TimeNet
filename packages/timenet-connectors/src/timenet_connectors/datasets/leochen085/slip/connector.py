"""Convert the SLIP release: its pretraining corpus and its eleven evaluation collections."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain, groupby
from pathlib import Path
from typing import TYPE_CHECKING

from timenet.connectors import BaseConnector
from timenet.dataset import OrdinalAxis, Record, RegularAxis, Source, TimeAxis, TimeFDataset
from timenet.errors import TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, ClassificationTask, InputModality, Split, TimeSeriesSpec
from timenet_connectors.datasets.leochen085.slip.shards import SlipRow, iter_rows
from timenet_connectors.sources.huggingface_hub import hub_snapshot


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

REPO = "LeoChen085/SlipDataset"
REVISION = "e5e4871a9f376ff5377ed16c598e7a84a160664a"

# The release keeps its pretraining corpus in data/ and each evaluation collection in a folder of
# its own. A folder's train-*.parquet shards are its train subset, and an evaluation folder's
# test-*.parquet shards are its test subset, which becomes the test split.
CORPUS = "data"
COLLECTIONS = (
    "AsphaltObstacles",
    "Beijing_AQI",
    "PPG_CVA",
    "PPG_DM",
    "PPG_HTN",
    "ptbxl",
    "sleepEDF",
    "studentlife",
    "uci_har",
    "wesad",
    "wisdm",
)
_SPLITS = {"train": Split.TRAIN, "test": Split.TEST}
_META = "meta.csv"
_SERIES = "time_series"  # the nested column of a corpus row: one inner list per signal
_WINDOW = "X"  # the nested column of a collection window: one inner list per channel
_CAPTION_COLUMNS = ("caption0", "caption1", "caption2", "caption3")
# meta.csv describes each corpus the rows were drawn from. Its Source and Freq cells remain annotations
# exactly as the release wrote them. Freq does not define the axis because the table can disagree with
# the released rows: it calls London Smart Meters hourly, while its rows retain half-hourly samples.
_CORPUS_COLUMNS = (("source_url", "Source"), ("source_rate", "Freq"))
_NOT_STATED = "-"  # what a meta.csv cell holds when the release states nothing
_EVALUATION_RATES_HZ: dict[str, int | Fraction] = {
    "AsphaltObstacles": 100,
    "Beijing_AQI": Fraction(1, 3600),
    "PPG_CVA": 65,
    "PPG_DM": 65,
    "PPG_HTN": 65,
    "ptbxl": 100,
    "sleepEDF": 100,
    "studentlife": Fraction(1, 60),
    "uci_har": 50,
    "wesad": 700,
    "wisdm": 30,
}

# The release states no unit for its values. It stores a corpus row as float32 and a window as float64.
SERIES = TimeSeriesSpec(spec_type="slip_series", name="SLIP sensor series", unit_value=None)
WINDOW = TimeSeriesSpec(spec_type="slip_eval_series", name="SLIP evaluation series", unit_value=None, dtype="float64")


@dataclass(frozen=True)
class SlipSource:
    """What ``download`` hands ``convert``: paths, and no rows."""

    corpus: tuple[Path, ...]  # the data/*.parquet shards, in path order
    collections: tuple[Path, ...]  # the shards of every evaluation folder, in path order
    meta_csv: Path  # the table describing the corpora the corpus rows were drawn from


class SlipConnector(BaseConnector[SlipSource]):
    """Connector for the SLIP release (Hub repo ``LeoChen085/SlipDataset``)."""

    def download(self, cache_dir: Path) -> list[SlipSource]:  # noqa: PLR6301 - BaseConnector override
        """Fetch the corpus shards, ``meta.csv``, and the shards of every evaluation folder.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One handle naming every shard and the table beside them.
        """
        patterns = (_META, *(f"{folder}/*.parquet" for folder in (CORPUS, *COLLECTIONS)))
        root = hub_snapshot(REPO, REVISION, cache_dir, patterns)
        return [_discover_source(root)]

    def convert(self, raw_refs: list[SlipSource]) -> TimeFDataset:
        """Build one record per corpus row and per collection window, and stream their tasks.

        Args:
            raw_refs: The single handle :meth:`download` gave back.

        Returns:
            The populated dataset.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        captioned_records = _add_corpus_records(dataset, source)
        classification_tasks = _add_evaluation_records(dataset, source)
        # The evaluation collections have few enough tasks to hold in memory. A dataset either streams
        # every task or holds every task, so they ride the much larger caption-task stream.
        dataset.set_task_stream(
            [AnswerTask, ClassificationTask],
            lambda: chain(_caption_tasks(captioned_records), classification_tasks),
        )
        return dataset


def _discover_source(root: Path) -> SlipSource:
    """Find every required SLIP artifact below a downloaded release root.

    Returns:
        Paths to the pretraining shards, evaluation shards, and corpus metadata table.

    Raises:
        TimeNetDownloadError: If the release has no pretraining or evaluation shards.
    """
    corpus = tuple(sorted((root / CORPUS).glob("*.parquet")))
    collections = tuple(sorted(path for folder in COLLECTIONS for path in (root / folder).glob("*.parquet")))
    if not corpus or not collections:
        raise TimeNetDownloadError(f"{REPO!r} at {REVISION!r} returned no corpus or collection shards under {root}")
    return SlipSource(corpus=corpus, collections=collections, meta_csv=root / _META)


def _add_corpus_records(dataset: TimeFDataset, source: SlipSource) -> list[tuple[Record, Split]]:
    """Add the captioned pretraining records and return each record with its task split.

    Returns:
        The added records paired with the split of their caption tasks.
    """
    corpus_metadata = _read_corpus_metadata(source.meta_csv)
    captioned_records: list[tuple[Record, Split]] = []
    for _, split, record_id, row in _walk(source.corpus, _SERIES):
        corpus_name = row.scalars["dataset"]
        # The pinned table names every corpus a shard row can hold.
        metadata = corpus_metadata[corpus_name]
        record = _record(record_id, corpus_name, row, SERIES, OrdinalAxis())
        dataset.add_record(record=record)
        record.add_annotations(_corpus_annotations(record_id, row, metadata))
        captioned_records.append((record, split))
    return captioned_records


def _corpus_annotations(record_id: str, row: SlipRow, metadata: dict[str, str]) -> list[Annotation]:
    """Build one pretraining record's provenance and caption annotations.

    Returns:
        The annotations to attach to the record.
    """
    annotations = [
        Annotation(key="source_dataset", value=row.scalars["dataset"]),
        Annotation(key="domain", value=row.scalars["category"]),
    ]
    for key, column in _CORPUS_COLUMNS:
        if metadata[column] != _NOT_STATED:
            annotations.append(Annotation(key=key, value=metadata[column]))
    for column in _CAPTION_COLUMNS:
        annotations.append(Annotation(key=column, value=row.scalars[column], id=f"{record_id}-{column}"))
    return annotations


def _add_evaluation_records(dataset: TimeFDataset, source: SlipSource) -> list[ClassificationTask]:
    """Add the labeled evaluation records and return their classification tasks.

    Returns:
        One classification task per added evaluation record.
    """
    classification_tasks: list[ClassificationTask] = []
    for folder, split, record_id, row in _walk(source.collections, _WINDOW):
        record = _record(record_id, folder, row, WINDOW, RegularAxis.from_rate_hz(_EVALUATION_RATES_HZ[folder]))
        dataset.add_record(record=record)
        record.add_annotations(_evaluation_annotations(folder, row))
        # The class is the text label, which is what SLIP's own zero-shot evaluation classifies by.
        classification_tasks.append(
            ClassificationTask(
                id=f"{record_id}-classification",
                inputs=(record,),
                targets=(row.scalars["text_label"],),
                input_modalities=frozenset({InputModality.TIME_SERIES}),
                split=split,
            )
        )
    return classification_tasks


def _evaluation_annotations(folder: str, row: SlipRow) -> list[Annotation]:
    """Build one evaluation record's dataset, retrieval-template, label, and participant facts.

    Returns:
        The annotations to attach to the record.
    """
    annotations = [
        Annotation(key="source_dataset", value=folder),
        # The prompt column is the sentence template SLIP fills with each class name for
        # zero-shot retrieval ("The subject is $label."), not a question put to a model.
        Annotation(key="label_template", value=row.scalars["prompt"]),
    ]
    # The class index of the linear-probe setup. wisdm has none, its label column repeats
    # the text label, and studentlife writes its indices as whole floats.
    label = row.scalars["label"]
    if not isinstance(label, str):
        annotations.append(Annotation(key="label", value=int(label)))
    # sleepEDF and studentlife hold text ids and wisdm holds numbers; one key needs one type.
    if (participant := row.scalars.get("participant_id")) is not None:
        annotations.append(Annotation(key="participant_id", value=str(participant)))
    return annotations


def _walk(shards: Sequence[Path], column: str) -> Iterator[tuple[str, Split, str, SlipRow]]:
    """Walk the rows of the shards, one folder subset at a time.

    The shards are in path order, so a subset's shards are adjacent and its rows number on from one
    shard to the next. The release ships no row id, so a record is named by its folder, its subset,
    and its row index within that subset, padded so that ids sort in row order, which is the order
    the writer reads in.

    Args:
        shards: Parquet files at ``<folder>/<subset>-NNNNN-of-NNNNN.parquet``, in path order.
        column: The nested column holding each row's series.

    Yields:
        Each row with its folder, the split its subset is, and its record id.
    """
    for (folder, subset), group in groupby(shards, key=_folder_and_subset):
        rows = chain.from_iterable(iter_rows(shard, column) for shard in group)
        for index, row in enumerate(rows):
            yield folder, _SPLITS[subset], f"slip-{folder}-{subset}-{index:06d}", row


def _folder_and_subset(shard: Path) -> tuple[str, str]:
    """Read a shard's folder and subset off its path, ``<folder>/<subset>-NNNNN-of-NNNNN.parquet``.

    Returns:
        The folder name and the subset name.
    """
    return shard.parent.name, shard.stem.split("-", maxsplit=1)[0]


def _record(record_id: str, dataset_name: str, row: SlipRow, spec: TimeSeriesSpec, time_axis: TimeAxis) -> Record:
    """Build the record of one row, with the dataset the row was cut from as its one source.

    The release states which dataset a row was cut from and nothing finer, so that dataset is the
    one source every signal of the row has.

    Returns:
        The record, with its signals and no annotations yet.
    """
    return Record(
        record_id=record_id,
        sources=(
            Source(
                id=f"{record_id}-source",
                name=dataset_name,
                signals=row.signals(spec=spec, record_id=record_id, time_axis=time_axis),
            ),
        ),
    )


def _read_corpus_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Read ``meta.csv`` into one row per corpus.

    Args:
        path: The table's path.

    Returns:
        Each row keyed by the name a shard's ``dataset`` column holds, header names as the file
        writes them, cells stripped of the stray spaces the file has.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        rows = ({column: cell.strip() for column, cell in row.items()} for row in csv.DictReader(handle))
        return {row["Dataset"]: row for row in rows}


def _caption_tasks(captioned: Sequence[tuple[Record, Split]]) -> Iterator[AnswerTask]:
    """Yield one unprompted :class:`AnswerTask` per distinct caption of every corpus record.

    The captions are annotations of the record, so a task answers by reference to the occurrence the
    record carries and the stream reads no shard again. A caption column that repeats an earlier one
    of the same row adds no task: the ChatTS rows of the release hold one text in all four columns.

    Args:
        captioned: The corpus records with their split, in the order :meth:`SlipConnector.convert`
            built them.

    Yields:
        One task per distinct caption of each record, in column order, named after the first column
        holding that caption.
    """
    for record, split in captioned:
        distinct: dict[object, Annotation] = {}
        for annotation in record.annotations:
            if annotation.key in _CAPTION_COLUMNS:
                distinct.setdefault(annotation.value, annotation)
        for caption in distinct.values():
            yield AnswerTask(
                id=f"{caption.id}-task",
                inputs=(record,),
                target_annotations=(caption,),
                input_modalities=frozenset({InputModality.TIME_SERIES}),
                split=split,
            )


CONNECTOR = SlipConnector
