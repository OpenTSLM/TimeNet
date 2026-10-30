"""Load the MIT-BIH records into one lazy Record each, and their beat, rhythm and beat classification tasks."""

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple, Self

from timenet.dataset import Record
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Annotation, ClassificationTask, Task, TemporalLocalizationTask
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.bases.physionet import WfdbHeader
from timenet_connectors.datasets.physionet.mitdb.annotations import read_reference
from timenet_connectors.datasets.physionet.mitdb.header import read_header
from timenet_connectors.datasets.physionet.mitdb.keys import AnnotationKey
from timenet_connectors.datasets.physionet.mitdb.records import AXIS, LeadLoader, ParsedRecord, build_record
from timenet_connectors.datasets.physionet.mitdb.release import (
    PINNED_COUNTS,
    RECORDS_FILE,
    REFERENCE_ANNOTATOR,
    PinnedCounts,
)
from timenet_connectors.datasets.physionet.mitdb.tasks import (
    beat_annotation_task,
    beat_classification_task,
    rhythm_annotation_task,
)
from timenet_connectors.datasets.physionet.mitdb.vocabulary import BeatCode, RhythmLabel


class WfdbAccess(NamedTuple):
    """How the loader reads WFDB files, which the PhysioNet base connector provides."""

    read_header: Callable[[Path], WfdbHeader]
    lead_loader: LeadLoader
    module: Callable[[], Any]
    """Imports ``wfdb``, which reads the annotation files."""


def _record_names(root: Path) -> list[str]:
    """Read ``RECORDS`` and check each record's header, signal and reference annotation files exist.

    Returns:
        The record names, in file order.

    Raises:
        TimeFFormatError: If a name repeats, or a record misses one of its files.
    """
    path = root / RECORDS_FILE
    names = path.read_text(encoding="utf-8").split()
    if len(set(names)) != len(names):
        raise TimeFFormatError(f"{path}: a record is listed twice")
    for name in names:
        missing = [ext for ext in ("hea", "dat", REFERENCE_ANNOTATOR) if not (root / f"{name}.{ext}").is_file()]
        if missing:
            raise TimeFFormatError(f"{root}: record {name} has no {', '.join(missing)} file")
    return names


class MitdbDatasetLoader(DatasetLoader):
    """The MIT-BIH records, read and checked: one lazy Record each, with every reference label, and its tasks.

    Each kind of task has its own method:

    - Beat annotation (:meth:`beat_annotation_tasks`): the output the authors' ``bxb`` compares, one per record.
    - Rhythm annotation (:meth:`rhythm_annotation_tasks`): the output ``epicmp`` compares, one per record.
    - Beat classification (:meth:`beat_classification_tasks`): one per reference beat.

    :meth:`tasks` keeps the record order: each record's three kinds of tasks, then the next record's.

    The loader keeps the labels, not the samples. The signals reread their ``.dat``, so the files must stay
    available until writing ends.
    """

    TASK_TYPES = (TemporalLocalizationTask, ClassificationTask)

    def __init__(self, records: tuple[ParsedRecord, ...]) -> None:
        self._records = records

    @classmethod
    def parse_from(
        cls, refs: Iterable[Path], *, wfdb: WfdbAccess | None = None, counts: PinnedCounts = PINNED_COUNTS
    ) -> Self:
        """Read and check every record's header and reference annotations once, and build its lazy Record.

        Args:
            refs: The one extracted release directory, which holds ``RECORDS``.
            wfdb: How to read the WFDB files; by default, as ``MitdbConnector`` reads them.
            counts: What the directory must hold.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If ``refs`` isn't one directory, or its counts differ from ``counts``.
        """
        if wfdb is None:
            # The connector module imports this one, so import it only when it is needed.
            from timenet_connectors.datasets.physionet.mitdb.connector import MitdbConnector  # noqa: PLC0415

            wfdb = MitdbConnector().wfdb_access()
        roots = list(refs)
        if len(roots) != 1:
            raise TimeFValidationError(f"MIT-BIH loads from one directory, got {roots}")
        (root,) = roots
        parsed = []
        for name in _record_names(root):
            header = read_header(root / name, wfdb.read_header)
            reference = read_reference(header.base, header.wfdb.sig_len, wfdb.module())
            parsed.append(build_record(header, reference, wfdb.lead_loader))
        found = PinnedCounts(
            records=len(parsed),
            beats=sum(len(item.reference.beats) for item in parsed),
            rhythm_labels=sum(len(item.reference.rhythm_episodes) for item in parsed),
            quality_labels=sum(len(item.reference.quality_episodes) for item in parsed),
            non_beats=sum(len(item.reference.events) for item in parsed),
            comments=sum(len(item.reference.comments) for item in parsed),
        )
        counts.check(root, found)
        return cls(tuple(parsed))

    def dataset_annotations(self) -> tuple[Annotation, ...]:  # noqa: PLR6301 - the loader interface
        """Return the beat codes and rhythm labels, the closed sets the tasks' targets take.

        Returns:
            The beat vocabulary, which the beat classification tasks name as their ``target_schema``, and the
            rhythm vocabulary.
        """
        return (
            Annotation(key=AnnotationKey.BEAT_VOCABULARY, value=[code.value for code in BeatCode]),
            Annotation(key=AnnotationKey.RHYTHM_VOCABULARY, value=[label.value for label in RhythmLabel]),
        )

    def records(self) -> Iterator[Record]:
        """Return one Record per record, in ``RECORDS`` order, whose values stay in the files until written."""
        return (parsed.record for parsed in self._records)

    def tasks(self) -> Iterator[Task]:
        """Return each record's beat and rhythm annotation tasks, then one classification task per beat.

        Yields:
            The tasks, by record in ``RECORDS`` order, and its beats in sample order.
        """
        for parsed in self._records:
            yield beat_annotation_task(parsed.record, parsed.beats)
            yield rhythm_annotation_task(parsed.record, parsed.rhythms)
            yield from _beat_classification_tasks(parsed)

    def beat_annotation_tasks(self) -> Iterator[TemporalLocalizationTask]:
        """Return the beat annotation tasks, which ``bxb`` scores.

        Yields:
            One task per record, in ``RECORDS`` order.
        """
        for parsed in self._records:
            yield beat_annotation_task(parsed.record, parsed.beats)

    def rhythm_annotation_tasks(self) -> Iterator[TemporalLocalizationTask]:
        """Return the rhythm annotation tasks, from which ``epicmp`` takes its episodes.

        Yields:
            One task per record, in ``RECORDS`` order.
        """
        for parsed in self._records:
            yield rhythm_annotation_task(parsed.record, parsed.rhythms)

    def beat_classification_tasks(self) -> Iterator[ClassificationTask]:
        """Return the beat classification tasks.

        Yields:
            One task per reference beat, by record in ``RECORDS`` order, and its beats in sample order.
        """
        for parsed in self._records:
            yield from _beat_classification_tasks(parsed)


def _beat_classification_tasks(parsed: ParsedRecord) -> Iterator[ClassificationTask]:
    for beat in parsed.reference.beats:
        yield beat_classification_task(parsed.record, AXIS.time_offset_us(beat.sample), beat.code)
