"""The Time-MMD connector: multi-domain time series paired with dated text.

Time-MMD (Liu et al., 2024) pairs the numerical series of each of nine domains with two textual
series: what a language model extracted from the reports the authors selected for the domain,
and from the web search results of each week. Every row of every file states the days it
covers.

One record is one domain. This connector covers two of the nine. ``energy`` is the weekly US
retail gasoline price with the eight regional prices EIA publishes beside it. ``environment``
is the daily Air Quality Index of the New York core-based statistical area with the columns
EPA states beside it. The two cadences meet on one timeline, because a record measures
everything in microseconds from one midnight
(:mod:`~timenet_connectors.datasets.adityalab.time_mmd.timeline`).

The numerical file of a domain becomes the signals of its record
(:mod:`~timenet_connectors.datasets.adityalab.time_mmd.numerical`). Each text becomes an
annotation over the days it states
(:mod:`~timenet_connectors.datasets.adityalab.time_mmd.text`). The forecasting questions the
release was built for, and the captions its reports answer, are tasks with a scope on that one
record (:mod:`~timenet_connectors.datasets.adityalab.time_mmd.tasks`).
:mod:`~timenet_connectors.datasets.adityalab.time_mmd.domains` states what each domain holds,
so the loop here is written for no domain in particular.

The release is fetched from its GitHub repository at one pinned commit, and every file is
checked against the digest recorded here, so two builds read the same bytes.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
import hashlib
from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, ForecastingTask, TimeInterval
from timenet_connectors.datasets.adityalab.time_mmd import numerical, tasks, text
from timenet_connectors.datasets.adityalab.time_mmd.domains import DOMAINS, DomainShape
from timenet_connectors.datasets.adityalab.time_mmd.keys import AnnotationKey, TextKind, record_id_of
from timenet_connectors.datasets.adityalab.time_mmd.timeline import ONE_DAY, Timeline
from timenet_connectors.download import Artifact, download_files


# The release lives in a GitHub repository with no tagged version. The commit pins the bytes.
TIME_MMD_REPOSITORY = "AdityaLab/Time-MMD"
TIME_MMD_COMMIT = "00281e2d86058286d5548b15a7670e8eda57ef62"

# The SHA-256 of every file this connector reads, at that commit. A download that gives other
# bytes stops the build, so a changed release never builds under this version.
RELEASE_DIGESTS: dict[str, str] = {
    "numerical/Energy/Energy.csv": "94313cefb3459f04b58ec814da5d817a694e767cc52de0ce09d4de4efdad02ee",
    "numerical/Environment/Environment.csv": "ec2ac594525f77628f728d7ee87b156a934780960d206659a642d0d51459ca47",
    "textual/Energy/Energy_report.csv": "36fa229dca003f40d4fb8c816a8ddfde2e020fca1f14572b3ce81a368e948fe8",
    "textual/Energy/Energy_search.csv": "f9023e6fcb635731a74f248dcf5ceb8aeca48870f3fc559f2173b7aae790b595",
    "textual/Environment/Environment_report.csv": "48363ad20d81b671133cdee290a6f5f6a29fe2622c72942ace34efe8f3fdbcc5",
    "textual/Environment/Environment_search.csv": "afe47dda61eab150ce60e7dd2b36a398d315fbe07df2116678b037a6e1c0aa38",
}


def raw_url(relative: str) -> str:
    """Give the URL of one file of the release, at the pinned commit.

    Args:
        relative: The path of the file inside the repository, with forward slashes.

    Returns:
        The URL of its raw bytes.
    """
    return f"https://raw.githubusercontent.com/{TIME_MMD_REPOSITORY}/{TIME_MMD_COMMIT}/{relative}"


def verify_digests(root: Path, digests: Mapping[str, str]) -> None:
    """Check every file of the release under ``root`` against the SHA-256 recorded for it.

    Args:
        root: The directory that holds the fetched files, in the layout of the repository.
        digests: The relative path of each file and the digest it must have.

    Raises:
        TimeNetDownloadError: If a file is missing, or holds other bytes than the digest records.
    """
    for relative, expected in digests.items():
        path = root / relative
        if not path.is_file():
            raise TimeNetDownloadError(f"{path} is missing, and the release at commit {TIME_MMD_COMMIT} holds it")

        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise TimeNetDownloadError(
                f"{path} holds bytes whose SHA-256 is {actual}, and the release at commit {TIME_MMD_COMMIT} records "
                f"{expected}. Delete the file and build again to fetch it"
            )


@dataclass(frozen=True)
class TimeMmdSource:
    """A handle to the fetched release: the directory that holds its ``numerical`` and ``textual`` trees.

    A clone of the repository has that layout too, so ``convert`` reads one as well as it reads
    a download.
    """

    root: Path


class TimeMmdConnector(BaseConnector[TimeMmdSource]):
    """Connector for Time-MMD (GitHub ``AdityaLab/Time-MMD``): its energy and environment domains."""

    async def download_async(self, cache_dir: Path) -> list[TimeMmdSource]:  # noqa: PLR6301 (override: the files are fixed, so nothing here reads the instance)
        """Fetch the six files of the two domains, each checked against its recorded digest.

        The files download concurrently and keep the layout of the repository under
        ``cache_dir``. A file already there is not fetched again, but it is checked again: the
        download helper verifies the digest of a file it fetches and skips one it finds, so a
        cached copy that changed on disk would otherwise pass. This reads no row of any file,
        and ``convert`` reads each one time.

        Args:
            cache_dir: The directory for the fetched files.

        Returns:
            A list of one :class:`TimeMmdSource` handle.
        """
        await download_files(
            [
                Artifact(raw_url(relative), cache_dir / relative, sha256=digest)
                for relative, digest in RELEASE_DIGESTS.items()
            ]
        )
        await asyncio.to_thread(verify_digests, cache_dir, RELEASE_DIGESTS)
        return [TimeMmdSource(root=cache_dir)]

    def convert(self, raw_refs: list[TimeMmdSource]) -> TimeFDataset:
        """Turn the fetched release into a dataset of one record for each domain.

        The tasks stream from the records, so a build holds them nowhere as a list. See
        :mod:`~timenet_connectors.datasets.adityalab.time_mmd.tasks`.

        Args:
            raw_refs: The list of one handle from :meth:`download`.

        Returns:
            The dataset.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        for shape in DOMAINS:
            _add_domain(dataset, source.root, shape)
        dataset.set_task_stream([ForecastingTask, AnswerTask], lambda: tasks.iter_tasks(dataset.records, DOMAINS))
        return dataset


def _add_domain(dataset: TimeFDataset, root: Path, shape: DomainShape) -> Record:
    """Read the three files of one domain and add its record to the dataset.

    The record's zero sits on the cadence of the values, at or before the earliest text, and its
    session reaches to the later of the last value and the last text. Every text of the domain
    then has a place in the record, whether or not a value sits beneath it.

    Args:
        dataset: The dataset under construction.
        root: The directory that holds the ``numerical`` and ``textual`` trees.
        shape: What the domain holds.

    Returns:
        The record, with its signals and every annotation attached.
    """
    record_id = record_id_of(shape.name)
    table = numerical.read_table(shape.numerical_path(root), shape)
    reports = text.read_rows(shape.textual_path(root, TextKind.REPORT))
    searches = text.read_rows(shape.textual_path(root, TextKind.SEARCH))
    rows = (*reports, *searches)

    timeline = Timeline.covering(
        table.first, shape.period_days, earliest=min(table.first, *(row.start for row in rows))
    )
    series = numerical.build_series(table, shape, timeline, record_id)
    session_end = max(
        timeline.offset_us(table.last + timedelta(days=shape.period_days)),
        *(timeline.offset_us(row.end + ONE_DAY) for row in rows),
    )
    record = dataset.add_record(
        time_series=series,
        record_id=record_id,
        start_time=timeline.start_time,
        time_span=TimeInterval.micros(0, session_end),
    )
    record.add_annotations(_describe_domain(record_id, shape))
    record.add_annotations(numerical.build_constants(table, shape, record_id))
    record.add_annotations(text.build_annotations(record_id, TextKind.REPORT, reports, timeline))
    record.add_annotations(text.build_annotations(record_id, TextKind.SEARCH, searches, timeline))
    return record


def _describe_domain(record_id: str, shape: DomainShape) -> list[Annotation]:
    """Give the annotations that state where the signals of a record came from.

    Args:
        record_id: The record they belong to.
        shape: What the domain holds.

    Returns:
        Two annotations with no span: the signal that holds ``OT``, and the header each signal
        was read from.
    """
    return [
        Annotation(
            key=AnnotationKey.TARGET_SIGNAL,
            value=shape.target,
            description="The signal that holds the OT column of the release, its default forecasting target.",
            id=f"{record_id}-{AnnotationKey.TARGET_SIGNAL}",
        ),
        Annotation(
            key=AnnotationKey.SOURCE_COLUMNS,
            value={one.signal: one.column for one in shape.signals},
            description="The header of the release's numerical file that each signal was read from.",
            id=f"{record_id}-{AnnotationKey.SOURCE_COLUMNS}",
        ),
    ]


CONNECTOR = TimeMmdConnector
