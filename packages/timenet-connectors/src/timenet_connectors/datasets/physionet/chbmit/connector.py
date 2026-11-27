"""The CHB-MIT connector: scalp EEG of children with epilepsy, with every seizure marked.

The release holds one directory for each case. A case is days of EEG from one child, recorded as a
sequence of EDF files with gaps between them. ``chbNN-summary.txt`` beside the files lists, for
each file, its channels and the onset and end of each seizure in seconds from the file start.

One record is one EDF file, and not one case. An EDF file is continuous, and the gaps between the
files of a case range from 2 s to four days. TimeNet has no axis yet that holds one signal with
gaps in it (TNET-167). A record for each case would therefore need one source for each file, and
its exhaustive seizure task would state that no seizure happened in time that nobody recorded.
Each record instead carries ``case_offset``, the seconds from the start of its case to its own
start, so a consumer can place the files of a case on one timeline and see each gap.

The EDF headers state surrogate dates: the release replaced each real date, and kept the time
between the files of a case. The connector sets no ``Record.start_time`` and uses the header
clock only to measure those offsets.

chb24 is left out. The release added it later with no subject information, so it may be one of the
other subjects. Its summary leaves out 10 of its 22 files, and the one file where a ``.seizures``
file and a summary disagree is in chb24. The cases chb01 to chb23 hold the 182 seizures that the
dataset page states.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from timenet.dataset import Record, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import US_PER_S, Annotation, ClassificationTask, Split, Task, TemporalLocalizationTask, TimeInterval
from timenet_connectors.bases.edf import reader, timeseries
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.bases.seizures import RecordingSeizures
from timenet_connectors.datasets.physionet.chbmit.channels import select_channels
from timenet_connectors.datasets.physionet.chbmit.summary import CaseSummary
from timenet_connectors.download import Artifact, download_files


_BUCKET_ROOT = "s3://physionet-open/chbmit/1.0.0"

# The checksum file of the release pins every other file the connector downloads.
SHA256SUMS_URL = f"{_BUCKET_ROOT}/SHA256SUMS.txt"
SHA256SUMS_SHA256 = "f1698efeb9fef887e55cedf6eda13386d9a888db663aec48b2aa194b7b753d2b"

# The cases the connector builds: chb01 to chb23. See the module docstring for chb24.
CASES = tuple(f"chb{number:02d}" for number in range(1, 24))

# chb21 is chb01, recorded 1.5 years later. Every other case is its own subject.
_SUBJECT_OF_CASE = {"chb21": "chb01"}

# The release defines no split. This one assigns whole subjects, so no child is in two splits. It
# was chosen once, by hand: validation and test each hold four subjects and about 15 % of the 182
# seizures. chb12 holds 40 seizures and three montage changes, and stays in train.
SUBJECT_SPLITS = {
    **dict.fromkeys(("chb03", "chb10", "chb14", "chb23"), Split.VALIDATION),
    **dict.fromkeys(("chb05", "chb13", "chb20", "chb22"), Split.TEST),
    **dict.fromkeys(
        (
            "chb01", "chb02", "chb04", "chb06", "chb07", "chb08", "chb09",
            "chb11", "chb12", "chb15", "chb16", "chb17", "chb18", "chb19",
        ),
        Split.TRAIN,
    ),
}  # fmt: skip

# The length of a window that a classification task asks about. Shoeb and Guttag (ICML 2010),
# whose paper the release ships, split the EEG into 2 s epochs.
WINDOW_US = 2 * US_PER_S

_SOURCE_NAME = "CHB-MIT scalp EEG"

# A line of the checksum file holds a digest and a path.
_CHECKSUM_FIELDS = 2


@dataclass(frozen=True)
class ChbMitCase:
    """The fetched files of one case: its summary, and its EDF files in name order."""

    case_id: str
    summary_path: Path
    edf_paths: tuple[Path, ...]

    @property
    def subject_id(self) -> str:
        """The child this case recorded. Two cases can share one."""
        return _SUBJECT_OF_CASE.get(self.case_id, self.case_id)


@dataclass(frozen=True)
class ChbMitSource:
    """A small handle to the fetched release: the files of each case, by path."""

    cases: tuple[ChbMitCase, ...]


@dataclass(frozen=True)
class _ConvertedFile:
    """One record with the seizures it holds and the split its subject belongs to."""

    record: Record
    seizures: RecordingSeizures
    split: Split


def parse_checksums(path: Path) -> dict[str, str]:
    """Read ``SHA256SUMS.txt``: one ``<digest> <path>`` line for each file of the release.

    Args:
        path: The checksum file.

    Returns:
        The digest of each file, keyed by its path inside the release.

    Raises:
        TimeNetDownloadError: If a line does not have that form.
    """
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != _CHECKSUM_FIELDS:
            raise TimeNetDownloadError(f"{path}: a line that is not '<sha256> <path>': {line!r}")

        digest, release_path = parts
        checksums[release_path] = digest
    return checksums


def _iter_case_files(checksums: dict[str, str], case_id: str) -> Iterator[str]:
    """Give the release path of every EDF file of one case, in name order.

    Yields:
        Paths such as ``chb01/chb01_03.edf``.
    """
    yield from sorted(path for path in checksums if path.startswith(f"{case_id}/") and path.endswith(".edf"))


class ChbMitConnector(BasePhysioNetConnector[ChbMitSource]):
    """Connector for the CHB-MIT Scalp EEG Database (PhysioNet ``chbmit`` 1.0.0)."""

    async def download_async(self, cache_dir: Path) -> ChbMitSource:  # ruff: ignore[no-self-use]
        """Get the summary and the EDF files of each case, each checked against its digest.

        The release is about 42 GB of separate files with no archive. The checksum file is
        pinned, and it pins every other file. A second run downloads no file it already holds.

        Args:
            cache_dir: The directory for the release.

        Returns:
            One :class:`ChbMitSource` handle.

        Raises:
            TimeNetDownloadError: If the checksum file lists no summary or no EDF file for a case.
        """
        root = cache_dir / "chbmit-1.0.0"
        (checksums_path,) = await download_files(
            [Artifact(url=SHA256SUMS_URL, dest=root / "SHA256SUMS.txt", sha256=SHA256SUMS_SHA256)]
        )
        checksums = parse_checksums(checksums_path)

        cases: list[ChbMitCase] = []
        artifacts: list[Artifact] = []
        for case_id in CASES:
            summary = f"{case_id}/{case_id}-summary.txt"
            edf_files = tuple(_iter_case_files(checksums, case_id))
            if summary not in checksums or not edf_files:
                raise TimeNetDownloadError(f"the checksum file lists no summary or no EDF file for {case_id}")

            for release_path in (summary, *edf_files):
                artifacts.append(
                    Artifact(
                        url=f"{_BUCKET_ROOT}/{release_path}", dest=root / release_path, sha256=checksums[release_path]
                    )
                )

            cases.append(
                ChbMitCase(
                    case_id=case_id,
                    summary_path=root / summary,
                    edf_paths=tuple(root / release_path for release_path in edf_files),
                )
            )

        await download_files(artifacts)
        return ChbMitSource(cases=tuple(cases))

    def convert(self, source: ChbMitSource) -> TimeFDataset:
        """Turn the fetched release into a dataset with one record for each EDF file.

        The records hold lazy signals, so this reads EDF headers and no sample. The tasks stream:
        the window tasks of the release number about 1.7 million.

        Args:
            source: The handle from :meth:`download`.

        Returns:
            The dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        converted: list[_ConvertedFile] = []
        for case in source.cases:
            converted.extend(_add_case_records(dataset, case))

        dataset.set_task_stream(
            [TemporalLocalizationTask, ClassificationTask],
            lambda: _iter_tasks(converted),
        )
        return dataset


def _add_case_records(dataset: TimeFDataset, case: ChbMitCase) -> list[_ConvertedFile]:
    """Add one record for each EDF file of a case.

    Args:
        dataset: The dataset to add the records to.
        case: The fetched files of the case.

    Returns:
        Each added record with its seizures and split, in file name order.

    Raises:
        TimeFFormatError: If the summary does not list a file, or the subject has no split.
    """
    split = SUBJECT_SPLITS.get(case.subject_id)
    if split is None:
        raise TimeFFormatError(f"{case.case_id}: subject {case.subject_id} has no split")

    summary = CaseSummary.from_file(case.summary_path)
    files = [reader.open_edf(path) for path in case.edf_paths]
    # The first file in name order is not always the first in time: chb03_25 starts before
    # chb03_24. The earliest header clock starts the case.
    case_start = min(file.header.start_time for file in files)
    converted = []
    for file in files:
        entry = summary.entry_for(file.path.name)
        entry.check_channels(file.header.signals, file.path)
        channels = select_channels(file.header.signals, file.header.units, file.path.name)
        series = timeseries.build(
            file,
            {channel.label: channel.spec for channel in channels},
            loader=reader.build_signal_loader,
            names={channel.index: channel.name for channel in channels},
        )

        duration_us = reader.compute_signal_end_microseconds(file.header)
        seizures = RecordingSeizures(recording_id=file.path.name, duration_us=duration_us, seizures=entry.seizures)
        case_offset_s = int((file.header.start_time - case_start).total_seconds())
        record = Record(
            sources=[Source(_SOURCE_NAME, signals=series)],
            time_span=TimeInterval.micros(0, duration_us),
            annotations=[
                Annotation("subject_ids", value=[case.subject_id]),
                Annotation(
                    "case_offset",
                    value=case_offset_s,
                    unit="second",
                    description="Seconds from the start of the first file of this case to the start of this file.",
                ),
                *seizures.create_annotations(),
            ],
            metadata={"case_id": case.case_id, "file_name": file.path.name},
        )
        dataset.add_record(record)
        converted.append(_ConvertedFile(record=record, seizures=seizures, split=split))
    return converted


def _iter_tasks(converted: list[_ConvertedFile]) -> Iterator[Task]:
    """Give the tasks of every record: where its seizures are, then one task for each window.

    The writer calls the stream more than once, so the caller wraps this in a callable that gives
    a fresh iterator each time.

    Yields:
        The tasks, one record at a time.
    """
    for one in converted:
        yield one.seizures.create_localization_task(one.record, split=one.split)
        yield from one.seizures.iter_window_tasks(one.record, window_us=WINDOW_US, split=one.split)


CONNECTOR = ChbMitConnector
