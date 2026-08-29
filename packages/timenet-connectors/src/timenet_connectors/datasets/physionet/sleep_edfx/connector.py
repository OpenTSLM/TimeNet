"""The Sleep-EDF connector: whole-night polysomnograms with expert sleep scoring.

One sample is one recording. A recording is a ``*-PSG.edf`` file of body signals with a
``*-Hypnogram.edf`` file beside it. A technician scored that hypnogram in epochs of 30 s.

The release holds two studies. ``sleep-cassette`` measured the effect of age on sleep. It
recorded each subject at home, on a cassette recorder. ``sleep-telemetry`` measured the effect
of temazepam. It recorded each subject in hospital, on a telemetry system.

The two studies used different equipment. As a result, their channel sets differ. Their spans
differ too. A cassette recording covers about a day, and a telemetry recording about ten hours.

The loop that walks the release is in ``convert``, so one place states what a sample is made of.
:mod:`~timenet_connectors.datasets.physionet.sleep_edfx.reader` reads the EDF container, and
:mod:`~timenet_connectors.datasets.physionet.sleep_edfx.specs` states what each channel measures.

:mod:`~timenet_connectors.datasets.physionet.sleep_edfx.tables` turns the rows of a subject table
into the facts it states, behind one reader that a per-sheet description drives.
:mod:`~timenet_connectors.datasets.physionet.sleep_edfx.metadata` turns the values that the header
and the table state into annotations.

No two facts about a recording come from the same place. The directory names the study. The
filename names the night. The EDF header states the clock and a second reading of the age and the
sex of the subject. The subject table states the age, the sex and the lights-off time. For
telemetry it states the drug condition in the shape of its columns and nowhere else.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import TimeInterval
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.sleep_edfx import annotations, metadata, reader, tables, timeseries
from timenet_connectors.datasets.physionet.sleep_edfx.specs import SPECS
from timenet_connectors.download import ensure_archive, find_dir_containing


# The full release, as one zip on the open S3 bucket of PhysioNet. boto3 reads it anonymously.
SLEEP_EDFX_ZIP_URL = "s3://physionet-open/sleep-edfx/sleep-edfx-1.0.0.zip"

# The prefix of every id this connector writes.
_ID_PREFIX = "sleep-edfx"


# The two studies of the release. Each writes its own three characters at the front of a
# recording id, and each numbers its subjects on its own.
_CASSETTE_STUDY = "sleep-cassette"

_TELEMETRY_STUDY = "sleep-telemetry"

# The three characters that each study writes at the front of its recording ids.
_STUDY_CODES = {_CASSETTE_STUDY: "SC4", _TELEMETRY_STUDY: "ST7"}

# A recording id is eight characters: the three its study writes, the two digits of the subject
# number, the night, and two more for the recorder and a trailing zero.
_RECORDING_ID_SHAPE = re.compile(r"(?P<code>.{3})(?P<subject>[0-9]{2})(?P<night>[0-9]).{2}")

# The nights a subject slept. Each study recorded two, and a sheet holds a column for each.
_NIGHTS = (1, 2)


@dataclass(frozen=True)
class SleepEdfxSource:
    """A small handle to the fetched release, which ``convert`` walks in place of a list of refs.

    ``download`` returns one handle and not an entry for each recording. During ``convert``,
    :func:`_iter_recordings` finds the recordings. Nothing holds them as a list.

    It names the subject table of every study by path, and carries no row of any of them.
    ``download`` resolves paths and opens nothing. ``convert`` reads the table of each study it
    converts one time, and joins each recording against that table as the recording streams past.
    """

    studies: tuple[tuple[str, Path], ...]  # (study, study_dir) per study
    subject_tables: tuple[tuple[str, Path], ...]  # (study, table path) per study


@dataclass(frozen=True)
class SleepEdfxRecording:
    """One recording: the file of signals, and the id that its name states.

    A recording name has the form ``<SC4|ST7><subject><night><recorder>0``. The name holds the
    subject and the night. The scoring, the subject table and the sample ids all need them.

    A recording is not one night of sleep. It starts before the night and stops after it. A
    ``*-Hypnogram.edf`` file of sleep stages is beside each recording, which
    :mod:`~timenet_connectors.datasets.physionet.sleep_edfx.annotations` reads.
    """

    recording_id: str
    study: str
    subject_id: str
    subject_number: int  # the number that identifies this person inside its own study
    night: int  # 1 or 2, which of the subject's recordings this is
    psg_path: Path
    hypnogram_path: Path


def _find_hypnogram(psg_path: Path) -> Path:
    """Find the scoring for one PSG recording.

    Args:
        psg_path: The ``*-PSG.edf`` file to pair.

    Returns:
        The ``*-Hypnogram.edf`` scored for the same night.

    Raises:
        TimeNetDownloadError: If the directory holds no scoring for this recording, or more than
            one.
    """
    # A hypnogram name ends with the initial of the technician who scored it. The PSG name does
    # not predict that letter. Match on the prefix that the two names share instead.
    prefix = psg_path.name[:7]
    matches = sorted(psg_path.parent.glob(f"{prefix}?-Hypnogram.edf"))
    if len(matches) != 1:
        found = ", ".join(match.name for match in matches) or "none"
        raise TimeNetDownloadError(f"expected one scoring for {psg_path.name}, found {found}")

    return matches[0]


def _parse_subject_id(study: str, recording_id: str, psg_path: Path) -> str:
    """Give the id of the person a recording belongs to, from the name of its file.

    A subject id names a person and not a recording, so the recordings of one person share an
    id. The study qualifies the number, because each study numbers its subjects on its own.

    Args:
        study: The study directory the file was walked in, which qualifies the number.
        recording_id: The name of the file with ``-PSG`` taken off.
        psg_path: The file itself, for the error message.

    Returns:
        The subject id: the name of the study and the two subject digits, joined by a hyphen.

    Raises:
        TimeFFormatError: If the study is not one this connector knows, if the name does not
            fit the shape the release uses, or if it states a night the release does not hold.
    """  # noqa: DOC502 (raised by _parse_recording_number, not directly here)
    subject, _ = _parse_recording_number(study, recording_id, psg_path)
    return f"{study}-{subject:02d}"


def _parse_recording_number(study: str, recording_id: str, psg_path: Path) -> tuple[int, int]:
    """Give the subject number and the night number that the name of a recording states.

    The cassette table keys its rows by that pair. The telemetry table matches the night number
    alone against its two night columns.

    Args:
        study: The study directory the file was walked in, which states the code of the name.
        recording_id: The name of the file with ``-PSG`` taken off.
        psg_path: The file itself, for the error message.

    Returns:
        The subject number and the night number.

    Raises:
        TimeFFormatError: If the study is not one this connector knows, if the name does not
            fit the shape the release uses, or if it states a night the release does not hold.
    """
    code = _STUDY_CODES.get(study)
    if code is None:
        raise TimeFFormatError(f"{psg_path}: lies in {study!r}, which is not a study of this release")

    match = _RECORDING_ID_SHAPE.fullmatch(recording_id)
    if match is None or match["code"] != code:
        raise TimeFFormatError(
            f"{psg_path.name}: a {study} recording is named {code}<2 subject digits><night><recorder>0, "
            f"as in {code}001E0, thus {recording_id!r} names no subject"
        )

    night = int(match["night"])
    if night not in _NIGHTS:
        raise TimeFFormatError(f"{psg_path.name}: states night {night}, and a subject slept night 1 or night 2")

    return int(match["subject"]), night


def _iter_recordings(source: SleepEdfxSource) -> Iterator[SleepEdfxRecording]:
    """Walk the release and give one recording for each PSG file, with no list up front.

    Args:
        source: The download handle, which names the directory of each study.

    Yields:
        Each recording, in the order the studies are declared, and inside a study in
        recording-id order.

    Raises:
        TimeFFormatError: If a study directory holds a PSG file whose name this connector
            cannot read.
    """  # noqa: DOC502 (raised by _parse_subject_id, not directly here)
    for study, study_dir in source.studies:
        for psg_path in sorted(study_dir.glob("*-PSG.edf")):
            # The filename is the only identifier. The EDF header names no subject, because the
            # patient field is anonymous in every file of this release.
            recording_id = psg_path.stem.removesuffix("-PSG")
            subject_number, night = _parse_recording_number(study, recording_id, psg_path)
            yield SleepEdfxRecording(
                recording_id=recording_id,
                study=study,
                subject_id=_parse_subject_id(study, recording_id, psg_path),
                subject_number=subject_number,
                night=night,
                psg_path=psg_path,
                hypnogram_path=_find_hypnogram(psg_path),
            )


class SleepEdfxConnector(BasePhysioNetConnector[SleepEdfxSource]):
    """Connector for Sleep-EDF (PhysioNet ``sleep-edfx``)."""

    # Each study with the subject table that describes it. The order keeps sample ids stable.
    _STUDIES: ClassVar[tuple[tuple[str, tables.SheetShape], ...]] = (
        (_CASSETTE_STUDY, tables.CASSETTE_SHEET),
        (_TELEMETRY_STUDY, tables.TELEMETRY_SHEET),
    )

    async def download_async(self, cache_dir: Path) -> list[SleepEdfxSource]:
        """Get the Sleep-EDF archive and give one small handle to it.

        One archive holds the full release, and this is one fetch.
        :func:`~timenet_connectors.download.ensure_archive` extracts it. A second run does
        neither step again.

        This method pairs no recording, opens no EDF header and reads no row of a subject table.
        It names each table by path, and ``convert`` reads the table. ``convert`` walks the
        recordings when it needs them.

        Args:
            cache_dir: The directory for the archive and its extracted contents.

        Returns:
            A list of one :class:`SleepEdfxSource` handle.

        Raises:
            TimeNetDownloadError: If the archive holds no directory for a study, or no subject table.
        """
        # The directory of the first study locates the release. Every study is checked below.
        root = find_dir_containing(await ensure_archive(SLEEP_EDFX_ZIP_URL, cache_dir), self._STUDIES[0][0])
        studies: list[tuple[str, Path]] = []
        subject_tables: list[tuple[str, Path]] = []
        for study, shape in self._STUDIES:
            study_dir = root / study
            table_path = root / shape.table_name
            if not study_dir.is_dir() or not table_path.is_file():
                raise TimeNetDownloadError(f"Sleep-EDF archive at {root} is missing {study!r} or {shape.table_name!r}")

            # The handle names the table of every study, and carries no row of any of them.
            subject_tables.append((study, table_path))
            studies.append((study, study_dir))

        return [SleepEdfxSource(studies=tuple(studies), subject_tables=tuple(subject_tables))]

    # PLR0914: the loop holds one name for each fact a sample is made of, on purpose. A reader
    # of it sees the join and each annotation, and not one call that hides both behind a name.
    def convert(self, raw_refs: list[SleepEdfxSource]) -> TimeFDataset:
        """Turn the fetched release into a dataset.

        This reads the subject table of each study one time, before the loop. It then joins each
        recording against the table of its own study as the recording streams past.

        The annotations that state a study, a night, a sex or a drug condition come from sets
        the release fixes. ``metadata_annotation`` holds one annotation for each value it meets.
        Every sample that states a value carries that same instance, and the writer stores it
        one time.

        Args:
            raw_refs: The list of one handle from :meth:`download`.

        Returns:
            The dataset, with one sample for each recording it walked.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        metadata_annotation = metadata.MetadataAnnotation(_ID_PREFIX)
        subject_tables = metadata.SubjectTables(
            {study: reader.read_table_rows(path) for study, path in source.subject_tables}, dict(self._STUDIES)
        )
        for recording in _iter_recordings(source):
            # Named and not generated, thus two builds of one archive give one set of ids.
            sample_id = f"{_ID_PREFIX}-{recording.recording_id}"

            file = reader.open_edf(recording.psg_path)
            series = timeseries.build(sample_id, file, SPECS, loader=reader.build_channel_loader)

            entries = reader.read_annotations(recording.hypnogram_path)
            signal_end = reader.compute_signal_end_microseconds(file.header)
            # The session may end after the signals stop, when the scoring reaches past them.
            session_end = signal_end + reader.measure_overrun_microseconds(entries, signal_end)

            sleep_stages = annotations.build(sample_id, entries, series)

            recording_metadata = subject_tables.build_annotations(
                sample_id, metadata_annotation, recording, file.header
            )

            sample = dataset.add_sample(
                time_series=series,
                sample_id=sample_id,
                subject_ids=(recording.subject_id,),
                time_span=TimeInterval.micros(0, session_end),
            )
            sample.add_annotations(sleep_stages)
            sample.add_annotations(recording_metadata)

        return dataset


CONNECTOR = SleepEdfxConnector
