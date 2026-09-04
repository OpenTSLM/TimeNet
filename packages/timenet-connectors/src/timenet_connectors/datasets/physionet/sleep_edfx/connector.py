"""The Sleep-EDF connector: whole-night polysomnograms with expert sleep scoring.

One sample is one recording. A recording is a ``*-PSG.edf`` file of body signals with a
``*-Hypnogram.edf`` file beside it. A technician scored that hypnogram in epochs of 30 s.

The release holds two studies. ``sleep-cassette`` measured the effect of age on sleep. It
recorded each subject at home, on a cassette recorder. ``sleep-telemetry`` measured the effect
of temazepam. It recorded each subject in hospital, on a telemetry system.

The two studies used different equipment. As a result, their channel sets differ. Their spans
differ too. A cassette recording covers about a day, and a telemetry recording about ten hours.

The loop that walks the release is in ``convert``, so one place states what a sample is made of.
:mod:`~timenet_connectors.bases.edf.reader` reads the EDF container, and
:mod:`~timenet_connectors.datasets.physionet.sleep_edfx.specs` states what each channel measures.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import TimeInterval
from timenet_connectors.bases.edf import reader, timeseries
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.sleep_edfx import annotations
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
# number, and three more for the night, the recorder and a trailing zero.
_RECORDING_ID_SHAPE = re.compile(r"(?P<code>.{3})(?P<subject>[0-9]{2}).{3}")


@dataclass(frozen=True)
class SleepEdfxSource:
    """A small handle to the fetched release, which ``convert`` walks in place of a list of refs.

    ``download`` returns one handle and not an entry for each recording. During ``convert``,
    :func:`_iter_recordings` finds the recordings. Nothing holds them as a list.
    """

    studies: tuple[tuple[str, Path], ...]  # (study, study_dir) per study


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
        TimeFFormatError: If the study is not one this connector knows, or if the name does not
            fit the shape the release uses.
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

    return f"{study}-{match['subject']}"


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
            yield SleepEdfxRecording(
                recording_id=recording_id,
                study=study,
                subject_id=_parse_subject_id(study, recording_id, psg_path),
                psg_path=psg_path,
                hypnogram_path=_find_hypnogram(psg_path),
            )


class SleepEdfxConnector(BasePhysioNetConnector[SleepEdfxSource]):
    """Connector for Sleep-EDF (PhysioNet ``sleep-edfx``)."""

    # Each study with the subject table that describes it. The order keeps sample ids stable.
    _STUDIES: ClassVar[tuple[tuple[str, str], ...]] = (
        (_CASSETTE_STUDY, "SC-subjects.xls"),
        (_TELEMETRY_STUDY, "ST-subjects.xls"),
    )

    async def download_async(self, cache_dir: Path) -> list[SleepEdfxSource]:
        """Get the Sleep-EDF archive and give one small handle to it.

        One archive holds the full release, and this is one fetch.
        :func:`~timenet_connectors.download.ensure_archive` extracts it. A second run does
        neither step again.

        This method pairs no recording and opens no EDF header. ``convert`` walks the recordings
        when it needs them.

        Args:
            cache_dir: The directory for the archive and its extracted contents.

        Returns:
            A list of one :class:`SleepEdfxSource` handle.

        Raises:
            TimeNetDownloadError: If the archive holds no directory for a study, or no subject table.
        """
        root = find_dir_containing(await ensure_archive(SLEEP_EDFX_ZIP_URL, cache_dir), "SC-subjects.xls")
        studies: list[tuple[str, Path]] = []
        for study, table_name in self._STUDIES:
            study_dir = root / study
            if not study_dir.is_dir() or not (root / table_name).is_file():
                raise TimeNetDownloadError(f"Sleep-EDF archive at {root} is missing {study!r} or {table_name!r}")

            studies.append((study, study_dir))

        return [SleepEdfxSource(studies=tuple(studies))]

    def convert(self, raw_refs: list[SleepEdfxSource]) -> TimeFDataset:
        """Turn the fetched release into a dataset.

        Args:
            raw_refs: The list of one handle from :meth:`download`.

        Returns:
            The dataset, with one sample for each recording it walked.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        for recording in _iter_recordings(source):
            # Named and not generated, thus two builds of one archive give one set of ids.
            sample_id = f"{_ID_PREFIX}-{recording.recording_id}"

            file = reader.open_edf(recording.psg_path)
            series = timeseries.build(sample_id, file, SPECS, loader=reader.build_channel_loader)

            entries = reader.read_annotations(reader.open_edf(recording.hypnogram_path))
            signal_end = reader.compute_signal_end_microseconds(file.header)
            # The session may end after the signals stop, when the scoring reaches past them.
            session_end = signal_end + annotations.measure_overrun_microseconds(entries, signal_end)

            sleep_stages = annotations.build(sample_id, entries, series)

            sample = dataset.add_sample(
                time_series=series,
                sample_id=sample_id,
                subject_ids=(recording.subject_id,),
                time_span=TimeInterval.micros(0, session_end),
            )
            sample.add_annotations(sleep_stages)

        return dataset


CONNECTOR = SleepEdfxConnector
