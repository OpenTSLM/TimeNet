"""The Sleep-EDF connector: whole-night polysomnograms with expert sleep scoring.

One sample is one recording. A recording is a ``*-PSG.edf`` file of body signals with a
``*-Hypnogram.edf`` file beside it. A technician scored that hypnogram in epochs of 30 s.

The release holds two studies. ``sleep-cassette`` measured the effect of age on sleep. It
recorded each subject at home, on a cassette recorder. ``sleep-telemetry`` measured the effect
of temazepam. It recorded each subject in hospital, on a telemetry system.

The two studies used different equipment. As a result, their channel sets differ. Their spans
differ too. A cassette recording covers about a day.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetDownloadError
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.download import ensure_archive, find_dir_containing


# The full release, as one zip on the open S3 bucket of PhysioNet. boto3 reads it anonymously.
SLEEP_EDFX_ZIP_URL = "s3://physionet-open/sleep-edfx/sleep-edfx-1.0.0.zip"


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
    Nothing reads them yet, and this class does not parse them yet.

    A recording is not one night of sleep. A cassette recording covers about a day, with one
    night in the middle of it. A ``*-Hypnogram.edf`` file of sleep stages is beside each
    recording. Nothing reads that file yet.
    """

    recording_id: str
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


def _iter_recordings(source: SleepEdfxSource) -> Iterator[SleepEdfxRecording]:
    """Walk the release and give one recording for each PSG file, with no list up front.

    Args:
        source: The download handle, which names the directory of each study.

    Yields:
        Each recording, cassette study first, and inside a study in recording-id order.
    """
    for _, study_dir in source.studies:
        for psg_path in sorted(study_dir.glob("*-PSG.edf")):
            # The filename is the only identifier. The EDF header names no subject, because the
            # patient field is anonymous in every file of this release.
            yield SleepEdfxRecording(
                recording_id=psg_path.stem.removesuffix("-PSG"),
                psg_path=psg_path,
                hypnogram_path=_find_hypnogram(psg_path),
            )


class SleepEdfxConnector(BasePhysioNetConnector[SleepEdfxSource]):
    """Connector for Sleep-EDF (PhysioNet ``sleep-edfx``)."""

    # Each study with the subject table that describes it. The order keeps sample ids stable.
    # The download gets both studies. Only a study with a module of its own becomes samples.
    _STUDIES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("sleep-cassette", "SC-subjects.xls"),
        ("sleep-telemetry", "ST-subjects.xls"),
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

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


CONNECTOR = SleepEdfxConnector
