"""The Sleep-EDF connector: whole-night polysomnograms with expert sleep scoring.

One record is one recording: a ``*-PSG.edf`` file of signals with the ``*-Hypnogram.edf`` scoring
beside it, scored in 30 s epochs. The release holds two studies. ``sleep-cassette`` recorded
subjects at home to study age, and ``sleep-telemetry`` recorded them in hospital to study
temazepam. README.md explains how each fact of a record maps to the release.
"""

from pathlib import Path
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetDownloadError
from timenet.types import ClassificationTask, ScalarPredictionTask, TemporalLocalizationTask
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.sleep_edfx import records, source, tasks
from timenet_connectors.datasets.physionet.sleep_edfx.source import (
    SLEEP_EDFX_ZIP_SHA256,
    SLEEP_EDFX_ZIP_URL,
    STUDIES,
    SleepEdfxSource,
    iter_recordings,
)
from timenet_connectors.download import ensure_archive, find_dir_containing


class SleepEdfxConnector(BasePhysioNetConnector[SleepEdfxSource]):
    """Connector for Sleep-EDF (PhysioNet ``sleep-edfx``)."""

    # Each study with the subject sheet that describes it. The order keeps the record order stable.
    _STUDIES: ClassVar[tuple[tuple[str, source.SheetShape], ...]] = STUDIES

    async def download_async(self, cache_dir: Path) -> list[SleepEdfxSource]:
        """Fetch and extract the release, and name each study's directory and subject table.

        Returns:
            A list of one :class:`SleepEdfxSource` handle.

        Raises:
            TimeNetDownloadError: If the archive holds no directory for a study, or no subject table.
        """
        # The directory of the first study locates the release. Every study is checked below.
        root = find_dir_containing(
            await ensure_archive(SLEEP_EDFX_ZIP_URL, cache_dir, sha256=SLEEP_EDFX_ZIP_SHA256), self._STUDIES[0][0]
        )
        studies: list[tuple[str, Path]] = []
        subject_tables: list[tuple[str, Path]] = []
        for study, shape in self._STUDIES:
            study_dir = root / study
            table_path = root / shape.table_name
            if not study_dir.is_dir() or not table_path.is_file():
                raise TimeNetDownloadError(f"Sleep-EDF archive at {root} is missing {study!r} or {shape.table_name!r}")

            subject_tables.append((study, table_path))
            studies.append((study, study_dir))

        return [SleepEdfxSource(studies=tuple(studies), subject_tables=tuple(subject_tables))]

    def convert(self, raw_refs: list[SleepEdfxSource]) -> TimeFDataset:
        """Build one record for each recording, and a stream of their tasks.

        Returns:
            The dataset.
        """
        release = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        subject_tables = records.SubjectTables(source.read_subject_tables(release), dict(self._STUDIES))
        for recording in iter_recordings(release):
            records.add_record(dataset, recording, source.read_recording(recording), subject_tables)

        dataset.set_task_stream(
            [ClassificationTask, TemporalLocalizationTask, ScalarPredictionTask],
            lambda: tasks.iter_tasks(dataset.records),
        )
        return dataset


CONNECTOR = SleepEdfxConnector
