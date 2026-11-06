"""Orchestrate CODE-15 downloads, source validation, records and tasks."""

import asyncio
from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Annotation,
    ClassificationTask,
    ScalarPredictionTask,
)
from timenet_connectors.datasets.zenodo.code_15.records import build_record
from timenet_connectors.datasets.zenodo.code_15.source import (
    TraceReader,
    check_counts,
    download_url,
    part_zip,
    read_exams,
    scan_release,
)
from timenet_connectors.datasets.zenodo.code_15.tasks import iter_tasks
from timenet_connectors.datasets.zenodo.code_15.vocabulary import (
    CSV_FILE,
    CSV_SHA256,
    PART_SHA256,
    RECORD_DOI,
    RECORD_URL,
    AnnotationKey,
)
from timenet_connectors.download import Artifact, download_files, ensure_archive


class Code15Connector(BaseConnector[Path]):
    """Build CODE-15% from ``exams.csv`` and the 18 zips of its Zenodo record."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # ruff: ignore[no-self-use] (the connector interface)
        """Fetch ``exams.csv`` and every zip, check their digests, and extract each zip once.

        Each zip is deleted once extracted.

        Returns:
            The one directory that holds ``exams.csv`` and the parts.
        """
        root = cache_dir / "code15"
        await download_files([Artifact(download_url(CSV_FILE), root / CSV_FILE, sha256=CSV_SHA256)])
        await asyncio.gather(
            *(
                ensure_archive(download_url(part_zip(part)), root, filename=part_zip(part), sha256=sha)
                for part, sha in enumerate(PART_SHA256)
            )
        )
        return [root]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each exam with a trace into one lazy Record, and stream its age and abnormality tasks.

        Reads ``exams.csv`` and scans every part once, which checks each row and trace. The samples stay in
        the parts until written, so the files must stay available until then.

        Args:
            raw_refs: The one directory that holds ``exams.csv`` and the extracted parts.

        Returns:
            A dataset with the provenance annotations, the Records, and a stream of every task:
            each exam's age task, then each exam's abnormality task.

        Raises:
            TimeNetConnectorError: If ``raw_refs`` is not one directory.
            TimeNetConnectorError: If a file is missing or malformed.
        """
        if len(raw_refs) != 1 or not raw_refs[0].is_dir():
            raise TimeNetConnectorError(f"CODE-15% converts one directory, got {raw_refs}")
        root = raw_refs[0]
        if not (root / CSV_FILE).is_file():
            raise TimeNetConnectorError(f"{root}: {CSV_FILE} is missing")
        exams = list(scan_release(root, read_exams(root / CSV_FILE)))
        check_counts(root, exams)
        reader = TraceReader()
        records = [build_record(exam, reader) for exam in exams]

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(Annotation(key=AnnotationKey.SOURCE_URL, value=RECORD_URL))
        dataset.annotate(Annotation(key=AnnotationKey.SOURCE_DOI, value=RECORD_DOI))
        for record in records:
            dataset.add_record(record=record)

        dataset.set_task_stream((ScalarPredictionTask, ClassificationTask), lambda: iter_tasks(exams, records))
        return dataset


CONNECTOR = Code15Connector
