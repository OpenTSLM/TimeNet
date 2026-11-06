"""Orchestrate EngineMT-QA source validation, records and repeatable tasks."""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    AnswerTask,
    ClassificationTask,
)
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.pandalin98.enginemt_qa.records import build_record, unlabelled_annotation
from timenet_connectors.datasets.pandalin98.enginemt_qa.source import (
    FileCounts,
    QAFile,
    make_channel_loader,
    read_recording_count,
    scan_questions,
)
from timenet_connectors.datasets.pandalin98.enginemt_qa.tasks import iter_tasks
from timenet_connectors.datasets.pandalin98.enginemt_qa.vocabulary import QA_PARTS, ClosedQuestion, Part, Stage


class EngineMTQAConnector(BaseHuggingFaceConnector):
    """Build EngineMT-QA from the pinned files of its Hugging Face release."""

    HF_REPO = "pandalin98/EngineMT-QA"
    SOURCE_REVISION = "ea2051a15512212976c22c1a075ec549c7d48eef"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {part.value: part.source_file for part in Part}
    # Counted on the pinned files. They match the authors' README: 101,565 train and 42,477 test pairs.
    SOURCE_COUNTS: ClassVar[Mapping[Part, FileCounts]] = {
        Part.TRAIN: FileCounts(
            lines=13_291,
            pairs={
                Stage.UNDERSTANDING: 41_128,
                Stage.PERCEPTION: 35_987,
                Stage.REASONING: 16_300,
                Stage.DECISION_MAKING: 8_150,
            },
            unlabelled={ClosedQuestion.FAULT_CAUSE: 1_054, ClosedQuestion.HEALTH_STATE: 215},
        ),
        Part.TEST: FileCounts(
            lines=5_539,
            pairs={
                Stage.UNDERSTANDING: 17_240,
                Stage.PERCEPTION: 15_085,
                Stage.REASONING: 6_768,
                Stage.DECISION_MAKING: 3_384,
            },
            unlabelled={ClosedQuestion.FAULT_CAUSE: 422, ClosedQuestion.HEALTH_STATE: 78},
        ),
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the files into one lazy Record per named row, and a stream of one task per labelled pair.

        Conversion checks the signals file's layout and every QA line once, and compares each QA file's
        counts with the pinned ones. The values stay in the HDF5 file until written, so the files must stay
        available until then. A closed pair with an empty answer is a missing label and gets no task.

        Args:
            raw_refs: The downloaded files, one per part.

        Returns:
            A dataset with the Records, the unlabelled pairs as an annotation, and a stream of every task.

        Raises:
            TimeNetConnectorError: If a part is missing, or a file is malformed or has other counts.
            TimeNetConnectorError: If a file cannot be read.
        """
        paths = {Part(ref.part): ref.path for ref in raw_refs}
        if paths.keys() != set(Part):
            raise TimeNetConnectorError(f"EngineMT-QA needs the parts {[part.value for part in Part]}")
        signals_path = paths[Part.SIGNALS]
        n_rows = read_recording_count(signals_path)
        qa_files = [QAFile(paths[part], part, n_rows) for part in QA_PARTS]
        scan = scan_questions(qa_files, self.SOURCE_COUNTS)
        reader = make_channel_loader(signals_path)
        records = {row_id: build_record(row_id, reader) for row_id in sorted(scan.named_rows)}

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(unlabelled_annotation(scan.unlabelled))
        for record in records.values():
            dataset.add_record(record=record)

        dataset.set_task_stream((AnswerTask, ClassificationTask), lambda: iter_tasks(qa_files, records, scan))
        return dataset


CONNECTOR = EngineMTQAConnector
