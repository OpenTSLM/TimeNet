"""Import the complete, frozen TimeSeriesExam round-three exam as test-only QA."""

import asyncio
import hashlib
from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.types import AnswerTask, Split, TimeSeriesSpec, ureg
from timenet_connectors.datasets.moment_timeseries_foundation_model.time_series_exam.question import (
    QUESTIONS,
    ExamQuestion,
)
from timenet_connectors.download import Artifact, download_files


class TimeSeriesExamConnector(BaseConnector[list[Path]]):
    """Preserve all published exam questions, their ordered signals, and source identities.

    The release has no train/validation split. Earlier rounds are developmental exam
    revisions, not training partitions, so only the paper's final round is imported.
    """

    SOURCE_REVISION = "384cf50864860c65e962b441eaa4c201857a06f8"
    SOURCE_SHA256 = "4ee3b62c8b57d6cb5f539ee0f53da7b4a558aa6a07f1f96f48892c6f9aa9e2e2"
    SOURCE_FILE = "qa_dataset.json"
    QUESTION_COUNT = 763
    SIGNAL_SPEC = TimeSeriesSpec(
        spec_type="timeseriesexam_synthetic",
        name="Synthetic time series",
        unit_value=ureg.dimensionless,
        dtype="float64",
    )

    def download(self, cache_dir: Path) -> list[Path]:
        """Fetch the pinned source JSON and verify its checksum."""
        url = (
            "https://raw.githubusercontent.com/moment-timeseries-foundation-model/TimeSeriesExam/"
            f"{self.SOURCE_REVISION}/output/round_3_folder/{self.SOURCE_FILE}"
        )
        artifact = Artifact(url=url, dest=cache_dir / self.SOURCE_FILE, sha256=self.SOURCE_SHA256)
        return asyncio.run(download_files([artifact]))

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Validate the frozen exam and construct one test AnswerTask per source question.

        Returns:
            All 763 questions with float64 observations and an ordinal axis.

        Raises:
            TimeFValidationError: If the supplied file is not the complete pinned exam.
        """
        if len(raw_refs) != 1:
            raise TimeFValidationError("TimeSeriesExam requires exactly one source JSON")
        raw = raw_refs[0].read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.SOURCE_SHA256:
            raise TimeFValidationError("TimeSeriesExam source checksum differs from the frozen release")
        questions = QUESTIONS.validate_json(raw)
        if tuple(question.id for question in questions) != tuple(range(1, self.QUESTION_COUNT + 1)):
            raise TimeFValidationError("TimeSeriesExam requires source question IDs 1 through 763 in order")
        dataset = TimeFDataset(metadata=self.metadata())
        for question in questions:
            record = self.make_record(question)
            dataset.add_record(record=record)
            dataset.add_task(
                task=AnswerTask(
                    inputs=(record,),
                    prompt=question.render_prompt(),
                    targets=(question.render_answer(),),
                    split=Split.TEST,
                    metadata=question.model_dump(mode="json", exclude={"ts", "ts1", "ts2"}),
                )
            )
        return dataset

    def make_record(self, question: ExamQuestion) -> Record:
        """Represent source order without inventing a sample rate or physical timestamps."""
        signals = tuple(
            Signal.from_values(values, spec=self.SIGNAL_SPEC, name=name, time_axis=OrdinalAxis())
            for name, values in question.iter_signals()
        )
        return Record(sources=(Source(name="TimeSeriesExam", signals=signals),), metadata={"source_id": question.id})


CONNECTOR = TimeSeriesExamConnector
