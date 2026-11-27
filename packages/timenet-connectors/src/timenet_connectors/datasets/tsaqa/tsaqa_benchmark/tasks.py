"""Represent TSAQA tasks and convert each task into TimeNet."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import ClassVar

from timenet.dataset import OrdinalAxis, Record, Signal, Source
from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask as TimeNetClassificationTask,
    Label,
    LabelSet,
    Split,
    Task,
    TimeSeriesSpec,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.source import (
    AnalysisTask,
    QuestionFormat,
    QuestionRepresentation,
    TSAQAMetadata,
    TSAQARawSeries,
    TSAQARow,
)


class TSAQATask(ABC):
    """One TSAQA question, its raw records, and the TimeNet task built from it."""

    records: list[Record]
    input_annotations: list[Annotation]
    metadata: TSAQAMetadata
    _timenet_task: Task

    analysis_task: ClassVar[AnalysisTask]
    input_count: ClassVar[int] = 1

    signal_spec: ClassVar[TimeSeriesSpec] = TimeSeriesSpec(
        spec_type="tsaqa_raw_series", name="TSAQA raw series", dtype="float64", unit_value=None
    )
    true_false_labels: ClassVar[LabelSet] = LabelSet(
        [
            Label(name="T", description="True."),
            Label(name="F", description="False."),
        ]
    )

    @abstractmethod
    def __init__(self, row: TSAQARow, index: int, *, split: Split, filename: str) -> None:
        """Check the source row and build the corresponding TimeNet task."""
        ...

    def to_timenet_task(self) -> Task:
        """Return the task built during construction, without allocating or registering anything."""
        return self._timenet_task

    def get_input_record(self) -> Record:
        """Return the first record, which contains all signals supplied to the model."""
        return self.records[0]

    def make_signal_from_row(self, row: TSAQARow, series_index: int) -> Signal:
        """Create a signal from one raw series, keeping its values and position in the row."""
        return Signal(
            spec=self.signal_spec,
            name="raw_ts",
            time_axis=OrdinalAxis(),
            data=row.raw_series[series_index].to_arrow(),
            metadata={"raw_series_index": series_index},
        )

    def create_raw_records(self, row: TSAQARow, *, input_count: int) -> None:
        """Group model inputs in one record and keep other raw series in separate records."""
        self.input_annotations = []
        signals = []
        sources = []
        for index in range(len(row.raw_series)):
            signal = self.make_signal_from_row(row, series_index=index)
            signals.append(signal)
            sources.append(Source(f"raw_ts_{index}", signals=[signal]))
        input_record = Record(
            sources=sources[:input_count],
            metadata={"file": self.metadata.file, "row": self.metadata.row, "is_task_input": True},
        )
        self.records = [input_record]
        for source in sources[input_count:]:
            self.records.append(
                Record(
                    sources=[source],
                    metadata={"file": self.metadata.file, "row": self.metadata.row, "is_task_input": False},
                )
            )
        self.annotate_series_descriptions(row, signals, input_count=input_count)

    def annotate_series_descriptions(self, row: TSAQARow, signals: list[Signal], *, input_count: int) -> None:
        """Attach each description to the signals identified by the row parser."""
        for description in row.series_descriptions:
            if description.series_indices is None:
                # Keep meta_info unchanged because we do not know which series each description belongs to.
                self.metadata.source_issues.append("ambiguous_series_context")
                context = self.get_input_record().annotate(Annotation("meta_info", value=row.meta_info))
                self.input_annotations.append(context)
                return
            for series_index in description.series_indices:
                signal = signals[series_index]
                annotation = signal.annotate(Annotation("meta_info", value=description.text))
                if series_index < input_count:
                    self.input_annotations.append(annotation)


class AnomalyDetectionTask(TSAQATask):
    """Detect anomalies in one raw series using a True/False answer.

    Example from train.parquet, zero-based row 21000 (question excerpt):
        Question: Determine whether the given time series contains anomalies.
        T: True.
        F: False.
        Published answer: F
    """

    analysis_task = AnalysisTask.ANOMALY_DETECTION

    def __init__(self, row: TSAQARow, index: int, *, split: Split, filename: str) -> None:
        """Validate the source row and build its TimeNet task.

        Raises:
            TimeNetConnectorError: If the row does not match this task's structure.
        """
        try:
            row.require_analysis_task(self.analysis_task)
            row.require_question_format(QuestionFormat.TRUE_OR_FALSE)
            row.require_num_raw_series(self.input_count)
        except TimeFValidationError as exc:
            raise TimeNetConnectorError(f"{filename}: row {index + 1}: {exc}") from exc
        self.metadata = row.to_task_metadata(filename, index)
        self.create_raw_records(row, input_count=self.input_count)
        self._timenet_task = TimeNetClassificationTask(
            inputs=[self.get_input_record()],
            prompt=row.question,
            input_annotations=self.input_annotations,
            targets=row.answer,
            split=split,
            label_set=self.true_false_labels,
            metadata=self.metadata.to_timenet_metadata(),
        )


class ClassificationTask(TSAQATask):
    """Classify one raw series using the choices in the question.

    Example from train.parquet, zero-based row 0 (question excerpt):
        Question: Classify the given time series into one of the categories below.
        A: down-down (1306 cases)
        B: up-down (1248 cases)
        C: down-up (1245 cases)
        D: up-up (1201 cases)
        Published answer: A
    """

    analysis_task = AnalysisTask.CLASSIFICATION

    def __init__(self, row: TSAQARow, index: int, *, split: Split, filename: str) -> None:
        """Validate the source row and build its TimeNet task.

        Raises:
            TimeNetConnectorError: If the row does not match this task's structure.
        """
        try:
            row.require_analysis_task(self.analysis_task)
            row.require_question_format(QuestionFormat.MULTIPLE_CHOICE)
            row.require_num_raw_series(self.input_count)
        except TimeFValidationError as exc:
            raise TimeNetConnectorError(f"{filename}: row {index + 1}: {exc}") from exc
        self.metadata = row.to_task_metadata(filename, index)
        self.create_raw_records(row, input_count=self.input_count)
        self._timenet_task = AnswerTask(
            inputs=[self.get_input_record()],
            prompt=row.question,
            input_annotations=self.input_annotations,
            targets=row.answer,
            split=split,
            metadata=self.metadata.to_timenet_metadata(),
        )


class CharacterizationTask(TSAQATask):
    """Describe properties of one raw series, such as trend or seasonality.

    Example from train.parquet, zero-based row 42000 (question excerpt):
        Question: Which sub-topic of irregularity is most evident from the time series of daily
        flow measurements for the Saugeen River at Walkerton?
        A: Mean shift
        B: Variance shift
        C: Trend shift
        D: Autocorrelation change
        Published answer: A
    """

    analysis_task = AnalysisTask.CHARACTERIZATION

    def __init__(self, row: TSAQARow, index: int, *, split: Split, filename: str) -> None:
        """Validate the source row and build its TimeNet task.

        Raises:
            TimeNetConnectorError: If the row does not match this task's structure.
        """
        try:
            row.require_analysis_task(self.analysis_task)
            row.require_question_format(QuestionFormat.TRUE_OR_FALSE, QuestionFormat.MULTIPLE_CHOICE)
            row.require_num_raw_series(self.input_count)
        except TimeFValidationError as exc:
            raise TimeNetConnectorError(f"{filename}: row {index + 1}: {exc}") from exc
        self.metadata = row.to_task_metadata(filename, index)
        self.create_raw_records(row, input_count=self.input_count)
        if row.question_format is QuestionFormat.TRUE_OR_FALSE:
            self._timenet_task = TimeNetClassificationTask(
                inputs=[self.get_input_record()],
                prompt=row.question,
                input_annotations=self.input_annotations,
                targets=row.answer,
                split=split,
                label_set=self.true_false_labels,
                metadata=self.metadata.to_timenet_metadata(),
            )
        else:
            self._timenet_task = AnswerTask(
                inputs=[self.get_input_record()],
                prompt=row.question,
                input_annotations=self.input_annotations,
                targets=row.answer,
                split=split,
                metadata=self.metadata.to_timenet_metadata(),
            )


class ComparisonTask(TSAQATask):
    """Compare two raw series, retaining their published order and individual lengths.

    Example from train.parquet, zero-based row 84000 (question excerpt):
        Question: In the provided time series, is there a variance shift in the second half of
        Series 2, accompanied by a decline in trend strength?
        T: True.
        F: False.
        Published answer: T
    """

    analysis_task = AnalysisTask.COMPARISON
    input_count = 2

    def __init__(self, row: TSAQARow, index: int, *, split: Split, filename: str) -> None:
        """Validate the source row and build its TimeNet task.

        Raises:
            TimeNetConnectorError: If the row does not match this task's structure.
        """
        try:
            row.require_analysis_task(self.analysis_task)
            row.require_question_format(QuestionFormat.TRUE_OR_FALSE, QuestionFormat.MULTIPLE_CHOICE)
            row.require_num_raw_series(self.input_count)
        except TimeFValidationError as exc:
            raise TimeNetConnectorError(f"{filename}: row {index + 1}: {exc}") from exc
        self.metadata = row.to_task_metadata(filename, index)
        self.create_raw_records(row, input_count=self.input_count)
        if row.question_format is QuestionFormat.TRUE_OR_FALSE:
            self._timenet_task = TimeNetClassificationTask(
                inputs=[self.get_input_record()],
                prompt=row.question,
                input_annotations=self.input_annotations,
                targets=row.answer,
                split=split,
                label_set=self.true_false_labels,
                metadata=self.metadata.to_timenet_metadata(),
            )
        else:
            self._timenet_task = AnswerTask(
                inputs=[self.get_input_record()],
                prompt=row.question,
                input_annotations=self.input_annotations,
                targets=row.answer,
                split=split,
                metadata=self.metadata.to_timenet_metadata(),
            )


class DataTransformationTask(TSAQATask):
    """Relate one input series to the transformation candidates printed in the question.

    Example from train.parquet, zero-based row 105000 (question excerpt):
        Question: Which of the following choices is most likely the Wavelet Transform of the
        given time series?
        Candidate arrays, abbreviated here:
        A: [1.557, 1.5605, 1.5476] ...
        B: [-0.5466, -0.5425, -0.5297] ...
        C: [-0.1198, -0.1132, -0.1066] ...
        D: [0.0073, -0.0222, -0.0446] ...
        Published answer: B
    """

    analysis_task = AnalysisTask.DATA_TRANSFORMATION

    def __init__(self, row: TSAQARow, index: int, *, split: Split, filename: str) -> None:
        """Validate the source row and build its TimeNet task.

        Raises:
            TimeNetConnectorError: If the row does not match this task's structure.
        """
        try:
            row.require_analysis_task(self.analysis_task)
            row.require_question_format(QuestionFormat.TRUE_OR_FALSE, QuestionFormat.MULTIPLE_CHOICE)
            series_count = self.input_count + row.question_format.num_answer_options()
            row.require_num_raw_series(series_count)
        except TimeFValidationError as exc:
            raise TimeNetConnectorError(f"{filename}: row {index + 1}: {exc}") from exc
        self.metadata = row.to_task_metadata(filename, index)
        self.create_raw_records(row, input_count=self.input_count)
        if row.question_format is QuestionFormat.TRUE_OR_FALSE:
            self._timenet_task = TimeNetClassificationTask(
                inputs=[self.get_input_record()],
                prompt=row.question,
                input_annotations=self.input_annotations,
                targets=row.answer,
                split=split,
                label_set=self.true_false_labels,
                metadata=self.metadata.to_timenet_metadata(),
            )
        else:
            self._timenet_task = AnswerTask(
                inputs=[self.get_input_record()],
                prompt=row.question,
                input_annotations=self.input_annotations,
                targets=row.answer,
                split=split,
                metadata=self.metadata.to_timenet_metadata(),
            )


@dataclass(frozen=True)
class TemporalCandidate:
    """A candidate printed in a temporal question, with its label and expected series length."""

    source_line: str
    label: str | None
    n_values: int

    @classmethod
    def parse_line(cls, line: str) -> "TemporalCandidate | None":
        """Parse a candidate array line; return None for ordinary question text."""
        match = re.fullmatch(r"(?:(?P<label>[A-D]): )?(?P<values>\[.*\])\n?", line)
        if match is None:
            return None
        values = TSAQARawSeries.model_validate_json(match.group("values"))
        return cls(source_line=line, label=match.group("label"), n_values=len(values.root))

    @property
    def time_series_option_name(self) -> str:
        """The name shared by the input annotation and the rewritten question."""
        if self.label is None:
            return "Candidate series"
        return f"Time series {self.label}"

    def require_num_values(self, count: int) -> None:
        """Require the raw record to have the same length as the published candidate."""
        if count != self.n_values:
            raise TimeFValidationError("candidate array length differs from its raw series")


class TemporalRelationshipTask(TSAQATask):
    """Identify a continuation or order patches using labelled raw candidate inputs.

    Example from train.parquet, zero-based row 63000 (question excerpt):
        Question: Is the following patch the future continuation of the given time series?
        Proposed patch: [-0.7429, -0.7429, -0.7429, -0.7429, -0.7429, -0.7429, -0.7429, -0.7429,
        -0.7429, -0.7429, -0.7429, -0.7429, -0.7429, -0.7429, -0.7429, -0.7429, -0.7429,
        -0.7429, -0.7429, -0.742, -0.3652, 0.5507]
        Choices: T: True. F: False.
        Published answer: F
    """

    analysis_task = AnalysisTask.TEMPORAL_RELATIONSHIP

    def __init__(self, row: TSAQARow, index: int, *, split: Split, filename: str) -> None:
        """Validate the source row and build its TimeNet task.

        Raises:
            TimeNetConnectorError: If the row does not match this task's structure.
        """
        try:
            row.require_analysis_task(self.analysis_task)
            series_count = 1 + row.question_format.num_answer_options()
            row.require_num_raw_series(series_count)
        except TimeFValidationError as exc:
            raise TimeNetConnectorError(f"{filename}: row {index + 1}: {exc}") from exc
        self.metadata = row.to_task_metadata(filename, index)
        self.metadata.question_representation = QuestionRepresentation.RAW_CANDIDATE_REFERENCES
        self.create_raw_records(row, input_count=series_count)
        try:
            prompt = self.label_signals_and_rewrite_question(row)
        except TimeFValidationError as exc:
            raise TimeNetConnectorError(f"{filename}: row {index + 1}: {exc}") from exc
        if row.question_format is QuestionFormat.TRUE_OR_FALSE:
            self._timenet_task = TimeNetClassificationTask(
                inputs=[self.get_input_record()],
                prompt=prompt,
                targets=row.answer,
                input_annotations=self.input_annotations,
                split=split,
                label_set=self.true_false_labels,
                metadata=self.metadata.to_timenet_metadata(),
            )
        else:
            self._timenet_task = AnswerTask(
                inputs=[self.get_input_record()],
                prompt=prompt,
                targets=row.answer,
                input_annotations=self.input_annotations,
                split=split,
                metadata=self.metadata.to_timenet_metadata(),
            )

    def label_signals_and_rewrite_question(self, row: TSAQARow) -> str:
        """Give each candidate signal a name and refer to it by that name in the question.

        The pinned release stores temporal candidates in the same order as their question
        lines. Preserve that order, including identical candidates; never match by the answer.
        The original prompt is retained in metadata because the text arrays use normalized values.

        Returns:
            A prompt referring to labelled records instead of embedding candidate arrays.

        Raises:
            TimeFValidationError: If candidate arrays do not match the number or lengths of records.
        """
        candidates = self.parse_time_series_candidates(row.question)
        given_signal, *candidate_signals = self.get_input_record().signals
        if len(candidates) != len(candidate_signals):
            raise TimeFValidationError("question candidates must match the number of raw candidate series")

        given_label = given_signal.annotate(Annotation("series_label", value="Given series"))
        self.input_annotations.append(given_label)
        for candidate, signal in zip(candidates, candidate_signals, strict=True):
            candidate.require_num_values(signal.n_values)
            label = signal.annotate(Annotation("series_label", value=candidate.time_series_option_name))
            self.input_annotations.append(label)

        return self.replace_candidate_arrays_with_option_names(row.question, candidates)

    @staticmethod
    def parse_time_series_candidates(question: str) -> list[TemporalCandidate]:
        """Read the candidate arrays in question order, including repeated arrays."""
        candidates = []
        for line in question.splitlines(keepends=True):
            candidate = TemporalCandidate.parse_line(line)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def replace_candidate_arrays_with_option_names(question: str, candidates: list[TemporalCandidate]) -> str:
        """Replace each printed array with its signal's name, keeping labels and line breaks."""
        for candidate in candidates:
            option_line = candidate.time_series_option_name
            if candidate.label is not None:
                option_line = f"{candidate.label}: {option_line}"
            if candidate.source_line.endswith("\n"):
                option_line += "\n"
            question = question.replace(candidate.source_line, option_line, 1)
        return question
