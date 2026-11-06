"""Build repeatable description, question and counterfactual tasks from checked rows."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
import random

from timenet.dataset import Record
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonValue
from timenet.types import AnswerTask, Split
from timenet_connectors.datasets.mikeam.time_series_reasoning.records import RecordIndex
from timenet_connectors.datasets.mikeam.time_series_reasoning.source import (
    CounterfactualFields,
    CounterfactualKey,
    Description,
    DescriptionFields,
    FileReader,
    Question,
    QuestionFields,
    Release,
    RoundedCopyKey,
    RowLocation,
    RowT,
    read_counterfactual,
    read_description,
    read_question,
)
from timenet_connectors.datasets.mikeam.time_series_reasoning.vocabulary import (
    DESCRIPTION_FILES,
    DESCRIPTION_PREFIX,
    LETTERS,
    QUESTION_PREFIX,
    SPLITS,
    AnnotationKey,
    File,
    QuestionCategory,
    TaskMeta,
)


def _prompt(prefix: str, columns: tuple[str, ...], options: Sequence[str]) -> str:
    # The prefix and the first column meet without a separator, as in the authors' loader.
    lettered = "\n".join(f"{letter}) {option}" for letter, option in zip(LETTERS, options, strict=True))
    return prefix + "\n".join(columns) + "\n" + lettered


def _provenance(ref: RowLocation) -> dict[str, JsonValue]:
    return {TaskMeta.FILE: ref.file.value, TaskMeta.LINE: ref.location.line}


def _description_task(row: Description, record: Record, split: Split) -> AnswerTask:
    """Return the task that picks the series' description among four (the authors' ``ts2desc_mcq``).

    Returns:
        The task, its options in the row's seeded order and its target the description's letter.
    """
    options = _shuffled_options(row)
    letter = LETTERS[options.index(row.series.texts[AnnotationKey.DESCRIPTION])]
    metadata = _provenance(row.series.ref) | {TaskMeta.SCENARIO_CATEGORY: row.scenario_category.value}
    return AnswerTask(
        inputs=(record,),
        prompt=_prompt(DESCRIPTION_PREFIX, (), options),
        targets=(letter,),
        split=split,
        metadata=metadata,
    )


def _question_task(row: Question, records: tuple[Record, ...], category: QuestionCategory | None = None) -> AnswerTask:
    """Return the task that answers a question (the authors' ``qa_mcq`` or ``qa_counterfactual``).

    The release gives these tasks no split. The options keep their published order, and the target is
    the letter at ``answer_index``.

    Args:
        row: The question row.
        records: The input Records: the series, then for a counterfactual its changed version, in the
            order the authors' loader joins them.
        category: A counterfactual question's published category, ``None`` for a one-series question.

    Returns:
        The task.
    """
    texts = row.series.texts
    prompt = _prompt(
        QUESTION_PREFIX,
        (texts[AnnotationKey.DESCRIPTION_TINY], texts[AnnotationKey.METADATA], row.question),
        row.options,
    )
    issues: list[JsonValue] = [issue.value for issue in row.issues]
    metadata = _provenance(row.series.ref) | {
        TaskMeta.ANSWER_INDEX: row.answer_index,
        TaskMeta.ANSWER_TEXT: row.answer_text,
        TaskMeta.SOURCE_ISSUES: issues,
    }
    if category is not None:
        metadata[TaskMeta.CATEGORY] = category.value
    return AnswerTask(inputs=records, prompt=prompt, targets=(LETTERS[row.answer_index],), metadata=metadata)


@dataclass
class TaskStream:
    """Reread checked source rows and build tasks against the registered records."""

    release: Release
    records: RecordIndex

    def tasks(self) -> Iterator[AnswerTask]:
        """Return every task by rereading the five task files: description, question, counterfactual.

        Each call rereads them, so the tasks can be iterated more than once. The files must stay
        available until writing ends.

        Yields:
            The description tasks, then the question tasks, then the counterfactual tasks.
        """
        for file in DESCRIPTION_FILES:
            yield from self._stream(file, DescriptionFields, partial(self._from_description_row, split=SPLITS[file]))
        yield from self._stream(File.QUESTIONS, QuestionFields, self._from_question_row)
        yield from self._stream(File.COUNTERFACTUALS, CounterfactualFields, self._from_counterfactual_row)

    def _stream(
        self, file: File, model: type[RowT], build: Callable[[RowLocation, RowT], AnswerTask]
    ) -> Iterator[AnswerTask]:
        """Build a task from each row of a file read again, and check at its end that the file is unchanged.

        Yields:
            The task of each row.

        Raises:
            TimeNetConnectorError: If the file's bytes differ from pass one's, or a row no longer reads
                or names a series that pass one did not find.
        """
        reader = FileReader(self.release.paths[file], file, model)
        try:
            for ref, row in reader.rows():
                yield build(ref, row)
        except (TimeNetConnectorError, KeyError) as exc:
            raise TimeNetConnectorError(f"{reader.path}: {exc}; the file changed since conversion") from exc
        if reader.digest != self.release.digests[file]:
            raise TimeNetConnectorError(f"{reader.path}: the file changed since conversion")

    def _from_description_row(self, ref: RowLocation, row: DescriptionFields, split: Split) -> AnswerTask:
        description = read_description(ref, row)
        copy = self.records.rounded_copies.get(
            RoundedCopyKey(description.series.uuid, description.series.fingerprint.digest)
        )
        record = self.records.series[description.series.uuid] if copy is None else copy
        return _description_task(description, record, split)

    def _from_question_row(self, ref: RowLocation, row: QuestionFields) -> AnswerTask:
        question = read_question(ref, row)
        return _question_task(question, (self.records.series[question.series.uuid],))

    def _from_counterfactual_row(self, ref: RowLocation, row: CounterfactualFields) -> AnswerTask:
        counterfactual = read_counterfactual(ref, row)
        uuid = counterfactual.question.series.uuid
        counterfactuals = self.records.counterfactuals[CounterfactualKey(uuid, counterfactual.counterfactual)]
        return _question_task(
            counterfactual.question, (self.records.series[uuid], counterfactuals), counterfactual.category
        )


def _shuffled_options(row: Description) -> list[str]:
    """Shuffle the options once, with a seed of the row's own.

    The authors' loader shuffles them with ``random.shuffle`` and no seed, so each load shows its own
    order, while the published order puts the answer first. A stored prompt needs one order, so this
    seeds ``random.Random`` with the row's file and line: every build gives the row the same order.

    Returns:
        The options in the order the task's prompt shows them.
    """
    options = list(row.options)
    ref = row.series.ref
    random.Random(f"{ref.file.value}:{ref.location.line}").shuffle(options)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - a shuffle, not cryptography
    return options
