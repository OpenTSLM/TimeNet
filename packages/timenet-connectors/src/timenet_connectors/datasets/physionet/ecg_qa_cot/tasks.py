"""Stream the ECG-QA CoT tasks: one answer task per chain-of-thought row."""

from collections.abc import Iterator, Mapping

from timenet.dataset import Record, TimeFDataset
from timenet.types import (
    Annotation,
    AnswerTask,
)
from timenet_connectors.datasets.physionet.ecg_qa_cot.source import EcgQaCotSource, iter_cot_rows


def iter_tasks(
    source: EcgQaCotSource,
    records: Mapping[int, Record],
    dataset: TimeFDataset,
    contexts: Mapping[int, str],
    answers: Mapping[int, tuple[str, ...]],
) -> Iterator[AnswerTask]:
    """Yield one :class:`AnswerTask` per CoT row, referencing its recording and metadata annotations.

    Each call reads the CoT CSVs again, so the whole set of questions never lives in memory.

    Args:
        source: The download handle naming the CoT CSVs.
        records: Input Records keyed by PTB-XL ECG ID.
        dataset: The dataset whose shared annotations the tasks reference.
        contexts: Each recording's clinical context, keyed by PTB-XL ECG ID.
        answers: Per-template answer options.

    Yields:
        Each question as an answer task.
    """
    for split, csv_path in source.cot_csvs:
        for row in iter_cot_rows(csv_path):
            input_annotations = [dataset.annotate_shared(_question_type_annotation(row.question_type))]
            if options := answers.get(row.template_id):
                input_annotations.append(dataset.annotate_shared(_answer_options_annotation(options)))
            input_annotations.append(dataset.annotate_shared(_clinical_context_annotation(contexts[row.ecg_id])))
            yield AnswerTask(
                inputs=(records[row.ecg_id],),
                prompt=row.question,
                targets=(row.answer,),
                rationale=row.rationale,
                input_annotations=tuple(input_annotations),
                split=split,
                metadata={"template_id": row.template_id},
            )


def _question_type_annotation(question_type: str) -> Annotation:
    """Return the annotation of a question type."""
    return Annotation(key="question_type", value=question_type)


def _answer_options_annotation(options: tuple[str, ...]) -> Annotation:
    """Return the annotation of a template's answer options."""
    return Annotation(key="answer_options", value=list(options))


def _clinical_context_annotation(context: str) -> Annotation:
    """Return the annotation of a recording's clinical context."""
    return Annotation(key="clinical_context", value=context)
