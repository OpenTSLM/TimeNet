"""The ECG-QA CoT connector: PTB-XL 12-lead ECGs with chain-of-thought question answering.

Each record is one PTB-XL recording: 12 leads at 500 Hz in millivolts. Every chain-of-thought row is
an :class:`~timenet.types.AnswerTask` on that recording. Its ``prompt`` is the question, ``targets``
is the short evaluation answer, and ``rationale`` is the reasoning target. So a recording carries many
QA tasks. There is no record per question. The connector stores the question type, the answer options,
and the clinical context once, as value-deduped annotations that the tasks reference as their input.
The task metadata stores the question template ID for provenance. The question text already expresses
the template, so the ID is excluded from the input annotations. Each recording carries its dataset
split and the patient's subject ID, sex, and age from PTB-XL. The ~230k tasks stream to disk, so they
never all live in memory.

The connector imports the recordings and their patient metadata from the pinned PTB-XL TimeF
parent. The per-template answer options come from the ``Jwoo5/ecg-qa`` GitHub repo. The precomputed
CoT rows (question, answer, rationale, template) come from the OpenTSLM release, the only public
source for the rationales. Build the PTB-XL parent first, or pass ``--build-parents``.

The clinical context comes from the annotations of the imported PTB-XL records. It states the patient's
age and sex, the recording device, and the signal-quality issues, extra beats, and pacemaker that PTB-XL
records for the recording.
"""

import asyncio
from functools import partial
from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector
from timenet.connectors.context import BuildContext
from timenet.dataset import TimeFDataset
from timenet.types import AnswerTask, Split
from timenet.types.reference_keys import ParentAlias
from timenet_connectors.datasets.physionet.ecg_qa_cot.records import add_records
from timenet_connectors.datasets.physionet.ecg_qa_cot.source import (
    ECG_QA_COT_SHA256,
    ECG_QA_COT_URL,
    ECG_QA_TEMPLATE_ANSWERS_SHA256,
    ECG_QA_TEMPLATE_ANSWERS_URL,
    EcgQaCotSource,
    load_template_answers,
    scan_cot_rows,
)
from timenet_connectors.datasets.physionet.ecg_qa_cot.tasks import iter_tasks
from timenet_connectors.download import Artifact, download_files, ensure_archive, find_dir_containing


class EcgQaCotConnector(BaseConnector[EcgQaCotSource]):
    """Connector for the ECG-QA CoT dataset (PTB-XL signals + OpenTSLM chain-of-thought QA)."""

    _COT_CSVS: ClassVar[tuple[tuple[Split, str], ...]] = (
        (Split.TRAIN, "ecg_qa_cot_train.csv"),
        (Split.VALIDATION, "ecg_qa_cot_val.csv"),
        (Split.TEST, "ecg_qa_cot_test.csv"),
    )

    async def download_async(self, cache_dir: Path) -> list[EcgQaCotSource]:
        """Fetch the template answers and CoT CSVs and return one lightweight handle.

        The template-answers CSV and the CoT archive download concurrently. This method reads no
        CoT row. ``compose`` and the task stream read the rows lazily.

        Args:
            cache_dir: The directory that holds downloaded archives.

        Returns:
            A single-element list holding the :class:`EcgQaCotSource` handle.
        """
        answers_path = cache_dir / "answers_for_each_template.csv"
        # The two HTTP artifacts download concurrently.
        _, cot_root = await asyncio.gather(
            download_files(
                [Artifact(ECG_QA_TEMPLATE_ANSWERS_URL, answers_path, sha256=ECG_QA_TEMPLATE_ANSWERS_SHA256)]
            ),
            ensure_archive(ECG_QA_COT_URL, cache_dir, sha256=ECG_QA_COT_SHA256),
        )
        cot_csvs = tuple(
            (split, find_dir_containing(cot_root, csv_name) / csv_name) for split, csv_name in self._COT_CSVS
        )
        return [EcgQaCotSource(answers_path=answers_path, cot_csvs=cot_csvs)]

    def compose(self, raw_refs: list[EcgQaCotSource], context: BuildContext) -> TimeFDataset:
        """Build one record per recording and stream one :class:`AnswerTask` per CoT row.

        One full read of the CoT CSVs collects each recording's split.
        The records come from the PTB-XL parent. The task stream reads the CSVs again on each call, so
        the ~230k questions never all live in memory.

        Args:
            raw_refs: The single-element list from :meth:`download`.
            context: The opened PTB-XL parent, under the parent alias ``ptbxl``.

        Returns:
            The dataset: recording records plus a task stream.

        Raises:
            TimeNetConnectorError: If the PTB-XL parent has no recording for an ECG that the rows
                name, or if an imported Record has no patient, age, or sex, or has a sex other than
                male or female.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by add_records
        source = raw_refs[0]
        answers = load_template_answers(source.answers_path)
        dataset = TimeFDataset(metadata=self.metadata())
        records, contexts = add_records(dataset, context.parents[ParentAlias("ptbxl")], scan_cot_rows(source))
        # The task stream attaches each question type, answer option list, and clinical context with
        # annotate_shared on first use, and finds the same occurrence on every later use and pass.
        dataset.set_task_stream([AnswerTask], partial(iter_tasks, source, records, dataset, contexts, answers))
        return dataset


CONNECTOR = EcgQaCotConnector
