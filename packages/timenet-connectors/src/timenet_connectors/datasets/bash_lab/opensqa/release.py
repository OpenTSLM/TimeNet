"""Describe the immutable OpenSQA source release converted by this connector."""

from dataclasses import dataclass


REPOSITORY = "BASH-Lab/OpenSQA"
"""Hugging Face dataset repository."""

REVISION = "ed16a14854a531ba9e71a6786e92a8f427b06b95"
"""Immutable source revision."""


@dataclass(frozen=True, slots=True)
class ReleaseFile:
    """Expected contents of one OpenSQA training JSONL file."""

    path: str
    corpus: str
    version: str
    rows: int
    qa_tasks: int

    @property
    def max_qa_pairs(self) -> int:
        """Return the number of QA pairs requested for each source row.

        Returns:
            Ten pairs for v1 and five pairs for v2.
        """
        return 10 if self.version == "v1" else 5


FILES = (
    ReleaseFile("data/train_hhar_v1-00000-of-00001.jsonl", "hhar", "v1", 9_166, 88_961),
    ReleaseFile("data/train_hhar_v2-00000-of-00001.jsonl", "hhar", "v2", 9_166, 45_825),
    ReleaseFile("data/train_motion_v1-00000-of-00001.jsonl", "motion", "v1", 4_534, 44_162),
    ReleaseFile("data/train_motion_v2-00000-00001.jsonl", "motion", "v2", 4_534, 22_650),
    ReleaseFile("data/train_pamap2_v2-00000-of-00001.jsonl", "pamap2", "v2", 9_672, 48_345),
    ReleaseFile("data/train_shoaib_v1-00000-of-00001.jsonl", "shoaib", "v1", 10_500, 101_972),
    ReleaseFile("data/train_shoaib_v2-00000-of-00001.jsonl", "shoaib", "v2", 10_500, 52_480),
    ReleaseFile("data/train_uci_v1-00000-of-00001.jsonl", "uci", "v1", 2_088, 20_343),
    ReleaseFile("data/train_uci_v2-00000-of-00001.jsonl", "uci", "v2", 2_088, 10_440),
)
"""Training files in stable conversion order, with verified release counts."""

RECORDS = 35_960
"""Unique sensor windows after v1/v2 deduplication."""

CAPTION_TASKS = sum(source.rows for source in FILES)
"""One embedded SensorCaps completion per source row."""

QA_TASKS = sum(source.qa_tasks for source in FILES)
"""Complete question-answer pairs recoverable from the generated text."""

ANSWER_TASKS = CAPTION_TASKS + QA_TASKS
"""Total caption and question-answer tasks in the converted release."""
