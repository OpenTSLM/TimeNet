"""Load EngineMT-QA's three files into one lazy Record per row and one task per QA pair."""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask, ClassificationTask, Task
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.pandalin98.enginemt_qa.keys import vocabulary_key
from timenet_connectors.datasets.pandalin98.enginemt_qa.qa import QAPair, read_pairs
from timenet_connectors.datasets.pandalin98.enginemt_qa.questions import Stage, Vocabulary
from timenet_connectors.datasets.pandalin98.enginemt_qa.records import RowLoader, build_record
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import (
    PINNED_RELEASE,
    QA_PARTS,
    Part,
    PinnedInventory,
    PinnedRelease,
)
from timenet_connectors.datasets.pandalin98.enginemt_qa.signals import check_layout
from timenet_connectors.datasets.pandalin98.enginemt_qa.tasks import closed_task, open_task, stage_task


class EngineMTQADatasetLoader(DatasetLoader):
    """EngineMT-QA's files, read and checked: one lazy Record per row, and one task per QA pair.

    EngineMT-QA defines four tasks, the paper's stages, each under the authors' split. Two stages are
    ``AnswerTask``s and two are ``ClassificationTask``s, so each stage has its own method:

    - Understanding (:meth:`understanding_tasks`): describe one cycle's change in a named signal.
    - Perception (:meth:`perception_tasks`): from one cycle, choose a component's health, the fault's
      causes, or the engine's health state.
    - Reasoning (:meth:`reasoning_tasks`): from 10 cycles, choose the engine's condition or its failure
      probability range.
    - Decision-making (:meth:`decision_tasks`): from 10 cycles, recommend maintenance.

    :meth:`tasks` keeps the file order, in which the stages alternate.

    The loader keeps the QA pairs, not the values: the signals read their row from the HDF5 file, so the
    files must stay available until writing ends.
    """

    TASK_TYPES = (AnswerTask, ClassificationTask)

    def __init__(self, pairs: tuple[QAPair, ...], records: Mapping[int, Record]) -> None:
        self._pairs = pairs
        self._records = records

    @classmethod
    def parse_from(
        cls, refs: Iterable[HubFileRef], *, all_rows: bool = False, release: PinnedRelease = PINNED_RELEASE
    ) -> Self:
        """Read and check the three files once, and build one lazy Record per row.

        A closed QA pair with an empty answer has no label, so it gets no task; ``release`` pins how many
        there are. Every other malformed pair, and any count that differs from ``release``, raises.

        Args:
            refs: The downloaded files, one per part.
            all_rows: Build a Record for every HDF5 row, as the authors' encoder pretraining reads them,
                rather than only for the rows a QA names.
            release: What the files must hold.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If a part is missing or unknown, or the files differ from ``release``.
        """
        files = list(refs)
        paths = {Part(ref.part): ref.path for ref in files if ref.part in set(Part)}
        if set(paths) != set(Part) or len(paths) != len(files):
            raise TimeFValidationError(f"EngineMT-QA loads exactly the parts {[part.value for part in Part]}")
        signals = paths[Part.SIGNALS]
        check_layout(signals, n_rows=release.rows)
        pairs: list[QAPair] = []
        for part in QA_PARTS:
            pairs.extend(_read_labelled_pairs(paths[part], part, release.qa[part]))
        referenced = sorted({row for pair in pairs for row in pair.rows})
        release.check_referenced_rows(len(referenced))
        if referenced[-1] > release.rows:
            raise TimeFValidationError(f"a QA names row id {referenced[-1]}, past the file's {release.rows} rows")
        # One loader for the Records of this load, so a row's channels share one read.
        rows = RowLoader(signals)
        row_ids = range(1, release.rows + 1) if all_rows else referenced
        return cls(tuple(pairs), {row_id: build_record(row_id, rows) for row_id in row_ids})

    def dataset_annotations(self) -> tuple[Annotation, ...]:  # noqa: PLR6301 - the loader interface
        """Return each closed question's option letters, with the option each letter names.

        A closed task names its vocabulary's key as its ``target_schema``. The connector adds the provenance.

        Returns:
            One vocabulary per option list.
        """
        return tuple(
            Annotation(
                key=vocabulary_key(vocabulary),
                value=list(vocabulary.letters),
                description="The option letters a closed question offers. Its answer is one or more of them.",
                metadata={"options": dict(zip(vocabulary.letters, vocabulary.options, strict=True))},
            )
            for vocabulary in Vocabulary
        )

    def records(self) -> Iterator[Record]:
        """Return one Record per row, in row id order, whose values stay in the HDF5 file until written."""
        return iter(self._records.values())

    def tasks(self) -> Iterator[Task]:
        """Return one task per labelled QA pair: the train file's pairs, then the test file's, in file order.

        Yields:
            The tasks.
        """
        for pair in self._pairs:
            yield stage_task(pair, self._inputs(pair))

    def understanding_tasks(self) -> Iterator[AnswerTask]:
        """Return the understanding tasks (stage 1), in file order.

        Yields:
            One answer task for each understanding pair.
        """
        for pair in self._pairs_of(Stage.UNDERSTANDING):
            yield open_task(pair, self._inputs(pair))

    def perception_tasks(self) -> Iterator[ClassificationTask]:
        """Return the perception tasks (stage 2), in file order.

        Yields:
            One classification task for each labelled perception pair.
        """
        for pair in self._pairs_of(Stage.PERCEPTION):
            yield closed_task(pair, self._inputs(pair))

    def reasoning_tasks(self) -> Iterator[ClassificationTask]:
        """Return the reasoning tasks (stage 3), in file order.

        Yields:
            One classification task for each reasoning pair.
        """
        for pair in self._pairs_of(Stage.REASONING):
            yield closed_task(pair, self._inputs(pair))

    def decision_tasks(self) -> Iterator[AnswerTask]:
        """Return the decision-making tasks (stage 4), in file order.

        Yields:
            One answer task for each decision-making pair.
        """
        for pair in self._pairs_of(Stage.DECISION_MAKING):
            yield open_task(pair, self._inputs(pair))

    def _pairs_of(self, stage: Stage) -> Iterator[QAPair]:
        return (pair for pair in self._pairs if pair.stage is stage)

    def _inputs(self, pair: QAPair) -> tuple[Record, ...]:
        return tuple(self._records[row] for row in pair.rows)


def _read_labelled_pairs(path: Path, part: Part, inventory: PinnedInventory) -> list[QAPair]:
    """Read one QA file's pairs, leave out the closed ones with an empty answer, and check the inventory.

    Returns:
        The labelled pairs, in file order.
    """
    lines: set[int] = set()
    per_stage: Counter[int] = Counter()
    empty: Counter[str] = Counter()
    labelled = []
    for pair in read_pairs(path, part):
        lines.add(pair.ref.line)
        per_stage[pair.stage.value] += 1
        if pair.closed is not None and not pair.answer:
            empty[pair.closed.value] += 1
        else:
            labelled.append(pair)
    inventory.check(path, PinnedInventory(len(lines), dict(per_stage), dict(empty)))
    return labelled
