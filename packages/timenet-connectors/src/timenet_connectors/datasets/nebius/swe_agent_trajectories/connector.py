"""The SWE-agent trajectories connector: 80,036 coding-agent runs, cut into checkpoints for early warning.

Nebius released the trajectories of SWE-agent runs that try to resolve GitHub issues, each with
whether its final patch passed the tests of the issue's pull request. Each checkpoint of a run (its
first 5, 10, 20 or 30 steps, if the run is still going) becomes one Record of per-step signals, with
the agent's thoughts for those steps as text. The issue text is a dataset annotation, stored once per
distinct text. :mod:`~timenet_connectors.datasets.nebius.swe_agent_trajectories.tasks` describes the
three tasks of each checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import functools
from typing import ClassVar

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.json import JsonMapping
from timenet.types import Annotation, AnswerTask, ClassificationTask, Task, TimeSeriesSpec, ureg
from timenet.types.splits import Split
from timenet.types.text import LazyTextList
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.nebius.swe_agent_trajectories.source import (
    ActionKind,
    ObservationKind,
    Run,
    read_runs,
)
from timenet_connectors.datasets.nebius.swe_agent_trajectories.tasks import (
    ENDING_PROMPT,
    ENDINGS,
    OUTCOMES,
    RESOLVE_PROMPT,
    RESOLVED,
    STATUS_PROMPT,
    UNRESOLVED,
    Checkpoint,
    checkpoints,
    split_runs,
    status,
)


AXIS = OrdinalAxis()
SOURCE_NAME = "agent_steps"
THOUGHTS = "agent_thoughts"
ISSUE = "issue"


def _count_spec(spec_type: str, name: str) -> TimeSeriesSpec:
    return TimeSeriesSpec(spec_type=spec_type, name=name, unit_value=ureg.dimensionless, dtype="int32")


# One signal per step property, in Record order. Lengths count characters of the release's text.
SPECS = {
    "thought_chars": _count_spec("agent_thought_length", "Agent thought length (characters)"),
    "action_chars": _count_spec("agent_action_length", "Agent command length (characters)"),
    "observation_chars": _count_spec("agent_observation_length", "Environment reply length (characters)"),
    "context_chars": _count_spec("agent_context_length", "Conversation length so far (characters)"),
    "repeated_action": TimeSeriesSpec(
        spec_type="agent_repeated_action", name="Command repeats an earlier one", unit_value=None, dtype="bool"
    ),
    "action_kind": TimeSeriesSpec(
        spec_type="agent_action_kind",
        name="Agent command kind",
        unit_value=ureg.dimensionless,
        dtype="enum",
        categories=tuple(ActionKind),
    ),
    "observation_kind": TimeSeriesSpec(
        spec_type="agent_observation_kind",
        name="Environment reply kind",
        unit_value=ureg.dimensionless,
        dtype="enum",
        categories=tuple(ObservationKind),
    ),
}


def _values(run: Run, name: str, steps: int) -> pa.Array:
    """The first ``steps`` values of one step signal of a run."""
    values = getattr(run.steps, name)[:steps]
    if name in {"action_kind", "observation_kind"}:
        # An enum series loads as dictionary-encoded labels.
        return pa.array([str(value) for value in values], type=pa.string()).dictionary_encode()
    return pa.array(values)


class SWEAgentTrajectoriesConnector(BaseHuggingFaceConnector):
    """Connector for nebius/SWE-agent-trajectories on the Hugging Face Hub."""

    HF_REPO = "nebius/SWE-agent-trajectories"
    SOURCE_REVISION = "68195a1450865274106246d0d0296a1d6807b88e"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {"train": "data/train-*.parquet"}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Read every run, then build one Record per checkpoint and stream three tasks per Record.

        Every shard is read once, which checks every row before anything is written. Each run keeps
        only the signals of its first 30 steps and its thoughts, compressed, so the loaders and the
        task stream need no second read.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        runs = list(read_runs(self.ref_paths(raw_refs, "train")["train"]))
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (("source_repository", self.HF_REPO), ("source_revision", self.SOURCE_REVISION)):
            dataset.annotate(Annotation(key=key, value=value))
        issues: dict[str, Annotation] = {}
        for run in runs:
            if run.issue not in issues:
                issues[run.issue] = dataset.annotate(
                    Annotation(
                        key=ISSUE,
                        value=run.issue,
                        description="The GitHub issue a run tries to resolve, as the agent read it.",
                        metadata={"instance_id": run.instance_id},
                    )
                )
        split_of = split_runs(runs)
        records: list[tuple[Checkpoint, Record]] = []
        for run in runs:
            for checkpoint in checkpoints(run):
                records.append((checkpoint, dataset.add_record(record=self._make_record(checkpoint))))
        dataset.set_task_stream(
            [ClassificationTask, AnswerTask], functools.partial(_tasks, records, issues, split_of.__getitem__)
        )
        return dataset

    @staticmethod
    def _make_record(checkpoint: Checkpoint) -> Record:
        """Build one checkpoint's Record: a signal per step property, and the thoughts as text.

        Returns:
            The Record. It holds nothing about the run after the checkpoint.
        """
        run, steps = checkpoint.run, checkpoint.steps
        signals = tuple(
            Signal.from_loader(
                spec=spec,
                time_axis=AXIS,
                name=name,
                loader=functools.partial(_values, run, name, steps),
                n_values=steps,
            )
            for name, spec in SPECS.items()
        )
        record = Record(
            sources=(Source(name=SOURCE_NAME, signals=signals),),
            metadata={
                "run_id": run.run_id,
                "instance_id": run.instance_id,
                "repository": run.repository,
                "model_name": run.model_name,
                "checkpoint_steps": steps,
            },
        )
        record.annotate(
            Annotation(
                key=THOUGHTS,
                value=LazyTextList(functools.partial(run.steps.thought_texts, steps), length=steps),
                description="The agent's thought at each step, in step order: its message without the command.",
            )
        )
        return record


def _tasks(
    records: list[tuple[Checkpoint, Record]], issues: dict[str, Annotation], split_of: Callable[[str], Split]
) -> Iterator[Task]:
    """Yield the three tasks of every checkpoint, in Record order.

    Yields:
        The outcome and ending classifications, then the status report, of each checkpoint.
    """
    for checkpoint, record in records:
        run = checkpoint.run
        context = (issues[run.issue], *(a for a in record.annotations if a.key == THOUGHTS))
        metadata: JsonMapping = {
            "run_id": run.run_id,
            "instance_id": run.instance_id,
            "repository": run.repository,
            "model_name": run.model_name,
            "checkpoint_steps": checkpoint.steps,
        }
        inputs, split = (record,), split_of(run.run_id)
        yield ClassificationTask(
            inputs=inputs,
            split=split,
            input_annotations=context,
            metadata=metadata,
            prompt=RESOLVE_PROMPT,
            targets=(RESOLVED if run.resolved else UNRESOLVED,),
            label_set=OUTCOMES,
        )
        yield ClassificationTask(
            inputs=inputs,
            split=split,
            input_annotations=context,
            metadata=metadata,
            prompt=ENDING_PROMPT,
            targets=(str(run.ending),),
            label_set=ENDINGS,
        )
        yield AnswerTask(
            inputs=inputs,
            split=split,
            input_annotations=context,
            metadata=metadata,
            prompt=STATUS_PROMPT,
            targets=(status(checkpoint),),
        )


CONNECTOR = SWEAgentTrajectoriesConnector
