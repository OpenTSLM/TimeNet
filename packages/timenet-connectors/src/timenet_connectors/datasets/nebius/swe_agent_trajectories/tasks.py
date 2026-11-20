"""Cut each run at fixed checkpoints and ask, from the steps so far, how the run will turn out.

A checkpoint is the first ``k`` steps of a run that is still going after them: the run has more than
``k`` steps, so its final action is never among them. How a run ends (a submission, the context limit,
its length) would give its outcome away, and a monitor never sees it before the end. Each checkpoint
gives three tasks: will the run resolve its issue, how will it end, and a short status report. The
answers come from the release's ``target`` and ``exit_status`` and the steps' signals by fixed rules.
No model writes them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from timenet.types import Label, LabelSet
from timenet.types.splits import Split
from timenet_connectors.datasets.nebius.swe_agent_trajectories.source import (
    ActionKind,
    Ending,
    ObservationKind,
    Run,
)
from timenet_connectors.splitting import StratifiedSplitter


CHECKPOINTS = (5, 10, 20, 30)

RESOLVED = "resolved"
UNRESOLVED = "unresolved"
OUTCOMES = LabelSet(
    name="run_outcome",
    labels=(
        Label(name=RESOLVED, description="The run's final patch passed the tests of the issue's pull request."),
        Label(name=UNRESOLVED, description="The run did not resolve the issue."),
    ),
)
ENDINGS = LabelSet(
    name="run_ending",
    labels=(
        Label(name=Ending.SUBMITTED, description="The agent submitted its changes."),
        Label(
            name=Ending.CONTEXT_LIMIT,
            description="The run reached the model's context limit; SWE-agent may have submitted the changes so far.",
        ),
        Label(name=Ending.EARLY_EXIT, description="The release's early_exit status."),
        Label(name=Ending.SUBMITTED_NO_PATCH, description="The agent submitted without a patch."),
        Label(
            name=Ending.FORMAT_ERRORS,
            description="SWE-agent stopped the run after repeated malformed agent messages.",
        ),
        Label(name=Ending.COST_LIMIT, description="The run reached its cost limit."),
    ),
)

_CONTEXT = (
    "These are the first steps of a software engineering agent (SWE-agent) that is trying to resolve "
    "the GitHub issue above. The run is still going."
)
RESOLVE_PROMPT = f"{_CONTEXT} Will the agent's final patch resolve the issue?"
ENDING_PROMPT = f"{_CONTEXT} How will the run end?"
STATUS_PROMPT = (
    f"{_CONTEXT} Report how the agent is doing so far, then say how the run will end and whether it "
    "will resolve the issue."
)

_ENDING_TEXT = {
    "submitted": "the agent submitted its changes",
    "submitted (exit_context)": "it reached the context limit and SWE-agent submitted its changes",
    "exit_context": "it reached the context limit",
    "early_exit": "it ended with the status early_exit",
    "submitted_no_patch": "the agent submitted without a patch",
    "submitted (exit_format)": "SWE-agent stopped it after repeated malformed messages and submitted its changes",
    "exit_format": "SWE-agent stopped it after repeated malformed messages",
    "submitted (exit_cost)": "it reached its cost limit and SWE-agent submitted its changes",
    "exit_cost": "it reached its cost limit",
}


@dataclass(frozen=True)
class Checkpoint:
    """The first ``steps`` steps of one run."""

    run: Run
    steps: int


def checkpoints(run: Run) -> list[Checkpoint]:
    """List the checkpoints of a run: every ``k`` of :data:`CHECKPOINTS` that the run outlasts.

    Returns:
        The checkpoints, earliest first.
    """
    return [Checkpoint(run, k) for k in CHECKPOINTS if run.n_steps > k]


def split_runs(runs: list[Run]) -> dict[str, Split]:
    """Assign each run to a partition by its repository, so no repository is in two partitions.

    A repository's runs share its code, and an issue's runs share the issue, so a model could
    otherwise recognise a test run's repository or issue from training.

    Returns:
        The partition of each run, keyed by run id.
    """
    splitter = StratifiedSplitter(
        train_folds=8,
        validation_folds=1,
        test_folds=1,
        stratify_by=lambda run: run.resolved,
        group_by=lambda run: run.repository,
        seed=42,
    )
    return {run.run_id: split for split, members in splitter.split(runs).items() for run in members}


def _count(number: int, singular: str, plural: str | None = None) -> str:
    return f"{number} {singular if number == 1 else plural or singular + 's'}"


def status(checkpoint: Checkpoint) -> str:
    """Write the status report of a checkpoint from its steps' signals and the run's outcome.

    Returns:
        Three sentences on the steps so far, then one on how the run ended and whether it resolved
        the issue.
    """
    k, run = checkpoint.steps, checkpoint.run
    actions = Counter(run.steps.action_kind[:k])
    replies = Counter(run.steps.observation_kind[:k])
    repeats = int(run.steps.repeated_action[:k].sum())
    looked = actions[ActionKind.READ] + actions[ActionKind.SEARCH]
    rest = run.n_steps - k
    return (
        f"In its first {k} steps the agent edited files {_count(actions[ActionKind.EDIT], 'time')}, "
        f"created {_count(actions[ActionKind.CREATE], 'file')}, ran code {_count(actions[ActionKind.RUN], 'time')}, "
        f"read or searched the repository {_count(looked, 'time')} "
        f"and repeated an earlier command {_count(repeats, 'time')}. "
        f"The linter rejected {_count(replies[ObservationKind.EDIT_REJECTED], 'edit')}, "
        f"{_count(replies[ObservationKind.TRACEBACK], 'reply', 'replies')} showed a Python traceback, "
        f"{_count(replies[ObservationKind.COMMAND_ERROR], 'reply', 'replies')} a command error "
        f"and {_count(replies[ObservationKind.TIMEOUT], 'command')} timed out. "
        f"The conversation holds {int(run.steps.context_chars[k - 1]):,} characters. "
        f"The run went on for {_count(rest, 'more step')}, then {_ENDING_TEXT[run.exit_status]}, "
        f"and {'resolved' if run.resolved else 'did not resolve'} the issue."
    )
