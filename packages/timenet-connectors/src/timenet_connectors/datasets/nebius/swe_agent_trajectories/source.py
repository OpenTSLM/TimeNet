"""Read the SWE-agent trajectories release and turn each run's first steps into per-step signals.

A run is one attempt of SWE-agent to resolve one GitHub issue. Its trajectory is a system prompt, the
issue message, then alternating agent and environment messages. One agent message and the
environment's reply to it make one step. The connector keeps the first :data:`MAX_STEPS` steps of
every run, because no checkpoint looks further, and the run's step count and outcome. The outcome
is never part of a step.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
import re
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeNetConnectorError


# The twelve parquet shards of the pinned revision and their row counts.
SHARD_ROWS = {
    "train-00000-of-00012.parquet": 6670,
    "train-00001-of-00012.parquet": 6670,
    "train-00002-of-00012.parquet": 6670,
    "train-00003-of-00012.parquet": 6670,
    "train-00004-of-00012.parquet": 6670,
    "train-00005-of-00012.parquet": 6670,
    "train-00006-of-00012.parquet": 6670,
    "train-00007-of-00012.parquet": 6670,
    "train-00008-of-00012.parquet": 6669,
    "train-00009-of-00012.parquet": 6669,
    "train-00010-of-00012.parquet": 6669,
    "train-00011-of-00012.parquet": 6669,
}
# The latest checkpoint is at 30 steps, so no step after the 30th is kept.
MAX_STEPS = 30
# A trajectory opens with the system prompt and the issue message; the steps follow.
_OPENING = 2
_COLUMNS = ["instance_id", "model_name", "target", "trajectory", "exit_status"]
_BATCH_ROWS = 256
_ISSUE = re.compile(r"ISSUE:\n(.*?)\n\nINSTRUCTIONS:", re.DOTALL)
# An instance id is "<owner>__<repository>-<issue or pull request number>".
_INSTANCE = re.compile(r"(?P<repository>.+__.+)-\d+")
# SWE-agent's ThoughtActionParser: an opening fence may name a language, a closing fence is bare.
_FENCE = re.compile(r"^```(\S*)\s*\n|^```\s*$", re.MULTILINE)


@unique
class Ending(StrEnum):
    """How a run ended, grouped from the release's ``exit_status``."""

    SUBMITTED = "submitted"
    CONTEXT_LIMIT = "context_limit"
    EARLY_EXIT = "early_exit"
    SUBMITTED_NO_PATCH = "submitted_no_patch"
    FORMAT_ERRORS = "format_errors"
    COST_LIMIT = "cost_limit"


# Every exit_status of the release. "submitted (exit_x)" means SWE-agent hit limit x and submitted the
# changes so far by itself.
ENDINGS = {
    "submitted": Ending.SUBMITTED,
    "submitted (exit_context)": Ending.CONTEXT_LIMIT,
    "exit_context": Ending.CONTEXT_LIMIT,
    "early_exit": Ending.EARLY_EXIT,
    "submitted_no_patch": Ending.SUBMITTED_NO_PATCH,
    "submitted (exit_format)": Ending.FORMAT_ERRORS,
    "exit_format": Ending.FORMAT_ERRORS,
    "submitted (exit_cost)": Ending.COST_LIMIT,
    "exit_cost": Ending.COST_LIMIT,
}


@unique
class ActionKind(StrEnum):
    """What a step's command does, by its first word."""

    EDIT = "edit"
    CREATE = "create"
    RUN = "run"
    READ = "read"
    SEARCH = "search"
    SUBMIT = "submit"
    OTHER = "other"


_ACTION_WORDS = {
    "edit": ActionKind.EDIT,
    "create": ActionKind.CREATE,
    "python": ActionKind.RUN,
    "python3": ActionKind.RUN,
    "pytest": ActionKind.RUN,
    "open": ActionKind.READ,
    "goto": ActionKind.READ,
    "scroll_down": ActionKind.READ,
    "scroll_up": ActionKind.READ,
    "cat": ActionKind.READ,
    "search_dir": ActionKind.SEARCH,
    "search_file": ActionKind.SEARCH,
    "find_file": ActionKind.SEARCH,
    "find": ActionKind.SEARCH,
    "grep": ActionKind.SEARCH,
    "ls": ActionKind.SEARCH,
    "submit": ActionKind.SUBMIT,
}


@unique
class ObservationKind(StrEnum):
    """What the environment's reply to a step shows, by the first rule of :func:`observation_kind` it meets."""

    EDIT_REJECTED = "edit_rejected"
    TIMEOUT = "timeout"
    TRACEBACK = "traceback"
    COMMAND_ERROR = "command_error"
    SILENT_SUCCESS = "silent_success"
    OUTPUT = "output"


_EDIT_REJECTED = "Your proposed edit has introduced new syntax error"
_TIMEOUT = "EXECUTION TIMED OUT"
_TRACEBACK = "Traceback (most recent call last)"
_SILENT_SUCCESS = "Your command ran successfully and did not produce any output"
_COMMAND_ERRORS = ("command not found", "No such file or directory", "syntax error near unexpected token")


def split_message(text: str) -> tuple[str, str]:
    """Split an agent message into its thought and its command, as SWE-agent's ThoughtActionParser does.

    The command is the content of the last complete top-level fenced block. The thought is the rest of
    the message. A message without a complete block is all thought, with an empty command.

    Returns:
        The thought and the command, both stripped.
    """
    stack: list[re.Match[str]] = []
    last: tuple[re.Match[str], re.Match[str]] | None = None
    for match in _FENCE.finditer(text):
        if stack and not match.group(1):
            start = stack.pop()
            if not stack:
                last = (start, match)
        elif match.group(1) is not None:
            stack.append(match)
    if last is None:
        return text.strip(), ""
    start, end = last
    return (text[: start.start()] + text[end.end() :]).strip(), text[start.end() : end.start()].strip()


def action_kind(command: str) -> ActionKind:
    """Classify a command by its first word.

    Returns:
        The kind; ``other`` for an empty command or a word not in the table.
    """
    words = command.split(maxsplit=1)
    return _ACTION_WORDS.get(words[0], ActionKind.OTHER) if words else ActionKind.OTHER


def observation_kind(text: str) -> ObservationKind:
    """Classify the environment's reply, without its leading whitespace, by the first of these rules that holds.

    1. It starts with SWE-agent's message for an edit that its linter rejected.
    2. It starts with ``EXECUTION TIMED OUT``.
    3. It contains a Python traceback.
    4. Its first line starts with ``Usage:`` or ``Error:``, or names a shell error.
    5. It is SWE-agent's message for a command that printed nothing.
    6. Otherwise it is plain output.

    Returns:
        The kind.
    """
    # Some replies open with a blank line, e.g. the timeout message.
    text = text.lstrip()
    if text.startswith(_EDIT_REJECTED):
        return ObservationKind.EDIT_REJECTED
    if text.startswith(_TIMEOUT):
        return ObservationKind.TIMEOUT
    if _TRACEBACK in text:
        return ObservationKind.TRACEBACK
    first = text.split("\n", 1)[0]
    if first.startswith(("Usage:", "Error:")) or any(error in first for error in _COMMAND_ERRORS):
        return ObservationKind.COMMAND_ERROR
    if text.startswith(_SILENT_SUCCESS):
        return ObservationKind.SILENT_SUCCESS
    return ObservationKind.OUTPUT


@dataclass(frozen=True)
class Steps:
    """The per-step signals of a run's first steps, one entry per step."""

    thought_chars: np.ndarray
    action_chars: np.ndarray
    observation_chars: np.ndarray
    context_chars: np.ndarray
    """The characters of the whole conversation after the step's reply, system prompt included."""
    repeated_action: np.ndarray
    """Whether the step's command equals an earlier command of the run, character for character."""
    action_kind: tuple[ActionKind, ...]
    observation_kind: tuple[ObservationKind, ...]
    thoughts: bytes
    """The thoughts, NUL-joined and compressed, because the release holds about 0.7 GB of them."""

    def thought_texts(self, count: int) -> list[str]:
        """The thoughts of the first ``count`` steps.

        Returns:
            One string per step.
        """
        return zlib.decompress(self.thoughts).decode().split("\0")[:count]


@dataclass(frozen=True)
class Run:
    """One run of the release: its identity, its outcome, and the signals of its first steps."""

    shard: str
    row: int
    instance_id: str
    repository: str
    model_name: str
    resolved: bool
    exit_status: str
    issue: str
    n_steps: int
    """The run's step count. Its last step is the run's final action."""
    steps: Steps

    @property
    def run_id(self) -> str:
        """The run's place in the release: shard stem and row."""
        return f"{self.shard.removesuffix('.parquet')}:{self.row}"

    @property
    def ending(self) -> Ending:
        """How the run ended."""
        return ENDINGS[self.exit_status]


def _steps(messages: list[dict], opening: int, context: str) -> tuple[int, Steps]:
    """Parse the agent and environment messages that follow the issue message.

    Args:
        messages: The messages after the issue message.
        opening: The characters the agent reads before its first step: system prompt and issue message.
        context: The row, for error messages.

    Returns:
        The run's step count and the signals of its first :data:`MAX_STEPS` steps.

    Raises:
        TimeNetConnectorError: If the messages do not alternate agent, environment, ending on the agent.
    """
    for index, message in enumerate(messages):
        if message["role"] != ("ai" if index % 2 == 0 else "user"):
            raise TimeNetConnectorError(f"{context}: message {index + 2} is {message['role']!r} out of turn")
    if messages and messages[-1]["role"] != "ai":
        raise TimeNetConnectorError(f"{context}: the trajectory does not end on an agent message")
    n_steps = (len(messages) + 1) // 2
    kept = min(n_steps, MAX_STEPS)
    numbers = np.zeros((4, kept), dtype=np.int32)
    repeated = np.zeros(kept, dtype=bool)
    actions, observations, thoughts, seen = [], [], [], set()
    for step in range(kept):
        thought, command = split_message(messages[2 * step]["text"] or "")
        # The final step has no reply; every kept step of a checkpoint does.
        reply = messages[2 * step + 1]["text"] or "" if 2 * step + 1 < len(messages) else ""
        numbers[:3, step] = len(thought), len(command), len(reply)
        numbers[3, step] = len(messages[2 * step]["text"] or "") + len(reply)
        repeated[step] = command in seen
        seen.add(command)
        actions.append(action_kind(command))
        observations.append(observation_kind(reply))
        thoughts.append(thought.replace("\0", ""))
    return n_steps, Steps(
        thought_chars=numbers[0],
        action_chars=numbers[1],
        observation_chars=numbers[2],
        context_chars=(opening + np.cumsum(numbers[3])).astype(np.int32),
        repeated_action=repeated,
        action_kind=tuple(actions),
        observation_kind=tuple(observations),
        thoughts=zlib.compress("\0".join(thoughts).encode()),
    )


def parse_run(row: dict, shard: str, index: int) -> Run:
    """Check one release row and parse it into a :class:`Run`.

    Returns:
        The run.

    Raises:
        TimeNetConnectorError: If the row breaks the release's shape: an unknown exit status, a
            malformed instance id, a trajectory that does not open with the system prompt and the
            issue, or messages out of turn.
    """
    context = f"{shard} row {index}"
    instance = _INSTANCE.fullmatch(row["instance_id"] or "")
    if instance is None:
        raise TimeNetConnectorError(f"{context}: malformed instance_id {row['instance_id']!r}")
    if row["exit_status"] not in ENDINGS:
        raise TimeNetConnectorError(f"{context}: unknown exit_status {row['exit_status']!r}")
    if not isinstance(row["target"], bool) or not row["model_name"]:
        raise TimeNetConnectorError(f"{context}: target must be true or false and model_name set")
    messages = row["trajectory"] or []
    if len(messages) < _OPENING or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise TimeNetConnectorError(f"{context}: the trajectory must open with the system prompt and the issue")
    issue = _ISSUE.search(messages[1]["text"] or "")
    if issue is None:
        raise TimeNetConnectorError(f"{context}: the first message holds no ISSUE section")
    opening = len(messages[0]["system_prompt"] or "") + len(messages[0]["text"] or "") + len(messages[1]["text"])
    n_steps, steps = _steps(messages[_OPENING:], opening, context)
    return Run(
        shard=shard,
        row=index,
        instance_id=row["instance_id"],
        repository=instance["repository"],
        model_name=row["model_name"],
        resolved=row["target"],
        exit_status=row["exit_status"],
        issue=issue.group(1),
        n_steps=n_steps,
        steps=steps,
    )


def read_runs(root: Path) -> Iterator[Run]:
    """Read every run of the release, shard by shard, in row order.

    Args:
        root: The directory that holds the parquet shards.

    Yields:
        Each run.

    Raises:
        TimeNetConnectorError: If a shard is missing, unreadable, or holds a different number of rows
            than the pinned revision.
    """
    for shard, expected in SHARD_ROWS.items():
        path = root / shard
        if not path.is_file():
            raise TimeNetConnectorError(f"{path} is missing")
        count = 0
        try:
            batches = pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS, columns=_COLUMNS)
            for batch in batches:
                for row in batch.to_pylist():
                    yield parse_run(row, shard, count)
                    count += 1
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{path}: {exc}") from exc
        if count != expected:
            raise TimeNetConnectorError(f"{path}: expected {expected} rows, found {count}")
