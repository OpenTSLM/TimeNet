from collections import Counter
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types import AnswerTask, ClassificationTask
from timenet.types.text import read_value
from timenet_connectors.datasets.nebius.swe_agent_trajectories import SWEAgentTrajectoriesConnector, source
from timenet_connectors.datasets.nebius.swe_agent_trajectories.source import (
    ActionKind,
    ObservationKind,
    Run,
    SourceRow,
    action_kind,
    observation_kind,
    parse_run,
    read_runs,
    split_message,
)
from timenet_connectors.datasets.nebius.swe_agent_trajectories.tasks import ENDING_LABELS, checkpoints, status


# A synthetic release shaped like the Hub parquet: one shard of runs over twelve repositories, so the
# split has enough groups for its ten folds. The texts are hand-made, so the repo ships no dataset bytes.
SHARD = "train-00000-of-00012.parquet"
SYSTEM = "SETTING: You are an autonomous programmer."
ISSUE = "Parser crashes on empty input\nCalling parse('') raises IndexError."


def _message(role: str, text: str, system_prompt: str | None = None) -> dict:
    return {"cutoff_date": None, "mask": role == "ai", "role": role, "system_prompt": system_prompt, "text": text}


def _agent(thought: str, command: str) -> str:
    return f"{thought}\n\n```\n{command}\n```"


# One cycle of steps; a run of n steps repeats it and ends on a submit.
CYCLE = (
    (_agent("Let's reproduce the bug first.", "create reproduce.py"), "[File: /repo/reproduce.py (1 lines total)]"),
    (
        _agent("Write the script.", "edit 1:1\nprint(parse(''))\nend_of_edit"),
        "Your proposed edit has introduced new syntax error(s).",
    ),
    (_agent("Run it.", "python reproduce.py"), "Traceback (most recent call last):\n  File ...\nIndexError"),
    (_agent("Look at the parser.", "open src/parser.py"), "[File: /repo/src/parser.py (120 lines total)]"),
    (_agent("Search for the call.", "search_dir parse"), 'Found 3 matches for "parse"'),
)


def _trajectory(n_steps: int) -> list[dict]:
    messages = [_message("system", "", SYSTEM), _message("user", f"ISSUE:\n{ISSUE}\n\nINSTRUCTIONS:\nFix it.")]
    for step in range(n_steps - 1):
        agent, reply = CYCLE[step % len(CYCLE)]
        messages += [_message("ai", agent), _message("user", reply)]
    return [*messages, _message("ai", _agent("Done.", "submit"))]


def _row(repository: int, number: int, n_steps: int, *, resolved: bool, exit_status: str = "submitted") -> dict:
    return {
        "instance_id": f"owner{repository}__repo{repository}-{number}",
        "model_name": "swe-agent-llama-70b",
        "target": resolved,
        "trajectory": _trajectory(n_steps),
        "exit_status": exit_status,
        "generated_patch": "",
        "eval_logs": "",
    }


def _parse(row: dict) -> Run:
    # The reader selects the columns the connector reads; the patch and the test log are not among them.
    return parse_run(SourceRow.model_validate({name: row[name] for name in SourceRow.model_fields}), SHARD, 0)


def _rows() -> list[dict]:
    rows = []
    for repository in range(12):
        # Instance 3 has only a 4-step run, so no checkpoint and no issue annotation.
        rows.append(_row(repository, 3, 4, resolved=True))
        rows.append(_row(repository, 1, 12, resolved=repository % 3 == 0))
        rows.append(_row(repository, 2, 33, resolved=False, exit_status="submitted (exit_context)"))
    return rows


def _write(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), root / SHARD)
    return root


@pytest.fixture
def one_shard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def release(rows: list[dict]) -> Path:
        monkeypatch.setattr(source, "SHARD_ROWS", {SHARD: len(rows)})
        return _write(tmp_path / "release", rows)

    return release


@pytest.fixture
def dataset(one_shard):
    return SWEAgentTrajectoriesConnector().convert({"train": one_shard(_rows())})


def _target(task) -> str:
    assert task.targets is not None
    return read_value(task.targets[0])


def _thoughts(record) -> list[str]:
    (annotation,) = record.annotations
    assert annotation.key == "agent_thoughts"
    value = read_value(annotation.value)
    assert isinstance(value, list)
    return value


def test_split_message_takes_the_last_fenced_block_as_the_command() -> None:
    text = "Here is code:\n```python\nx = 1\n```\nNow run it.\n```\npython x.py\n```\n"
    assert split_message(text) == ("Here is code:\n```python\nx = 1\n```\nNow run it.", "python x.py")
    assert split_message("Only thinking.") == ("Only thinking.", "")


@pytest.mark.parametrize(
    ("command", "kind"),
    [
        ("edit 1:2\nx\nend_of_edit", ActionKind.EDIT),
        ("create a.py", ActionKind.CREATE),
        ("python a.py", ActionKind.RUN),
        ("goto 40", ActionKind.READ),
        ("find_file a.py", ActionKind.SEARCH),
        ("submit", ActionKind.SUBMIT),
        ("git diff", ActionKind.OTHER),
        ("", ActionKind.OTHER),
    ],
)
def test_action_kind_reads_the_first_word(command: str, kind: ActionKind) -> None:
    assert action_kind(command) == kind


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        ("Your proposed edit has introduced new syntax error(s). Traceback (most recent call last)", "edit_rejected"),
        ("\nEXECUTION TIMED OUT", "timeout"),
        ("Running...\nTraceback (most recent call last):\n", "traceback"),
        ("Usage: open <file>", "command_error"),
        ("/bin/bash: line 1: foo: command not found", "command_error"),
        ("Your command ran successfully and did not produce any output.", "silent_success"),
        ("[File: /repo/a.py (10 lines total)]", "output"),
    ],
)
def test_observation_kind_applies_its_rules_in_order(reply: str, kind: str) -> None:
    assert observation_kind(reply) == ObservationKind(kind)


def test_a_run_keeps_its_first_steps_and_its_outcome_apart() -> None:
    run = _parse(_row(0, 1, 7, resolved=False))
    assert run.n_steps == 7
    assert run.issue == ISSUE
    assert run.repository == "owner0__repo0"
    assert list(run.steps.action_kind) == ["create", "edit", "run", "read", "search", "create", "submit"]
    assert list(run.steps.observation_kind[:3]) == ["output", "edit_rejected", "traceback"]
    # The final step has no reply.
    assert run.steps.observation_chars[-1] == 0
    assert list(run.steps.repeated_action) == [False, False, False, False, False, True, False]
    opening = len(SYSTEM) + len(f"ISSUE:\n{ISSUE}\n\nINSTRUCTIONS:\nFix it.")
    first = len(CYCLE[0][0]) + len(CYCLE[0][1])
    assert run.steps.context_chars[0] == opening + first
    assert all(later > earlier for earlier, later in pairwise(run.steps.context_chars))
    assert run.steps.thought_texts(2) == ["Let's reproduce the bug first.", "Write the script."]


def test_a_thought_is_kept_as_written() -> None:
    row = _row(0, 1, 7, resolved=False)
    row["trajectory"][2]["text"] = _agent("A thought with a NUL \0 and ```inline``` fences.", "ls")
    run = _parse(row)
    assert run.steps.thought_texts(2) == ["A thought with a NUL \0 and ```inline``` fences.", "Write the script."]
    assert run.steps.thought_chars[0] == len(run.steps.thought_texts(1)[0])


def test_only_a_run_that_goes_on_gets_a_checkpoint() -> None:
    assert [c.steps for c in checkpoints(_parse(_row(0, 1, 5, resolved=False)))] == []
    assert [c.steps for c in checkpoints(_parse(_row(0, 1, 6, resolved=False)))] == [5]
    assert [c.steps for c in checkpoints(_parse(_row(0, 1, 33, resolved=False)))] == [5, 10, 20, 30]


def test_records_hold_the_steps_so_far_and_nothing_about_the_outcome(dataset) -> None:
    records = list(dataset.records)
    # Per repository: the 4-step run gives none, the 12-step run 2, the 33-step run 4.
    assert len(records) == 12 * 6
    for record in records:
        k = record.metadata["checkpoint_steps"]
        assert set(record.metadata) == {"run_id", "instance_id", "repository", "model_name", "checkpoint_steps"}
        assert {signal.name for signal in record.sources[0].signals} == set(source.Steps.__annotations__) - {"thoughts"}
        assert all(signal.n_values == k for signal in record.sources[0].signals)
        assert len(_thoughts(record)) == k
        # The last visible step is never the run's submit.
        kinds = next(s for s in record.sources[0].signals if s.name == "action_kind")
        assert "submit" not in kinds.to_arrow().to_pylist()


def test_each_checkpoint_gives_three_tasks_with_the_issue_and_thoughts_as_input(dataset) -> None:
    tasks = list(dataset.iter_tasks())
    assert len(tasks) == 3 * 12 * 6
    assert Counter(type(task) for task in tasks) == {ClassificationTask: 2 * 72, AnswerTask: 72}
    # One issue annotation per instance with a checkpoint: instances 1 and 2 of each repository.
    issues = [a for a in dataset.annotations if a.key == "issue"]
    assert len(issues) == 24
    assert not any(a.metadata["instance_id"].endswith("-3") for a in issues)
    for task in tasks:
        assert [a.key for a in task.input_annotations] == ["issue", "agent_thoughts"]
        assert task.input_annotations[0].metadata["instance_id"] == task.metadata["instance_id"]
        assert task.metadata == task.inputs[0].metadata
    endings = {
        read_value(t.targets[0]) for t in tasks if isinstance(t, ClassificationTask) and t.label_set == ENDING_LABELS
    }
    assert endings == {"submitted", "context_limit"}


def test_the_status_report_states_the_steps_so_far_and_what_comes() -> None:
    run = _parse(_row(0, 1, 12, resolved=True))
    (first, *_) = checkpoints(run)
    assert status(first) == (
        "In its first 5 steps the agent edited files 1 time, created 1 file, ran code 1 time, "
        "read or searched the repository 2 times and repeated an earlier command 0 times. "
        "The linter rejected 1 edit, 1 reply showed a Python traceback, 0 replies a command error "
        f"and 0 commands timed out. The conversation holds {int(run.steps.context_chars[4]):,} characters. "
        "The run will go on for 7 more steps, then the agent will submit its changes, and the agent will "
        "resolve the issue."
    )


def test_no_repository_is_in_two_partitions(dataset) -> None:
    splits: dict[str, set] = {}
    for task in dataset.iter_tasks():
        splits.setdefault(task.metadata["repository"], set()).add(task.split)
    assert all(len(found) == 1 for found in splits.values())
    assert len({split for found in splits.values() for split in found}) == 3


def test_dataset_survives_storage_round_trip(dataset, tmp_path: Path) -> None:
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(dataset)
    with TimeFReader(open_local_version(registry.root / "nebius/swe_agent_trajectories/1.0.0")) as reader:
        restored = reader.read()
        record = next(iter(restored.records))
        signals = {signal.name: signal for signal in record.sources[0].signals}
        assert signals["action_kind"].to_arrow().to_pylist() == ["create", "edit", "run", "read", "search"]
        assert signals["observation_kind"].to_arrow().to_pylist()[1] == "edit_rejected"
        assert _thoughts(record)[0] == "Let's reproduce the bug first."
        restored_tasks = list(restored.iter_tasks())
        original = list(dataset.iter_tasks())
        assert [t.split for t in restored_tasks] == [t.split for t in original]
        assert [_target(t) for t in restored_tasks] == [_target(t) for t in original]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda row: row.update(exit_status="exit_unknown"), "unknown exit_status"),
        (lambda row: row.update(instance_id="no-separator"), "malformed instance_id"),
        (lambda row: row["trajectory"][1].update(text="no issue here"), "no ISSUE section"),
        (lambda row: row["trajectory"][3].update(role="ai"), "out of turn"),
        (lambda row: row["trajectory"].pop(), "does not end on an agent message"),
        (lambda row: row["trajectory"].pop(0), "must open with the system prompt"),
        (lambda row: row["trajectory"][2].update(text=None), "message 2 has no text"),
        (lambda row: row["trajectory"][0].update(text="more prompt"), "system_prompt only"),
        (lambda row: row.update(model_name=""), "model_name is empty"),
    ],
)
def test_a_malformed_row_fails_the_build(change, message: str) -> None:
    row = _row(0, 1, 6, resolved=False)
    change(row)
    with pytest.raises(TimeNetConnectorError, match=message):
        _parse(row)


def test_a_row_without_a_target_fails_the_build(one_shard) -> None:
    rows = _rows()
    rows[1]["target"] = None
    with pytest.raises(TimeNetConnectorError, match="target"):
        list(read_runs(one_shard(rows)))


def test_a_shard_with_another_row_count_fails_the_build(one_shard, monkeypatch: pytest.MonkeyPatch) -> None:
    root = one_shard(_rows())
    monkeypatch.setattr(source, "SHARD_ROWS", {SHARD: 1})
    with pytest.raises(TimeNetConnectorError, match="expected 1 rows"):
        list(read_runs(root))
