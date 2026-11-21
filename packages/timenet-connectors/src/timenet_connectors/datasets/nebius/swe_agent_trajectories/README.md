# SWE-agent trajectories: the release, and the decisions of this connector

Nebius released 80,036 runs of SWE-agent, a coding agent, each trying to resolve one real GitHub
issue. Each run is labelled with whether its final patch passed the tests of the issue's pull request.

This connector turns the runs into an **early-warning** dataset: a model sees the first steps of a run
that is still going and says how the run will turn out, like a monitor watching an agent at work.

This document covers:
- what the release holds;
- how the connector maps it onto TimeF;
- why no input shows how a run ends;
- what a user should know before training or evaluating on it.

*(measured)* marks a number taken from the files, not from the dataset page.

## Source, licence and citation

**Source:** the Hugging Face dataset [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), pinned at revision `68195a1450865274106246d0d0296a1d6807b88e` (last modified 2024-12-23).
- 12 parquet shards, 1.1 GB to download, one `train` split of 80,036 rows *(measured)*.
- The connector pins the revision and each shard's row count, so a changed release fails the build.

**Licence:** CC BY 4.0, as stated on the dataset page. The page adds two conditions that this card's
single licence field cannot hold:
- the repositories behind the issues keep their own licences, listed in nebius/SWE-bench-extra;
- using the model outputs requires complying with the Llama 3.1 License, because Llama models wrote them.

**Citation:** the release has no citation entry. Cite the dataset page and the pinned revision:
> Nebius. SWE-agent trajectories (nebius/SWE-agent-trajectories). Hugging Face, 2024. Revision
> 68195a1450865274106246d0d0296a1d6807b88e.

## What a run is

Each row is one run of one model on one issue:
- `instance_id`: `<owner>__<repository>-<number>`;
- `model_name`: `swe-agent-llama-70b` (74,792 runs), `-8b` (4,053) or `-405b` (1,191) *(measured)*;
- `target`: whether the run resolved the issue: 13,389 true, 66,647 false *(measured)*;
- `exit_status`: how the run ended;
- `trajectory`: the messages;
- `generated_patch` and `eval_logs`: the final patch and its test log. **The connector reads neither.**

**The trajectory** is a system prompt, the issue message, then alternating agent and environment
messages, ending on an agent message. Every run follows this shape *(measured)*.
- **A step** is one agent message and the environment's reply to it.
- **An agent message** is a thought followed by one command in a fenced block.
- **The connector splits them** as SWE-agent's `ThoughtActionParser` does: the command is the last complete fenced block, and the thought is the rest.

| | Value *(measured)* |
| --- | --- |
| Steps per run | median 17, 90th percentile 50, maximum 408; one run has none |
| Repositories, issues | 1,276 repositories, 3,591 issues; 807 issues have both resolved and unresolved runs |
| Largest repositories | pydantic/pydantic 6,279 runs, iterative/dvc 5,646 |

**How runs end** (`exit_status` *(measured)*), and the ending label the connector gives it:

| `exit_status` | Runs | Ending label |
| --- | --- | --- |
| `submitted` | 51,087 | `submitted` |
| `submitted (exit_context)` | 21,026 | `context_limit` |
| `exit_context` | 3,568 | `context_limit` |
| `early_exit` | 3,176 | `early_exit` |
| `submitted_no_patch` | 1,066 | `submitted_no_patch` |
| `submitted (exit_format)`, `exit_format` | 80, 20 | `format_errors` |
| `submitted (exit_cost)`, `exit_cost` | 10, 3 | `cost_limit` |

`submitted (exit_x)` means SWE-agent reached limit x and submitted the changes so far itself. The
release does not explain `early_exit`; the connector keeps the name.

## Why the inputs never show the end of a run

The end of a run gives its outcome away:
- **The final step:** every `submitted` run ends with the command `submit` *(measured)*.
- **Running out of context:** 36 % of unresolved runs stop at the context limit, against 5 % of resolved ones *(measured)*.
- **Length:** unresolved runs are longer.

A model that sees whole runs can learn "long, or cut off, means failure". That is not monitoring, and
a live monitor never sees the end before it happens.

**So the connector only cuts checkpoints.** A checkpoint is the first *k* steps of a run, for
*k* = 5, 10, 15, 20 and 30, and only if the run has more than *k* steps. Its last visible step is
therefore never the run's final action.
- **What no input holds:** the outcome, the exit status, the run's length, the patch and the test log.
- **What the inputs do hold:** a growing conversation, which is real information a monitor has. The longest checkpoint ends at 30 steps.

## Records

Each checkpoint is **one Record**: 235,978 in all *(measured)*.

| Checkpoint | Runs still going | Resolved among them |
| --- | --- | --- |
| 5 steps | 76,182 | 16.6 % |
| 10 steps | 59,708 | 13.8 % |
| 15 steps | 43,550 | 9.2 % |
| 20 steps | 34,059 | 6.4 % |
| 30 steps | 22,479 | 4.2 % |

**Signals.** The Record has one source, `agent_steps`, with nine signals of *k* values each, one per
step, on an ordinal axis. The steps have an order but no timestamps: the release records none.

| Signal | dtype | Value at each step |
| --- | --- | --- |
| `thought_chars` | int32 | characters of the agent's thought |
| `action_chars` | int32 | characters of the command |
| `observation_chars` | int32 | characters of the environment's reply |
| `context_chars` | int32 | characters of the whole conversation after the reply, from the system prompt on |
| `repeated_action` | bool | the command equals an earlier command of the run, character for character |
| `action_kind` | enum | `edit`, `create`, `run`, `read`, `search`, `submit`, `other`, by the command's first word |
| `observation_kind` | enum | `edit_rejected`, `timeout`, `traceback`, `command_error`, `silent_success`, `output` |
| `same_file_edit` | bool | the step is an `edit` of the same file as the run's previous edit |
| `edit_target` | enum | for an `edit`, the kind of file it changes: `source`, `test` or `own_script`; `none` for every other step |

**`action_kind` rules**, by the command's first word:
- `edit`: `edit`;
- `create`: `create`;
- `run`: `python`, `python3`, `pytest`;
- `read`: `open`, `goto`, `scroll_down`, `scroll_up`, `cat`;
- `search`: `search_dir`, `search_file`, `find_file`, `find`, `grep`, `ls`;
- `submit`: `submit`;
- `other`: every other command, or none.

**`observation_kind` rules.** The first of these that holds wins:
1. `edit_rejected`: the reply starts with SWE-agent's message for an edit its linter rejected ("Your proposed edit has introduced new syntax error").
2. `timeout`: it starts with `EXECUTION TIMED OUT`.
3. `traceback`: it contains `Traceback (most recent call last)`. This includes a file view that
   merely shows such text, e.g. after `open` or `edit`. In shard 0 that is 69 of 9,375 tracebacks
   *(measured)*.
4. `command_error`: its first line starts with `Usage:` or `Error:`, or contains `command not found`, `No such file or directory` or `syntax error near unexpected token`.
5. `silent_success`: it starts with SWE-agent's message for a command that printed nothing.
6. `output`: anything else.

**File rules.** An `edit` command names no file: SWE-agent applies it to the file open at that
moment. Every reply ends with that file, `(Open file: <absolute path>)`, or `n/a`, so the connector
follows it. An `edit` changes the file the reply before it names, and its `edit_target` is:
- `own_script`: the agent created that file earlier in the run. A `create` opens the new file, so the reply after it names the file. Reproduction scripts are the usual case.
- `test`: otherwise, if a folder of the path below the repository's folder is `test`, `tests` or `testing`, or the file is `conftest.py`, `test_*` or `*_test.py`. The repository's folder is skipped, so the files of a repository called `pytest` are not all tests.
- `source`: any other file.
- `none`: the step is not an `edit`, or no file is open (326 of 406,189 edits *(measured)*).

`same_file_edit` compares an edit's file with the file of the run's previous edit, so the first edit
of a run is never a same-file edit.

Over the first 30 steps of every run *(measured)*:
- **Replies:** plain output 1,030,976; linter rejections 179,147; tracebacks 114,145; silent successes 55,182; command errors 49,989; timeouts 6,235.
- **Repeats:** 28 % of steps repeat an earlier command. Rerunning a script counts as a repeat.

Over the steps the checkpoints show (each run's longest checkpoint, 1,292,285 steps) *(measured)*:
- **Edits:** 406,189. Of these, 235,311 change source files, 163,682 files the agent created and 6,870 test files.
- **Same file:** 67 % of edits change the file the edit before changed.

**Text.** The agent's thoughts for the *k* steps are the record annotation `agent_thoughts`, a list
of *k* strings in step order, kept exactly as written. They are text, not a signal, because a text
series has no tensor form in `TimeFTorchDataset`.

**Metadata** (provenance, never model input): `run_id` (shard and row), `instance_id`, `repository`,
`model_name`, `checkpoint_steps`.

**The issue** is a dataset annotation with the key `issue`, stored once per instance, because each
issue has many runs. Its text is the issue message's `ISSUE:` section, and its metadata names the
`instance_id`. Only instances with a checkpoint get one: 3,599 annotations *(measured)*.
- 13 instances appear with two slightly different texts *(measured)*; each text is its own annotation.
- 16 texts are shared by two instances of the same repository, e.g. `iterative__dvc-5335` and
  `iterative__dvc-5336` *(measured)*; each instance gets its own annotation.

## Tasks

Each checkpoint gives three tasks. All take the Record as input and name the issue and the thoughts
in `input_annotations`. That makes 707,934 tasks *(measured)*.

| Task | Target | Source of the answer |
| --- | --- | --- |
| `ClassificationTask` `run_outcome` | `resolved` or `unresolved` | `target` |
| `ClassificationTask` `run_ending` | the ending label (table above) | `exit_status` |
| `AnswerTask` | a status report: the steps so far, then how many more steps the run will take, how it will end and whether it will resolve the issue | the signals, `target`, `exit_status`, the step count |

**How the text is made.** A fixed template fills in the numbers; no model writes it:

> In its first {k} steps the agent edited files {n} times, created {n} files, ran code {n} times,
> read or searched the repository {n} times and repeated an earlier command {n} times. The linter
> rejected {n} edits, {n} replies showed a Python traceback, {n} replies a command error and {n}
> commands timed out. {n} edits changed the same file as the edit before, and {n} edits changed a
> file the agent had created. The conversation holds {context} characters. The run will go on for
> {m} more steps, then {ending}, and the agent will {resolve | not resolve} the issue.

"Read or searched" adds the `read` and `search` steps. {ending} is one fixed phrase per `exit_status`,
e.g. "it will reach the context limit and SWE-agent will submit its changes". The prompt asks for
the same three things in the same tense.

**Example.** The first checkpoint of the release (run `train-00000-of-00012:0`, issue
`AnalogJ__lexicon-336`, 5 steps) has the target:

> In its first 5 steps the agent edited files 0 times, created 0 files, ran code 0 times, read or
> searched the repository 4 times and repeated an earlier command 0 times. The linter rejected 0
> edits, 1 reply showed a Python traceback, 1 reply a command error and 0 commands timed out. 0
> edits changed the same file as the edit before, and 0 edits changed a file the agent had created.
> The conversation holds 12,857 characters. The run will go on for 41 more steps, then it will
> reach the context limit and SWE-agent will submit its changes, and the agent will not resolve the
> issue.

**Task metadata:** the Record's metadata.

## The split

**Runs are grouped by repository.** `StratifiedSplitter` (8/1/1 folds, stratified on `target`,
seed 42) assigns whole repositories, and each checkpoint takes its run's partition.
- **Why:** a repository's runs share its code, and an issue's runs share the issue. With 807 issues holding both resolved and unresolved runs, a random split would test on issues seen in training.
- **The release** has a single `train` split, so there is no official split to follow.

**Result** *(measured)*:

| Partition | Repositories | Runs with a checkpoint | Tasks | Resolved at 5 / 10 / 15 / 20 / 30 steps |
| --- | --- | --- | --- | --- |
| train | 993 | 60,985 | 570,120 | 16.6 / 13.7 / 9.1 / 6.2 / 3.9 % |
| validation | 139 | 7,557 | 67,254 | 16.8 / 14.6 / 9.9 / 6.8 / 4.4 % |
| test | 143 | 7,640 | 70,560 | 16.4 / 14.1 / 9.4 / 7.1 / 5.9 % |

No repository and no issue is in two partitions. The checks count only the 1,275 repositories that
have a checkpoint.

## Evaluating on it

**Most runs fail.** A model that always answers `unresolved` is right 83 % of the time at 5 steps and
96 % at 30 steps. Accuracy therefore says little. We suggest:
- **Report each checkpoint separately.** The base rate changes with *k*, so pooled results mix five problems.
- **Use measures that do not reward the majority class:** AUROC, the precision-recall curve of `resolved` (its baseline is the resolved rate), balanced accuracy, and calibration (Brier score) for a health score.
- **Count each run once.** A long run gives up to five checkpoints, and long runs fail more often. When training, weight a checkpoint by 1 / (checkpoints of its run), or sample one checkpoint per run.
- **Compare against simple baselines:** always `unresolved`; the numeric signals alone; the thoughts alone; the issue text alone, which never sees the agent.
- **Compare runs on the same issue.** AUROC over all runs also rewards telling easy issues from hard
  ones. 807 issues have both resolved and unresolved runs: the share of such pairs where the
  resolved run scores higher measures only what a model reads from the run itself.
- **Score only the prediction in the status report.** Its first four sentences count what happened
  so far, which the model can read off its input. Only the last sentence predicts: how many more
  steps, how the run will end, and whether it will resolve the issue. A word-overlap score (ROUGE,
  BLEU) on the whole answer therefore stays high even when that prediction is wrong. Score the last
  sentence, which starts with "The run will go on for", and check its three facts one by one.

## Release details a user should know

- **The labels depend on the model as much as the issue.** 93 % of runs come from one model (Llama 70B), and each issue is attempted 22 times on average *(measured)*.
- **The thoughts are long:** 5.8 kB per run over the first 30 steps, on average *(measured)*.
- **Reply lengths count the logged text.** That includes SWE-agent's status lines (open file, current directory).
- **`mask` and `cutoff_date`** are not explained by the release. The connector does not use them.

## Limitations and intended use

**Limitations:**
- One agent framework (SWE-agent) and one model family (Llama 3.1).
- The signals and the kinds are rule-made from the logged text. The rules are fixed and listed above, but they are not exhaustive: a shell command the table does not list is `other`.
- Status reports are templates, not expert assessments.
- The outcome is the release's test result, which may miss correct fixes or accept incomplete ones.

**Intended use:**
- Training and evaluating time-series language models that monitor a coding agent while it works, and warn before a run fails.
- Studying which step patterns precede success or failure.

*Contributed by Team RAMadan during the TimeNet hackathon, Round 1 (November 2026).*
