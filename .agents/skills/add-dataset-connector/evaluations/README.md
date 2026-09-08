# Evaluations for `add-dataset-connector`

Three scenarios that test what this skill exists to fix. Each one targets a release the repo has
already converted, so the right answer is known and a run can be marked against the connector that
shipped.

There is no built-in runner. Give the `query` to an agent with this skill loaded, and mark the run
against `expected_behavior` by reading the plan it wrote and the code it produced.

## Why these three

Each names a decision the skill was written to force, and a way to get it wrong that a real run
took.

| file | the release | what it tests |
| --- | --- | --- |
| `01-row-shaped-no-identity.json` | `chengsenwang/tsqa` | a source that states no id and no rate |
| `02-two-shapes-one-container.json` | `physionet/sleep_edfx` | a survey that has to find what varies |
| `03-tasks-outnumber-samples.json` | `physionet/ecg_qa_cot` | the count that decides the task API |

## Marking a run

An expected behaviour is met or it is not. Two rules keep the marking honest:

- **The gate is pass or fail on its own.** A run that writes connector code before the user has
  ruled fails the scenario whatever else it got right.
- **A number in the plan is marked against the survey that produced it**, not against the number the
  shipped connector states. A run that predicts a wrong count and says how it counted has failed
  differently from one that states no count at all.
