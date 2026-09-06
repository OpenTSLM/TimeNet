---
name: review-connector-stack
description: Use when reviewing a TimeNet dataset connector before it merges — a whole `gh stack` chain, one PR of one, a branch, or the working tree. Runs an adversarial review against the repo's written and unwritten conventions: lazy vs top-level imports of connector dependencies, the I/O-apart-from-meaning split, ids, streamed tasks, fixtures, error types, and whether each PR of the stack stands on its own.
---

# Reviewing a connector stack

This review is adversarial. The default verdict on every check is **fail**, and the diff has to
produce the evidence that clears it. A check you did not open a file for is not a pass, it is
"not reviewed", and it says so in the report.

Read `references/checks.md` for the full checklist. The conventions it checks are written down in
three places, and where any of them disagrees with the code, the code wins:

- `AGENTS.md` — repo-wide rules: errors, `noqa` style, type checking, the habits under
  **Python Habits**.
- `.agents/skills/add-dataset-connector/references/fidelity.md` — what a connector may do to its
  source, ids, annotations, tasks.
- `.agents/skills/add-dataset-connector/references/layout.md` — how the modules divide, naming,
  keys, dependencies.

## 1. Fix the target

Ask what is under review if it is not obvious, then resolve it to a list of commits and a base:

```bash
gh stack view --json                 # the chain, when there is one
git log --oneline <base>..HEAD       # the commits under review
git diff <base>...HEAD --stat        # what the whole stack touches
```

A stack is reviewed **twice**: each PR against its own base, and the whole chain against `main`.
The two find different faults. A per-PR read finds a commit that does not stand alone. A
whole-chain read finds a convention the stack broke in the middle and never restored.

## 2. Read the source before the diff

A connector review that never looks at the dataset cannot judge the mapping. Before reading
`connector.py`, get the answers to these from the PR body, the connector `README.md`, the
`dataset.yaml` card, or by asking:

- What is one record made of — which files, and which row of which table beside them?
- Where does the join key come from? A key read out of file contents is a design fault, not a
  style one.
- How many records, and how many tasks per record? The ratio decides `add_tasks` against
  `set_task_stream`.
- What does the release state inconsistently? Every inconsistency needs a README entry with a
  decision and a state (**Handled**, **Not built**, **Open**).

## 3. What produced what you are reviewing

This review is the second half of a pair. `add-dataset-connector` builds a connector in six phases,
and every group below checks the output of one of them. Use the map to find the artifact behind a
check: if a check has no artifact, the phase that should have produced it was skipped, and that is
itself the finding.

| the phase that produced it | what this review checks |
| --- | --- |
| 0 — the card, and the source's own prose | the card is valid and matches the id; §3, the README quotes the sentences the scoping rests on |
| 1 — heads, census, record design, task count | §4 ids come from the source; §5 the task count decides `add_tasks` against `set_task_stream`; §7 the handle shape and the naming |
| 2 — the gate | §3, every ruling the user gave is a README entry with a state |
| 3 — the README | §3, one entry per inconsistency: evidence, decision, **Handled** / **Not built** / **Open**, numbers marked *(measured)* |
| 4 — the build | §1 imports and dependencies · §2 the I/O split · §3 faithfulness · §4 ids · §5 annotations and tasks · §6 errors · §7 naming · §8 tests · §9 docs |
| 5 — smoke test | §11 the checks that must have been run, and whether the build matched the numbers the plan predicted |
| the stack it shipped as | §10, each PR standing on its own |

**A connector built without the skill is still reviewable.** The map says where to look for evidence,
not that the phases must have been run. Where there is no plan and no README, say so under **Not
reviewed** and check what the code alone can show.

## 4. Run the checks

Work through `references/checks.md` in order. Eleven groups: imports and dependencies, the I/O
split, data faithfulness, ids, annotations and tasks, errors, naming and shape, tests, docs and
prose, the stack itself, and the checks that must have been run.

For each check, write down one of three verdicts and nothing else:

| Verdict | What it takes |
| --- | --- |
| **pass** | A `file:line` you opened, and one sentence saying what it shows. |
| **fail** | A `file:line`, what the convention says, and the smallest change that fixes it. |
| **not reviewed** | You could not get the evidence. Say why. |

Never write "looks fine". Cite or say you did not look.

## 5. Verify the claims you are least sure of

Before reporting, attack the two or three findings you are least certain about. A finding that
does not survive is dropped, not softened.

- Does the "convention" have two or more citations in the tree, or is it one connector's habit?
  A rule the references mark as unsettled is not a finding — `layout.md § Not a rule: docstring
  voice` is the current example. Do not report a divergence the references already call unsettled.
- Does the check actually run? `make check`, `make test`, and `make test-connectors` for a
  connector change. State which of the three you ran, and the result.
- Would the fix be a gratuitous rename of an existing function or test? Then it is not a finding.

## 6. Report

```markdown
## Verdict
<merge / merge after the fixes below / do not merge>, one sentence saying why.

## Blocking
One line each: `file:line` — what breaks, and the smallest fix.

## Non-blocking
Same shape.

## Not reviewed
What you could not check, and why.

## Checks run
`make check`, `make test`, `make test-connectors` — the result of each, or that it was not run.
```

Order findings by severity, not by file. A wrong mapping of the source outranks every style
finding in the file, however many of those there are.

## 7. Say what to run next

End by naming the next step, so nobody has to guess:

| what the review found | what to say |
| --- | --- |
| blocking findings | fix them on the branch that owns each one, then run `/review-connector-stack` again on the result |
| findings only in a lower PR of a stack | fix there and `gh stack rebase --upstack`, rather than patching the higher PR |
| a rule that looks wrong rather than the code | the finding is against the document: `add-dataset-connector`'s references or `AGENTS.md`, and the fix is a PR to it |
| an assumption nobody has ruled on | it belongs in the connector's README as **Open**, and the user decides it |
| nothing blocking | say the stack is ready to merge, and that merging is the user's call |

Never end a review with findings and no next step.
