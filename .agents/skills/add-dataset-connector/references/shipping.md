# How a connector ships

Reference for phase 5 of the `add-dataset-connector` skill, and for every commit before it. A
connector ships as a stack of small pull requests, not as one commit. `AGENTS.md § Stacked PRs`
holds the `gh stack` mechanics. This file holds only what is specific to a connector.

## Contents

- [The size of one pull request](#the-size-of-one-pull-request)
- [The seam](#the-seam)
- [Each pull request stands alone](#each-pull-request-stands-alone)
- [What enters the stack](#what-enters-the-stack)
- [Naming a commit and a pull request](#naming-a-commit-and-a-pull-request)

## The size of one pull request

**Target 100 to 500 changed lines per pull request.** One run shipped two connectors as PRs of 1290
and 1420 lines, and neither could be reviewed. "As small as possible" is not actionable without a
number and a seam.

## The seam

**The seam for a connector is this, bottom to top.** Each step is reviewable without the ones above
it:

1. **The card** — `dataset.yaml` and the org `__init__.py`. It is what a reviewer checks the model
   against.
2. **The pure modules with their tests** — `tables.py`, `specs.py`, `keys.py`. They take values and
   give values, so their tests need no fixture and the review needs no dataset.
3. **`connector.py` with its tests** — the loop, the download shape, and the synthetic release the
   test writes.
4. **The `README.md`** — the assumptions and the inconsistencies, last, because it describes what the
   PRs below it built.

Put a mid-stack change on the branch that owns it and run `gh stack rebase --upstack`. Do not fold it
into a higher branch.

## Each pull request stands alone

**This skill's own phases create the forward-reference trap, so watch for it.** Each PR must stand
alone: no comment, docstring or README line that a later PR deletes, and no forward-looking chatter
such as "the next PR adds the tasks". The plan names a README that has not landed yet, and a module
docstring easily names a `connector.py` two branches up. Write each file as though the branch it sits
on is the last one.

## What enters the stack

**One rule decides what enters the stack: a file that `download` or `convert` imports and calls.**
Everything else you wrote to build the connector stays out — `heads.py`, the survey script, the
plan. Those live under `docs/notes/`, which is never `git add`ed.

## Naming a commit and a pull request

**Say what the dataset is, in the commit subject and the PR title.** `feat(slip): read the SLIP
pretraining corpus` says nothing to a reader who has not met the dataset. Name the release and say
that this adds a connector.
