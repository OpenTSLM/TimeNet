# The skills in this repository

A skill is a procedure an agent loads when it needs it, rather than a rule it carries all the time.
The rules live in `AGENTS.md`, which is read every session and stays short for that reason. Anything
that only matters while doing one particular job lives here instead, and is loaded on demand
(see `docs/adrs/ADR-0008-agents-md-as-the-single-agent-contract.md`).

**This file routes. It states no rule of its own**, so there is nothing here to keep in step with
anything else. Where it names a subject, the file it points at is the authority.

## Which one to reach for

| you want to | invoke | it needs from you |
| --- | --- | --- |
| find, download or load a dataset | `timenet-datasets` | nothing |
| convert an external dataset into TimeF | `add-dataset-connector` | the dataset card confirmed, and a ruling at its gate |
| check a connector before it merges | `review-connector-stack` | nothing, though it reads better with the connector's plan and README |

Invoke one by name, or with `/<name>`. Each `SKILL.md` opens with a contents list.

## The two connector skills are a pair

`add-dataset-connector` builds a connector in six phases with one gate.
`review-connector-stack` checks the result. They are written against each other: the review's check
groups map onto the phases whose output they read, and it **cites** the author skill's references
rather than copying them, so an author and a reviewer cannot end up following two versions of one
rule.

That means the review skill depends on the author skill being present. Its `SKILL.md` says what to
do when it is not.

## Where a rule lives

| a rule about | lives in |
| --- | --- |
| the repository: errors, lint, type checking, commits, stacked PRs | `AGENTS.md` |
| what a connector may do to its source | `add-dataset-connector/references/fidelity.md` |
| how a connector's modules divide | `add-dataset-connector/references/layout.md` |
| reading a release before designing against it | `add-dataset-connector/references/discovery.md` |
| the `BaseConnector` API surface | `add-dataset-connector/references/connector-anatomy.md` |
| what a reviewer checks | `review-connector-stack/references/checks.md` |

One rule, one home. A rule that appears in two of these is a bug in the documents, and the fix is to
delete one copy and cite the other.
