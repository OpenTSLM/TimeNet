# `<org>/<name>` — connector plan

<!--
This file IS the template, and phase 0 copies it whole to
`docs/notes/connectors/<org>/<name>/plan.md`. Keep the section order and the headings, so every
connector's plan reads the same and a reader knows where to look. Delete this comment and every
`<placeholder>` as you fill them in.

The plan is scratch and untracked. Never `git add` it.
-->

| phase | state | artifact |
| --- | --- | --- |
| 0 card and description | not started | |
| 1 heads, census, map | not started | |
| 2 gate | not started | |
| 3 assumptions to README | not started | |
| 4 build | not started | |
| 5 smoke and review | not started | |

States are `not started`, `in progress`, `done`, and for the gate `approved <date>`.

## 0. The source

- **id**: `<org>/<name>`
- **source_url**: `<url>`
- **licence**: <as confirmed by the user>
- **one dataset or several**: <one, or the list. A release that bundles a corpus with its benchmarks
  gets one plan and one card per dataset, and this line says which this plan is>

### Where the code is built

- **built on**: <the base ref this connector branches from>
- **why not `main`**: <the API or the fix this branch needs, or "no reason: it is `main`">
- **skill ref**: <the branch that holds `add-dataset-connector`>. Read every reference with
  `git show <skill ref>:.agents/skills/add-dataset-connector/references/<file>`, because the
  connector branch does not carry the skill.

### The fetch

- **command**: <the exact command, such as `git clone git@hf.co:datasets/<org>/<name>`>
- **size on disk**: <bytes, and how long the fetch took>
- **where it lives**: `~/.cache/timenet/cache/<dataset_id>/`, outside the working tree
- **what the fetch needs**: <such as `git lfs`, a token, `wget -r`>

### What the description states

Quote the sentences the design relies on, each with where it came from. These move to the README in
phase 3.

> <quoted sentence>

- What it decides: <e.g. stage annotations are scoped to Fpz-Cz and Pz-Oz>

### What the metadata files claim

A file that describes the release rather than holding it: a `dataset_info.json`, a manifest, a data
dictionary, a `state.json`. Read these before you open anything, because what they claim is what the
heads then check. Phase 1 fills in the last column.

| file | what it claims | what the files show |
| --- | --- | --- |
| `<path>` | <the declared schema, the shard list, the licence> | <checked in phase 1> |

## 1. What ships

### The inventory

| path | kind | count | belongs to a sample |
| --- | --- | --- | --- |
| `<glob>` | signals | | yes |
| `<glob>` | labels | | yes |
| `<path>` | table | | joined, not owned |
| `<path>` | index / checksums | | no |
| `<path>` | describes the release | | no |

Every kind in this table is a node in the map below. Every file that belongs to no sample is named
here once, so nobody looks for it again.

### Heads

One per kind, named for the kind the release ships. Give the command first and the unedited output
after it, so the user can run it again.

#### signals — `<one file of this kind>`

```bash
uv run python docs/notes/connectors/<org>/<name>/heads.py <path>
```

```text
<head output>
```

### The census

*(measured)* over the whole release.

| | shape A | shape B |
| --- | --- | --- |
| samples | | |
| channels | | |
| rates | | |
| dtypes | | |
| labels | | |
| table row | | |

- **Total samples**: `<count>`, counted by `<how>`.
- **Anomalies**: what differs in only a few files, with the count.
- **Free text**: whether every value terminates, and whether the lengths have a cliff.
- **Against the declaration**: every file checked against the declared schema, and every column the
  declaration got wrong.

### The map — which module does what

```mermaid
flowchart LR
```

A dashed arrow is a part that is not written yet.

### The sample model — where every fact comes from

Every object TimeF will hold, and the exact part of the source that states it. This is the diagram
the user accepts or rejects, so name the part of a file rather than the file, and label every edge
with what it carries.

```mermaid
flowchart LR
```

Check it before you show it:

- every TimeF node has an inbound edge — one without is invented
- every source node has an outbound edge, or is in the inventory as belonging to no sample
- the edge that scopes an annotation says whether it came from prose or a header
- every task traces to an annotation or to the sentence stating the question

## 2. The model

This is the section the user argues with, so every line carries the evidence that produced it. Cite
the head or the census row above — "signals head, line 4", "census: 117 distinct ranges" — not a
belief. A line with no evidence is an assumption, and belongs in section 5 instead.

Real values, not placeholders.

- **sample_id**: `<prefix>-<source id>`, from <where the source states it>. Where the source states
  none, name the position it is built from and say that a re-release invalidates it.
- **subject_ids**: `<value>`, qualified by `<what>`
- **time_span**: <what fixes it>
- **start_time**: set / unset, because `<reason>`

### Signals

| signal | spec | unit | rate | axis | source dtype | spec dtype | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |

`TimeSeriesSpec.dtype` defaults to `float32`. A source that stores `double` fails when the writer
first calls a loader, which is long after `convert` returned, so the two dtype columns must agree
here. Take the source dtype from the head.

### Annotations

| key | value | span | scoped to which signals | evidence |
| --- | --- | --- | --- | --- |

The scope is the line most often wrong and least often questioned. It comes from the description's
prose, never from a header, so quote the sentence.

### Tasks

- **type**: <Task class>, because the answer is <a category / free text / a number / regions>
- **evidence**: <what in the source says this is the question being asked>
- **one question is**: <what a single task asks>
- **expansion**: <run-length? exact? what the boundary rule is>
- **count**: <tasks per sample> and `<total>`, derived from the census rows `<which>`
- **therefore**: `add_tasks` / `set_task_stream`, because <the count>

### How the answer is stored

Name every facility this connector uses, from
`connector-anatomy.md § Where an answer, an annotation and a task can live`. A facility you did not
know about is not a choice you made.

| the choice | used here | why |
| --- | --- | --- |
| `add_task` / `add_tasks` / `set_task_stream` | | |
| `sample.add_annotation` / `dataset.register_annotations` | | |
| `Task.target` / `Task.target_annotation_ids` | | |
| `Task.input_annotation_ids`, `Task.from_tasks` | | |

## 3. The connector

### Download shape

A list per sample, or one handle walked at convert time. Say which and why.
`BaseHuggingFaceConnector` gives the list shape only, so say whether this release fits in it.

### What already exists

List what the tree already has for each part, before naming a module. Reuse is judged against the
whole tree, which is the thing an author writing one connector cannot see.

| this connector needs | what already does it | reused | if not, why not |
| --- | --- | --- | --- |
| fetching the archive | `download.ensure_archive`, `find_dir_containing` | | |
| opening the container | `bases/<...>` | | |
| the connector base | `BaseHuggingFaceConnector` / `BasePhysioNetConnector` | | |
| <a modelling question> | <the connector that already answered it> | | |

### Modules

| module | responsibility | touches disk | test needs a fixture |
| --- | --- | --- | --- |
| `tables.py` | rows to facts | no | no, literal tuples |
| `metadata.py` | facts to annotations | no | no, literal values |
| `specs.py` | the signal-name to spec map | no | no |

The half that **opens** a file is a base, not a module here — see `layout.md`. A test that needs a
file writes a synthetic one into `tmp_path`; no real bytes are checked in. `heads.py` is not a module
of the connector: it stays in this folder, beside this plan.

### Dependencies

| library | why | declared in |
| --- | --- | --- |

## 4. What the build should produce

The smoke test in phase 5 checks these. A number that does not match means the plan is wrong or the
code is.

**A number that can be derived from the census is derived from it, and the derivation is shown
here.** A prediction that contradicts a table two sections above it is a mistake nobody has to make.

| | expected | derived from | measured (phase 5) |
| --- | --- | --- | --- |
| samples | | | |
| series per sample | | | |
| annotations | | | |
| tasks | | | |
| warnings, by kind and reason | | | |

## 5. Decisions and open questions

Two headings, and the split matters. A gate holding ten questions trains the user to approve the
list rather than argue with it, which is the failure the gate exists to prevent.

### Decisions taken

Everything a repo rule or an existing connector already answers. Take the decision, cite the rule,
and record it here. The user can overturn any of these on reading them.

| # | the question | what was decided | the rule that decided it |
| --- | --- | --- | --- |
| D1 | | | `fidelity.md § <section>` |

### Open questions

A question belongs here only when all three hold: the source is silent, **and** no repo rule covers
it, **and** two defensible answers lead to different samples, tasks or counts. Anything else is a
decision, above.

#### A1. <the question in one line>

- **What the source states**: <evidence, with the file or the sentence it came from>
- **What it does not state**: <the gap>
- **The two answers**: <what each one changes about the samples, the tasks or the counts>
- **Proposed**: <what you would do>
- **Ruling**: <filled in at the gate>
