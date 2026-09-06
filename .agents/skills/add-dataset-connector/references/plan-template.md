# The plan format

**Phase 0 copies this file** to `docs/notes/connectors/<org>/<name>/plan.md` before it does anything
else, ledger included, with every row `not started`. Every phase after it writes its own section and
sets its own ledger row. Keep the section order and the headings, so every connector's plan reads the
same and a reader knows where to look.

The plan is scratch and untracked. Never `git add` it.

---

```markdown
# <org>/<name> — connector plan

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
- **source_url**: <url>
- **licence**: <as confirmed by the user>
- **archive**: <what is downloaded, and how big>

### What the description states

Quote the sentences the design relies on, each with where it came from. These move to the README in
phase 3.

> <quoted sentence>

- What it decides: <e.g. stage annotations are scoped to Fpz-Cz and Pz-Oz>

## 1. What ships

### The inventory

| path | kind | count | belongs to a record |
| --- | --- | --- | --- |
| `<glob>` | signals | | yes |
| `<glob>` | labels | | yes |
| `<path>` | table | | joined, not owned |
| `<path>` | index / checksums | | no |

Every kind in this table is a node in the map below. Every file that belongs to no record is named
here once, so nobody looks for it again.

### Heads

One per kind. Paste the output of `heads.py`, unedited.

#### signals — `<one file of this kind>`

```text
<head output>
```

### The census

*(measured)* over the whole release.

| | shape A | shape B |
| --- | --- | --- |
| records | | |
| signals | | |
| rates | | |
| labels | | |
| table row | | |

- **Total records**: <count>, counted by <how>.
- **Anomalies**: what differs in only a few files, with the count.

### The map

```mermaid
flowchart LR
```

A dashed arrow is a part that is not written yet.

## 2. The model

This is the section the user argues with, so every line carries the evidence that produced it. Cite
the head or the census row above — "signals head, line 4", "census: 117 distinct ranges" — not a
belief. A line with no evidence is an assumption, and belongs in section 5 instead.

Real values, not placeholders.

- **record_id**: `<prefix>-<source id>`, from <where the source states it>
- **subject_ids**: `<value>`, qualified by <what>
- **time_span**: <what fixes it>
- **start_time**: set / unset, because <reason>

### Signals

| signal | spec | unit | rate | axis | evidence |
| --- | --- | --- | --- | --- | --- |

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
- **count**: <tasks per record> and <total> *(measured)*
- **therefore**: `add_tasks` / `set_task_stream`, because <the count>

## 3. The connector

### Download shape

A list per record, or one handle walked at convert time. Say which and why.

### Modules

| module | responsibility | touches disk | test needs a fixture |
| --- | --- | --- | --- |
| `tables.py` | rows to facts | no | no, literal tuples |
| `metadata.py` | facts to annotations | no | no, literal values |
| `specs.py` | the signal-name to spec map | no | no |

The half that **opens** a file is a base, not a module here — see `layout.md`. A test that needs a
file writes a synthetic one into `tmp_path`; no real bytes are checked in.

### Dependencies

| library | why | lazily imported in |
| --- | --- | --- |

## 4. What the build should produce

The smoke test in phase 5 checks these. A number that does not match means the plan is wrong or the
code is.

| | expected |
| --- | --- |
| records | |
| series per record | |
| annotations | |
| tasks | |
| warnings, by kind and reason | |

## 5. Assumptions and open questions

One numbered entry each. These are what the gate is for, and they move to the README once ruled on.

### A1. <the question in one line>

- **What the source states**: <evidence, with the file or the sentence it came from>
- **What it does not state**: <the gap>
- **Proposed**: <what you would do>
- **Ruling**: <filled in at the gate>
```
