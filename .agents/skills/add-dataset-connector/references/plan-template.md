# The plan format

Copy this shape into `docs/notes/connectors/<org>/<name>/plan.md`. Every connector's plan reads the
same, so a reader knows where to look. Keep the section order and the headings.

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

| path | kind | count | belongs to a sample |
| --- | --- | --- | --- |
| `<glob>` | signals | | yes |
| `<glob>` | labels | | yes |
| `<path>` | table | | joined, not owned |
| `<path>` | index / checksums | | no |

Every kind in this table is a node in the map below. Every file that belongs to no sample is named
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
| samples | | |
| channels | | |
| rates | | |
| labels | | |
| table row | | |

- **Total samples**: <count>, counted by <how>.
- **Anomalies**: what differs in only a few files, with the count.

### The map

```mermaid
flowchart LR
```

A dashed arrow is a part that is not written yet.

## 2. What one sample is

Real values, not placeholders.

- **sample_id**: `<prefix>-<source id>`, from <where the source states it>
- **subject_ids**: `<value>`, qualified by <what>
- **time_span**: <what fixes it>
- **start_time**: set / unset, because <reason>

### Series

| channel | spec | unit | rate | axis | source of the rate |
| --- | --- | --- | --- | --- | --- |

### Annotations

| key | value | span | scoped to which series | where it comes from |
| --- | --- | --- | --- | --- |

### Tasks

- **type**: <Task class>, because the answer is <a category / free text / a number / regions>
- **one question is**: <what a single task asks>
- **expansion**: <run-length? exact? what the boundary rule is>
- **count**: <tasks per sample> and <total> *(measured)*
- **therefore**: `add_tasks` / `set_task_stream`, because <the count>

## 3. The connector

### Download shape

A list per sample, or one handle walked at convert time. Say which and why.

### Modules

| module | responsibility | touches disk | test needs a fixture |
| --- | --- | --- | --- |
| `reader.py` | opens files, decodes nothing | yes | yes, a truncated real file |
| `tables.py` | rows to facts | no | no, literal tuples |

### Dependencies

| library | why | lazily imported in |
| --- | --- | --- |

## 4. What the build should produce

The smoke test in phase 5 checks these. A number that does not match means the plan is wrong or the
code is.

| | expected |
| --- | --- |
| samples | |
| series per sample | |
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
