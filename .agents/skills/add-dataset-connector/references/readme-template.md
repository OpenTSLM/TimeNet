# The connector README format

Copy this shape into `packages/timenet-connectors/src/timenet_connectors/datasets/<org>/<name>/README.md`.
This file ships with the connector. It is the document somebody reads a year after the build.

The warning reaches the person running the build. The README reaches the person reading the data.
Both are needed.

Rules:

- **These seven sections, in this order, and no others.** A reader must be able to find a fact
  without reading the whole file, and a section nobody expects is a section nobody finds. A new kind
  of fact goes in the section that already covers it, or the standard changes for every connector.
- **Each section has a shape, given below, and not just a heading.** A fixed heading over freeform
  prose is not a standard.
- **Every section below is required, even when it is empty.** Write `None.` under a heading that
  does not apply. A section saying "None" is evidence somebody looked; a missing section is
  ambiguous, and a reader cannot tell the difference between a clean release and an unasked
  question. `scripts/check_connector_readmes.py` enforces the headings.
- **One entry per inconsistency and per assumption**, with three parts: the evidence, the decision,
  and the state.
- **The state is `Handled`, `Not built`, or `Open`.** An open entry is worth more than a tidy file,
  because it names what nobody has decided.
- **Mark any number you measured over the release yourself as `*(measured)*`**, so a reader can tell
  it from one copied off the dataset's page. A number nobody can re-measure is a claim, not evidence.
- **Quote the description sentences the design relies on**, beside `source_url`, so the next reader
  can check them rather than trust them.
- Use the `simple-english` skill on the prose. Do not run it over quoted evidence.

## Contents

The README this template produces has these sections, in this order:

1. The source of truth
2. What the description states
3. What one record holds
4. The tasks this connector builds
5. Inconsistencies and decisions
6. Warnings this build emits
7. What is not built

## What each section holds

| section | shape | says `None.` when |
| --- | --- | --- |
| The source of truth | a table: fact, source of truth, the other reading — plus one line naming *why* that source wins | nothing in the release disagrees with itself |
| What the description states | the quoted sentences, each with where it came from, each followed by one line saying what it decides | the design leans on no prose (rare — say so deliberately) |
| What one record holds | signals with units and rates, annotations with what each is scoped to, and where each came from | never; a connector that builds no records builds nothing |
| The tasks this connector builds | a table: the question, its type, its count, its scope | the connector builds no tasks |
| Inconsistencies and decisions | one `###` entry each, its state in the heading — `— **Handled**`, `**Not built**` or `**Open**` — then **Problem.**, **Decision.**, **Consequence.** in that order | the release ships none you found |
| Warnings this build emits | a table: warning, count *(measured)*, why | a build emits none |
| What is not built | prose naming the parts of the release the connector does not convert, and why | everything is converted |

Two rules hold inside every section:

- **A number you counted yourself is marked `*(measured)*`.** One copied from the dataset's page is
  not. A reader has to be able to tell which claims they can re-measure.
- **A decision states its reason.** "The table wins" is not a decision; "the table wins, because
  published work joins against it" is.

`packages/timenet-connectors/src/timenet_connectors/datasets/physionet/sleep_edfx/README.md` is the
worked example.

---

```markdown
# <Dataset name>

<One paragraph: what the release holds and what one record is.>

- **id**: `<org>/<name>`
- **source**: <source_url>
- **licence**: <licence>

## The source of truth

Where two parts of the release state the same fact differently, one wins for a named reason and the
connector keeps the other beside it. `None.` when nothing disagrees.

| fact | source of truth | the other reading |
| --- | --- | --- |

## What the description states

The prose fixes things no file header states. These sentences decide the conversion.

> <quoted sentence>
> — <where it came from>

<What it decides, in one line.>

## What one record holds

- **Series**: <count and kinds, with units and rates, and where each rate comes from>
- **Annotations**: <kinds, and which series each is scoped to>
- **Tasks**: <type, what one question asks, and how many there are>

## The tasks this connector builds

What the release ships is not a task. What it becomes is a decision, and this states it. `None.`
when the connector builds no tasks.

| the question | type | count | scope |
| --- | --- | --- | --- |

## Inconsistencies and decisions

### <Short title> — **Handled**

**Problem.** <What the release states, with the count if you measured it *(measured)*.>

**Decision.** <What the connector does, and the reason it does that.>

**Consequence.** <What somebody reading the data sees because of this.>

### <Short title> — **Open**

**Problem.** <...>

**Decision.** <What would have to be decided, and by whom.>

**Consequence.** <What a reader should not assume until it is.>

## Warnings this build emits

One row per kind, with the count and the reason. Anyone running the build can check their output
against this table.

| warning | count *(measured)* | why |
| --- | --- | --- |

## What is not built

<Parts of the release the connector does not convert, and why. A dashed arrow in the plan's map ends
up here.>
```
