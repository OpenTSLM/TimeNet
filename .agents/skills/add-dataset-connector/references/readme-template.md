# The connector README format

Copy this shape into `packages/timenet-connectors/src/timenet_connectors/datasets/<org>/<name>/README.md`.
This file ships with the connector. It is the document somebody reads a year after the build.

The warning reaches the person running the build. The README reaches the person reading the data.
Both are needed.

Rules:

- **One entry per inconsistency and per assumption**, with three parts: the evidence, the decision,
  and the state.
- **The state is `Handled`, `Not built`, or `Open`.** An open entry is worth more than a tidy file,
  because it names what nobody has decided.
- **Mark any number you measured over the release yourself as `*(measured)*`**, so a reader can tell
  it from one copied off the dataset's page. A number nobody can re-measure is a claim, not evidence.
- **Quote the description sentences the design relies on**, beside `source_url`, so the next reader
  can check them rather than trust them.
- Use the `simple-english` skill on the prose. Do not run it over quoted evidence.

`packages/timenet-connectors/src/timenet_connectors/datasets/physionet/sleep_edfx/README.md` is the
worked example.

---

```markdown
# <Dataset name>

<One paragraph: what the release holds and what one record is.>

- **id**: `<org>/<name>`
- **source**: <source_url>
- **licence**: <licence>

## What the description states

The prose fixes things no file header states. These sentences decide the conversion.

> <quoted sentence>
> — <where it came from>

<What it decides, in one line.>

## What one record holds

- **Series**: <count and kinds, with units and rates, and where each rate comes from>
- **Annotations**: <kinds, and which series each is scoped to>
- **Tasks**: <type, what one question asks, and how many there are>

## Inconsistencies and decisions

### <Short title>

- **Evidence**: <what the source states, with the count if you measured it *(measured)*>
- **Decision**: <what the connector does>
- **State**: Handled

### <Short title>

- **Evidence**: <...>
- **Decision**: <what would have to be decided>
- **State**: Open

## Warnings this build emits

One row per kind, with the count and the reason. Anyone running the build can check their output
against this table.

| warning | count *(measured)* | why |
| --- | --- | --- |

## What is not built

<Parts of the release the connector does not convert, and why. A dashed arrow in the plan's map ends
up here.>
```
