# RATS40K connector

`time-ra/rats40k@1.0.0` converts the numeric observations and generated diagnostic supervision
from [Time-RA/RATs40K](https://huggingface.co/datasets/Time-RA/RATs40K) at source commit
`dc968d6a88f396215be50023c93c3928e06096df`. The source license is Apache-2.0. Access requires
accepting the Hugging Face gate; use an existing login or an externally supplied `HF_TOKEN`.
The image archives (plots of the same series) are not downloaded; `FigurePath` is kept as
provenance.

## Inventory and records

| Original file | Train | Test | Total | Null entries excluded |
| --- | ---: | ---: | ---: | ---: |
| `RATs-Uni-TSImage_Reason.json` | 30,266 | 6,034 | 36,300 | 11 test entries |
| `RATs-Multi-TSImage_Reason.json` | 2,728 | 546 | 3,274 | 0 |
| Total | 32,994 | 6,580 | 39,574 | 11 |

Each usable entry becomes one Record with one Source: a single signal, or the named multivariate
channels in their original order (kept in Source metadata). IDs contain the variant, split, and
original key. Signals are lazy float64 with unknown units and ordinal axes, since the release
gives neither units nor sampling cadence. Source file, key, variant, split, and figure path are
Record annotations; repository and revision are dataset annotations. Downloads must match the
pinned split counts and the eleven null keys in `NULL_KEYS`.

## Tasks

| Task | Type | Target | Target schema | Rationale |
| --- | --- | --- | --- | --- |
| `-detection` | `ClassificationTask` | `Label` | Normal / Anomaly | `Thought` |
| `-type` | `ClassificationTask` | `ActionID` as a string | Variant vocabulary | `Thought` |
| `-reasoning` | `AnswerTask` | `Thought` | Free text | none |

Every task keeps its record's split and receives the `source_domain` input annotation. The
category task keeps `Action` and any extra row fields (such as `Rank`) in its metadata. The
answer task duplicates `Thought` as a target because TimeNet treats a rationale as optional
supporting text, not a scored output.

**Category vocabularies.** `uni_action_vocabulary` and `multi_action_vocabulary` list the class
codes as strings, sorted numerically. Their metadata `observed_names` records every `Action` name
seen per split and ID, and `prompt_names` the one name the prompt defines per code. Observed names
are not canonical: univariate data uses 0–14, multivariate data uses
0–6 with different meanings, and in univariate test data classes 1–6 each appear under two names
(class 2 is both `Periodic Change Anomaly` and `Temporal Dependency Anomaly`). Targets are never
remapped.

**Label/Action disagreements.** 1,184 rows have a `Label` that disagrees with whether `Action` is
`Normal Sequence`: 1,117 univariate train, 66 univariate test, and one multivariate train. Both
targets are kept as published, and the count is the `label_action_disagreements` dataset
annotation. `Thought` was generated with `Action`, never `Label`, so in these rows the detection
task's rationale argues against its own target.

## Prompt

All tasks of a variant reference one dataset annotation with key `prompt` (`rats40k-uni-prompt`
or `rats40k-multi-prompt`), listed first in `input_annotations`; give its value to the model as
the instruction. `task.prompt` is unset because TimeF stores it once per task, which grew the
real build from 47 MB to 773 MB.

The text in `prompts/` is the authors' [evaluation prompt][prompts]: role, category definitions,
worked examples, and the `\boxed1{}`/`\boxed2{}` answer format. The domain sentence and the
`Observation:` line are removed, since `source_domain` and the Record supply them. Wording and
typos are otherwise unchanged, and the system message (`You are a helpful assistant.`) is not
stored. The multivariate prompt defines classes 0–7; the data uses 0–6.

The prompt asks for `Thought` and the category *name*, never `Label`, so each task reads its own
part of the answer. To score the category task, map the answered name to a code with the
vocabulary metadata `prompt_names`, the name the prompt defines for each code. For the detection
task, `Normal Sequence` means `Normal` and any other category `Anomaly`; that rule disagrees with
the stored `Label` in the 1,184 rows above.

## Data flow and validation

The release has no `refs/convert/parquet` revision, so the connector implements `BaseConnector`
directly and downloads the two pinned JSON files with `hub_snapshot`. `_json.py` streams the
nested split/key objects one entry at a time and records each entry's byte range, so signal
loaders seek straight to their row; a loader caches one decoded observation across its channels.
Tasks are re-read from the raw files on each pass, which must stay available until writing ends.

Validation rejects malformed JSON, duplicate splits or keys, invalid row fields, empty, nonnumeric,
or nonfinite observations, and duplicate or unequal channels, with file, split, and key in the
error. Only explicit null entries are skipped. Tests cover the download pin, inventory check,
stored prompts, category-name conflicts, malformed inputs, and a full build round trip with UTF-8
and CRLF offsets across small read buffers.

## Local build

After granting access and configuring Hugging Face authentication outside the repository:

```bash
uv run timenet-build build time-ra/rats40k --out /tmp/rats40k-registry
```

```python
from timenet.client import TimeNet

dataset = TimeNet(registry="/tmp/rats40k-registry").load(
    "time-ra/rats40k@1.0.0"
)
dataset.describe()
```

[prompts]: https://github.com/yyysjz1997/Time-RA/tree/200b244f75b134348fad39c9ba140d9b2f1e1279/prompts
