# RATS40K connector

`time-ra/rats40k@1.0.0` converts the original numeric observations and generated diagnostic
supervision from [Time-RA/RATs40K](https://huggingface.co/datasets/Time-RA/RATs40K). All downloads
use source commit `dc968d6a88f396215be50023c93c3928e06096df`. The source license is Apache-2.0;
access requires accepting the Hugging Face gate with an account. Use an existing Hugging Face
login or an externally supplied `HF_TOKEN`.

The visual data is intentionally omitted: the images are plots of the time-series observations.
The connector retains the numeric observations, source context, channel names, and original
`FigurePath`, but does not download or store the image archives. This supports the authors'
numeric-series setting; the original image-based experiments require the upstream plots.

## Inventory and mapping

| Original file | Train records | Test records | Total records | Null entries excluded |
| --- | ---: | ---: | ---: | ---: |
| `RATs-Uni-TSImage_Reason.json` | 30,266 | 6,034 | 36,300 | 11 test entries |
| `RATs-Multi-TSImage_Reason.json` | 2,728 | 546 | 3,274 | 0 |
| Total | 32,994 | 6,580 | 39,574 | 11 |

Every usable entry becomes an independent Record, with one Source containing either a single
signal or the published named multivariate channels. IDs contain the variant, original split, and
original entry key. Original channel order is retained in Source metadata because hierarchy
traversal can sort signals by name. Equal values do not imply shared recording identities.

Signals use lazy float64 values, unknown units (`unit_value=None`), and ordinal axes. The release
does not establish physical units or sampling cadence. Preserve the published numbers without
rescaling. Float64 preserves decoded JSON values without a float32 downcast. Each Record has three
independent tasks:

| Task | Target | Target schema |
| --- | --- | --- |
| Anomaly detection | Original `Label` string | Normal / Anomaly |
| Anomaly category | `ActionID` written as a string | Variant-specific vocabulary |
| Diagnostic answer | Complete original `Thought` | Free text |

Detection and category tasks use `ClassificationTask`; diagnostic answers use `AnswerTask`.

Each task references the registered Record and retains its train/test split. The original `Source`
is supplied through the `source_domain` input annotation. `ActionID` is a category class label,
not an example identifier. The category task retains the original `Action` name in its metadata.
Both classification tasks carry the complete `Thought` as their `rationale`, since it explains
why a series is or is not anomalous. A rationale is optional supporting text, so the diagnostic
answer task also keeps `Thought` as a separate target that can be scored on its own.

The `Thought` was generated together with `Action`, never with `Label`. In the 1,184 rows where
the two disagree (see below), the detection task's rationale argues for the category and against
its own `Label` target. For example, a row labelled `Anomaly` with `Normal Sequence` describes no
abnormal changes. Filter on `label_action_disagreements` provenance or compare the two targets
when that matters.

Every task of a variant shares one prompt. It is a dataset annotation with key `prompt` (ids
`rats40k-uni-prompt` and `rats40k-multi-prompt`), and each task lists it first in
`input_annotations`, before `source_domain`. `task.prompt` is unset: TimeF stores `Task.prompt`
once per task, which grew the build from 47 MB to 773 MB, while a shared annotation is stored once
and referenced by every task. Give the annotation's value to the model as the instruction.

Each prompt is the authors' [evaluation prompt][prompts] for univariate or multivariate data,
stored in `prompts/`:
role, category definitions, worked examples, and the `\boxed1{}`/`\boxed2{}` answer instruction.
Two per-row parts are removed because TimeNet supplies them separately: the sentence naming the
domain (the `source_domain` input annotation) and the `Observation: ...` line (the Record's
Signals). Trailing whitespace is trimmed; the wording, including the original typos, is kept.
The authors' system message, `You are a helpful assistant.`, is not stored.

The prompt asks for both `Thought` and the category *name*. It does not ask for `Label`, and the
category target is the `ActionID` code, so a consumer evaluating one task should read the matching
part of the answer. The univariate prompt defines classes 0–14. The multivariate prompt defines
0–7, while the data uses 0–6 with names that do not always match. Evaluation metrics and
classification averaging are outside TimeNet's scope.

Labels, category labels/names, and reasoning are kept on tasks, not input Records. Source file,
key, variant, split, and figure path remain Record provenance. Repository and source revision are
dataset annotations. Optional `Rank` and any additional row fields are preserved in the category
task's metadata.

Dataset annotations `uni_action_vocabulary` and `multi_action_vocabulary` are referenced by the
corresponding category tasks' `target_schema`. Each annotation's value is the list of string class
codes, sorted numerically before conversion (for example, `"2"` before `"10"`). Targets use those
same strings. Its JSON metadata contains `observed_names`, label/name associations by split
represented as `{split, action_id, actions}` entries, with original integer IDs and every observed
name retained. There is no universal canonical name mapping: univariate training uses classes
0–14, while multivariate data uses 0–6 with different meanings for classes 1–6. In univariate test
data, each of classes 1–6 occurs with two names. For example, class 2 occurs as both `Periodic
Change Anomaly` and `Temporal Dependency Anomaly`. Preserve these associations without remapping
the targets.

The source contains **1,184 disagreements** between `Label` and whether `Action` is `Normal
Sequence`, also independently recomputed from the built dataset:

| Variant | Train disagreements | Test disagreements | Total |
| --- | ---: | ---: | ---: |
| Univariate | 1,117 | 66 | 1,183 |
| Multivariate | 1 | 0 | 1 |
| Total | 1,118 | 66 | 1,184 |

Preserve these independent targets without relabeling. Dataset annotations expose their count and
the number of skipped null entries. Nulls are not usable observations; downloaded files must match
both pinned split counts and the eleven exact null keys listed in `NULL_KEYS`. Original action
names and IDs are preserved even where the published taxonomy and code disagree.

## Data flow and validation

The connector implements `BaseConnector` directly because this release has no
`refs/convert/parquet` revision for the row-based Hugging Face base to read. It reuses
`hub_snapshot` for pinned, selected JSON downloads and existing authentication.

`_json.py` handles the source's nested split/key objects, which are not JSONL. It decodes one
entry at a time and retains its UTF-8 byte range so a signal loader can seek directly to that row.
Using `json.load()` instead would retain the entire decoded file. Each loader caches at most one
decoded observation across its channels. Diagnostic text, optional ranking provenance, and Tasks
are re-iterated from the local files rather than retained across the corpus. The
Record/Source/Signal hierarchy and IDs remain in memory; total memory still grows with record
count. Raw files must remain available until writing completes. Every task-stream pass rereads
both JSON files in full, including observations. This is acceptable for roughly 40K rows, but
repeated iteration repeats that work.

Validation rejects malformed JSON, duplicate split or entry keys, invalid row fields, empty or
nonnumeric observations, duplicate channel names, unequal channel lengths, and nonfinite numbers.
Numeric decoding uses JSON, never executable expressions. Row errors identify the file, split, and
original key. Only explicit null entries are excluded; other malformed rows fail conversion.

The small synthetic fixtures include a deliberate label/category disagreement. One pipeline round
trip verifies values, all three targets, separate category vocabularies, observed category names,
source context, provenance, and splits after raw-cache cleanup. It also exercises UTF-8 and CRLF
byte offsets across small buffer boundaries and writing after a task-stream preview. Focused tests
cover category-name conflicts, pinned download selection, and the pinned inventory check. Others
reject malformed split structure (duplicate entry keys, unknown or missing splits) and malformed
rows with source context: duplicate or unequal channels, nonfinite values, a boolean `ActionID`,
an unknown `Label`, and non-object rows. These tests cover the main mapping, build behavior, and
likely upstream drift, rather than every validation branch.

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
