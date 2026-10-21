# RATS40K connector

`time-ra/rats40k@1.0.0` converts the numeric observations and generated diagnostic supervision
from [Time-RA/RATs40K](https://huggingface.co/datasets/Time-RA/RATs40K) at source commit
`dc968d6a88f396215be50023c93c3928e06096df`. The source license is Apache-2.0. Access requires
accepting the Hugging Face gate; use an existing login or an externally supplied `HF_TOKEN`.

## Inventory and records

| Original file | Train | Test | Total | Null entries excluded |
| --- | ---: | ---: | ---: | ---: |
| `RATs-Uni-TSImage_Reason.json` | 30,266 | 6,034 | 36,300 | 11 test entries |
| `RATs-Multi-TSImage_Reason.json` | 2,728 | 546 | 3,274 | 0 |
| Total | 32,994 | 6,580 | 39,574 | 11 |

Each usable entry becomes one Record with one Source: a single signal, or the named multivariate
channels, whose original order is the Source's `channel_order` metadata, since TimeF does not keep
signal order on read. IDs contain the variant, split, and original key. Signals are lazy float64
with unknown units and ordinal axes, since the release gives neither units nor sampling cadence.
The original key (`source_key`), variant (`source_variant`), and figure path (`figure_path`) are
`Record.metadata`; the domain (`source_domain`) is a Record annotation, since the task uses it as an
input annotation. The source repository and revision, and each variant's category vocabulary,
are dataset annotations. Downloads must match the pinned split counts and the eleven null keys in `NULL_KEYS`.

## Task

Each record has one `ClassificationTask`, `rats40k-<variant>-<split>-<key>-category`:

| Field | Value |
| --- | --- |
| `prompt` | The variant's evaluation prompt (see Prompt) |
| `input_annotations` | `source_domain` |
| `targets` | The published category name, `Action` |
| `target_schema` | `rats40k-uni-category-vocabulary` or `rats40k-multi-category-vocabulary` |
| `rationale` | The published explanation, `Thought` |
| `metadata` | `ActionID`, `Label`, `target_in_vocabulary`, and `Rank` on test rows |

This mirrors the authors, who ask one question per row and read the category and the explanation
from the same answer. The target is the category *name* because that is what the prompt asks the
model to answer, so an answer is compared with the target directly, with no code mapping.
`ActionID` and `Label` stay in metadata so the authors' code-based and detection metrics remain
reproducible.

**Category vocabularies.** Each vocabulary lists the category names its prompt defines, in the
authors' code order: 15 for univariate (0–14) and 8 for multivariate (0–7). The data does not
always follow the prompt:

- 3,790 of the 6,034 univariate test rows (63%) carry a *multivariate* category name, so their
  target is outside the univariate vocabulary and a model following the prompt cannot match it.
  The authors' univariate labelling script imports the multivariate prompt module, the likely
  cause; the `Thought` of these rows reasons with the multivariate categories. The rows are kept
  as published, and their task metadata sets `target_in_vocabulary` to `false`, so a score can
  exclude them with one filter.
- Multivariate `ActionID` uses 0–6, matching the six anomaly types on the dataset card, while the
  prompt numbers its categories 0–7. Its class 4 (`Local Structural Jump Anomaly`) never occurs,
  and `Joint Space`, `Principal Component Space`, and `Collinearity Shift Anomaly` are stored as
  4, 5, and 6 where the prompt numbers them 5, 6, and 7 (1,923 of the 3,274 rows). The name target
  avoids this; it affects only comparisons by `ActionID`.

**Label/Action disagreements.** 1,184 rows have a `Label` that disagrees with whether `Action` is
`Normal Sequence`: 1,117 univariate train, 66 univariate test, and one multivariate train. Both are
kept as published. `Label` is the source benchmark's label: the authors' labelling script passes
it to the model and parses `Thought` and `Action` from the same response, so `Thought` argues for
`Action`.

## Prompt

`prompts/uni.txt` and `prompts/multi.txt` are the authors' [evaluation prompts][prompts], the ones
every evaluation script uses, for zero-shot and fine-tuned models alike. Each states the role, the
category definitions, worked examples, and the `\boxed1{}`/`\boxed2{}` answer format, and asks for
a `Thought` and a category name. Every task carries its variant's prompt as `task.prompt`. The
sources are `prompt_llama_anoclf_reason.py` (univariate) and `prompt_llama_multi_ano.py`
(multivariate) in that directory.
TimeF stores the prompt once per task, and DuckDB does not compress strings this long, so the
6 KB prompts make the full build about 277 MB; that cost is accepted.

The domain sentence and the `Observation:` line are removed, since `source_domain` and the Record
supply them; a consumer restores them when building the request. Wording and typos are otherwise
unchanged.
The system message (`You are a helpful assistant.`) is not stored. The authors' text ends 15
univariate and 2 multivariate lines with trailing spaces, which the repository's
`trailing-whitespace` hook strips from the stored files.

## How the connector follows the authors

The [dataset card][card] frames RATS40K as a multi-objective diagnosis: anomaly detection,
fine-grained classification, and diagnostic reasoning. The authors' [evaluation][eval] asks one
question per row and derives all three scores from the answer: binary F1 against `Label`, macro F1
against `ActionID`, and text similarity against `Thought`. The connector mirrors that setup:

| Authors' setup | Connector choice |
| --- | --- |
| One question per row, three scores from the answer | One task per row; metadata keeps the rest |
| Evaluation prompt per variant | That prompt as `task.prompt` |
| Domain and series filled into the prompt | `source_domain` input annotation and the Record |
| Model answers a category name and a `Thought` | Target `Action`, rationale `Thought` |
| `Label` and `ActionID` scored from the answer | Both kept in task metadata |
| `TSAD_train` and `TSAD_test` | The same two splits, with no validation split |
| Series rendered as figures for multimodal models | Numeric series only; `FigurePath` as provenance |
| Gated release | `credentialed` access, pinned commit and inventory |

Within TimeNet, each source row is one Record carrying its series once. The category vocabularies
are dataset annotations that the task names through `target_schema`, which is how TimeNet attaches
class lists to classification targets. Source quirks are kept in the data as published and
documented below instead of being silently fixed.

## Data flow and validation

The release has no `refs/convert/parquet` revision, so the connector subclasses
`BaseHubSnapshotConnector`, which downloads the two pinned JSON files. `_json.py` streams the
nested split/key objects one entry at a time and records each entry's byte range, so signal
loaders seek straight to their row; a loader caches one decoded observation across its channels.
Tasks are re-read from the raw files on each pass, which must stay available until writing ends.

Validation rejects malformed JSON, duplicate splits or keys, invalid row fields, empty, nonnumeric,
or nonfinite observations, and duplicate or unequal channels, with file, split, and key in the
error. Only explicit null entries are skipped. Tests cover the download pin, inventory check,
stored prompts, the authors' category order, published names as targets, malformed inputs, and a
conversion whose lazy reads follow UTF-8 and CRLF offsets across small read buffers.

## Known source issues

The connector keeps the release as the authors publish and evaluate it. These issues are in the
source data and are documented here, not corrected:

1. **Multivariate class codes.** Three multivariate categories have an `ActionID` one below the
   prompt's numbering, which the authors' evaluation uses (see the category vocabularies above).
2. **Univariate test rows with multivariate names.** 3,790 univariate test rows (63%, nearly all
   `Anomaly`) carry a multivariate category name, and their `Thought` often reasons about
   variables or channels. The first model in `Rank` predicts this: no row where `gpt-4o` ranks
   first is affected, against 81–89% where `gemini`, `llama-70b`, or `deepseek-r1` ranks first.
   Their targets are outside the univariate vocabulary, and `target_in_vocabulary` is `false` on
   their tasks. Any test score should state whether these rows are kept.
3. **Label/Action disagreements.** 1,184 rows, described above. In train, 758 of the 1,117 are
   `Label` Normal with an anomalous `Action`; in univariate test, 45 of the 66 are the reverse.
4. **Repeated series.** 43 univariate and 10 multivariate test series also appear, value for
   value, in train, and 182 univariate and 18 multivariate train series appear more than once.
5. **Rationale names the category.** `Thought` names its `Action` in 41% of univariate and 28% of
   multivariate rows.
6. **`Rank` is parsed model text.** Multivariate entries always list the four generators.
   Univariate entries list one to four, with occasional malformed names such as `llaama-70b`.

## Open questions

1. **Domain tags.** `dataset.yaml` has no tag for the synthetic `Artificial` series or the
   `Twitter-Ticker_Symbol` counts.

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

[card]: https://huggingface.co/datasets/Time-RA/RATs40K
[eval]: https://github.com/yyysjz1997/Time-RA/blob/200b244f75b134348fad39c9ba140d9b2f1e1279/eval2file.py
[prompts]: https://github.com/yyysjz1997/Time-RA/tree/200b244f75b134348fad39c9ba140d9b2f1e1279/prompts
