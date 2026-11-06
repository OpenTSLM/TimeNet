# RATs40K

`time-ra/rats40k` converts the [TIME-RA benchmark][paper]: univariate and multivariate anomaly
series with published category targets, binary labels and model-generated explanations.

The connector pins the [Hugging Face release][card] to
`dc968d6a88f396215be50023c93c3928e06096df`. Access requires accepting the dataset gate and
configuring Hugging Face authentication outside the repository.

The release declares Apache-2.0. Its upstream datasets and generated outputs may have additional
terms; in particular, Yahoo S5 uses Yahoo Webscope research-use terms. The release license does
not replace those terms.

## Source and conversion

The two JSON files contain `TSAD_train` and `TSAD_test` objects keyed by decimal row IDs.
`SourceInventory` checks usable row counts and the exact null-entry keys for each split.

| File | Train rows | Test rows | Null entries skipped |
| --- | ---: | ---: | ---: |
| `RATs-Uni-TSImage_Reason.json` | 30,266 | 6,034 | 11 test entries |
| `RATs-Multi-TSImage_Reason.json` | 2,728 | 546 | 0 |

Each usable row becomes one Record and one classification task. Only explicit null entries are
skipped; malformed rows fail with `TimeNetConnectorError` and source context.

- **Signals:** published values as float64, with ordinal axes and unknown units. The release gives
  no sample rates or timestamps. Multivariate channel names and order are preserved; the order is
  stored in Source metadata as `channel_order`.
- **Identity:** Record metadata stores `variant`, `split` and `key`. Together they identify a row.
  The release provides no subject or upstream series identity.
- **Target and rationale:** the published `Action` category and `Thought` explanation.
- **Model inputs:** the signals, evaluation prompt and `source_domain` annotation. The binary
  `Label` and `ActionID` are task annotations, never input annotations.
- **Provenance:** task metadata retains `figure_path`, optional `rank`, and
  `target_in_vocabulary`. Dataset annotations retain the source repository and revision.
- **Splits:** source train and test are preserved; there is no validation split.

Each variant's label set starts with the categories in its evaluation prompt, in published code
order. Other published targets follow in sorted order and have no category code. No target is
corrected or dropped because it disagrees with the prompt vocabulary.

## Prompts and evaluation

The files in `prompts/` preserve the authors' [evaluation templates][prompts]. Their domain clause
and observation line are omitted from `task.prompt`, because input annotations and Records supply
those inputs. The complete templates remain in the dataset's `uni_request_template` and
`multi_request_template` annotations, with `{our_source}` and `{our_observation}` placeholders.
Use `channel_order` when reconstructing a multivariate observation.

`authors_answer(task)` produces the authors' boxed `Thought` and `Action` answer format. Their
[training script][sft] uses a different, shorter prompt, so the evaluation prompt and this answer
do not reproduce their training pairs.

The authors' [evaluation][eval] compares predicted category codes with `action_id` and binary
predictions with `label`. It excludes predictions outside the prompt vocabulary. For comparisons,
state how out-of-vocabulary targets and conflicting duplicate inputs were handled; the connector
applies neither filter.

## Source limitations

- Many univariate test targets are multivariate category names. `target_in_vocabulary` identifies
  targets outside the relevant prompt vocabulary.
- Some `ActionID` values disagree with the prompt numbering, and some binary labels disagree with
  whether the category is normal. All are retained.
- Train and test contain overlapping windows and some identical series. Identical inputs can also
  have conflicting targets. No upstream file or offset is available to reconstruct those groups.
- Explanations are generated text and often name the category; they are rationales, not inputs.
  `Rank` is also parsed model output and can contain inconsistent model names.

## Building and verification

```bash
uv run timenet-build build time-ra/rats40k --out /tmp/rats40k-registry
```

Raw files must remain available until writing finishes. Lazy rereads check channel names, lengths
and value fingerprints against the conversion scan.

Tests use small fixtures. A full build on the real release has not been verified; the pinned
inventory and source-quality claims above have not been independently rechecked by these tests.

[paper]: https://arxiv.org/abs/2507.15066
[card]: https://huggingface.co/datasets/Time-RA/RATs40K
[prompts]: https://github.com/yyysjz1997/Time-RA/tree/200b244f75b134348fad39c9ba140d9b2f1e1279/prompts
[sft]: https://github.com/yyysjz1997/Time-RA/blob/200b244f75b134348fad39c9ba140d9b2f1e1279/train/train_sft_parser.py
[eval]: https://github.com/yyysjz1997/Time-RA/blob/200b244f75b134348fad39c9ba140d9b2f1e1279/eval2file.py
