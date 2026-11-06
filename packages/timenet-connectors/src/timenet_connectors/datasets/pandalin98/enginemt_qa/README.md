# EngineMT-QA

`pandalin98/enginemt-qa` converts questions about simulated aero-engine signals from the
[ITFormer paper][paper] and [authors' code][code]. The [Hugging Face release][card] is pinned to
`ea2051a15512212976c22c1a075ec549c7d48eef`.

The Hub metadata declares MIT, but its license section is an unfilled placeholder and the paper
states no dataset license. The local card follows that metadata; confirmation from the authors
remains outstanding.

## Source and records

| File | Content |
| --- | --- |
| `time_series_data.h5` | `seq_data`: float64, shape `(118921, 600, 33)`; `data_ID`: IDs 1–118,921 |
| `train_qa.jsonl` | 13,291 lines, 101,565 question-answer pairs |
| `test_qa.jsonl` | 5,539 lines, 42,477 question-answer pairs |

The connector creates one Record for each row referenced by a labelled pair: 7,755 rows in the
pinned release. Tasks referring to the same row share its Record. Unreferenced rows, which the
authors also use for encoder pretraining, are not included.

Each Record contains 33 float64 signals named `channel_00` through `channel_32`, with 600 values
and unknown units. Values are unchanged. The 1 Hz regular axis follows the paper's description of
600-point, ten-minute segments; the HDF5 file itself states no rate. Record metadata retains the
published `row_id`.

The release supplies no channel names, engine identity, flight identity or scaling parameters.
The [authors' channel audit][audit] does not establish which of the paper's 32 channels corresponds
to each of the 33 published columns. The connector therefore retains all 33 without assigning
physical names.

## Tasks and splits

Each labelled pair becomes one task, in source order. Train and test follow the two QA files.

| Stage | TimeNet task | Inputs | Target |
| --- | --- | --- | --- |
| Understanding | `AnswerTask` | 1 Record | Published answer |
| Perception | `ClassificationTask` | 1 Record | Published option letters |
| Reasoning | `ClassificationTask` | 10 Records | Published option letter |
| Decision-Making | `AnswerTask` | 10 Records | Published answer |

Closed-question label sets preserve the printed options. Fault-cause questions allow several
letters; other closed questions require one. Published question text supplies the prompt.

The authors take the first 60 steps of each of ten Records for multi-cycle questions. Tasks keep
the complete Records and store `steps_per_cycle: 60`; consumers reproducing that input must apply
this truncation themselves. The selected IDs are not necessarily consecutive cycles.

Task metadata retains `stage`, the closed-question name where applicable, `source_file`,
`source_line`, `qa_offset`, and `source_issues`. `source_line` and `qa_offset` are zero-based to
match the authors' evaluation; error messages use one-based physical file lines.

## Missing labels and source overlap

The connector omits tasks for 1,769 empty closed answers and records their locations in the
`unlabelled_pairs` dataset annotation. An empty fault-cause answer is not interpreted as “no
fault.” The authors trained on these pairs, so this is a deliberate conversion difference.

Duplicate pairs remain in the dataset, with flags in `source_issues`:

| Flag | Meaning |
| --- | --- |
| `repeated_across_splits` | Both files contain the same rows, question and answer |
| `repeated_question` | The same file repeats those rows, question and answer |

In the pinned QA files, 27,955 of 41,977 labelled test tasks also appear in train. Scores on this
split therefore include substantial overlap. Consumers evaluating unseen pairs can exclude the
flagged test tasks and should report that choice.

Other limitations remain: component-health and fault-cause answers can disagree, generated open
answers vary in wording, and published counts and channel descriptions differ between the files,
card and paper. The connector follows the pinned files and does not repair their answers.

## Building and verification

```bash
uv run timenet-build build pandalin98/enginemt-qa --out .timenet-registry
```

The download includes an 18.8 GB signals file; the referenced rows contain about 1.2 GB of values.
Keep source files available until writing finishes. Conversion checks the HDF5 structure, QA
schema, option vocabulary and pinned QA inventory. Task rereads verify the QA line fingerprints.
Invalid sources raise `TimeNetConnectorError` with context.

The QA parser has been run on the pinned QA files and the HDF5 header inspected. A complete build
and an end-to-end scan of the signals file have not been verified. Tests use small fixtures.

[paper]: https://arxiv.org/abs/2506.20093
[code]: https://github.com/Pandalin98/ITFormer-ICML25
[card]: https://huggingface.co/datasets/pandalin98/EngineMT-QA
[audit]: https://github.com/Pandalin98/ITFormer-ICML25/blob/b80142f/docs/ISSUE_23_CHANNEL_AUDIT.md
