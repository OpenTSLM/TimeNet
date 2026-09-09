# ECG-QA CoT

ECG-QA CoT pairs PTB-XL 12-lead ECG recordings with clinical questions, short gold answers and
chain-of-thought rationales. One record is one PTB-XL recording: 12 leads at 500 Hz in millivolts.
Every question about that recording is one `AnswerTask`, so a record carries many tasks and there
is no record per question. A build gives 231,543 tasks over 16,024 recordings *(measured)*.

- **id**: `physionet/ecg-qa-cot`
- **source**: <https://physionet.org/content/ptb-xl/> for the signals,
  <https://github.com/Jwoo5/ecg-qa> for the per-template answer options, and the OpenTSLM release
  for the chain-of-thought rows
- **licence**: CC-BY-4.0

*(measured)* marks a number counted over the release itself: the three CoT CSVs and the 16,024
PTB-XL headers they name. A number with no mark comes from a dataset page.

## The source of truth

| fact | source of truth | the other reading |
| --- | --- | --- |
| a lead's rate, gain and unit | that recording's `.hea` header | the PTB-XL page states one rate |
| the answers a question offers | its row's `prompt` column | the template vocabulary, kept too |
| the split a recording is in | the CoT CSV the row came from | PTB-XL's `strat_fold`, not carried |

The header wins for the signal, because a per-file gain is part of the data and a page cannot state
it. The prompt wins for the answers, because it states the choice the benchmark put to a reader.
The template vocabulary is wider, and it stays under `template_answer_options`. The CoT CSV wins
for the split, because the rationales were written under that split. That split is not PTB-XL's:
each of the three CoT splits holds recordings from all ten `strat_fold` values *(measured)*. A
recording carries one `split` annotation, and no recording appears in two CoT splits *(measured)*.

## What the description states

> The waveform files are stored in WaveForm DataBase (WFDB) format with 16 bit precision at a
> resolution of 1μV/LSB and a sampling frequency of 500Hz (records500/).
> — the PTB-XL page, <https://physionet.org/content/ptb-xl/1.0.3/>

The connector reads `records500/<bucket>/<ecg_id>_hr`, the 500 Hz copy, and takes the rate and the
gain from each header rather than from this sentence.

> Cross-validation Folds: recommended 10-fold train-test splits (strat_fold) obtained via
> stratified sampling while respecting patient assignments, i.e. all records of a particular
> patient were assigned to the same fold.
> — the PTB-XL page, <https://physionet.org/content/ptb-xl/1.0.3/>

The `split` annotation is not this. It names the CoT CSV the questions came from.

> The date of birth only as age at the time of the ECG recording, where ages of more than 89 years
> appear in the range of 300 years in compliance with HIPAA standards.
> — the PTB-XL page, <https://physionet.org/content/ptb-xl/1.0.3/>

A clinical context that opens `300-year-old` states an age of more than 89. The connector keeps
that text as the release writes it.

> `answers_for_each_template.csv` provides the possible answer options for each template ID.
> — the ECG-QA repository, <https://github.com/Jwoo5/ecg-qa>

That file states a template's whole label space, so it becomes `template_answer_options`. It does
not state the two answers one question offers.

> ecg_id: a list of ecg IDs of the source ECG dataset. [...] For comparison questions, it contains
> two corresponding ecg IDs. Otherwise, it has only one element.
> — the ECG-QA repository, <https://github.com/Jwoo5/ecg-qa>

The connector reads one recording id per row. No row of this release names two *(measured)*.

> ECG-QA contains single-ECG and multi-ECG questions. Multi-ECG questions are about comparing two
> ECGs. The latter are not used for training OpenTSLM.
> — OpenTSLM, `src/opentslm/time_series_datasets/ecg_qa/README.md`

This is why the CoT release holds only the three `single-` question types *(measured)*.

> This question has one of two possible answers:
> — every `prompt` value in the three CoT CSVs

The two answers a question offers appear here and nowhere else, so the connector parses them out
of the prompt. All 231,543 rows state this heading and exactly two answers under it *(measured)*.

> ECG Lead I - sampled at ~100Hz, normalized (mean=0.003, std=0.132)
> — OpenTSLM, `src/opentslm/time_series_datasets/ecg_qa/README.md`

OpenTSLM trained on a decimated and normalized copy of these signals. The artifact keeps the
archival values that PTB-XL states, and the card tells a consumer to do that step at load time.

## What one record holds

- **Series**: 12 leads, all 500 Hz, all in millivolts, 5000 values each. The rate, the gain and
  the unit come from that recording's `.hea` header, one header for each record. All 16,024
  headers state 12 signals, 500 Hz, 5000 samples and a gain of `1000.0(0)/mV` *(measured)*. A
  lead keeps the name the header writes, so the augmented leads read `AVR`, `AVL` and `AVF` and
  not `aVR`, `aVL`, `aVF` *(measured over all 16,024 headers)*.
- **Annotations**: each record carries its `split`. The question metadata is registered once for
  each distinct value and referenced by id from the tasks: 3 `question_type`, 42 `template_id`,
  42 `template_answer_options`, 2,056 `answer_options` and 1,184 `clinical_context` *(measured)*.
  No annotation carries a span, so none is scoped to a series.
- **Tasks**: one `AnswerTask` for each CoT row. Its `prompt` is the question, its `target` is the
  short gold answer, and its `rationale` is the chain-of-thought text. The tasks stream, so the
  231,543 of them never all live in memory.
- **Record ids**: `ptbxl-<ecg_id>` from PTB-XL's own `ecg_id`, and `ecg-<ecg_id>-<lead name>` for
  a series. Neither is positional. A task id is positional, `ecgqa-<split>-<row index>`, because
  the one CoT column that looks like an id repeats in the training split (see below). A re-release
  that adds or reorders a row invalidates a task id.

## The tasks this connector builds

| the question | type | count | scope |
| --- | --- | --- | --- |
| is this stated finding present (`single-verify`) | `AnswerTask` | 78,356 *(measured)* | record |
| which of these findings is it (`single-choose`) | `AnswerTask` | 69,842 *(measured)* | record |
| what does this recording show (`single-query`) | `AnswerTask` | 83,345 *(measured)* | record |

Every one is a two-way choice, whatever its type. The two answers it offers are its
`answer_options` annotation, and the gold answer is one of the two on all 231,543 rows
*(measured)*.

## Inconsistencies and decisions

### The answers a question offers are stated only in its prompt — **Handled**

**Problem.** A CoT row states its two candidate answers inside the `prompt` text and in no column
of its own. `answers_for_each_template.csv` states something wider: the whole label space of the
template. On 26 of the 42 templates in use, that space is strictly wider than the pair. The widest
holds 37 answers against the question's 2. Those 26 templates carry 161,459 of the 231,543
questions *(measured)*.

**Decision.** The connector parses the pair out of the prompt and stores it as `answer_options`.
The template vocabulary stays under `template_answer_options`, because it is a different fact and
nothing else records it. A task that carries the vocabulary alone poses a wider question than the
benchmark. Accuracy measured on it does not compare to published results.

**Consequence.** Score against `answer_options`. `template_answer_options` is the label space of
the template, not the choice a reader was given.

### The pair keeps the order the prompt lists it in — **Handled**

**Problem.** Sorting the pair would make the artifact smaller: 2,056 distinct ordered pairs
against 1,029 sorted ones *(measured)*. The order carries no answer: the gold answer is listed
first on 115,510 rows and second on 116,033 *(measured)*.

**Decision.** Keep the prompt's order. Option position is a known bias in multiple-choice scoring,
so the order is part of the question that was asked. A consumer that reorders the pair runs a
different task.

**Consequence.** 2,056 `answer_options` annotations rather than 1,029. Two questions that offer the
same two answers in opposite orders are two annotations.

### The prompt column is not stored — **Handled**

**Problem.** The `prompt` column is 378,940,032 B, 56% of all field text in the three CSVs
*(measured)*. It restates what the artifact holds elsewhere.

**Decision.** Store the parts, not the text. Every one of the 231,543 prompts rebuilds byte for
byte from five pieces: a constant 138 B preamble, the clinical context, the question, the two
answers, and an instruction block *(measured over all 231,543 rows)*. The context, the question and
the pair are stored. The preamble is one constant. The instruction block is 104 distinct texts.
Each is one constant 983 B prefix plus a line that quotes the gold answer, and that quoted answer
equals the `answer` column on every row *(measured)*. Storing it puts the label inside the model
input.

**Consequence.** A consumer who wants the released prompt text rebuilds it from the stored pieces,
the two constant blocks and the target. Nothing is lost, but it is not one field to read.

### A prompt that is not a two-way choice raises — **Handled**

**Problem.** The pair is parsed out of prose. A re-release that rewords the heading, drops the
list, or offers three answers leaves the connector without the choice the question poses.

**Decision.** Raise `TimeFFormatError`, and name the `ecg_id` in the message. A question converted
as something the release does not contain is the larger error, and the vocabulary is not a safe
substitute.

**Consequence.** A reworded release fails the build rather than converting to a wider task. All
231,543 rows of this release parse, and each offers exactly two answers *(measured)*.

### A row with two recordings fails on the id, not on the format — **Open**

**Problem.** ECG-QA states that a comparison question carries two `ecg_id`s. This CoT release
holds no comparison question: all 231,543 rows carry one bracketed id, and all three question
types are `single-` types *(measured)*. A row with two reaches `int()` inside the id parse and
stops the build with a `ValueError` instead of a `TimeFFormatError`.

**Decision.** Left as it is. The parse of `template_id` and the parse of the template `classes`
cell have the same shape. The fix is one change over all three sites, not one over this site.

**Consequence.** On this release, nothing. On a release that adds comparison rows, the build stops
with an error that does not name the file a reader has to open.

### The training split repeats seven `sample_id` values — **Handled**

**Problem.** The training CSV holds 159,313 rows and 159,306 distinct `sample_id` values: the ids 1
to 7 each appear twice, on two different questions with a different `ecg_id`, question, answer and
rationale *(measured)*. The validation and test files number their rows 0 upward with no repeat
*(measured)*. 159,306 is the number of training samples OpenTSLM's ECG-QA page reports, so the
seven repeats are exactly the difference between that count and the file.

**Decision.** Convert all 159,313 rows, and build the task id from the row position rather than
from `sample_id`. Every row is a question with its own answer and rationale, so dropping one drops
a question, and a repeated id cannot identify a task.

**Consequence.** A consumer who counts tasks against OpenTSLM's page finds seven more. The 14 rows
that share those seven ids are 14 tasks.

### The clinical context states an age of 300 — **Handled**

**Problem.** 5,743 of the 231,543 rows carry a clinical context that opens `300-year-old`, over 18
of the 1,184 distinct contexts. No context states an age between 90 and 299 *(measured)*.

**Decision.** Keep the text as the release states it. The PTB-XL page quotes above explains the
number: an age of more than 89 appears as 300 for privacy. It is a convention, not a fault, and
the connector repairs no stated value.

**Consequence.** A consumer who reads an age out of the context text reads 300 for those patients.
It means older than 89.

### `answer_options` changed meaning under an unchanged `dataset_version` — **Handled**

**Problem.** `answer_options` used to hold the template vocabulary and now holds the two answers
one question offers, and the annotation id changed with it. Two builds of version 1.0.0 read the
same key as two different facts.

**Decision.** Keep 1.0.0. No 1.0.0 artifact exists to disagree with it: the hosted registry
publishes no dataset at all, and this dataset id resolves to a 404 there. A private registry cannot
be checked from a branch. An artifact published somewhere else moves the version to 2.0.0 instead.

**Consequence.** An artifact that carries a `template_answer_options` annotation is this reading of
the key. One that has only `answer_options` is the older one.

### Two of the three sources cannot be pinned — **Open**

**Problem.** The signals come from a versioned PhysioNet release, `ptb-xl-1.0.3.zip`. The template
answers come off the `master` branch of a GitHub repository, and the CoT rows come off a file
share link. Neither of those two states a version, and the connector checks no digest, so both can
change under identical connector code.

**Decision.** Kept as they are. The share link is the only public source of the rationales, so
there is nothing to fall back to. Both URLs are module-level constants a mirror can replace.

**Consequence.** Two builds from the same connector can disagree, and nothing in the artifact says
which bytes it was built from.

## Warnings this build emits

None. The connector logs nothing of its own, and no annotation it writes carries a span, so TimeF
emits no `SpanOutsideWindowWarning` either.

## What is not built

The `prompt` column, for the reason above, and the CoT `sample_id` column, which the training
split repeats on seven pairs of different questions.

PTB-XL's own metadata is not converted: `ptbxl_database.csv` (age, sex, the SCP statements, the
report text, `strat_fold`), `scp_statements.csv`, and the 100 Hz copies under `records100`. This
connector converts the CoT release, and a record it builds holds what a question is asked about.

PTB-XL recordings that no question names are not converted: 16,024 of the release's 21,799
recordings carry a question *(measured)*, and the rest have nothing to answer.

ECG-QA's comparison questions and its paraphrased question sets are not here. The CoT release
ships neither.
