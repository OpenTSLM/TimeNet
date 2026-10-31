# Time-MMD: Economy and Energy

This connector brings **two of Time-MMD's domains** into TimeNet as
`adityalab/time-mmd`, version `0.1.0`. Economy has monthly observations; Energy
has weekly observations. We chose these two to exercise different calendars and
units under one schema while keeping the initial contribution small. The other
Time-MMD domains are outside this connector's scope.

## Mapping

One Dataset keeps Time-MMD recognizable as a collection. Separate Records keep
each domain's timeline, units and context together. A Source groups related
measurements, and each Signal represents one numerical column across time.
Reports can concern several measurements, so their text belongs to the domain
Record rather than a single Signal.

```text
Dataset: Time-MMD (Economy + Energy)
|
+-- Record: Economy
|   +-- Source: US Census goods trade
|   |   +-- Signal: Exports
|   |   +-- Signal: Imports
|   |   `-- Signal: OT (reported trade balance)
|   `-- Annotations: original report/search fact and preds
|
`-- Record: Energy
    +-- Source: US EIA gasoline prices
    |   +-- Signal: OT (national price)
    |   `-- Signal: each of eight regional price columns
    `-- Annotations: original report/search fact and preds
```

The [visual overview](shape.svg) shows this same mapping and the task inputs.

### Economy

The selected snapshot has 447 monthly observations of Exports, Imports and OT,
in nominal million USD, not seasonally adjusted. OT is the source-reported
goods-only trade balance; a negative balance means a deficit.

We preserve reported OT rather than recalculate it from the other columns.
Displayed figures are rounded: for April 2023, Exports minus Imports gives
-88,688.1, while reported OT is -88,688.2. Recalculating would change the source.

We also retain actual monthly dates. Months have different lengths, so a fixed
number of days would misrepresent the calendar. All three channels remain
available even though the example tasks use only OT.

### Energy

The selected snapshot has 1,622 weekly observations: the national OT and eight
regional price columns. These are all-grades/all-formulations retail gasoline
prices in USD per US gallon.

We retain the regional channels because they describe variation that a national
price cannot capture. We keep seven-day spacing rather than resample Energy to
match Economy. Sharing a schema does not require inventing aligned observations.
Text may quote regular-grade gasoline or a particular region; such a quote does
not necessarily describe the national all-grades OT.

### Reports and search text

An annotation is text attached to a period in a Record. It provides context
alongside measurements; it is not another numerical channel or a task answer.
We keep it on the domain Record because a report can discuss several channels
or broader events rather than one individual measurement.

Each report/search CSV row produces two separate annotations: one for its `fact`
field and one for its `preds` field. The annotation keeps the original field text,
the period it concerns and enough source information to find that field again:
file, revision, checksum, row index and saved row ID. Its key distinguishes
`report_fact`, `report_preds`, `search_fact` and `search_preds`.

For example, a report row covering January 2000 creates a `report_fact` annotation
covering January on its domain's timeline. The row's `preds` field becomes a
separate `report_preds` annotation over that same period. Neither annotation says
the text was available in January. Publication and availability remain unknown.

Empty fields and missing markers remain visible, and rows sharing dates remain
distinct. This gives 5,498 annotations for Economy and 5,336 for Energy. Those
counts describe stored fields, including empty fields, not that many unique
reports or independently verified statements.

The Time-MMD authors collected paragraphs from selected reports and results from
keyword-based web searches. They used Llama3-70B to filter for relevance, separate
factual material from predictions, and summarize the text. Their preprocessing
template also requests analysis and short/long-term predictions, allowing the
model's own opinions with an LLM label. We import the released text; this
connector does not generate or recalculate it. See the paper's
[text preparation and prompt](https://arxiv.org/pdf/2406.08627v3#page=24).

Consequently, `fact` is not a guarantee of verified truth, and `preds` may contain
processed or model-generated outlooks. Neither is the measured next OT used as
our forecasting reference. The authors' manual audit also found fabricated
factual statements. Economy text can discuss goods and services while OT
concerns goods only. Preserving the wording and provenance makes these differences
visible rather than treating every excerpt as a label.

Coverage tells us which period an excerpt concerns, not when it was published
or processed. Those availability times are unknown in the released CSVs, so
**neither example task receives the stored report or search annotations**.
An old report date would not prove its processed excerpt existed at that time.
Adding past reports requires independent evidence of availability at the task's
cutoff; numbers in a report are not, by themselves, a reason to exclude it.

## Example tasks

Both tasks use the same 64 prior OT observations. Keeping a shared input makes
their different questions easy to inspect: QA asks about observed history;
forecasting asks about the next observation that has been withheld.

### QA: a simple integration check

The `AnswerTask` compares the mean of the first eight observations with the mean
of the last eight. It is deliberately simple and easy to solve with arithmetic.
Its purpose is an end-to-end smoke test of the time-series input shape, task
scope, reference answer and write/read path, not a demanding reasoning benchmark.

Answers are `higher`, `lower` or `within`. `Within` means the difference is between
-1 and +1 million USD for Economy, or -0.001 and +0.001 USD per gallon for Energy,
including equality. For example, Energy means of 1.2000 and 1.2008 differ by
0.0008, so the answer is `within`; means of 1.2000 and 1.2020 give `higher`.
These thresholds define this example question. They do not estimate measurement
error or say whether a change matters economically.

### Forecasting: predict the next observation

The `ForecastingTask` asks for the next measured OT in its original units. This
is a prediction problem: its answer is not present in the supplied history.
The reference comes from the next numerical source row, not a report's outlook.
The connector supplies a task and a reference, not evidence that a model can
forecast well. A last-value persistence forecast provides a useful baseline.

The full Record stores all observations, but each task's scope identifies only
the 64 values permitted as input. A model adapter applies that scope and sends
the history through the model's native numerical pathway. Standard model
normalization and summary metadata are allowed. History length, QA comparison
length and threshold are part of the task definition, so switching models does
not change the question or its answer.

Chronological splits keep training targets before test targets; rolling histories
overlap. Observation dates do not establish historical release dates or revision
vintages. These are tasks on an archived snapshot, not verified real-time inputs.
The exact construction rules are in [connector.py](connector.py).

## Verification and reproduction

[reproduce.py and usage instructions](https://gist.github.com/haznai/b7f1326ea9cefccbdadf7cce1085381b)
rebuild the dataset and check it against the original source.

The tests and reproduction check the following properties:

- **Source fidelity:** numerical values and dates survive conversion and
  write/read. The full-source script compares every channel and original text
  field with the pinned CSVs, including source-row identity and coverage dates.
- **Calendar and context:** monthly dates are not replaced by a fixed day count;
  weekly spacing is retained. Earlier text fits its Record's timeline. Quoted
  text, newlines, empty fields, missing markers and repeated dates are preserved.
- **Lazy loading:** conversion creates Signal loaders without reading their
  numerical arrays. Additional numerical columns are kept, with OT required.
- **Invalid inputs:** malformed numerical values, duplicate observation dates,
  calendar gaps, missing files and corrupt cached bytes are rejected.
- **Task boundaries:** QA covers all three answers and equality at both threshold
  edges. Changing the future target leaves the permitted history and QA answer
  unchanged. Changing evaluator targets, rationale or metadata does not alter
  the selected numerical input. No source annotations are selected by either task.
- **Complete task delivery:** the full-source script checks both task types at
  every eligible origin and their chronological splits, rejecting missing or
  duplicate tasks and incorrect train/test assignments.

These checks test conversion and task construction. They do not prove that an
arbitrary model adapter respects the scope, that historical data was available
at its observation date, or that a model forecasts accurately.

TimeNet generates `manifest.json` in the output's dataset version directory
(`registry/adityalab/time-mmd/0.1.0`). It describes metadata, schemas, data files
and checksums. It is a build output, not a file to maintain in this connector.
The [fixture provenance](tests/fixtures/README.md) explains which samples are
real and which are synthetic. The [offline tests](tests/test_connector.py) and
[connector](connector.py) contain the detailed checks and construction rules.

Source attribution and licensing are recorded in [dataset.yaml](dataset.yaml);
the exact source revision and checksums are in [connector.py](connector.py).
The authors state ODC-By 1.0 in Appendix V of the
[Time-MMD paper](https://arxiv.org/pdf/2406.08627v3).
