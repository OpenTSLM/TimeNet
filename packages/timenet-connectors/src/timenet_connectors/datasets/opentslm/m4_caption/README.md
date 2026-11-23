# M4-Caption

M4-Caption pairs the 100,000 M4 competition series with captions written by GPT-4o. The connector
leaves out 706 series whose captions are API errors (253 daily, 57 monthly, 396 quarterly), leaving
99,294 series.

Each series becomes a `Record` with a float32 `Signal` on an ordinal axis, M4 id metadata and a
frequency annotation. An `AnswerTask` targets the caption and uses the OpenTSLM loader's prompt.

## Split

The OpenTSLM release has no split. We ran `generate_splits.py` once to make an 80/10/10 split with
`StratifiedSplitter`, stratified by M4 frequency. Its `group_by` option keeps the 90 pairs of series
with equal values in one split, preventing identical inputs from crossing splits. This is not the
OpenTSLM paper's random split.

`splits.json.xz` stores the validation and test M4 ids (9,929 each) and the release archive's
SHA-256. All remaining series go into train. The build fails if the SHA-256 does not match the
pinned release.

## Relation to the M4 data

Each series contains the train values for the same M4 id in the M4-methods repository's
`Dataset/Train` files. Each value is the float32 rounding of the M4 train value. The release omits
the test horizon and the metadata in `M4-info.csv`.

## Future work

1. An M4 parent dataset, such as `mcompetitions/m4`, could provide train and test values and
`M4-info.csv` metadata. M4-Caption could declare it under `parents`, import its `Record`s and scope
each caption task to the train part with `StepInterval`. This would store the 24 million values
once and expose M4 categories and forecast horizons. It would also replace the float32 values
OpenTSLM plotted for GPT-4o with M4's float64 values.

2. We use ordinal axes because some frequencies, such as monthly, are regular only in calendar terms.
Revisit this choice if the
[calendar axes proposal](https://github.com/OpenTSLM/TimeNet/issues/555) lands.
