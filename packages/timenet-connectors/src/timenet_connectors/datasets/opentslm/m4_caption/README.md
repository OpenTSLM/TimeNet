# M4-Caption

M4-Caption is a dataset introduced by OpenTSLM, derived from the M4 forecasting competition. Each
series has one caption that GPT-4o wrote from a plot of the series. M4 has 100,000 series, but 706
have a caption API error message instead of a caption (253 daily, 57 monthly, 396 quarterly). The
connector leaves them out, so the dataset has 99,294 series.

Each series becomes a `Record` with a float32 `Signal` on an ordinal axis, M4 id metadata and a
frequency annotation. An `AnswerTask` targets the caption and uses the OpenTSLM loader's prompt.

## Split

The OpenTSLM release has no split. We ran `generate_splits.py` once to make an 80/10/10 split with
`StratifiedSplitter`, stratified by M4 frequency. Its `group_by` option keeps the 90 pairs of series
with equal values in one split, preventing identical inputs from crossing splits. This is not the
OpenTSLM paper's random split.

`splits.json.xz` stores the train, validation and test M4 ids (79,436, 9,929 and 9,929) and the
release archive's SHA-256. The build fails if the SHA-256 does not match the pinned release.

## Relation to the M4 data

Each series contains the train values for the same M4 id in the M4-methods repository's
`Dataset/Train` files. Each value is the float32 rounding of the M4 train value. The release omits
the test horizon and the metadata in `M4-info.csv`.
