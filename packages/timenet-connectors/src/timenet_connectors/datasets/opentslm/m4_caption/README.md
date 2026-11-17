# M4-Caption

M4-Caption pairs each of the 100,000 series of the M4 forecasting competition with one caption that
GPT-4o wrote. OpenTSLM publishes it as a zip archive of CSV files, one series file and one caption
file for each M4 frequency.

The connector converts each series into one Record and one `AnswerTask` without a prompt. The
Record holds the series as one float32 Signal on an ordinal axis, the M4 id in its metadata, and
the M4 frequency as an annotation. The task target is the caption.

## Split

The release has no split column. The OpenTSLM loader splits the series at load time with the
`train_test_split` function of the datasets library. The connector computes the same 80/10/10 split
with NumPy. If the result does not match `OPENTSLM_SPLIT_SHA256`, the build stops with an error.
`generate_splits.py` computed that digest with the OpenTSLM code path and the library versions of
the OpenTSLM `uv.lock`.

## Relation to the M4 data

Each series of the release is the train part of the M4 series with the same id, from the
`Dataset/Train` files of the M4-methods repository. The release does not hold the test horizon or
the series information of `M4-info.csv`. Each value is the float32 rounding of the M4 value. The
largest relative difference is 6e-8, and in 49,610 of the 100,000 series the values are equal bit
for bit.

## Future work: an M4 parent dataset

TimeNet can model this relation with a parent dataset. A connector for M4, for example
`mcompetitions/m4`, can convert the `Dataset/` folder of M4-methods: the train and test files and
`M4-info.csv`. M4-Caption can then declare that release under `parents` and import its Records, as
`physionet/ecg-qa-cot` imports PTB-XL. The caption task then names the train part of each series
with a `StepInterval` scope.

With a parent, the 24 million values exist once, and a reader gets the M4 category, horizon and
test part. The values become the float64 M4 values, not the float32 values that OpenTSLM plotted
for GPT-4o. The card must document that difference. The connector must also make sure that each
release series equals the float32 rounding of the parent series.
