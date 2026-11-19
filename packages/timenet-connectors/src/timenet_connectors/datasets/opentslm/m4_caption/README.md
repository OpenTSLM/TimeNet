# M4-Caption

M4-Caption pairs each of the 100,000 series of the M4 forecasting competition with one caption that
GPT-4o wrote. OpenTSLM publishes it as a zip archive of CSV files, one series file and one caption
file for each M4 frequency.

The connector converts each series into one Record and one `AnswerTask`. The Record holds the
series as one float32 Signal on an ordinal axis, the M4 id in its metadata, and the M4 frequency as
an annotation. The task target is the caption. The task prompt is the prompt of the OpenTSLM loader
(the post-prompt of `M4QADataset`), the same for every series.

## Split

The release archive stores no split. `generate_splits.py` makes an 80/10/10 split with
`StratifiedSplitter`, stratified by M4 frequency. Validation and test each hold a tenth of every
frequency, so the small hourly and weekly frequencies keep their share in each split. The result
depends on the version of scikit-learn, so the script ran once. It wrote the 10,000 validation and
10,000 test series, by M4 id, to `splits.json.xz`. The connector reads that file and puts every
other series in the train split.

This split is not the split of the OpenTSLM paper. The OpenTSLM loader makes a random split of all
frequencies together, so its share of the small frequencies drifts. For example, its test split
holds 6.5% of the hourly series.

The file is JSON compressed with xz, about 19 KB. Python reads it with the `lzma` module of the
standard library. To read it yourself, run `xz -dc splits.json.xz`. The file also states the SHA-256
of the release archive. If it does not match the pinned archive, the build stops with an error.

## Relation to the M4 data

Each series of the release is the train part of the M4 series with the same id, from the
`Dataset/Train` files of the M4-methods repository. The release does not hold the test horizon or
the series information of `M4-info.csv`. Each value is the float32 rounding of the M4 value. The
largest relative difference is 6e-8.

## Future work: an M4 parent dataset

TimeNet can model this relation with a parent dataset. A connector for M4, for example
`mcompetitions/m4`, can convert the `Dataset/` folder of M4-methods: the train and test files and
`M4-info.csv`. M4-Caption can then declare that release under `parents` and import its Records.
The caption task then names the train part of each series with a `StepInterval` scope.

With a parent, the 24 million values exist once, and a reader gets the M4 category, horizon and
test part. The values become the float64 M4 values, not the float32 values that OpenTSLM plotted
for GPT-4o.

## Future work: change axis types

We currently use an ordinal axis for consistency, as some of the frequencies are only regular in
calendar terms, e.g. "monthly". If
[Proposal: calendar axes, value cells and civil-date origins](https://github.com/OpenTSLM/TimeNet/issues/555)
lands we can reconsider this choice.
