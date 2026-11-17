# CAPTURE-24 mapping

One record is one participant file, `P<nnn>.csv.gz`. Its `x`, `y` and `z` columns become three
float64 signals in g at 100 Hz, under the source `Axivity AX3`. The release resamples every file to
100 Hz, so its timestamps step by 10 ms. Eight files jump forward once: P008, P060, P077, P094,
P097 and P142 stop recording for 1.5 to 3.9 hours, and in P016 and P017 the local clock moves one
hour ahead. Each stretch of rows 10 ms apart becomes one child source, `segment 1`, `segment 2`,
with a regular axis that starts at the offset its first timestamp states. Nothing fills a gap, and
a step under 10 ms raises. The first timestamp is kept as the `recording_start_local` annotation:
a local wall clock that names no zone, so the record sets no `start_time`.

Each run of equal non-blank annotation rows inside one segment becomes one span annotation with
the key `annotation` and one `ClassificationTask` scoped to that run. The target is the annotation
text of the release, from the `annotation` label set, which lists every text of
`annotation-label-dictionary.csv`. Rows nobody annotated get no annotation and no task.

The dictionary maps each text to six coarser label schemes. The dataset keeps it as the
dataset annotation `annotation_label_dictionary`, a JSON object keyed by annotation text.
`metadata.csv` gives each record its `age` group and `sex` annotations.
