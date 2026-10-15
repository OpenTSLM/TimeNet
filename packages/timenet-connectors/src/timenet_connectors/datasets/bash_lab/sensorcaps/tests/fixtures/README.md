# SensorCaps fixtures

`captions.jsonl` contains two synthetic chat rows and no upstream dataset samples. The rows have
unequal signal lengths, UTF-8 text, and unparsed NumPy-style statistics.

A test helper derives temporary partition files from this common fixture, giving each partition
distinct text and offsetting its sensor values by 100. It adds the optional PAMAP2 location.
Expected channel arrays are defined separately, without using the connector parser. Storage tests
remove the temporary source files before checking the complete restored artifact.
