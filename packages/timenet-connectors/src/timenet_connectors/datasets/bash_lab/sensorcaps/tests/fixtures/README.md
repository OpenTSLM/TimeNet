# SensorCaps fixtures

`captions.jsonl` contains two synthetic chat rows and no upstream dataset samples. The rows have
60 readings per sensor, UTF-8 text, and unparsed NumPy-style statistics. The first row repeats
a two-sample pattern and the second repeats a three-sample pattern.

A test helper derives temporary partition files from this common fixture, giving each partition
distinct text and offsetting its sensor values by 100. It adds the optional PAMAP2 location.
Generation instructions are shared across rows, with one location-specific variant for PAMAP2,
to verify that task prompts preserve the original system messages.
Expected channel arrays are defined separately, without using the connector parser. Storage tests
remove the temporary source files before checking the complete restored artifact.
