# SensorCaps fixtures

`captions.jsonl` contains two synthetic chat rows and no upstream dataset samples. The rows have
60 readings per sensor and UTF-8 text. The first row repeats a two-sample pattern and the second
repeats a three-sample pattern.

A test helper derives temporary partition files from this common fixture, giving each partition
distinct text and offsetting its sensor values by 100. It replaces the rows' statistics with all
12 derived features, NumPy-style and partly multiline, and adds PAMAP2's location after them.
Generation instructions are shared across rows, with one location-specific variant for PAMAP2,
to verify that task prompts preserve the original system messages.
Expected channel arrays are defined separately, without using the connector parser. Storage tests
remove the temporary source files before checking the complete restored artifact.
