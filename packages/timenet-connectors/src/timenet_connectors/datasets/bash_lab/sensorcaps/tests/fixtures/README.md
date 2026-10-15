# SensorCaps fixtures

These files are synthetic, hand-authored examples of the published chat-row structure. They contain
no upstream dataset samples. Two files cover v1 captions and v2 captions with unparsed NumPy-style
statistics. Each contains two rows, UTF-8 text, and different signal lengths. Tests derive the PAMAP2
variant by adding `Location: ankle` to the v2 rows and writing temporary JSONL. They reuse these
shapes across the named source partitions and exercise both caption selections.
