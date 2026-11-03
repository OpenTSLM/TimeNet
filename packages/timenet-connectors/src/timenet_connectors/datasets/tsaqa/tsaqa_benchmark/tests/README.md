# Test fixtures

`fixtures/<split>.parquet` holds real rows of the pinned release, copied with every column as
published. Each row is the smallest one of its kind, plus the copies that make a source issue.

| File | Release rows, in fixture order |
| --- | --- |
| `train.parquet` | 31492, 344, 58715, 55206, 97985, 88586, 105647, 124117, 108026, 111614, 68522, 64476, 126163, 126422, 70268, 4591, 135829 |
| `val.parquet` | 2950 |
| `test.parquet` | 39798 |

- train holds one row of each task, published question type and format, and one data transformation
  question per transform.
- Rows 126163 and 126422 are `ordering` rows that ask a true-or-false and a multiple-choice question.
- Row 4591 asks the classification question of val row 2950, under another UCR dataset's name.
- Row 135829 asks the puzzle of test row 39798, with another answer.

To regenerate a file, download the release's `<split>.parquet` and write the rows above, in this
order, with the release's schema:

```python
import pyarrow.parquet as pq

table = pq.read_table("train.parquet")
rows = [31492, 344, ...]  # the train row list above
pq.write_table(table.take(rows).replace_schema_metadata(table.schema.metadata), "fixtures/train.parquet")
```
