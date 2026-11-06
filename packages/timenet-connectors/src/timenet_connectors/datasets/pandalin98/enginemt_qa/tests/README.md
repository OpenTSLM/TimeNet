# Test fixtures

`fixtures/<file>` holds real lines of the pinned release, copied byte for byte. The tests write the
signals file themselves, because the real one is 18.8 GB. Its values encode the row, the step and
the channel, so a misplaced value cannot pass.

| File | Release lines (0-based), in fixture order |
| --- | --- |
| `train_qa.jsonl` | 316, 92, 11228, 3503 |
| `test_qa.jsonl` | 2786, 456, 1082 |

- Train line 316 (row 278) has an empty health state.
- Train line 92 (row 8435) has the fault cause `a, b, e`.
- Train line 11228 (row 20) is the same line as test line 2786.
- Train line 3503 and test line 456 are multi-cycle lines. They share five of their ten rows.
- Test line 1082 (row 37669) has an empty fault cause.

To make a file again, download the release's QA file and copy the lines above, in this order:

```python
lines = open("train_qa.jsonl", "rb").read().splitlines(keepends=True)
open("fixtures/train_qa.jsonl", "wb").write(b"".join(lines[n] for n in (316, 92, 11228, 3503)))
```
