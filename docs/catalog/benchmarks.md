---
icon: lucide/gauge
description: "Benchmarks built on TimeNet datasets."
tags:
  - catalog
  - benchmarks
---

# Benchmarks

Benchmarks will pair TimeNet datasets with standard tasks and metrics. You can then compare models fairly.

!!! planned "Planned"
    This is not built yet. The benchmark suite and results are still in progress.

## Shared payload composition

Run the three-generation benchmark from the repository root:

```sh
uv run benchmarks/composition.py --values 100000 --tasks 5000 \
  --text-bytes 1000000 > /tmp/composition.json
```

It measures regular signals, irregular timestamps, large annotation and task text, and a large
replayable task set. Each child imports all parent tasks and adds one local task. The JSON output
separates catalog bytes, owned signal bytes, owned payload bytes, owner counts, and payload reads.

A local run with these inputs produced the following observations:

| Profile | Parent signal bytes | Parent other payload bytes | Child new payload bytes | Grandchild new payload bytes | Stored body reads during either descendant write |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular | 560,284 | 0 | 0 | 0 | 0 |
| Irregular | 560,279 | 323,646 | 0 | 0 | 0 |
| Large text | 560,285 | 128,804 | 0 | 0 | 0 |
| 5,000 streamed tasks | 560,279 | 0 | 0 | 0 | 0 |

Every descendant had one physical owner. Catalogs used about 4.47 MB; the task-heavy grandchild used
about 4.99 MB. DuckDB page allocation contributes fixed catalog overhead in small fixtures. More
copied tasks increase catalog size and write time. Descendant writes took about 0.08–0.09 seconds
for the first three profiles and 0.32–0.36 seconds for the task profile on this machine.

Timings depend on the machine and are not CI thresholds. The script fails if a descendant write
reads a stored body. See [Dataset composition](../composition.md) for the ownership model.
