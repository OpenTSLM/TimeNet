# TSAQA-clean

`tsaqa/tsaqa-clean@1.0.0` derives from `tsaqa/tsaqa-benchmark@1.0.0` and disables temporal
ordering tasks. Use this dataset ID for training and evaluation without ordering questions.
The complete parent release remains available under its original ID.

## Why exclude ordering?

Some questions cannot identify a unique label order. In test row 18000, patches B and D are
identical: both contain nine zeros. The published answer is `C,A,B,D`, but `C,A,D,B` reconstructs
exactly the same 45 observations. The paper's position-based scoring would give the second answer
only 50%, although the reconstructed numerical sequence is unchanged.

This variant conservatively excludes the entire ordering format. It does not claim that every
ordering sample is ambiguous or that all remaining tasks are free from source issues. It retains
the parent's raw-input representation, so scores are not directly comparable with the paper.

| Tasks | Complete TSAQA | TSAQA-clean |
| --- | ---: | ---: |
| True-or-false | 89,681 | 89,681 |
| Multiple-choice | 90,430 | 90,430 |
| Ordering | 29,889 | 0 |
| Total | 210,000 | 180,111 |

Selection uses the validated `question_format`, not the source's `question_type` column. The source
incorrectly labels 10,000 continuation questions as ordering; these true-or-false and multiple-choice
tasks remain enabled. Published prompts, answers, metadata, annotations, and splits are unchanged.

## Build and use the variant

```bash
uv run timenet-build build tsaqa/tsaqa-clean \
  --local-registry /tmp/timenet-tsaqa-registry
```

The build engine resolves the exact parent release first. If it is already in the local registry,
the clean connector reads that release without downloading the Hugging Face source again.

```python
from timenet import TimeNet
from timenet.types import Split

client = TimeNet(sources=["/tmp/timenet-tsaqa-registry"])
with client.open("tsaqa/tsaqa-clean@1.0.0") as reader:
    for task in reader.iter_tasks(split=Split.TRAIN):
        question = task.prompt.read()
        series = [signal.to_numpy() for record in task.inputs for signal in record.signals]
        answer = task.targets[0].read()
```

Ordering tasks are absent from the child task collection, so ordinary task iteration already applies
the exclusion. All 510,093 raw signals remain accessible through record iteration, including records
used only by disabled tasks. Their metadata describes their original roles in the parent.

## How inheritance works

The card declares an exact parent:

```yaml
parents:
  tsaqa: tsaqa/tsaqa-benchmark@1.0.0
```

The connector selects parent task IDs and delegates composition to TimeNet:

```python
parent = context.parents[self.parent_alias]
enabled_tasks = self.enabled_task_ids(parent)
dataset = TimeFDataset(metadata=self.metadata())
dataset.import_from(parent, records=parent.record_ids(), tasks=enabled_tasks)
```

TimeNet records parent lineage and the mapping between parent and child identities. The child has its
own control catalog and reuses the parent's stored signal payloads; it writes no new time-series
files. Importing tasks also preserves their input annotations and references. No source rows are
reparsed and no raw observations are changed. Missing or invalid resolved metadata fails explicitly.

Tests cover the filtered task set, mislabeled continuation questions, unchanged parent tasks, exact
raw-value preservation, and a persisted child whose manifest refers to its parent's payloads.
