---
icon: lucide/layers
description: "Build a dataset layer that reuses records and values from exact parent versions."
tags:
  - guide
  - architecture
  - composition
---

# Dataset composition

TimeNet can store a dataset as a layer over other dataset versions. A child layer owns only its
new objects. The child does not copy imported signal values.

This model supports a directed acyclic graph. One child can use many parents. Many children can
use the same parent.

## Dataset card

The card pins each parent as an exact `org/name@version` reference. Connectors address parents by
their full `org/name` dataset IDs. A card can pin one version of each direct parent dataset.

```yaml
yaml_schema_version: 1
dataset_id: physionet/ecg-qa-cot
dataset_version: 1.0.0
source_revision: 1.0.0
name: ECG-QA CoT
description: Questions and answers over PTB-XL records.
license: CC-BY-4.0
parents:
  - physionet/ptb-xl@1.0.0
```

`dataset_version` identifies the TimeNet release. `source_revision` identifies the upstream
release. These values do not have to match.

## Connector API

A connector whose card declares parents implements `compose()` instead of `convert()`. The engine
calls it with a `BuildContext` that contains lazy views of the direct parents. Parent tasks are not
part of the child unless the connector requests them.

```python
from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.types import AnswerTask


class ChildConnector(BaseConnector[RawRow]):
    def compose(self, raw_refs: list[RawRow], context: BuildContext) -> TimeFDataset:
        dataset = TimeFDataset(metadata=context.metadata)
        parent = context.parent("physionet/ptb-xl")
        for record in parent.iter_records(required_record_ids(raw_refs)):
            imported = dataset.import_record(record, parent="physionet/ptb-xl")
            dataset.add_task(
                task=AnswerTask(
                    inputs=(imported,),
                    prompt=question_for(imported.id),
                    targets=(answer_for(imported.id),),
                )
            )
        return dataset
```

`ParentDatasetView.iter_records()` keeps signal values lazy. `TimeFDataset.import_record()` keeps
the record under its parent id and marks it as owned by that parent. It does not transfer the
parent hierarchy to the child.

A connector can add record annotations after the import. TimeNet stores these annotations as
child overlays. The parent annotations remain in the parent layer.

## Build and storage

The build tool builds a missing parent before it builds the child. Each connector uses its own
isolated environment. The exact pinned parent version must exist after the parent build. Before
the engine writes the child, it checks that every imported record is held by the named parent.
An import from the wrong parent fails the build.

The child has one `control.duckdb` file. An imported record has a small proxy row in `records` and
a `record_imports` row that names its parent dataset ID. Child tasks and annotation overlays refer to
the proxy key.

The child manifest lists its exact direct-parent references under `metadata.parents`, and the
complete dependency closure under `dependencies`. Each `dependencies` row holds an exact dataset
version and the SHA-256 checksum of that version's manifest. The manifest lists the checksum of
every file, so one row pins the parent's complete content.

The child schema is derived from every record the child holds, imported ones included, so a search
by signal spec finds the child. Two parents that declare the same `spec_type` with different
contracts therefore cannot be composed into one child.

## Read behavior

`TimeNet.load("physionet/ecg-qa-cot")` resolves the full graph in one registry. The registry checks
each locked manifest checksum. A cycle, missing parent, or checksum mismatch stops the read.

The reader replaces each child proxy with the parent record, read from the parent in one call per
parent. It then adds the child record annotations and tasks. Signal loaders continue to read the
parent values plane.

Reading a composed dataset does not snapshot parent-owned fields. The first write or
`check_records()` call after a read hydrates the originals from the pinned parents again and
rejects any change to them, even when the reader that produced the dataset has closed.

`TimeNet.download()` resolves the same closure, downloads each version, and checks every local
copy against the manifest checksum the registry serves. Consumers use the normal dataset ID for
both root datasets and composed datasets.

## Current import boundary

The producer API imports complete records. This boundary covers shared signal values and
record-level task datasets. A finer import, such as one source or signal, would add a kind column
to `record_imports`.

Parent tasks require an explicit `ParentDatasetView.iter_tasks()` call. TimeNet never merges them
automatically. This rule prevents task duplication and split conflicts.
