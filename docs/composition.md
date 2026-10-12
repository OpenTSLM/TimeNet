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
        record_ids = required_record_ids(raw_refs)
        imported = parent.import_records(dataset, record_ids)
        for record_id, record in zip(record_ids, imported, strict=True):
            dataset.add_task(
                task=AnswerTask(
                    inputs=(record,),
                    prompt=question_for(record_id),
                    targets=(answer_for(record_id),),
                )
            )
        return dataset
```

`ParentDatasetView.iter_records()` yields parent records under their original IDs, for inspection.
`ParentDatasetView.import_records()` hydrates the requested records under qualified IDs and
registers them in the child in one step; signal values stay lazy. Use the returned records for
child tasks and annotations. Importing `ptbxl-1` from `physionet/ptb-xl@1.0.0` gives the child ID
`physionet/ptb-xl@1.0.0::ptbxl-1`. Source, signal, axis, and annotation identities use the same
prefix. Native raw recording IDs, annotation values, and metadata keep their original values.
`TimeFDataset.import_record()` is the registration step on its own, for a record the parent's
reader hydrated with that prefix.

Every import gets a prefix, even when IDs do not collide. Imports through different parents stay
separate. A grandchild adds its direct parent's prefix to the ID visible in that parent; reading
or rewriting a dataset preserves its existing IDs. Importing the same parent record twice fails.

A connector can add record annotations after the import. TimeNet stores these annotations as
child overlays. The parent annotations remain in the parent layer.

## Build and storage

The build tool builds a missing parent before it builds the child. Each connector uses its own
isolated environment. The exact pinned parent version must exist after the parent build. Before
the engine writes the child, it checks that every imported record is held by the named parent.
An import from the wrong parent fails the build.

The child has one `control.duckdb` file. An imported record has a small proxy row in `records` and
a `record_imports` row that names its parent dataset ID and original parent record ID. Child tasks
and annotation overlays refer to the proxy key.

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

The reader resolves imported records through their parents, one read per direct parent, and each
parent hydrates its hierarchy once under the final qualified IDs. The child then attaches its own
record annotations and restores its tasks. Signal loaders continue to read the parent values plane.

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
