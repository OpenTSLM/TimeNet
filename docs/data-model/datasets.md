---
icon: lucide/database
description: "Datasets in TimeNet: versioned, immutable collections addressed by an org/name id."
tags:
  - guide
  - concepts
---

# Datasets

A dataset is a versioned collection of [records](records.md). Its address is `org/name@version`, for
example `chengsenwang/tsqa@1.0.0`. The id is a HuggingFace-style `org/name` pair. The version is the
semantic version of the upstream source.

<figure markdown="span">
  ![Six small record signals under the header org/name@1.0.0, a versioned collection of records](../assets/figures/dataset-example.svg)
</figure>

## Versions are immutable

TimeNet commits a version atomically when its `manifest.json` lands. The version never changes after
that. A pinned `@version` gives you exactly those bytes. With no suffix (or `@latest`), you get the
newest committed version.

```python
from timenet.client import TimeNet

client = TimeNet()
client.open("chengsenwang/tsqa")          # latest committed version
client.open("chengsenwang/tsqa@1.0.0")    # a pinned, immutable snapshot
```

Immutability makes **full data lineage** possible downstream. Every batch a model trains on traces
back to the exact TimeF bytes of one version, not a moving target. It is also what lets the control
plane drop its keys and indexes: an invariant checked once at publish time holds for the rest of the
version's life.

## Where a dataset lives

A [registry](../registry.md) serves a dataset. It hands the manifest and the version's files to the
[SDK](../client.md), and it never runs producer code. It can be a local directory, an S3 prefix, or
a remote host. The output of a [`TimeFWriter`](../timef-writer.md) is itself a valid registry. The
same `org/name` id resolves across all of them.
