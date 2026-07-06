---
icon: lucide/database
description: "Registry backends that serve manifests and parquet to the SDK."
tags:
  - guide
  - registry
---

# Registry

A registry serves compiled manifests and parquet to the SDK. It never runs connector code. Lives in
`timenet.registry`. There can be several registries: one public, private internal ones, or a local
directory (the output of [curation](curation.md) is itself a valid local registry).

## Choosing a registry

```python
from timenet.registry import open_registry

registry = open_registry("~/.timenet/local")          # a local directory
registry = open_registry("s3://my-bucket/registry")   # S3 (deferred)
registry = open_registry("https://registry.example")  # remote HTTP (deferred)
registry = open_registry("timenet://")                # the hosted TimeNet registry (deferred)
```

`open_registry(uri)` dispatches by scheme:

| Scheme | Backend |
| --- | --- |
| `file://`, plain path | `LocalRegistry` |
| `s3://` | `S3Registry` (deferred) |
| `http(s)://` | `RemoteRegistry` (deferred) |
| `timenet://` | `RemoteRegistry`, an alias for the hosted `https://registry.timenet.ai` |

!!! warning "Only local registries today"
    `LocalRegistry` is the only working backend. The `s3://`, `http(s)://`, and `timenet://`
    backends are stubs that raise `NotImplementedError` until they land.

## `BaseRegistry`

The contract every backend implements (three data-access methods) plus a shared `search`:

| Method | Description |
| --- | --- |
| `list_datasets()` | Latest-version `DatasetMetadata` for every dataset, sorted by id. |
| `get_manifest(dataset_id, version=None)` | A dataset's [manifest](manifest.md) (latest if `version` is `None`). Raises `DatasetNotFoundError` for an unknown id/version. |
| `open_file(dataset_id, version, relpath)` | A file of a dataset version, opened for binary reading. |
| `search(...)` | Filter datasets (shared implementation). |

`LocalRegistry` serves a `<root>/<dataset_id>/<version>/` tree. `RemoteRegistry` is a placeholder for the
versioned REST contract (`GET /v1/datasets`, `/v1/datasets/{id}/{version}/manifest`, ...) and
`S3Registry` for the same layout under an S3 prefix; both currently raise `NotImplementedError`.

Dataset ids are an `org/name` pair (`chengsenwang/tsqa`), which nests one level deep on disk
(`<root>/chengsenwang/tsqa/<version>/`). `list_datasets` discovers them depth-agnostically. Prefer
lowercase ids to avoid casing clashes on case-insensitive filesystems.

## Writing to a registry

A `WritableRegistry` adds one write primitive to the read contract, so [curation](curation.md) can
publish into any backend, not just a local directory:

| Method | Description |
| --- | --- |
| `store(dataset, *, force=False, progress_cb=None)` | Compile a dataset and publish it; returns the stored version. Derives the schema first if absent, and skips an already-committed version unless `force`. |
| `exists(dataset_id, version)` | Whether a committed version already exists (shared implementation). |

`LocalRegistry` implements `store` by streaming the dataset through a [`TimeFWriter`](timef-writer.md),
which stages under `<version>.tmp-*` and publishes with a single atomic rename. `RemoteRegistry` and
`S3Registry` are write stubs for now. `open_writable_registry(uri)` resolves a URI like `open_registry`
but returns a `WritableRegistry`, raising if the backend can't be written to.

```python
from timenet.registry import open_writable_registry

registry = open_writable_registry("~/.timenet/local")
version = registry.store(dataset)   # schema derived if needed, atomic commit
```

## `search`

```python
registry.search(
    query=None, domain=None, task=None, license=None,
    time_series_spec=None, dataset_id=None, tag=None, limit=100,
)
```

Every filter takes a scalar or a list; `None` filters are ignored and non-`None` filters are ANDed. The
consumer CLI mirrors this one-to-one (`timenet search`).

| Filter | Matches |
| --- | --- |
| `query` | any term is a case-insensitive substring of name/description/tags |
| `domain` | dataset shares any of these domains |
| `task` | dataset's schema includes any of these task classes (reads the manifest) |
| `license` | dataset has any of these licenses |
| `time_series_spec` | dataset declares all of these `spec_type` values (reads the manifest) |
| `dataset_id` | dataset id is any of these |
| `tag` | dataset declares all of these tags |

The type-filters (`task`, `time_series_spec`) resolve each dataset's schema from its committed manifest —
no `precomputed_schema` is needed because the manifest always carries the derived schema.

---

See the [API reference for `timenet.registry`](api/registry.md) for the full symbol listing.
