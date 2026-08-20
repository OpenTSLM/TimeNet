---
icon: lucide/database
description: "Registry backends that serve compiled TimeF versions to the SDK."
tags:
  - guide
  - registry
---

# Registry

A registry serves compiled TimeF versions to the SDK. A compiled TimeF version has manifests,
Parquet control tables, and a values plane in Parquet or Zarr format. A registry never runs
connector code. The registry code lives in `timenet.registry`.

You can have several registries: a public registry, private internal registries, or a local
directory. The output of [curation](curation.md) is itself a valid local registry.

## Choosing a registry

```python
from timenet.registry import open_registry

registry = open_registry("./local_registry")
```

`open_registry(uri)` dispatches by scheme:

| Scheme | Backend |
| --- | --- |
| `file://`, plain path | `LocalRegistry` |
| `s3://` | `S3Registry` (deferred) |
| `http(s)://` | `RemoteRegistry` (deferred) |
| `timenet://` | `RemoteRegistry`, an alias for the hosted `https://registry.timenet.ai` |

An unrecognized scheme, for example `gs://` or `az://`, raises `ValueError` instead of becoming a
local path. `open_registry` expands `~` in a local path or a `file://` URI.

!!! warning "Only local registries today"
    `LocalRegistry` is the only working backend. The `s3://`, `http(s)://`, and `timenet://`
    backends are stubs. These backends raise `NotImplementedError` because they are not
    available yet.

## `BaseRegistry`

Every backend implements this contract: four data-access methods and one shared `search` method.

| Method | Description |
| --- | --- |
| `list_datasets()` | Latest-version `DatasetMetadata` for every dataset, sorted by id. |
| `get_manifest(dataset_id, version=None)` | A dataset's [manifest](manifest.md) (latest if `version` is `None`). Raises `DatasetNotFoundError` for an unknown id/version, or `TimeFFormatError` if the stored manifest's own id disagrees with the directory it was loaded from. |
| `open_file(dataset_id, version, relpath)` | A file of a dataset version, opened for seekable binary reading (an object store must return a range-capable handle, not a forward-only stream). |
| `open_version(dataset_id, version=None)` | A `DatasetVersion`: the parsed manifest plus a filesystem-rooted, picklable handle to the version's files. This is the storage seam the [reader](timef-reader.md) reads through, so a read never re-opens the registry nor re-parses `manifest.json`. |
| `search(...)` | Filter datasets (shared implementation). |

`LocalRegistry` serves a `<root>/<dataset_id>/<version>/` tree. `RemoteRegistry` is a placeholder
for the versioned REST contract, for example `GET /v1/datasets` and
`/v1/datasets/{id}/{version}/manifest`. `S3Registry` is a placeholder for the same layout under an
S3 prefix. Both backends raise `NotImplementedError` now.

A dataset id is an `org/name` pair, for example `chengsenwang/tsqa`. The id nests one level deep on
disk, at `<root>/chengsenwang/tsqa/<version>/`. `list_datasets` finds these ids at any depth. Use
lowercase ids to avoid casing clashes on case-insensitive filesystems.

## Writing to a registry

A `WritableRegistry` adds one write method to the read contract. As a result,
[curation](curation.md) can publish into any backend, not only a local directory:

| Method | Description |
| --- | --- |
| `store(dataset, *, force=False, progress_cb=None)` | Compile a dataset and publish it; returns the stored version. Derives the schema first if absent, and skips an already-committed version unless `force`. |
| `exists(dataset_id, version)` | Whether a committed version already exists (shared implementation). |

`LocalRegistry` implements `store` by streaming the dataset through a
[`TimeFWriter`](timef-writer.md). The writer stages the dataset under `<version>.tmp-*` and
publishes it with one atomic rename. `RemoteRegistry` and `S3Registry` are write stubs for now.
`open_writable_registry(uri)` resolves a URI like `open_registry` does, but it returns a
`WritableRegistry`.

```python
from timenet.registry import open_writable_registry

registry = open_writable_registry("./local_registry")
version = registry.store(dataset)   # schema derived if needed, atomic commit
```

Every backend is a `WritableRegistry`. As a result, this type alone does not show if a backend is a
directory the engine can write to or a remote stub. `local_registry_path(uri)` gives this
information. It returns the directory that a `file://` URI or a plain path names. It raises
`RegistryError` for a remote scheme. `default_registry_path()` uses `local_registry_path` to find
the default local registry. It uses `$TIMENET_REGISTRY` when its value is a local path. Otherwise,
it uses `<home>/registry`. This is how [`timenet-curate build`](cli/curate.md) resolves its output
when `--out` is absent. The `timenet_connectors.build` and `load` helpers use it too.

## `search`

```python
registry.search(
    query=None, domain=None, task=None, license=None,
    time_series_spec=None, dataset_id=None, tag=None, limit=100,
)
```

Every filter takes a scalar value or a list. The registry ignores `None` filters and combines all
non-`None` filters with AND logic. The consumer CLI, `timenet search`, mirrors this behavior
exactly. `limit` caps the number of results, with a default of 100. `limit=0` returns no results.
A negative `limit` raises `ValueError`.

| Filter | Matches |
| --- | --- |
| `query` | any term is a case-insensitive substring of name/description/tags |
| `domain` | dataset shares any of these domains |
| `task` | dataset's schema includes any of these task classes (reads the manifest) |
| `license` | dataset has any of these licenses |
| `time_series_spec` | dataset declares all of these `spec_type` values (reads the manifest) |
| `dataset_id` | dataset id is any of these |
| `tag` | dataset declares all of these tags |

The type-filters, `task` and `time_series_spec`, resolve each dataset's schema from its committed
manifest. The manifest always carries the derived schema, so the registry does not need a
`precomputed_schema`.

---

See the [API reference for `timenet.registry`](api/registry.md) for the full symbol listing.
