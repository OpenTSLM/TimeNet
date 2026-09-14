---
name: timenet-datasets
description: Use when finding, searching, inspecting, downloading, or reading TimeNet datasets through the Python `TimeNet` client (list, get, search, download, open) and the `TimeFReader` it hands back. Covers dataset ids, version pinning, and registry/storage configuration.
---

# Using TimeNet datasets

TimeNet's consumer side is the `timenet.client.TimeNet` SDK. It browses a **registry** (the catalog
of published datasets) and opens a dataset version as a `TimeFReader`. There is no CLI.

Writing datasets is a separate concern: build a `DeclarativeDataset` and hand it to `TimeFWriter`,
or to `LocalRegistry.store`. See `docs/timef-writer.md`.

## Dataset ids and versions

- A dataset id is `org/name` (HuggingFace-style, exactly one slash), for example `chengsenwang/tsqa`.
- Pin a version with `<id>@<version>` or `<id>@latest`, e.g. `chengsenwang/tsqa@1.0.0`. You can also
  pass the version as a separate argument. Don't do both: that raises `TimeFValidationError`.
- Omitting the version means the latest.

## Python API

```python
from timenet.client import TimeNet
from timenet.types import AnswerTask

tn = TimeNet()                      # registry: arg > $TIMENET_REGISTRY > timenet://
tn.list()                           # list[DatasetMetadata]
tn.get("chengsenwang/tsqa")         # Manifest (add a version or use id@version)
tn.search(task=AnswerTask, tag="ecg", limit=50)   # filters are scalar-or-list, ANDed
tn.download("chengsenwang/tsqa")    # -> local <storage>/<id>/<version>/ dir
```

- `TimeNet(registry=None, *, storage_path=None)`. `registry` accepts a `BaseRegistry`, a local path,
  or a `file://`, `s3://`, `timenet://`, or `http(s)://` URL.
- `search` takes `query`, `domain`, `task`, `license`, `time_series_spec`, `dataset_id`, `tag`, and
  `limit`. The task filter takes the task **class** (`from timenet.types import AnswerTask`), not a
  string.
- `open(id, version=None)` returns a `TimeFReader`. A local or S3 registry reads in place; a remote
  registry materializes the whole version into storage first.

## Reading a version

Everything after `open` happens on the reader, and the read contract is batched: a batch costs a
fixed number of queries however large it is, so never loop one record at a time.

```python
with tn.open("chengsenwang/tsqa") as reader:
    print(reader.counts())
    for batch in reader.iter_records(batch_size=64):
        print(len(batch))
    record = reader.record("record-000")
    values = reader.values(record.signals()[0].signal_id)
```

- Records and tasks: `record`/`records`/`records_by_id`, `task`/`tasks`/`tasks_by_id`,
  `iter_records`, `iter_tasks`, `record_ids()`, `task_ids()`, `counts()`.
- Annotations: `annotations_for`, `objects_with`, `records_with`, `tasks_for_record`, `subtree`.
  `reader.connection` is the open DuckDB connection for hand-written SQL.
- Values: `values`, `values_for`, `values_window`, `values_windows`, `chunk_locators`. They take the
  surrogate `signal_id`, not the external id.
- Framework views: `timenet.pandas` and `timenet.torch` (the `torch` extra) wrap a reader.

## Restricted datasets

A credentialed or restricted dataset is never served by a hosted registry. `download` and `open`
raise `TimeNetAccessError` and point at the access URL. Build it yourself and read it from a local
registry.

## Configuration

All local state lives under `TIMENET_HOME` (default `~/.cache/timenet`). Env vars, prefix
`TIMENET_`:

- `TIMENET_HOME` relocates everything.
- `TIMENET_REGISTRY` selects the registry (URL or path).
- `TIMENET_STORAGE` where downloads land (default `<home>/storage`).
- `TIMENET_CACHE` raw source download cache (default `<home>/cache`).
- `TIMENET_TOKEN` bearer token for a remote registry; unset reads anonymously.

Precedence for any value: explicit argument, then env var, then default.

## Further reading

`docs/client.md` (the SDK), `docs/timef-reader.md` (the full reader surface), `docs/registry.md`
(registry contract and search semantics), and `docs/usage.md` (pandas and PyTorch). The SDK lives in
`packages/timenet/src/timenet/client.py`, the reader in
`packages/timenet/src/timenet/control_plane/reader.py`.
