---
icon: lucide/file-json
description: "The manifest.json entry point, schema summary, file list, and commit marker."
tags:
  - reference
  - manifest
---

# Manifest

The **Dataset Manifest** (`manifest.json`) is the entry point for a TimeF version. It contains the
card metadata, derived schema, counts, and file descriptors. The
[writer](timef-writer.md) writes it last, so its presence marks a committed version. The
[reader](timef-reader.md) reads it before opening `control.duckdb` or Signal values. Manifest types
live in `timenet.manifest`.

The [`DatasetSchema`](types.md#datasetschema) type already holds flat descriptors. The manifest's
`schema` block is a direct serialization of this type. The manifest has no separate "entry" types
to keep in sync.

The packaged `manifest.schema.json` (JSON Schema draft 2020-12) pins the on-disk shape. This file
is the formal contract for external consumers. It is published as
[`manifest.schema.json`](https://docs.timenet.ai/schemas/manifest.schema.json) and is
available in Python as `timenet.schemas.MANIFEST_SCHEMA`. A test validates the output of
`to_dict()` against this schema.

---

## `Manifest`

```python
from timenet.manifest import Manifest, ManifestCounts, ManifestFiles

Manifest(
    dataset_id="physionet/ecg-qa-cot",
    metadata=metadata,          # DatasetMetadata
    files=files,                # ManifestFiles (required)
    schema=schema,              # DatasetSchema (default: empty)
    counts=counts,              # ManifestCounts (default: empty)
    build_env=None,             # environment provenance (see below)
    timef_format_version=1,     # validated against the supported set {1}
)
```

`build_env` records the environment that produced the version. It gives the interpreter version and
every installed package with its version. `timenet.provenance.build_env` collects this data.
`build_env` is provenance only. No code reads it to interpret the data.

The values locator is backend-neutral. One schema covers both scalar and multidimensional specs. A
multidimensional spec records its shape in `value_shape` and `dimension_names`. It does not need a
separate format version.

A spec records its `nullable` flag. Parquet stores nullability in Arrow validity bitmaps. Zarr stores
it in validity arrays next to the values.

If you construct or parse a `Manifest` with an unsupported `timef_format_version`, it raises
`TimeNetInvalidManifestError`. Format version 1 lists `files` as file groups. The reader does not
read version 2 manifests, so rebuild a version 2 dataset.

### Codec

| Method | Purpose |
| --- | --- |
| `to_dict()` / `to_json()` | Canonical serialization (all keys present, explicit nulls, no empty `encoding`). |
| `from_dict(data)` / `from_json(text)` | Parse, tolerating missing optional blocks. |

`from_dict` requires `timef_format_version`, `dataset_id`, `metadata`, and `files`. `schema` and
`counts` default to empty. The parser drops unmodeled metadata keys. A malformed block raises
`TimeNetInvalidManifestError`. This error names the offending block.

### Serialization notes

- Units serialize to their pint names (`"hertz"`, `"millivolt"`, `"dimensionless"`). The shared
  registry converts them back.
- Device and origin information belongs to the hierarchy's `Source`, not to `TimeSeriesSpec`.
- Tasks serialize as `{"task_type": ...}`. On read, the reader resolves them against the built-in
  `TASKS` registry. An unknown `task_type` raises `TimeNetInvalidManifestError`. The annotation
  `value_type` round-trips as a string. The reader uses it to decode values.

---

## `ManifestCounts`

The fields count `records`, `sources`, `signals`, `axes`, reusable `annotation_contents`,
`annotation_occurrences`, and `signal_chunks`. `tasks` maps task type to count, and
`signals_by_spec` maps specification type to Signal count. All fields default to `0` or `{}`.

## `ManifestFiles`

`ManifestFiles` lists every file of a version in `groups`. A reader uses this list and never uses a
directory glob.

Each group is a `FileGroup`. A group has a `kind`, the `backend` that wrote its files, and its
`parts`. Each kind occurs at most once.

| `kind` | `backend` | Files |
| --- | --- | --- |
| `control` | `duckdb` | Exactly one file, `control.duckdb`. The group is required. |
| `time_series` | `parquet` or `zarr` | The Signal values. The group is optional. |

The reader selects its values backend from the `time_series` group. A version without that group
has no Signal values. The writer always writes the group, even when it has no files.

A `time_series` group can also have an `encoding`. It gives the
[values encoding](timef-writer.md#values-settings) of each spec type. No code reads this field.
Parquet records the applied encoding in the footer of each file, so the reader does not need it.
The field lets a builder see which encoding a build selected. The writer omits it for a backend that
has no such choice.

Each part is a `FilePart`. A `FilePart` carries the file's `path` (version-relative), its
`checksum` (with the `sha256:` prefix), and its `size` in bytes. So the path and the digest never
live in separate structures.

`control` returns the single control file. `group(kind)` returns the group of one kind, or `None`.
`all_files()` returns every descriptor. `all_parts()` returns only the paths.

In `manifest.json`, `files` is a list of groups:

```json
"files": [
  {
    "kind": "control",
    "backend": "duckdb",
    "parts": [
      {
        "path": "control.duckdb",
        "checksum": "sha256:468a…",
        "size": 3944448
      }
    ]
  },
  {
    "kind": "time_series",
    "backend": "parquet",
    "encoding": {"cosine": "dictionary", "sine": "dictionary"},
    "parts": [
      {
        "path": "time_series/part-00000000.parquet",
        "checksum": "sha256:992f…",
        "size": 2455
      }
    ]
  }
]
```

A new kind of file needs a new `FileKind` member and a new format version. Code that only moves or
checks files, such as `verify()`, downloads, and publishing, reads every group. It does not change
when a kind is added.

---

See the [API reference for `timenet.manifest`](api/manifest.md) for the full symbol listing.
