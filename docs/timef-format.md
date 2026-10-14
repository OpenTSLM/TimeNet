---
icon: lucide/database
description: "TimeF format 3: independent catalogs and immutable payloads."
tags:
  - reference
  - format
---

# TimeF format

Format 3 separates a complete DuckDB catalog from immutable payload files. The catalog contains
object definitions and relationships. Signal values, irregular timestamps, and large text live
outside DuckDB. See [Dataset composition](composition.md) for import and storage policies.

## Files and ownership

```
org/name/version/
├── manifest.json
├── control.duckdb
├── time_series/           # new Parquet signal shards
├── time_series.zarr/      # new Zarr signal arrays
└── payloads/              # packed text, timestamps, copied artifacts
```

A version can use multiple codecs. Its manifest lists every owned artifact with its size and
SHA-256 checksum. `payload_sources` lists exact foreign physical owners and their manifest pins.
Build inputs are separate provenance and do not determine runtime dependencies.

Local references omit a source pin on disk. The reader binds them to the opened release's manifest
checksum. This avoids a manifest referring to its own checksum. Foreign references name their
physical owner directly; they do not name a parent Signal or require a parent catalog lookup.

## Catalog tables

| Table | Purpose |
| --- | --- |
| `datasets`, `records`, `clocks` | Dataset identity, recordings, and shared clock origins. |
| `sources` | Recursive Sources and their owning Record. |
| `signals` | Signal identity, axis, length, and inline specification fields. |
| `axes` | Regular, irregular, and ordinal axis definitions. |
| `tasks`, `task_targets` | Task definitions and ordered typed literal targets. |
| `task_record_refs`, `task_annotation_refs` | Ordered object and annotation relationships. |
| `task_dependencies` | Task derivations by identity. |
| `annotation_contents` | Reusable content descriptors and inline typed values. |
| `annotation_occurrences` | Attachments, spans, confidence, and occurrence metadata. |
| `dataset_refs` | Exact release identities shared by payload stores and lineage. |
| `payload_stores`, `payload_files` | Shared owner pins, codecs, prefixes, and file paths. |
| `payload_descriptors` | Shared content contracts. |
| `payloads`, `payload_chunks` | Object fields and ordered typed chunk coordinates. |
| `origins` | Typed source-object provenance, independent of payload ownership. |
| `signal_chunks` | Placements generated for newly written signal arrays. |

Object and relationship tables use internal integer keys and public IDs. Small fixed fields stay
queryable. Free-form metadata, shared physical descriptors, and classification schema references use JSON.
Release identities, file references, chunk coordinates, and lineage use typed columns. An external body leaves its native value column null and has an explicit `payloads` row.

The payload key is `(object_type, object_id, field, position)`. Signal values use the Signal ID;
timestamps use the axis ID; annotation values use the content ID. Prompts, rationales, and text
targets use task IDs and target positions. The payload descriptor records kind, shape, length, and
known encoded size. Each reference records its codec and physical chunk coordinates.

## Bodies

Regular axes compute offsets from a rational microsecond period and origin. Ordinal axes provide
order only. Irregular axes keep endpoints in the catalog and packed int64 offsets in Parquet.
Signals that share an axis share its timestamp payload. There is no `axis_offsets` bulk table.

Small annotation values stay in typed columns. Large text and text lists use packed Parquet cells.
One content ID can have many distinct occurrences without repeating its body. Prompts, rationales,
and text targets use the same policy. Numeric and boolean values remain inline.

Parquet array references identify files, row groups, and rows. Zarr references identify arrays and
ranges. Stored handles read these locations directly, including after their catalog reader closes.
Range reads select intersecting chunks. Each payload selects its own codec.

## Parquet recording groups

`TimeFWriter(..., row_group_layout="record")` finishes each recording's row group before the
next recording. A selected child then reads its recording without decoding its neighbors. Large
recordings still split at the configured byte cap. `row_group_layout="auto"` separates numeric
recordings of at least 128 KiB and packs smaller recordings together. The default, `packed`,
preserves the prior layout.

Grouped writes collect the same bounded encoding sample across recordings and reuse those decoded
arrays during writing. A short first recording does not force a different values encoding. Each
producer runs once. Recording groups can increase file sizes and storage request counts, so choose
a layout from measured consumer reads. The reader uses ordinary Parquet row-group metadata.

## Integrity and migration

The writer stages a release, writes catalog rows in a transaction, and audits the completed catalog.
The audit checks relationships, task scopes, typed chunk coverage, and store ownership. A successful
full write records its audit revision in `control_metadata`. The writer then flushes payload files,
records checksums, and publishes the directory atomically.

The reader verifies the control checksum for local and downloaded catalogs. For verified catalogs
with the supported audit revision, it restores task objects without repeating public constructor
validation. Unverified catalogs retain constructor checks. Public constructors still validate new
caller input. Shared owners and file placements pass a fresh preflight at each write, without reads
of inherited payload bodies. Runtime owner pins and decoded payload lengths remain checked.

`audit_control_database()` also checks external catalogs. `TimeFReader.verify()` checks owned files and required foreign payload artifacts.
Foreign catalogs are not needed for payload reads or verification.

Format 1 and 2 artifacts must be rebuilt. Artifact parsers raise `TimeFFormatError` for unsupported
formats
or corrupt manifests. There is no legacy artifact reader or automatic converter.
