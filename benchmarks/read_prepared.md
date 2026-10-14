# Prepare catalogs for consumer reads

The new core stack starts at `f24b2ce84a607df379ae3e67d47e7146797f3835`.
It removes the child catalog-load regression without changing consumer data.
The best tested PTB-XL storage configuration also makes the child waveform scan faster than the
standalone scan. This configuration uses recording groups, dictionary encoding, and LZ4 level 1.

## Changes

Format 3 stores each release identity, payload contract, and file path once. Typed chunk rows
replace nested physical-reference JSON. Readers share decoded contracts and release identities.
The public payload handles, lineage objects, and content-method API remain the same.

The writer audits the complete catalog before publication. It checks relational integrity, task
scopes, chunk coverage, and store ownership. It revalidates mutable task fields and starts a fresh
owner-pin preflight for each write. That preflight does not materialize inherited payload bodies.
Owned array and timestamp producers still validate their bodies during writing.

The reader verifies local and downloaded control checksums. Audited catalogs with the supported
validation revision use direct task hydration. Public constructors and unverified catalogs retain
their input checks. Runtime owner pins, decoded array lengths, and timestamp consistency still
have checks. Resolver caches reuse owner bindings and file lookups.

Parquet writers can separate recordings into row groups. Automatic layout separates numeric
recordings of at least 128 KiB and packs smaller recordings together. The byte cap still bounds
large recordings. Grouped writes collect the encoding sample across recordings and reuse those
arrays during writing. Each producer runs once, and a short first recording does not change the
encoding decision.

Formats 1 and 2 need a rebuild. The PTB-XL and ECG-QA connector commits remain in their separate
stack. The benchmark exports those modules from Sapling history and builds them with this core.

## Full dataset and measurements

The comparison uses all 21,799 cached PTB-XL recordings and all 231,543 ECG-QA CoT questions.
ECG-QA selects 16,024 recordings: 192,288 leads and 961,440,000 float32 samples.
Each recording has 12 leads at 500 Hz, in millivolts. No row limit applies.

The machine was an Apple M5 Max with 64 GiB RAM, macOS 26.6.2, and Python 3.13.15.
Each read used a fresh process and a warm filesystem cache. Timings are medians of three runs.
SDK load hydrated every record and task. Each waveform scan materialized and hashed every value
and null mask. Each task scan materialized and hashed every question, answer, rationale, and context.
These timings include fingerprint work. They do not measure isolated storage bandwidth.
The final standalone/LZ4 comparison alternated three fresh processes per variant, without concurrent
writers or checks. The exploratory sweep alternated the four other configurations. A brief lint and
type pass overlapped part of that sweep. Its packed/record difference is within observed variability.

| Variant | SDK load | Waveform scan | Peak read memory |
| --- | ---: | ---: | ---: |
| Original format 2 standalone | 11.11 s | 8.61 s | 2.57 GB |
| Original format 2 child | 15.19 s | 10.69 s | 3.19 GB |
| Format 3 standalone, packed Zstd (final comparison) | 10.26 s | 8.01 s | 2.53 GB |
| Format 3 child, packed Zstd | 10.33 s | 9.99 s | 2.69 GB |
| Format 3 child, recording groups and Zstd | 10.21 s | 9.93 s | 2.44 GB |
| Format 3 child, recording groups and byte-stream-split Zstd | 10.09 s | 11.42 s | 2.45 GB |
| Format 3 child, recording groups and dictionary LZ4 level 1 (final comparison) | 10.03 s | 7.58 s | 2.45 GB |

The child catalog fell from 127.15 MB to 113.26 MB with packed groups. The PTB-XL catalog fell
from 25.18 MB to 17.31 MB. The child still owns no waveform files and records zero inherited
materializations during building.

Recording groups prevent unrelated recordings from decoding, but they increase request counts.
They did not reliably improve this full scan with Zstd. Byte-stream-split was slower and increased
the parent artifact to 2.24 GB. The LZ4 parent occupies 1.69 GB, versus 1.36 GB for recording
groups with dictionary Zstd and 1.19 GB for packed Zstd. Artifact sizes use decimal units.

The generic defaults remain packed groups and Zstd. The measurements support an explicit
PTB-XL configuration for local training reads:

```python
with TimeFWriter(
    root,
    parent,
    row_group_layout="record",
    compression="lz4",
    compression_level=1,
    value_encoding="dictionary",
) as writer:
    writer.write()
```

The larger LZ4 artifact increases initial transfer size. These measurements cover local warm
reads. They do not establish the best configuration for cold-cache or remote reads.

## Fidelity

All 21 full read snapshots match the original standalone catalog, waveform, and task fingerprints.
The [results](read_prepared_results.json) include the individual runs, medians, build metrics,
entity counts, and complete SHA-256 fingerprints. The comparison covers all samples and tasks.

Catalog fidelity includes stable IDs, source hierarchy, metadata, annotations, signal contracts,
time axes, shared-axis groups, task memberships, and schema. Task fidelity includes prompts,
targets, rationales, input references, annotation context, scopes, splits, and modality declarations.

The comparison excludes generated axis and annotation occurrence IDs, lineage fields, release
version, and build-input declarations. Stable recording, source, signal, content, and task IDs
remain unchanged. Physical layouts, checksums, and artifact versions change as intended.

## Reproduce

Export the reviewed connectors without changing the checkout. The standalone connector is from
the commit immediately before the ECG-QA child change.

```sh
ECG_CONNECTORS="$(mktemp -d /tmp/timenet-ecg-connectors.XXXXXX)"
ECG_CACHE="$HOME/.cache/timenet/cache/physionet/ecg-qa-cot"
ECG_WORK="$(mktemp -d /tmp/timenet-read-prepared.XXXXXX)"

mkdir -p "$ECG_CONNECTORS/ptb_xl" "$ECG_CONNECTORS/before"
mkdir -p "$ECG_CONNECTORS/child"
ECG_SOURCE=packages/timenet-connectors/src/timenet_connectors/datasets
for file in connector.py dataset.yaml; do
  sl cat -r 34e2b55f0d62 "$ECG_SOURCE/physionet/ptb_xl/$file" \
    > "$ECG_CONNECTORS/ptb_xl/$file"
  sl cat -r 34e2b55f0d62 "$ECG_SOURCE/physionet/ecg_qa_cot/$file" \
    > "$ECG_CONNECTORS/before/$file"
  sl cat -r 44bc495dc5be "$ECG_SOURCE/physionet/ecg_qa_cot/$file" \
    > "$ECG_CONNECTORS/child/$file"
done
```

The raw cache needs the PTB-XL 1.0.3 directory, `answers_for_each_template.csv`, and the three
`ecg_qa_cot/*.csv` split files. Each build command runs in a fresh process.

```sh
uv run python benchmarks/build_ecg_prepared.py before \
  --connectors "$ECG_CONNECTORS" --cache "$ECG_CACHE" \
  --work "$ECG_WORK" --output "$ECG_WORK/before-build.json"
uv run python benchmarks/build_ecg_prepared.py parent \
  --connectors "$ECG_CONNECTORS" --cache "$ECG_CACHE" \
  --work "$ECG_WORK" --layout record --compression lz4 --level 1 \
  --encoding dictionary --output "$ECG_WORK/parent-build.json"
uv run python benchmarks/build_ecg_prepared.py after \
  --connectors "$ECG_CONNECTORS" --cache "$ECG_CACHE" \
  --work "$ECG_WORK" --output "$ECG_WORK/after-build.json"
```

After the builds finish, run the consumer comparison without other CPU-intensive work.

```sh
for run in 1 2 3; do
  uv run python benchmarks/read_prepared.py read \
    --registry "$ECG_WORK/before" \
    --dataset physionet/ecg-qa-cot@1.0.0 \
    --output "$ECG_WORK/before-read-$run.json"
  uv run python benchmarks/read_prepared.py read \
    --registry "$ECG_WORK/after" \
    --dataset physionet/ecg-qa-cot@1.1.0 \
    --output "$ECG_WORK/after-read-$run.json"
  uv run python benchmarks/read_prepared.py compare \
    --before "$ECG_WORK/before-read-$run.json" \
    --after "$ECG_WORK/after-read-$run.json" \
    --output "$ECG_WORK/fidelity-$run.json"
done
```

For other configurations, use another empty `ECG_WORK` directory. Select `--layout packed`,
`--layout record`, or `--layout auto`, and pass the desired codec and encoding.
The benchmark files contain no dataset bodies. CI does not download this corpus or impose timing
thresholds.
