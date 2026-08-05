#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyarrow>=21",
#   "numpy",
#   "zarr>=3",
#   "h5py",
#   "pylance",
#   "matplotlib",
#   "wfdb",
#   "requests",
# ]
# ///
"""Storage-format evaluation for TimeF: two datasets, several formats, one script.

Two experiments (real data), each comparing on-disk formats for ragged float32 time series:

  1. tsqa   - ChengsenWang/TSQA, 48k z-normalized series (high-entropy floats).
  2. ecg    - PTB-XL 500 Hz 12-lead ECG (physionet/ecg-qa-cot source), quantized microvolts.

For each dataset it measures, on byte-identical data:
  - Parquet with three value encodings: BYTE_STREAM_SPLIT, plain, dictionary (all + zstd L3).
  - A format bake-off: best Parquet, Zarr v3, HDF5, Lance, Arrow IPC, raw CSR binary.
  - Random-access amplification vs Parquet row-group size, and the reader row-group-cache effect.

Usage:
  ./format_eval.py all                 # prepare data, run benchmarks, make plots
  ./format_eval.py prepare             # download + cache both datasets as .npy
  ./format_eval.py run                 # benchmark (needs cached data)
  ./format_eval.py plot                # plots + tables from results.json
  ./format_eval.py all --ecg-records 1200 --data-dir ./data --out ./results

Data notes: tsqa comes from the HuggingFace datasets-server parquet URL; ECG records come from
physionet.org. Both are public; a normal network connection is enough. Re-runs reuse the .npy cache.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------------------------------------------------------------------------------
# Faithful TimeF Parquet settings (mirror packages/timenet/src/timenet/writer/encodings.py)
# ------------------------------------------------------------------------------------------------
ZSTD_LEVEL = 3


def parquet_kwargs(encoding: str) -> dict:
    """Write options for the value column under one of: byte_stream_split | plain | dictionary."""
    base = dict(
        compression="zstd",
        compression_level=ZSTD_LEVEL,
        write_statistics=True,
        write_page_index=True,
        write_page_checksum=True,
        use_content_defined_chunking=True,
    )
    if encoding == "byte_stream_split":
        return {**base, "use_dictionary": False, "column_encoding": {"values.list.element": "BYTE_STREAM_SPLIT"}}
    if encoding == "plain":
        return {**base, "use_dictionary": False}
    if encoding == "dictionary":
        return {**base, "use_dictionary": True}
    raise ValueError(encoding)


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------
def dir_size(path: str | Path) -> int:
    """Total bytes of a file or directory tree."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def list_table(arrs: list[np.ndarray]) -> pa.Table:
    """One row per series, values as list<float32> (the TimeF shard layout)."""
    values = pa.array(np.concatenate(arrs) if arrs else np.empty(0, np.float32), type=pa.float32())
    offsets = np.zeros(len(arrs) + 1, dtype=np.int32)
    np.cumsum([len(a) for a in arrs], out=offsets[1:])
    return pa.table({"values": pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values)})


def timed(fn):
    """Run fn(), return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# ------------------------------------------------------------------------------------------------
# Data preparation (download + cache as flat .npy + int64 offsets)
# ------------------------------------------------------------------------------------------------
def prepare_tsqa(data_dir: Path) -> Path:
    """Download ChengsenWang/TSQA and cache flat float32 values + offsets."""
    out = data_dir / "tsqa"
    out.mkdir(parents=True, exist_ok=True)
    flat_p, off_p = out / "flat.npy", out / "offsets.npy"
    if flat_p.exists() and off_p.exists():
        print(f"  tsqa cache present: {flat_p}")
        return out
    import requests

    meta = requests.get(
        "https://datasets-server.huggingface.co/parquet?dataset=ChengsenWang/TSQA", timeout=60
    ).json()
    url = meta["parquet_files"][0]["url"]
    print(f"  downloading tsqa parquet: {url}")
    pf_path = out / "tsqa.parquet"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with pf_path.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    arrs = []
    for batch in pq.ParquetFile(pf_path).iter_batches(batch_size=65536, columns=["Series"]):
        for s in batch.column("Series").to_pylist():
            v = json.loads(s)
            channels = v if v and isinstance(v[0], list) else [v]
            arrs.extend(np.asarray(c, dtype=np.float32) for c in channels)
    _save_ragged(arrs, flat_p, off_p)
    return out


def prepare_ecg(data_dir: Path, n_records: int) -> Path:
    """Download PTB-XL 500 Hz records and cache flat float32 leads + offsets."""
    out = data_dir / "ecg"
    out.mkdir(parents=True, exist_ok=True)
    flat_p, off_p = out / "flat.npy", out / "offsets.npy"
    if flat_p.exists() and off_p.exists():
        print(f"  ecg cache present: {flat_p}")
        return out
    import requests
    import wfdb

    base = "https://physionet.org/files/ptb-xl/1.0.3/records500"
    rec_dir = out / "rec"
    rec_dir.mkdir(exist_ok=True)
    print(f"  downloading up to {n_records} PTB-XL records from physionet.org ...")
    got = 0
    for eid in range(1, n_records + 1):
        bucket = f"{eid // 1000 * 1000:05d}"
        stem = f"{eid:05d}_hr"
        ok = True
        for ext in ("hea", "dat"):
            dest = rec_dir / f"{stem}.{ext}"
            if dest.exists() and dest.stat().st_size:
                continue
            try:
                r = requests.get(f"{base}/{bucket}/{stem}.{ext}", timeout=60)
                if r.status_code != 200:
                    ok = False
                    break
                dest.write_bytes(r.content)
            except requests.RequestException:
                ok = False
                break
        got += ok
        if got % 100 == 0 and ok:
            print(f"    {got} records")
    arrs = []
    for hea in sorted(rec_dir.glob("*.hea")):
        base_path = str(hea)[:-4]
        if not Path(base_path + ".dat").exists():
            continue
        try:
            sig, _ = wfdb.rdsamp(base_path)  # physical mV, float64
        except Exception:
            continue
        sig = sig.astype(np.float32)
        arrs.extend(sig[:, ch] for ch in range(sig.shape[1]))
    _save_ragged(arrs, flat_p, off_p)
    return out


def _save_ragged(arrs: list[np.ndarray], flat_p: Path, off_p: Path) -> None:
    flat = np.concatenate(arrs).astype(np.float32)
    offsets = np.zeros(len(arrs) + 1, dtype=np.int64)
    np.cumsum([len(a) for a in arrs], out=offsets[1:])
    np.save(flat_p, flat)
    np.save(off_p, offsets)
    print(f"  saved {len(arrs):,} series, {flat.size:,} values ({flat.nbytes / 2**20:.1f} MiB) -> {flat_p.name}")


def load_dataset(ds_dir: Path):
    """Return (arrs, flat, offsets, stats) for a cached dataset."""
    flat = np.load(ds_dir / "flat.npy")
    offsets = np.load(ds_dir / "offsets.npy")
    arrs = [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    uniq = np.unique(flat)
    stats = dict(
        series=len(arrs),
        values=int(flat.size),
        raw_bytes=int(flat.nbytes),
        distinct=int(uniq.size),
        distinct_frac=float(uniq.size / flat.size),
    )
    return arrs, flat, offsets, stats


# ------------------------------------------------------------------------------------------------
# Format benchmarks
# ------------------------------------------------------------------------------------------------
def bench_formats(arrs, flat, offsets, work: Path, raw_bytes: int, rng: random.Random) -> dict:
    """Run the encoding experiment, the format bake-off, and the random-access analysis."""
    work.mkdir(parents=True, exist_ok=True)
    n = len(arrs)
    tbl = list_table(arrs)
    picks = {k: rng.sample(range(n), min(k, n)) for k in (100, 1000)}
    results: dict = {"encodings": {}, "formats": {}, "rowgroup_sweep": {}, "lru": {}}

    # --- 1. Parquet encoding experiment: BSS vs plain vs dictionary -----------------------------
    for enc in ("byte_stream_split", "plain", "dictionary"):
        p = work / f"enc_{enc}.parquet"
        _, wt = timed(lambda: pq.write_table(tbl, p, **parquet_kwargs(enc)))
        _, rt = timed(lambda: pq.read_table(p))
        results["encodings"][enc] = dict(bytes=dir_size(p), ratio=raw_bytes / dir_size(p), write_s=wt, read_s=rt)

    # --- 2. Format bake-off (best-representative config each) ------------------------------------
    best_pq_enc = min(results["encodings"], key=lambda e: results["encodings"][e]["bytes"])
    results["formats"]["parquet_best"] = {"config": best_pq_enc, **results["encodings"][best_pq_enc]}
    results["formats"]["parquet_bss"] = {"config": "byte_stream_split", **results["encodings"]["byte_stream_split"]}
    results["formats"]["zarr"] = _bench_zarr(flat, offsets, work, raw_bytes, picks)
    results["formats"]["hdf5"] = _bench_hdf5(arrs, flat, offsets, work, raw_bytes, picks)
    results["formats"]["lance"] = _bench_lance(arrs, work, raw_bytes, picks)
    results["formats"]["arrow_ipc"] = _bench_arrow_ipc(tbl, work, raw_bytes)
    results["formats"]["raw_csr"] = _bench_raw(flat, offsets, work, raw_bytes, picks)

    # --- 3. Random-access amplification vs Parquet row-group size --------------------------------
    for tgt, label in [(256 * 1024, "256KiB"), (1 << 20, "1MiB"), (4 << 20, "4MiB"), (16 << 20, "16MiB")]:
        placements, rg_comp, nrg, store_bytes = _build_store(arrs, tgt, work / f"store_{label}.parquet")
        row = {"row_groups": nrg, "store_bytes": store_bytes}
        for k, sel in picks.items():
            touched = {placements[i][0] for i in sel}
            read = sum(rg_comp[g] for g in touched)
            target = sum(arrs[i].nbytes for i in sel)
            row[f"amp_k{k}"] = read / target
            row[f"rg_touched_k{k}"] = len(touched)
        results["rowgroup_sweep"][label] = row

    # --- 4. Reader row-group-cache (LRU) effect at 4 MiB -----------------------------------------
    placements, rg_comp, nrg, _ = _build_store(arrs, 4 << 20, work / "store_lru.parquet")
    pf = pq.ParquetFile(work / "store_lru.parquet")
    sel = picks[1000]
    distinct = len({placements[i][0] for i in sel})
    for cap in (16, distinct + 1):
        cache: OrderedDict = OrderedDict()
        phys = 0
        t0 = time.perf_counter()
        for i in sel:
            g = placements[i][0]
            if g in cache:
                cache.move_to_end(g)
                continue
            pf.read_row_group(g, columns=["values"])
            phys += 1
            cache[g] = 1
            while len(cache) > cap:
                cache.popitem(last=False)
        results["lru"][f"cache_{cap}"] = dict(physical_reads=phys, wall_ms=(time.perf_counter() - t0) * 1000)
    results["lru"]["distinct_row_groups"] = distinct
    return results


def _build_store(arrs, rg_target_bytes, path, encoding="byte_stream_split"):
    """TimeF-faithful store: byte-based row groups, one row per series.

    Returns (placements, rg_comp, n_row_groups, store_bytes) where placements[i] = (row_group,
    offset_within_row_group), matching how the real reader addresses a series.
    """
    tbl_schema = list_table(arrs[:1]).schema
    writer = pq.ParquetWriter(path, tbl_schema, **parquet_kwargs(encoding))
    buf, buf_bytes, rg, placements = [], 0, 0, {}

    def flush():
        nonlocal buf, buf_bytes, rg
        if not buf:
            return
        writer.write_table(list_table([arrs[i] for i in buf]))
        for offset, i in enumerate(buf):
            placements[i] = (rg, offset)
        rg += 1
        buf, buf_bytes = [], 0

    for i in range(len(arrs)):
        buf.append(i)
        buf_bytes += arrs[i].nbytes
        if buf_bytes >= rg_target_bytes:
            flush()
    flush()
    writer.close()
    md = pq.ParquetFile(path).metadata
    rg_comp = [
        sum(md.row_group(g).column(c).total_compressed_size for c in range(md.num_columns))
        for g in range(md.num_row_groups)
    ]
    return placements, rg_comp, md.num_row_groups, dir_size(path)


def _bench_zarr(flat, offsets, work, raw_bytes, picks):
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    p = work / "data.zarr"
    if p.exists():
        shutil.rmtree(p)
    chunk = 262144  # 1 MiB of float32
    arr = zarr.create_array(
        store=str(p), shape=(flat.size,), chunks=(chunk,), dtype="float32",
        compressors=[ZstdCodec(level=ZSTD_LEVEL)], serializer=BytesCodec(),
    )
    _, wt = timed(lambda: arr.__setitem__(slice(None), flat))
    _, rt = timed(lambda: arr[:])
    size = dir_size(p)
    n_units = (flat.size + chunk - 1) // chunk
    avg = size / n_units
    amp = {}
    for k, sel in picks.items():
        units = set()
        for i in sel:
            units.update(range(offsets[i] // chunk, (offsets[i + 1] - 1) // chunk + 1))
        target = sum((offsets[i + 1] - offsets[i]) for i in sel) * 4
        amp[f"amp_k{k}"] = len(units) * avg / target
    return dict(bytes=size, ratio=raw_bytes / size, write_s=wt, read_s=rt, config="CSR 1MiB chunks zstd", **amp)


def _bench_hdf5(arrs, flat, offsets, work, raw_bytes, picks):
    import h5py

    p = work / "data.h5"
    chunk = 262144
    def write():
        with h5py.File(p, "w") as f:
            d = f.create_dataset("values", shape=(flat.size,), dtype="float32",
                                 chunks=(chunk,), compression="gzip", compression_opts=4, shuffle=True)
            d[:] = flat
            f.create_dataset("offsets", data=offsets)
    _, wt = timed(write)
    def read():
        with h5py.File(p, "r") as f:
            return f["values"][:]
    _, rt = timed(read)
    size = dir_size(p)
    n_units = (flat.size + chunk - 1) // chunk
    avg = size / n_units
    amp = {}
    for k, sel in picks.items():
        units = set()
        for i in sel:
            units.update(range(offsets[i] // chunk, (offsets[i + 1] - 1) // chunk + 1))
        target = sum((offsets[i + 1] - offsets[i]) for i in sel) * 4
        amp[f"amp_k{k}"] = len(units) * avg / target
    return dict(bytes=size, ratio=raw_bytes / size, write_s=wt, read_s=rt, config="CSR gzip4+shuffle 1MiB", **amp)


def _bench_lance(arrs, work, raw_bytes, picks):
    import lance

    p = work / "data.lance"
    if p.exists():
        shutil.rmtree(p)
    tbl = list_table(arrs)
    _, wt = timed(lambda: lance.write_dataset(tbl, str(p)))
    ds = lance.dataset(str(p))
    _, rt = timed(lambda: ds.to_table())
    size = dir_size(p)
    take = {}
    for k, sel in picks.items():
        _, tt = timed(lambda: ds.take(sel, columns=["values"]))
        take[f"take_ms_k{k}"] = tt * 1000
    return dict(bytes=size, ratio=raw_bytes / size, write_s=wt, read_s=rt, config="list<float32> zstd", **take)


def _bench_arrow_ipc(tbl, work, raw_bytes):
    import pyarrow.ipc as ipc

    p = work / "data.arrow"
    opts = ipc.IpcWriteOptions(compression="zstd")

    def write():
        with ipc.new_file(str(p), tbl.schema, options=opts) as w:
            w.write_table(tbl)

    def read():
        with ipc.open_file(str(p)) as r:
            return r.read_all()

    _, wt = timed(write)
    _, rt = timed(read)
    return dict(bytes=dir_size(p), ratio=raw_bytes / dir_size(p), write_s=wt, read_s=rt, config="Arrow IPC zstd")


def _bench_raw(flat, offsets, work, raw_bytes, picks):
    p = work / "data.f32"
    _, wt = timed(lambda: flat.tofile(p))
    np.save(work / "raw_offsets.npy", offsets)
    _, rt = timed(lambda: np.fromfile(p, dtype=np.float32))
    size = p.stat().st_size + (work / "raw_offsets.npy").stat().st_size
    # mmap random access: O(1) seek, no decode, ~1x amplification
    mm = np.memmap(p, dtype=np.float32, mode="r")
    amp = {}
    for k, sel in picks.items():
        _, tt = timed(lambda: [np.array(mm[offsets[i] : offsets[i + 1]]) for i in sel])
        amp[f"mmap_ms_k{k}"] = tt * 1000
        amp[f"amp_k{k}"] = 1.0
    return dict(bytes=size, ratio=raw_bytes / size, write_s=wt, read_s=rt, config="flat f32 + offsets (mmap)", **amp)


# ------------------------------------------------------------------------------------------------
# Read-timing: random single-series, random batch (gather), sequential scan
# ------------------------------------------------------------------------------------------------
def _median_secs(fn, trials=3):
    """Median wall time of fn() over `trials` runs (warm cache)."""
    return sorted(timed(fn)[1] for _ in range(trials))[trials // 2]


def _timing_row(rand, batch, scan, *, k_single, n_batches, batch_size, raw_bytes, size):
    rs, bs, ss = _median_secs(rand), _median_secs(batch), _median_secs(scan)
    return dict(
        size_mb=size / 2**20,
        rand_us_per_series=rs / k_single * 1e6,
        batch_ms_per_batch=bs / n_batches * 1000,
        batch_us_per_series=bs / (n_batches * batch_size) * 1e6,
        scan_ms=ss * 1000,
        scan_mbps=(raw_bytes / 2**20) / ss,
    )


def bench_timing(arrs, flat, offsets, work: Path, raw_bytes: int, rng: random.Random) -> dict:
    """Time three read patterns per format: random single-series, random batch of 64, full scan.

    Warm-cache (the OS page cache cannot be dropped in-process), median of 3. Each format uses its
    native single-item / gather / scan API. Parquet random reads decode a whole row group per series
    with no cross-read reuse (the worst-case point-access cost); Parquet batch reads coalesce a batch's
    series by row group so a shared group decodes once.
    """
    work.mkdir(parents=True, exist_ok=True)
    n = len(arrs)
    k = min(1000, n)
    randset = rng.sample(range(n), k)
    bsize = 64
    n_batches = min(16, n // bsize)
    batches = [rng.sample(range(n), bsize) for _ in range(n_batches)]
    common = dict(k_single=k, n_batches=n_batches, batch_size=bsize, raw_bytes=raw_bytes)
    out: dict = {}

    # Parquet at three row-group sizes (dictionary encoding: recommended + fast decode)
    for label, tgt in [("Parquet 256KiB", 256 * 1024), ("Parquet 1MiB", 1 << 20), ("Parquet 4MiB", 4 << 20)]:
        p = work / f"t_{label.replace(' ', '_')}.parquet"
        placements, _, _, size = _build_store(arrs, tgt, p, encoding="dictionary")
        pf = pq.ParquetFile(p)

        def rand(pf=pf, placements=placements):
            for i in randset:
                rg, off = placements[i]
                pf.read_row_group(rg, columns=["values"]).column("values")[off]

        def batch(pf=pf, placements=placements):
            for bset in batches:
                by_rg: dict[int, list[int]] = {}
                for i in bset:
                    rg, off = placements[i]
                    by_rg.setdefault(rg, []).append(off)
                for rg, offs in by_rg.items():
                    col = pf.read_row_group(rg, columns=["values"]).column("values")
                    for off in offs:
                        col[off]

        out[label] = _timing_row(rand, batch, lambda p=p: pq.read_table(p, columns=["values"]),
                                 size=size, **common)

    # Zarr (CSR, 1 MiB chunks)
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    zp = work / "t.zarr"
    if zp.exists():
        shutil.rmtree(zp)
    za = zarr.create_array(store=str(zp), shape=(flat.size,), chunks=(262144,), dtype="float32",
                           compressors=[ZstdCodec(level=ZSTD_LEVEL)], serializer=BytesCodec())
    za[:] = flat
    out["Zarr 1MiB"] = _timing_row(
        lambda: [np.asarray(za[offsets[i]:offsets[i + 1]]) for i in randset],
        lambda: [np.asarray(za[offsets[i]:offsets[i + 1]]) for b in batches for i in b],
        lambda: za[:], size=dir_size(zp), **common)

    # HDF5 (CSR, 1 MiB chunks, gzip+shuffle), handle kept open
    import h5py

    hp = work / "t.h5"
    with h5py.File(hp, "w") as f:
        f.create_dataset("values", data=flat, chunks=(262144,), compression="gzip",
                         compression_opts=4, shuffle=True)
    hf = h5py.File(hp, "r")
    hd = hf["values"]
    out["HDF5 1MiB"] = _timing_row(
        lambda: [hd[offsets[i]:offsets[i + 1]] for i in randset],
        lambda: [hd[offsets[i]:offsets[i + 1]] for b in batches for i in b],
        lambda: hd[:], size=dir_size(hp), **common)
    hf.close()

    # Lance (native take / gather)
    import lance

    lp = work / "t.lance"
    if lp.exists():
        shutil.rmtree(lp)
    lance.write_dataset(list_table(arrs), str(lp))
    ds = lance.dataset(str(lp))
    out["Lance"] = _timing_row(
        lambda: [ds.take([i], columns=["values"]) for i in randset],
        lambda: [ds.take(b, columns=["values"]) for b in batches],
        lambda: ds.to_table(columns=["values"]), size=dir_size(lp), **common)

    # Raw CSR float32 + offsets, mmap (the floor)
    rp = work / "t.f32"
    flat.tofile(rp)
    mm = np.memmap(rp, dtype=np.float32, mode="r")
    out["raw mmap"] = _timing_row(
        lambda: [np.array(mm[offsets[i]:offsets[i + 1]]) for i in randset],
        lambda: [np.array(mm[offsets[i]:offsets[i + 1]]) for b in batches for i in b],
        lambda: np.array(mm[:]), size=rp.stat().st_size, **common)
    return out


# Chunk / row-group sizes (bytes of float32) swept in the matched Parquet-vs-Zarr comparison.
GRANULARITY = [(128 * 1024, "128KiB"), (256 * 1024, "256KiB"), (512 * 1024, "512KiB"),
               (1 << 20, "1MiB"), (2 << 20, "2MiB"), (4 << 20, "4MiB")]


def bench_pq_vs_zarr(arrs, flat, offsets, work: Path, rng: random.Random) -> dict:
    """Random single-series latency for Parquet vs Zarr at matched granularity.

    Both formats read one series by decoding the row group / chunk that holds it, so the fair comparison
    fixes the row-group target (Parquet) equal to the chunk size (Zarr). Warm cache, median of 3.
    """
    work.mkdir(parents=True, exist_ok=True)
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    n = len(arrs)
    randset = rng.sample(range(n), min(1000, n))
    rows = {"parquet": {}, "zarr": {}}
    for nbytes, label in GRANULARITY:
        # Parquet: row-group target = nbytes, dictionary encoding
        p = work / f"c_{label}.parquet"
        placements, _, nrg, _ = _build_store(arrs, nbytes, p, encoding="dictionary")
        pf = pq.ParquetFile(p)

        def pq_read(pf=pf, placements=placements):
            for i in randset:
                rg, off = placements[i]
                pf.read_row_group(rg, columns=["values"]).column("values")[off]

        rows["parquet"][label] = _median_secs(pq_read) / len(randset) * 1e6

        # Zarr: chunk = nbytes/4 float32 values
        zp = work / f"c_{label}.zarr"
        if zp.exists():
            shutil.rmtree(zp)
        za = zarr.create_array(store=str(zp), shape=(flat.size,), chunks=(nbytes // 4,), dtype="float32",
                               compressors=[ZstdCodec(level=ZSTD_LEVEL)], serializer=BytesCodec())
        za[:] = flat

        def z_read(za=za):
            for i in randset:
                np.asarray(za[offsets[i]:offsets[i + 1]])

        rows["zarr"][label] = _median_secs(z_read) / len(randset) * 1e6
        p.unlink(missing_ok=True)
        shutil.rmtree(zp, ignore_errors=True)
    return rows


# Row-group sizes swept when choosing a default, and the shard size the footer cost is normalised to.
ROWGROUP_SIZES = [(256 * 1024, "256KiB"), (512 * 1024, "512KiB"), (1 << 20, "1MiB"),
                  (2 << 20, "2MiB"), (4 << 20, "4MiB")]
SHARD_TARGET_BYTES = 128 << 20  # matches DEFAULT_SHARD_TARGET_BYTES


def _shard_table(arrs, idxs):
    """A realistic TimeF shard table (8 columns) so footer size reflects the real schema."""
    m = len(idxs)
    vals = pa.array(np.concatenate([arrs[i] for i in idxs]), type=pa.float32())
    o = np.zeros(m + 1, dtype=np.int32)
    np.cumsum([len(arrs[i]) for i in idxs], out=o[1:])
    return pa.table({
        "time_series_id": pa.array([f"ecg-{i}-lead{i % 12}" for i in idxs]),
        "spec_type": pa.array(["ecg"] * m),
        "channel": pa.array([f"c{i % 12}" for i in idxs]),
        "chunk_idx": pa.array(np.zeros(m, dtype=np.int32)),
        "t_start_s": pa.array(np.zeros(m, dtype=np.float64)),
        "n_values": pa.array([len(arrs[i]) for i in idxs], type=pa.int32()),
        "sampling_rate_hz": pa.array(np.full(m, 500.0)),
        "values": pa.ListArray.from_arrays(pa.array(o, type=pa.int32()), vals),
    })


def bench_rowgroup(arrs, work: Path, rng: random.Random, encoding: str = "dictionary") -> dict:
    """Sweep row-group target size on the real 8-column shard schema, per-dataset value encoding.

    Returns per size: on-disk size, size penalty vs the smallest, K=100 read amplification, random
    single-series latency, footer size, and footer per 128 MiB shard (the S3 single-GET-open budget is
    64 KB). This is the row-group-size tradeoff.
    """
    work.mkdir(parents=True, exist_ok=True)
    n = len(arrs)
    randset = rng.sample(range(n), min(1000, n))
    kw = parquet_kwargs(encoding)
    rows = {}
    for tgt, label in ROWGROUP_SIZES:
        p = work / f"rg_{label}.parquet"
        writer = pq.ParquetWriter(p, _shard_table(arrs, [0]).schema, **kw)
        buf, bb, rg, place = [], 0, 0, {}

        def flush():
            nonlocal buf, bb, rg
            if not buf:
                return
            writer.write_table(_shard_table(arrs, buf))
            for off, i in enumerate(buf):
                place[i] = (rg, off)
            rg += 1
            buf, bb = [], 0

        for i in range(n):
            buf.append(i)
            bb += arrs[i].nbytes
            if bb >= tgt:
                flush()
        flush()
        writer.close()
        md = pq.ParquetFile(p).metadata
        nrg = md.num_row_groups
        rg_comp = [sum(md.row_group(g).column(c).total_compressed_size for c in range(md.num_columns))
                   for g in range(nrg)]
        pf = pq.ParquetFile(p)

        def rand(pf=pf, place=place):
            for i in randset:
                r, off = place[i]
                pf.read_row_group(r, columns=["values"]).column("values")[off]

        sel = randset[:100]
        amp = sum(rg_comp[g] for g in {place[i][0] for i in sel}) / sum(arrs[i].nbytes for i in sel)
        footer_per_rg = md.serialized_size / max(nrg, 1)
        rgs_per_shard = max(1, SHARD_TARGET_BYTES // tgt)
        footer_per_shard = footer_per_rg * min(nrg, rgs_per_shard)
        rows[label] = dict(
            size_mb=dir_size(p) / 2**20,
            n_row_groups=nrg,
            amp_k100=amp,
            random_us=_median_secs(rand) / len(randset) * 1e6,
            footer_kb=md.serialized_size / 1024,
            footer_per_shard_kb=footer_per_shard / 1024,
            single_get_open=footer_per_shard < 65536,
        )
        p.unlink(missing_ok=True)
    smallest = min(r["size_mb"] for r in rows.values())
    for r in rows.values():
        r["size_pct_vs_min"] = (r["size_mb"] / smallest - 1) * 100
    return rows


# ------------------------------------------------------------------------------------------------
# Plots (dataviz method: validated categorical palette, direct labels, recessive grid, one axis)
# ------------------------------------------------------------------------------------------------
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
ENC_COLOR = {"byte_stream_split": "#2a78d6", "plain": "#1baf7a", "dictionary": "#eda100"}
ENC_LABEL = {"byte_stream_split": "BYTE_STREAM_SPLIT", "plain": "plain", "dictionary": "dictionary"}
BAR = "#2a78d6"
K_COLOR = {"amp_k100": "#2a78d6", "amp_k1000": "#eda100"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_plots(results: dict, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 130})
    datasets = list(results.keys())

    # Fig 1: encoding experiment, size relative to plain+zstd (lower is better), grouped by dataset
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _style(ax)
    encs = ["byte_stream_split", "plain", "dictionary"]
    x = np.arange(len(datasets))
    w = 0.26
    for j, enc in enumerate(encs):
        vals, labels = [], []
        for ds in datasets:
            e = results[ds]["encodings"]
            rel = e[enc]["bytes"] / e["plain"]["bytes"]
            vals.append(rel)
            labels.append(f"{e[enc]['bytes'] / 2**20:.0f}MB")
        bars = ax.bar(x + (j - 1) * w, vals, w, color=ENC_COLOR[enc], label=ENC_LABEL[enc], zorder=3)
        for b, lab in zip(bars, labels):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, lab, ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(ax.get_xlim()[1], 1.0, " plain+zstd baseline", va="center", ha="left", fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("on-disk size relative to plain+zstd")
    ax.set_title("Value-column encoding: BSS wins on continuous data, loses on quantized ECG",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig1_encoding.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: format bake-off, on-disk size (MB), small multiples per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(9.6, 4.4), squeeze=False)
    order = ["raw_csr", "arrow_ipc", "parquet_bss", "zarr", "hdf5", "parquet_best", "lance"]
    nice = {"raw_csr": "raw CSR", "arrow_ipc": "Arrow IPC", "parquet_bss": "Parquet BSS", "zarr": "Zarr",
            "hdf5": "HDF5", "parquet_best": "Parquet best", "lance": "Lance"}
    for ci, ds in enumerate(datasets):
        ax = axes[0][ci]
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        f = results[ds]["formats"]
        items = [(nice[k], f[k]["bytes"] / 2**20, f[k].get("config", "")) for k in order if k in f]
        items.sort(key=lambda t: t[1])
        names = [t[0] for t in items]
        vals = [t[1] for t in items]
        y = np.arange(len(names))
        ax.barh(y, vals, color=BAR, height=0.62, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.01, yi, f"{v:.0f}", va="center", ha="left", fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9, color=INK)
        ax.set_xlabel("on-disk size (MB)")
        raw_mb = results[ds]["stats"]["raw_bytes"] / 2**20
        ax.set_title(f"{ds.upper()}  ({raw_mb:.0f} MB raw)", color=INK, fontsize=10, loc="left")
    fig.suptitle("Format bake-off: on-disk size (lower is better)", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig2_formats.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: random-access amplification vs Parquet row-group size, small multiples per dataset.
    # Independent y-axes: amplification is a within-dataset ratio and the tiny-series dataset (TSQA)
    # reaches ~400x while ECG tops out near 64x; a shared axis would crush the ECG detail.
    fig, axes = plt.subplots(1, len(datasets), figsize=(9.6, 4.2), squeeze=False, sharey=False)
    labels = ["256KiB", "1MiB", "4MiB", "16MiB"]
    xs = np.arange(len(labels))
    for ci, ds in enumerate(datasets):
        ax = axes[0][ci]
        _style(ax)
        sweep = results[ds]["rowgroup_sweep"]
        for kkey, kname in [("amp_k100", "read 100 series"), ("amp_k1000", "read 1000 series")]:
            ys = [sweep[l][kkey] for l in labels]
            ax.plot(xs, ys, "-o", color=K_COLOR[kkey], linewidth=2, markersize=6, label=kname, zorder=3)
            ax.text(xs[-1], ys[-1], f"  {ys[-1]:.0f}x", va="center", ha="left", fontsize=8, color=K_COLOR[kkey])
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Parquet row-group target size")
        ax.set_ylabel("read amplification (bytes read / bytes needed)")
        ax.set_title(f"{ds.upper()}", color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Random single-series access: smaller row groups cut amplification for sparse reads",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig3_random_access.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig1_encoding.png, fig2_formats.png, fig3_random_access.png to {out}")


def make_timing_plots(timing: dict, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 130})
    datasets = list(timing.keys())
    order = ["raw mmap", "Lance", "Parquet 256KiB", "Zarr 1MiB", "Parquet 1MiB", "HDF5 1MiB", "Parquet 4MiB"]

    # Fig 4: per-series read latency, random single vs batched gather (same unit, us/series), log y
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.4, 4.6), squeeze=False)
    for ci, ds in enumerate(datasets):
        ax = axes[0][ci]
        _style(ax)
        f = timing[ds]["formats"]
        names = [k for k in order if k in f]
        x = np.arange(len(names))
        w = 0.4
        for j, (metric, color, lab) in enumerate(
            [("rand_us_per_series", "#2a78d6", "random single"), ("batch_us_per_series", "#1baf7a", "batch of 64")]
        ):
            vals = [f[k][metric] for k in names]
            bars = ax.bar(x + (j - 0.5) * w, vals, w, color=color, label=lab, zorder=3)
            for b, v in zip(bars, vals):
                txt = f"{v:.1f}" if v < 100 else f"{v:.0f}"
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08, txt, ha="center", va="bottom",
                        fontsize=7.5, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.5, color=INK)
        ax.set_ylabel("read latency (us per series, log)" if ci == 0 else "")
        ax.set_title(f"{ds.upper()}", color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Random single-series vs batched read latency (lower is better)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig4_read_latency.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 5: sequential scan throughput (MB/s, higher is better)
    fig, axes = plt.subplots(1, len(datasets), figsize=(9.6, 4.4), squeeze=False)
    for ci, ds in enumerate(datasets):
        ax = axes[0][ci]
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        f = timing[ds]["formats"]
        items = sorted([(k, f[k]["scan_mbps"]) for k in order if k in f], key=lambda t: t[1])
        names = [t[0] for t in items]
        vals = [t[1] for t in items]
        y = np.arange(len(names))
        ax.barh(y, vals, color=BAR, height=0.62, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.01, yi, f"{v:,.0f}", va="center", ha="left", fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9, color=INK)
        ax.set_xlabel("sequential scan throughput (MB/s)")
        ax.set_title(f"{ds.upper()}", color=INK, fontsize=10, loc="left")
    fig.suptitle("Sequential full-scan throughput (higher is better)", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig5_scan_throughput.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig4_read_latency.png, fig5_scan_throughput.png to {out}")


def make_compare_plot(compare: dict, out: Path):
    """Fig 6: Parquet vs Zarr random single-series latency at matched granularity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 130})
    datasets = list(compare.keys())
    labels = [lab for _, lab in GRANULARITY]
    xs = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.0, 4.4), squeeze=False)
    for ci, ds in enumerate(datasets):
        ax = axes[0][ci]
        _style(ax)
        for fmt, color in [("parquet", "#2a78d6"), ("zarr", "#eda100")]:
            ys = [compare[ds][fmt][lab] for lab in labels]
            ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6, label=fmt.capitalize(), zorder=3)
            ax.text(xs[0], ys[0], f"{ys[0]:.0f} ", va="center", ha="right", fontsize=8, color=color)
            ax.text(xs[-1], ys[-1], f" {ys[-1]:.0f}", va="center", ha="left", fontsize=8, color=color)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.set_xlabel("row-group (Parquet) / chunk (Zarr) size")
        ax.set_ylabel("random read latency (us/series)" if ci == 0 else "")
        ax.set_title(f"{ds.upper()}", color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Parquet vs Zarr random single-series latency, matched granularity (lower is better)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig6_parquet_vs_zarr.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig6_parquet_vs_zarr.png to {out}")


def make_rowgroup_plot(rowgroup: dict, out: Path):
    """Fig 7: the row-group size tradeoff (ECG) - faster random reads vs larger files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 130})
    ds = "ecg" if "ecg" in rowgroup else next(iter(rowgroup))
    data = rowgroup[ds]
    labels = [lab for _, lab in ROWGROUP_SIZES]
    xs = np.arange(len(labels))
    default_i = labels.index("512KiB")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for ax, key, color, ylab, title in [
        (ax1, "random_us", "#2a78d6", "random read latency (us/series)", "Benefit: faster random reads"),
        (ax2, "size_pct_vs_min", "#eda100", "on-disk size penalty (%)", "Cost: larger files"),
    ]:
        _style(ax)
        ys = [data[lab][key] for lab in labels]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6, zorder=3)
        ax.plot([xs[default_i]], [ys[default_i]], "o", color="#0b0b0b", markersize=11, zorder=4,
                markerfacecolor="none", markeredgewidth=2)
        for x, y in zip(xs, ys):
            ax.text(x, y * 1.08 if key == "random_us" else y + max(ys) * 0.03,
                    f"{y:.0f}" if y >= 10 else f"{y:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        if key == "random_us":
            ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.set_xlabel("row-group target size")
        ax.set_ylabel(ylab)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.suptitle(f"Row-group size tradeoff ({ds.upper()}); circled = 512 KiB default",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig7_rowgroup_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig7_rowgroup_tradeoff.png to {out}")


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def cmd_prepare(args):
    data_dir = Path(args.data_dir)
    print("Preparing tsqa ...")
    prepare_tsqa(data_dir)
    print("Preparing ecg ...")
    prepare_ecg(data_dir, args.ecg_records)


def cmd_run(args):
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in ("tsqa", "ecg"):
        ds_dir = data_dir / ds
        if not (ds_dir / "flat.npy").exists():
            sys.exit(f"missing cached data for {ds!r}; run `prepare` first")
        arrs, flat, offsets, stats = load_dataset(ds_dir)
        print(f"\n=== {ds.upper()}: {stats['series']:,} series, {stats['values']:,} values, "
              f"{stats['raw_bytes'] / 2**20:.1f} MiB raw, distinct={stats['distinct']:,} "
              f"({stats['distinct_frac'] * 100:.3f}%) ===")
        rng = random.Random(42)
        res = bench_formats(arrs, flat, offsets, out / f"work_{ds}", stats["raw_bytes"], rng)
        res["stats"] = stats
        results[ds] = res
        _print_summary(ds, res)
        shutil.rmtree(out / f"work_{ds}", ignore_errors=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out / 'results.json'}")
    _write_csv(results, out / "results_formats.csv")


def cmd_timing(args):
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timing = {}
    compare = {}
    for ds in ("tsqa", "ecg"):
        ds_dir = data_dir / ds
        if not (ds_dir / "flat.npy").exists():
            sys.exit(f"missing cached data for {ds!r}; run `prepare` first")
        arrs, flat, offsets, stats = load_dataset(ds_dir)
        print(f"\n=== {ds.upper()} read timing (warm cache, median of 3): "
              f"{stats['series']:,} series, {stats['raw_bytes'] / 2**20:.1f} MiB raw ===")
        rng = random.Random(42)
        work = out / f"twork_{ds}"
        rows = bench_timing(arrs, flat, offsets, work, stats["raw_bytes"], rng)
        timing[ds] = {"stats": stats, "formats": rows}
        print(f"  {'format':16s} {'random us/series':>16s} {'batch ms/64':>12s} {'scan MB/s':>10s} {'size MB':>8s}")
        for name, r in rows.items():
            print(f"  {name:16s} {r['rand_us_per_series']:16.1f} {r['batch_ms_per_batch']:12.2f} "
                  f"{r['scan_mbps']:10.0f} {r['size_mb']:8.0f}")
        compare[ds] = bench_pq_vs_zarr(arrs, flat, offsets, work, random.Random(42))
        print("  Parquet vs Zarr random us/series at matched granularity:")
        for _, lab in GRANULARITY:
            print(f"    {lab:7s}: parquet={compare[ds]['parquet'][lab]:8.1f}  zarr={compare[ds]['zarr'][lab]:8.1f}")
        shutil.rmtree(work, ignore_errors=True)
    (out / "compare_pq_zarr.json").write_text(json.dumps(compare, indent=2))
    make_compare_plot(compare, out)
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    rows = ["dataset,format,size_mb,random_us_per_series,batch_ms_per_batch,batch_us_per_series,scan_ms,scan_mbps"]
    for ds, blk in timing.items():
        for name, r in blk["formats"].items():
            rows.append(f"{ds},{name},{r['size_mb']:.1f},{r['rand_us_per_series']:.1f},"
                        f"{r['batch_ms_per_batch']:.2f},{r['batch_us_per_series']:.1f},{r['scan_ms']:.1f},{r['scan_mbps']:.0f}")
    (out / "timing.csv").write_text("\n".join(rows) + "\n")
    print(f"\nwrote {out / 'timing.json'} and {out / 'timing.csv'}")
    make_timing_plots(timing, out)


def cmd_rowgroup(args):
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rowgroup = {}
    for ds in ("tsqa", "ecg"):
        ds_dir = data_dir / ds
        if not (ds_dir / "flat.npy").exists():
            sys.exit(f"missing cached data for {ds!r}; run `prepare` first")
        arrs, _, _, stats = load_dataset(ds_dir)
        encoding = "byte_stream_split" if stats["distinct_frac"] > 0.5 else "dictionary"
        print(f"\n=== {ds.upper()} row-group tradeoff ({encoding}, 8-column shard schema) ===")
        work = out / f"rgwork_{ds}"
        rg = bench_rowgroup(arrs, work, random.Random(42), encoding=encoding)
        shutil.rmtree(work, ignore_errors=True)
        rowgroup[ds] = rg
        print(f"  {'rg':>7} {'nrg':>5} {'size_MB':>8} {'vs_min':>7} {'amp_K100':>9} {'random_us':>10} "
              f"{'footer/shard_KB':>16} {'1-GET open':>11}")
        for lab, r in rg.items():
            print(f"  {lab:>7} {r['n_row_groups']:5d} {r['size_mb']:8.1f} {r['size_pct_vs_min']:+6.1f}% "
                  f"{r['amp_k100']:8.1f}x {r['random_us']:10.0f} {r['footer_per_shard_kb']:16.0f} "
                  f"{'yes' if r['single_get_open'] else 'no':>11}")
    (out / "rowgroup.json").write_text(json.dumps(rowgroup, indent=2))
    make_rowgroup_plot(rowgroup, out)
    print(f"\nwrote {out / 'rowgroup.json'}")


def cmd_plot(args):
    out = Path(args.out)
    results = json.loads((out / "results.json").read_text())
    make_plots(results, out)


def cmd_all(args):
    cmd_prepare(args)
    cmd_run(args)
    cmd_plot(args)
    cmd_timing(args)
    cmd_rowgroup(args)


def _print_summary(ds, res):
    print(f"  encodings (list<float32>, zstd L3):")
    plain = res["encodings"]["plain"]["bytes"]
    for enc, r in res["encodings"].items():
        print(f"    {ENC_LABEL[enc]:20s} {r['bytes'] / 2**20:7.1f} MB  {r['ratio']:.2f}x raw  "
              f"{(r['bytes'] / plain - 1) * 100:+5.1f}% vs plain")
    print("  format sizes: " + "  ".join(
        f"{k}={v['bytes'] / 2**20:.0f}MB" for k, v in res["formats"].items()))
    lru = res["lru"]
    full_key = f"cache_{lru['distinct_row_groups'] + 1}"
    c16, cfull = lru["cache_16"], lru[full_key]
    print(f"  LRU (K=1000, 4MiB): cache16={c16['physical_reads']} reads / {c16['wall_ms']:.0f}ms  vs  "
          f"cache_full={cfull['physical_reads']} reads / {cfull['wall_ms']:.0f}ms")


def _write_csv(results, path):
    rows = ["dataset,format,config,size_mb,ratio_vs_raw,write_s,read_s,amp_k100"]
    for ds, res in results.items():
        for name, f in res["formats"].items():
            rows.append(f"{ds},{name},{f.get('config', '')},{f['bytes'] / 2**20:.1f},"
                        f"{f['ratio']:.2f},{f['write_s']:.2f},{f['read_s']:.2f},{f.get('amp_k100', '')}")
    path.write_text("\n".join(rows) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "timing", "rowgroup", "plot", "all"])
    ap.add_argument("--data-dir", default="./data", help="where datasets are cached")
    ap.add_argument("--out", default="./results", help="where results.json, CSV and plots go")
    ap.add_argument("--ecg-records", type=int, default=1200, help="max PTB-XL record ids to fetch")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "timing": cmd_timing, "rowgroup": cmd_rowgroup,
     "plot": cmd_plot, "all": cmd_all}[args.command](args)


if __name__ == "__main__":
    main()
