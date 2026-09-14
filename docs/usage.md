---
icon: lucide/plug
description: "Load a TimeNet dataset into pandas, polars, or PyTorch."
tags:
  - usage
  - pandas
  - polars
  - torch
---

# Load into your stack

Every dataset reads the same way: open a version, walk it in batches, and hand each batch to your
framework. Two consumer views ship with TimeNet, and both are thin wrappers over the same batched
reader calls.

- `timenet.pandas` delivers one record as a one-row frame whose cells hold whole value arrays.
- `timenet.torch` delivers one record as a dict of tensors.

Example status:

- [x] pandas: `TimeFPandasDataset` and `iter_record_frames`
- [x] polars: build a frame from the arrays `values_for()` returns
- [x] PyTorch: `TimeFTorchDataset` and `TimeFIterableStream`

=== "pandas"

    An array-cell frame is one row per record: `record_id`, `time_axis`, then one column per
    signal whose cell holds that signal's whole value array in the dtype it was stored in. Nothing
    pads, resamples, reindexes, or casts, so a 500 Hz ECG lead sits beside a 1 Hz temperature in
    the same row.

    ```python
    from timenet.client import TimeNet
    from timenet.pandas import TimeFPandasDataset

    with TimeNet("./local_registry").open("demo/hello-world") as reader:
        for record_id, frame in TimeFPandasDataset(reader).items():
            print(record_id, frame.columns.tolist())
    ```

    A record can name the same signal twice, at two rates or over two windows. Those need two
    columns, so a repeated name carries the signal's id as well: `I (record-000-lead-i)`.

=== "polars"

    polars has no array-cell idea, so build the frame from the arrays directly. One record's
    signals need not share a length, so pick the ones that belong in one frame rather than
    zipping all of them.

    ```python
    import polars as pl

    from timenet.client import TimeNet

    with TimeNet("./local_registry").open("demo/hello-world") as reader:
        record = reader.record("record-000")
        signals = record.signals()
        values = reader.values_for([s.signal_id for s in signals])
        frame = pl.DataFrame(
            {s.name: values[s.signal_id] for s in signals if s.name == "I"}
        )
    ```

=== "PyTorch"

    An item is one record: `record_id`, `series` as tensors that keep the stored dtype, plus
    `tasks`, `annotations`, and `series_masks`.

    ```python
    from torch.utils.data import DataLoader

    from timenet.client import TimeNet
    from timenet.torch import TimeFIterableStream

    # needs: pip install 'timenet[torch]'
    with TimeNet("./local_registry").open("demo/hello-world") as reader:
        stream = TimeFIterableStream(reader, batch_size=512)
        loader = DataLoader(
            stream,
            batch_size=8,
            collate_fn=lambda batch: [item["series"][0] for item in batch],
        )
        for batch in loader:
            ...
    ```

    Series lengths and dtypes differ between records, so a `DataLoader` that batches records needs
    a `collate_fn` of its own, or `batch_size=1`.

## Batch, do not loop

Both views hydrate a batch of records in one round of queries and read that batch's values in one
pass over the values plane. Reading a record at a time instead scans the control-plane tables per
record and re-decodes a row group per signal; on a 618,508-record corpus that is the difference
between 29 records a second and 15,540. A view of your own should keep the same shape:

```python
batch = list(itertools.islice(reader.iter_records(batch_size=512), 512))
wanted = [s.signal_id for record in batch for s in record.signals()]
values = reader.values_for(wanted)
```

## Splitting across workers

`TimeFIterableStream` reads the slice the reader partitions for the current `DataLoader` worker, so
`num_workers` workers together see every record exactly once. `TimeFPandasDataset` takes
`worker_index` and `num_workers` for the same split. The reader drops its database connection when
it is pickled into a worker and opens a new one there, so passing an open reader across the process
boundary is safe.

## Map-style access

`TimeFTorchDataset` addresses records by position, where position `i` is the record the control
plane gave surrogate id `i`. Use it when a sampler needs random access. A `DataLoader` with a batch
sampler calls `__getitems__`, which hydrates the whole batch in one round; plain `__getitem__` pays
a round of queries and a values pass for a single item.

## Windows

To train on windows rather than whole recordings, read the window instead of the recording.
`reader.values_window(signal_id, start, stop)` reads one, and `reader.values_windows(windows)` reads
many in one pass, coalescing the windows that share a row group.
