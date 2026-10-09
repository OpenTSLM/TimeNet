"""Shared Hugging Face download and conversion support for ChatTS."""

from abc import ABC
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.types import AnswerTask, Split, TimeSeriesSpec
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector


_SPEC = TimeSeriesSpec(
    spec_type="chatts_series",
    name="ChatTS series",
    unit_value=None,
    dtype="float64",
)
_TIMESERIES_MARKER = "<ts><ts/>"
# Keep nested time-series batches small to bound conversion and lazy-load memory.
_BUILD_BATCH_ROWS = 256


@lru_cache(maxsize=1)
def _timeseries_batch(path: str, row_group: int, batch_start: int) -> Any:
    """Load one bounded batch of time-series values for lazy signal loaders.

    Returns:
        The Arrow list array containing the requested batch's time-series rows.

    Raises:
        TimeFFormatError: If the requested batch is absent from the Parquet file.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    parquet = pq.ParquetFile(path)
    offset = 0
    for batch in parquet.iter_batches(
        batch_size=_BUILD_BATCH_ROWS,
        row_groups=[row_group],
        columns=["timeseries"],
    ):
        if offset == batch_start:
            return batch.column(0)
        offset += batch.num_rows
    raise TimeFFormatError(f"Parquet file {path!r} has no batch at row group {row_group}, row offset {batch_start}")


def _signal_values(path: str, row_group: int, batch_start: int, row_in_batch: int, signal_index: int) -> Any:
    """Load one series from a cached batch in the ChatTS source float64 representation.

    Returns:
        The selected series as a float64 Arrow array.
    """
    import pyarrow as pa  # noqa: PLC0415

    values = _timeseries_batch(path, row_group, batch_start)[row_in_batch].values[signal_index].values
    return values.cast(pa.float64())


class ChatTSConnector(BaseHuggingFaceConnector, ABC):
    """Base connector for one named ChatTS Hugging Face configuration."""

    HF_REPO = "ChatTSRepo/ChatTS-Training-Dataset"
    PARQUET_REVISION = "30e0c2813f390bb47f4a1619aa9f742bf6a21282"
    HF_CONFIG: ClassVar[str]
    TASK_SPLIT: ClassVar[Split] = Split.TRAIN

    def download(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Download this configuration's Parquet files without decoding their rows.

        Returns:
            Lightweight cached Parquet paths tagged with their source configuration.

        Raises:
            TimeNetDatasetNotFoundError: If the source has no data for :attr:`HF_CONFIG`.
        """
        configuration = self.HF_CONFIG
        filenames = [filename for filename in self._parquet_filenames() if filename.startswith(f"{configuration}/")]
        if not filenames:
            raise TimeNetDatasetNotFoundError(
                f"{self.HF_REPO!r} has no Parquet files for configuration {configuration!r} on "
                f"{self.PARQUET_REVISION!r}"
            )
        return [
            {"path": path, "__chatts_configuration": configuration}
            for path in self._download_parquet_files(cache_dir, filenames)
        ]

    def convert(self, raw_refs: list[dict[str, Any]]) -> TimeFDataset:
        """Build lazy records and a task stream from cached ChatTS Parquet files.

        Args:
            raw_refs: Cached Parquet paths from :meth:`download`, tagged with their configuration.

        Returns:
            The converted configuration dataset.

        Raises:
            TimeNetDatasetNotFoundError: If the selected Parquet files contain no rows.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        records = self._add_records(dataset, raw_refs)
        if not records:
            raise TimeNetDatasetNotFoundError(f"{self.HF_REPO!r} returned no rows for configuration {self.HF_CONFIG!r}")
        dataset.set_task_stream([AnswerTask], lambda: self._iter_tasks(raw_refs, records))
        return dataset

    def _add_records(self, dataset: TimeFDataset, raw_refs: list[dict[str, Any]]) -> dict[str, Record]:
        """Register one lazy record per Parquet row and return them by ID.

        Returns:
            The registered records keyed by their stable IDs.
        """
        records: dict[str, Record] = {}
        indexes: defaultdict[str, int] = defaultdict(int)
        for path, configuration, row_group, batch_start, batch in self._iter_batches(raw_refs, ["timeseries"]):
            series_column = batch.column(0)
            for row_in_batch in range(batch.num_rows):
                index = indexes[configuration]
                indexes[configuration] += 1
                record_id = f"{configuration}-{index}"
                series = series_column[row_in_batch].values
                signals = tuple(
                    Signal.from_loader(
                        spec=_SPEC,
                        name=f"c{signal_index}",
                        time_axis=OrdinalAxis(),
                        loader=self._signal_loader(path, row_group, batch_start, row_in_batch, signal_index),
                        id=f"{record_id}-c{signal_index}",
                        n_values=len(values),
                    )
                    for signal_index, values in enumerate(series)
                )
                record = dataset.add_record(
                    record=Record(
                        record_id=record_id,
                        sources=(Source(id=f"{record_id}-source", name="ChatTS series", signals=signals),),
                        metadata={"chatts_configuration": configuration},
                    )
                )
                records[record_id] = record
        return records

    def _iter_tasks(
        self,
        raw_refs: list[dict[str, Any]],
        records: Mapping[str, Record],
    ) -> Iterator[AnswerTask]:
        """Re-read prompts and answers to yield tasks without retaining them in memory.

        Yields:
            One answer task in :attr:`TASK_SPLIT` for each source row.

        Raises:
            TimeFFormatError: If a task row has no matching record, a record has no task row, or
                a prompt's time-series markers do not match its record signals.
        """
        indexes: defaultdict[str, int] = defaultdict(int)
        task_record_ids: set[str] = set()
        for _, configuration, _, _, batch in self._iter_batches(raw_refs, ["input", "output"]):
            input_column = batch.column(0)
            output_column = batch.column(1)
            for row_in_batch in range(batch.num_rows):
                index = indexes[configuration]
                indexes[configuration] += 1
                record_id = f"{configuration}-{index}"
                prompt = input_column[row_in_batch].as_py()
                record = records.get(record_id)
                if record is None:
                    raise TimeFFormatError(
                        f"{configuration} record {record_id!r}: task row has no corresponding converted record"
                    )
                marker_count = prompt.count(_TIMESERIES_MARKER)
                if marker_count != len(record.signals):
                    raise TimeFFormatError(
                        f"{configuration} record {record_id!r}: input has {marker_count} "
                        f"{_TIMESERIES_MARKER!r} markers for {len(record.signals)} time series"
                    )
                task_record_ids.add(record_id)
                yield AnswerTask(
                    id=f"{record_id}-answer",
                    inputs=(record,),
                    prompt=prompt,
                    targets=(output_column[row_in_batch].as_py(),),
                    metadata={"chatts_configuration": configuration},
                    split=self.TASK_SPLIT,
                )
        missing_task_record_ids = records.keys() - task_record_ids
        if missing_task_record_ids:
            missing = ", ".join(sorted(missing_task_record_ids))
            raise TimeFFormatError(f"ChatTS records have no corresponding task rows: {missing}")

    @staticmethod
    def _iter_batches(
        raw_refs: list[dict[str, Any]],
        columns: list[str],
    ) -> Iterator[tuple[Path, str, int, int, Any]]:
        """Yield bounded Arrow batches with enough location metadata to reload their values."""
        import pyarrow.parquet as pq  # noqa: PLC0415

        for raw_ref in raw_refs:
            path = Path(raw_ref["path"])
            configuration = str(raw_ref["__chatts_configuration"])
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                batch_start = 0
                for batch in parquet.iter_batches(
                    batch_size=_BUILD_BATCH_ROWS,
                    row_groups=[row_group],
                    columns=columns,
                ):
                    yield path, configuration, row_group, batch_start, batch
                    batch_start += batch.num_rows

    @staticmethod
    def _signal_loader(
        path: Path,
        row_group: int,
        batch_start: int,
        row_in_batch: int,
        signal_index: int,
    ) -> Callable[[], Any]:
        """Create a loader for one series in a bounded cached Parquet batch.

        Returns:
            A callable that loads the selected series on demand.
        """
        return lambda: _signal_values(str(path), row_group, batch_start, row_in_batch, signal_index)
