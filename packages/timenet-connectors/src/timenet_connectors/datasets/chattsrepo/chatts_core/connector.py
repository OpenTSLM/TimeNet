"""Shared Hugging Face download and conversion support for ChatTS."""

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeNetDatasetNotFoundError
from timenet.types import AnswerTask, TimeSeriesSpec, ureg
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector


_BATCH_ROWS = 65536
_SPEC = TimeSeriesSpec(
    spec_type="chatts_series",
    name="ChatTS series",
    unit_value=ureg.dimensionless,
)


class ChatTSConnector(BaseHuggingFaceConnector, ABC):
    """Base connector for one or more named ChatTS Hugging Face configurations."""

    HF_REPO = "ChatTSRepo/ChatTS-Training-Dataset"
    HF_CONFIG: ClassVar[str | None] = None
    HF_CONFIGS: ClassVar[tuple[str, ...]] = ()

    def _configurations(self) -> tuple[str, ...]:
        """Return the source configurations selected by this connector.

        Raises:
            ValueError: If the connector declares no source configuration.
        """
        if self.HF_CONFIGS:
            return self.HF_CONFIGS
        if self.HF_CONFIG is not None:
            return (self.HF_CONFIG,)
        raise ValueError(f"{type(self).__name__} must declare HF_CONFIG or HF_CONFIGS")

    def download(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Download rows from this connector's configuration only.

        Returns:
            Plain Hub rows from the configuration's auto-converted Parquet files.

        Raises:
            ImportError: If the connector's ``huggingface_hub`` dependency is unavailable.
            TimeNetDatasetNotFoundError: If the source has no data for :attr:`HF_CONFIG`.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"reading {self.HF_REPO!r} needs huggingface_hub, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc
        import pyarrow.parquet as pq  # noqa: PLC0415

        files = sorted(list_repo_files(self.HF_REPO, repo_type="dataset", revision=self.PARQUET_REVISION))
        rows: list[dict[str, Any]] = []
        for configuration in self._configurations():
            filenames = [
                filename
                for filename in files
                if filename.startswith(f"{configuration}/") and filename.endswith(".parquet")
            ]
            if not filenames:
                raise TimeNetDatasetNotFoundError(
                    f"{self.HF_REPO!r} has no Parquet files for configuration {configuration!r} on "
                    f"{self.PARQUET_REVISION!r}"
                )
            for filename in filenames:
                path = hf_hub_download(
                    self.HF_REPO,
                    filename,
                    repo_type="dataset",
                    revision=self.PARQUET_REVISION,
                    cache_dir=str(cache_dir),
                )
                for batch in pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS):
                    rows.extend({**row, "__chatts_configuration": configuration} for row in batch.to_pylist())
        if not rows:
            raise TimeNetDatasetNotFoundError(
                f"{self.HF_REPO!r} returned no rows for configurations {self._configurations()!r}"
            )
        return rows

    def convert(self, raw_refs: list[dict[str, Any]]) -> TimeFDataset:
        """Convert each ChatTS instruction and its signals into one answer task.

        Args:
            raw_refs: Plain rows with ``input``, ``timeseries``, and ``output`` fields.

        Returns:
            The converted configuration dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for index, row in enumerate(raw_refs):
            configuration = row.get("__chatts_configuration")
            if configuration is None:
                (configuration,) = self._configurations()
            record_id = f"{configuration}-{index}"
            signals = tuple(
                Signal.from_values(
                    values,
                    spec=_SPEC,
                    name=f"c{signal_index}",
                    time_axis=OrdinalAxis(),
                    id=f"{record_id}-c{signal_index}",
                )
                for signal_index, values in enumerate(row["timeseries"])
            )
            record = dataset.add_record(
                record=Record(
                    record_id=record_id,
                    sources=(Source(id=f"{record_id}-source", name="ChatTS series", signals=signals),),
                    metadata={"chatts_configuration": configuration},
                )
            )
            dataset.add_task(
                task=AnswerTask(
                    id=f"{record_id}-answer",
                    inputs=(record,),
                    prompt=row["input"],
                    targets=(row["output"],),
                    metadata={"chatts_configuration": configuration},
                )
            )
        return dataset
