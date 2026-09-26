"""Convert the eleven published SLIP evaluation sets as a separate TimeF dataset."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, ClassificationTask, InputModality, TimeSeriesSpec, ureg
from timenet_connectors.datasets.leochen085.slip_common import REPO, REVISION, download_slip, nested_row_group
from timenet_connectors.sources.hub import hub_files


_SPEC = TimeSeriesSpec(
    spec_type="slip_eval_series",
    name="SLIP evaluation series",
    unit_value=ureg.dimensionless,
    dtype="float64",
    nullable=True,
)


@dataclass(frozen=True)
class EvalSource:
    """Paths to the released evaluation Parquet shards."""

    shards: tuple[Path, ...]
    root: Path


def _signal_values(path: Path, group: int, row: int, channel: int) -> pa.Array:
    """Read one channel from one evaluation window.

    Returns:
        Arrow values with the published nulls intact.
    """
    return nested_row_group(path, group, "X")[row][channel].values


def _rows(path: Path) -> Iterator[tuple[int, int, dict[str, object], list[int]]]:
    """Yield scalar fields and channel lengths without retaining the value arrays.

    Yields:
        Row group, offset, scalar fields, and one length per channel.

    Raises:
        TimeFFormatError: If a shard lacks the required columns.
    """
    reader = pq.ParquetFile(path)
    columns = tuple(
        name for name in ("label", "text_label", "prompt", "participant_id") if name in reader.schema_arrow.names
    )
    if "X" not in reader.schema_arrow.names or "label" not in columns:
        raise TimeFFormatError(f"SLIP evaluation shard {path} has no X or label column")
    for group in range(reader.metadata.num_row_groups):
        offset = 0
        for batch in reader.iter_batches(batch_size=1024, row_groups=[group], columns=[*columns, "X"]):
            scalars = batch.select(columns).to_pylist()
            nested = batch.column("X")
            counts = nested.value_lengths().to_pylist()
            lengths = nested.flatten().value_lengths().to_pylist()
            cursor = 0
            for row, count in zip(scalars, counts, strict=True):
                yield group, offset, row, lengths[cursor : cursor + count]
                cursor += count
                offset += 1


class SlipEvalConnector(BaseConnector[EvalSource]):
    """Convert the train and test windows of all eleven SLIP evaluation components."""

    def download(self, cache_dir: Path) -> list[EvalSource]:  # noqa: PLR6301 - BaseConnector override
        """Fetch only evaluation shards from the pinned SLIP repository.

        Returns:
            A single handle with every evaluation shard.

        Raises:
            TimeNetDownloadError: If the release has no evaluation shards.
        """
        paths = tuple(
            sorted(
                path for path in hub_files(REPO, REVISION) if path.endswith(".parquet") and not path.startswith("data/")
            )
        )
        if not paths:
            raise TimeNetDownloadError(f"{REPO}@{REVISION} has no evaluation Parquet shards")
        root = download_slip(cache_dir, *paths)
        shards = tuple(root / path for path in paths)
        if any(not path.is_file() for path in shards):
            raise TimeNetDownloadError(f"{REPO}@{REVISION} did not return every evaluation shard")
        return [EvalSource(shards=shards, root=root)]

    def convert(self, raw_refs: list[EvalSource]) -> TimeFDataset:
        """Create one sensor record and classification task per published window.

        Returns:
            The evaluation dataset with native labels and split provenance.

        Raises:
            TimeFFormatError: If a shard contains an invalid split, label, or prompt.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        tasks = []
        for shard in source.shards:
            component = shard.parent.name
            split = shard.stem.split("-", maxsplit=1)[0]
            if split not in {"train", "test"}:
                raise TimeFFormatError(f"SLIP evaluation shard {shard} has no train or test split")
            shard_id = shard.stem
            for group, offset, row, lengths in _rows(shard):
                record_id = f"slip-eval-{component}-{shard_id}-g{group}-r{offset}"
                signals = tuple(
                    Signal.from_loader(
                        id=f"{record_id}-s{channel}",
                        name=f"s{channel}",
                        spec=_SPEC,
                        time_axis=OrdinalAxis(),
                        n_values=length,
                        source_id=record_id,
                        loader=lambda shard=shard, group=group, offset=offset, channel=channel: _signal_values(
                            shard, group, offset, channel
                        ),
                    )
                    for channel, length in enumerate(lengths)
                )
                record = Record(
                    record_id=record_id,
                    sources=(Source(id=f"{record_id}-source", name=component, signals=signals),),
                )
                dataset.add_record(record=record)
                annotations = [Annotation(key="component", value=component), Annotation(key="split", value=split)]
                for key in ("text_label", "participant_id"):
                    if row.get(key) is not None:
                        annotations.append(Annotation(key=key, value=row[key]))
                record.add_annotations(annotations)
                label = row["label"]
                if not isinstance(label, (str, int, float, bool)):
                    raise TimeFFormatError(f"SLIP evaluation row {record_id} has unsupported label {label!r}")
                prompt = row.get("prompt")
                if prompt is not None and not isinstance(prompt, str):
                    raise TimeFFormatError(f"SLIP evaluation row {record_id} has non-text prompt")
                modes = {InputModality.TIME_SERIES}
                if prompt:
                    modes.add(InputModality.TEXT)
                tasks.append(
                    ClassificationTask(
                        id=f"{record_id}-classification",
                        inputs=(record,),
                        targets=(label,),
                        prompt=prompt,
                        input_modalities=frozenset(modes),
                    )
                )
        dataset.add_tasks(tasks=tasks)
        return dataset


CONNECTOR = SlipEvalConnector
