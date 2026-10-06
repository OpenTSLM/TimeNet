"""Keep the complete continuous HARESPOD release, before benchmark window selection."""

import asyncio
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pyarrow as pa

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import IrregularAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.types import TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files
from timenet_connectors.time_axes import axis_for_offsets


URL = "https://ndownloader.figshare.com/files/40408430"
SHA256 = "c4d52694fe8ee2c35e65e2e89d546cea63cf30caf569d661e4281f9d950ad419"
CHANNELS = ("rsp", "hr", "spo", "spv", "prt")


class HarespodConnector(BaseConnector[Path]):
    """The 15 complete continuous recordings, with all five released channels."""

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch and extract the checksum-pinned continuous-data archive.

        Returns:
            The original Data_Cons directory.
        """
        import py7zr  # noqa: PLC0415 - connector dependency

        archive_path = cache_dir / "Data_Cons.7z"
        marker = cache_dir / f".{SHA256}.extracted"
        if not marker.exists():
            asyncio.run(download_files([Artifact(URL, archive_path, sha256=SHA256)]))
            with py7zr.SevenZipFile(archive_path) as archive:
                archive.extractall(cache_dir)
            marker.touch()
        return [cache_dir / "Data_Cons"]

    def convert(self, raw_refs: list[Path], context: BuildContext | None = None) -> TimeFDataset:  # noqa: ARG002
        """Keep complete released channels, clocks, and the experiment's timestamp notes.

        Returns:
            The taskless continuous corpus. Values keep the upstream normalization.

        Raises:
            TimeFFormatError: If the release has no subjects or a channel has invalid timestamps.
        """
        import pandas as pd  # noqa: PLC0415 - connector dependency

        subjects = sorted(path for path in raw_refs[0].iterdir() if path.is_dir() and not path.name.startswith("."))
        if not subjects:
            raise TimeFFormatError("HARESPOD continuous release contains no subjects")
        dataset = TimeFDataset(metadata=self.metadata())
        for subject in subjects:
            paths = {channel: subject / f"{channel}_5cut.csv" for channel in CHANNELS}
            times = {
                channel: pd.to_datetime(pd.read_csv(path, header=None, usecols=[0])[0]).to_numpy(dtype="datetime64[us]")
                for channel, path in paths.items()
            }
            if any(
                len(values) == 0 or np.isnat(values).any() or np.any(np.diff(values.astype(np.int64)) <= 0)
                for values in times.values()
            ):
                raise TimeFFormatError(f"HARESPOD {subject.name} has empty, missing, or unordered timestamps")
            origin = min(values[0] for values in times.values())
            record_id = f"harespod-{subject.name}"
            signals = []
            for channel, path in paths.items():
                offsets = (times[channel] - origin).astype("timedelta64[us]").astype(np.int64)
                axis = axis_for_offsets(offsets)
                signals.append(
                    Signal.from_loader(
                        id=f"{record_id}-{channel}",
                        name=channel,
                        n_values=len(offsets),
                        time_axis=axis,
                        time_offsets_loader=partial(pa.array, offsets) if isinstance(axis, IrregularAxis) else None,
                        spec=TimeSeriesSpec(
                            spec_type=f"harespod_{channel}",
                            name=f"HARESPOD {channel} (released scale)",
                            dtype="float64",
                            unit_value=ureg.dimensionless,
                        ),
                        loader=partial(_values, path),
                    )
                )
            dataset.add_record(
                record=Record(
                    record_id=record_id,
                    subject_ids=(subject.name,),
                    sources=(Source(id=f"{record_id}-source", name="Continuous recording", signals=tuple(signals)),),
                    metadata={
                        "recording_start_local": str(origin),
                        "key_timestamps": (subject / "key_timestamp.txt").read_text(),
                    },
                )
            )
        return dataset


@lru_cache(maxsize=4)
def _values(path: Path) -> pa.Array:
    import pandas as pd  # noqa: PLC0415 - connector dependency

    values = pd.read_csv(path, header=None, usecols=[1])[1].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise TimeFFormatError(f"HARESPOD {path} contains non-finite values")
    return pa.array(values)


CONNECTOR = HarespodConnector
