"""The BattLeDIM connector: two years of SCADA data from the L-Town water network, with its leaks.

BattLeDIM 2020 (Battle of the Leakage Detection and Isolation Methods) released simulated SCADA
data of the L-Town network for 2018 and 2019: 33 pressure sensors, 3 flow sensors, 1 tank level
sensor and 82 automated meter readings, every 5 minutes, and the 33 leaks that the simulation
contained. The two years form one Record. The leaks become record annotations, and two-week windows
of the record become the tasks of :mod:`~timenet_connectors.datasets.kios.battledim.tasks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import functools
from pathlib import Path

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, TimeInterval, TimeOrigin, TimePoint, TimeSeriesSpec, ureg
from timenet_connectors.datasets.kios.battledim.source import SCADA_FILES, STEP, Release
from timenet_connectors.datasets.kios.battledim.tasks import split_windows, window_tasks, windows
from timenet_connectors.download import Artifact, download_files


ZENODO_FILE = "https://zenodo.org/api/records/4017659/files/{name}/content"
# Zenodo record 4017659 (published 2020-09-07) is the only version. The two .xlsx files repeat the
# CSVs and L-TOWN_Real.inp (177 MB) is the simulation model, so the connector downloads neither.
SOURCE_FILES = {
    "2018_SCADA_Pressures.csv": "7782d890a7195699be94ddc01cba392da7ca285908afff619065a9616e04dcae",
    "2018_SCADA_Flows.csv": "7f7e8625c9d92b083ee079a54cffb9ef803876de1e35017c1da0455968daf3ff",
    "2018_SCADA_Levels.csv": "a1fc5f0229448a620b6220f4d079b3162a255b3070acdce448608fbfc760903c",
    "2018_SCADA_Demands.csv": "713d9f03ff94128f44c7d8036a032cc1d7eebcbbc94fda9def2bbc3efc047dd5",
    "2019_SCADA_Pressures.csv": "a4af6616059cc9b366bbab76d880b2c772df544970c6ff4d4a29a58704ce05b1",
    "2019_SCADA_Flows.csv": "c709ea5ecc03197e1fe70cf4c2de7225f71015acb50a7445ed7765b5029f7f15",
    "2019_SCADA_Levels.csv": "00624e97b72e63233beeccb48db528d43e59a87b116d0295d5c2a206ee0e75a8",
    "2019_SCADA_Demands.csv": "004266aa039833b5ae0f37740487db4d90216d2ee947f14871c5d5d3f89b4151",
    "2018_Leakages.csv": "e1abd9d549a18880d8ae704dc90c06c8a3d5aecbd2541fa091548ef392084ba6",
    "2019_Leakages.csv": "b2ffe56b6c2f3b6cdb6aae9aeb30b9c425ba911fff8e3716ae543a56ff7576cd",
    "dataset_configuration.yaml": "23471d8989d82c66a4909816ea248f7588d8ccc910e142265a7ea48db940b6a0",
    "2018_Fixed_Leakages_Report.txt": "5dba254c623b867a0a91150e2b5539842ca05a989fde6cf2754065d5b415a8f3",
    "L-TOWN.inp": "a7551b86745f4cc3433c78e60023077fa1386947d4d35372ffd3a0405622b436",
}

# README.txt of the release states each unit. Pressure is head in metres of water. The values stay
# float64, as parsed from the release's two-decimal text.
SPECS = {
    "pressure": TimeSeriesSpec(spec_type="pressure_head", name="Pressure head", unit_value=ureg.meter, dtype="float64"),
    "flow": TimeSeriesSpec(
        spec_type="volume_flow", name="Volume flow", unit_value=ureg.meter**3 / ureg.hour, dtype="float64"
    ),
    "tank_level": TimeSeriesSpec(
        spec_type="water_level", name="Tank water level", unit_value=ureg.meter, dtype="float64"
    ),
    "amr_demand": TimeSeriesSpec(
        spec_type="water_demand", name="Metered water demand", unit_value=ureg.liter / ureg.hour, dtype="float64"
    ),
}
SOURCE_NAMES = {
    "pressure": "Pressure sensors",
    "flow": "Flow sensors",
    "tank_level": "Tank level sensor",
    "amr_demand": "Automated meter readings",
}


@dataclass(frozen=True)
class BattLeDIMSource:
    """Handle to the downloaded release: the directory that holds the files of ``SOURCE_FILES``."""

    root: Path


class BattLeDIMConnector(BaseConnector[BattLeDIMSource]):
    """Connector for the BattLeDIM 2020 dataset (Zenodo record 4017659)."""

    async def download_async(self, cache_dir: Path) -> list[BattLeDIMSource]:  # ruff: ignore[no-self-use]
        """Download the release files once, checking each one's sha256.

        Returns:
            A list of one :class:`BattLeDIMSource`.
        """
        await download_files(
            Artifact(ZENODO_FILE.format(name=name), cache_dir / name, sha256=digest)
            for name, digest in SOURCE_FILES.items()
        )
        return [BattLeDIMSource(cache_dir)]

    def convert(self, raw_refs: list[BattLeDIMSource]) -> TimeFDataset:
        """Build the record, its leak annotations and the window tasks.

        The release is 190 MB of text, so every file is read once here, which also checks it before
        anything is written. The signal loaders then serve the parsed columns.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.

        Raises:
            TimeNetConnectorError: If ``raw_refs`` does not hold exactly one release.
        """
        if len(raw_refs) != 1:
            raise TimeNetConnectorError(f"expected one release, got {len(raw_refs)}")
        release = Release(raw_refs[0].root)
        dataset = TimeFDataset(metadata=self.metadata())
        record = dataset.add_record(record=self._make_record(release))
        found = windows(release)
        split_of = split_windows(found)
        for window in found:
            dataset.add_tasks(tasks=window_tasks(release, record, window, split_of[window.index]))
        return dataset

    @staticmethod
    def _make_record(release: Release) -> Record:
        """Build the one record: a source per sensor type and an annotation set per leak.

        Returns:
            The record, starting at the first step of 2018.
        """
        axis = RegularAxis(period_us=Fraction(int(STEP.total_seconds() * 1_000_000)))
        sources = []
        for kind in SCADA_FILES:
            columns = release.scada[kind][1]
            signals = tuple(
                Signal.from_loader(
                    spec=SPECS[kind],
                    time_axis=axis,
                    name=sensor,
                    loader=functools.partial(pa.array, columns[sensor]),
                    n_values=release.n_steps,
                )
                for sensor in release.sensors[kind]
            )
            sources.append(Source(name=SOURCE_NAMES[kind], signals=signals))
        record = Record(
            sources=tuple(sources),
            start_time=TimeOrigin(release.start),
            metadata={"network": "L-Town", "source_doi": "10.5281/zenodo.4017659"},
        )
        annotations = []
        for leak in release.leaks:
            start_us = int((leak.start - release.start).total_seconds() * 1_000_000)
            end_us = int((leak.end + STEP - release.start).total_seconds() * 1_000_000)
            span = TimeInterval.micros(start_us, end_us)
            outflow = release.leak_outflow[leak.pipe]
            annotations += [
                Annotation(key="leak_pipe", value=leak.pipe, span=span, description="The leaking pipe's EPANET id."),
                Annotation(key="leak_type", value=leak.leak_type, span=span, description="abrupt or incipient."),
                Annotation(
                    key="leak_hole_diameter",
                    value=round(leak.diameter_m * 1000, 4),
                    unit="mm",
                    span=span,
                    description="The diameter of the simulated hole.",
                ),
                Annotation(
                    key="leak_peak_outflow",
                    value=float(outflow.max()),
                    unit="m^3/h",
                    span=span,
                    description="The largest outflow in the leak tables.",
                ),
                Annotation(
                    key="leak_repaired",
                    value=leak.repaired,
                    span=span,
                    description="Whether the leak was repaired before the end of 2019.",
                ),
                Annotation(
                    key="leak_peak",
                    value=leak.pipe,
                    span=TimePoint.micros(int((leak.peak - release.start).total_seconds() * 1_000_000)),
                    description="When the leak reaches its full size; the start, for an abrupt leak.",
                ),
            ]
        record.add_annotations(annotations)
        return record


CONNECTOR = BattLeDIMConnector
