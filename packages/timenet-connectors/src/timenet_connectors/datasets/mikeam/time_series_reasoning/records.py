"""Build lazy records and provenance annotations from the checked reasoning release."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial

from timenet.dataset import Record, Signal, Source
from timenet.json import JsonValue
from timenet.types import Annotation
from timenet_connectors.datasets.mikeam.time_series_reasoning.source import (
    CounterfactualKey,
    Release,
    RoundedCopyKey,
    RowLocation,
    SeriesFingerprint,
    load_series,
)
from timenet_connectors.datasets.mikeam.time_series_reasoning.vocabulary import (
    AXIS,
    SERIES_NAME,
    AnnotationKey,
    SeriesField,
)


def _record(
    uuid: str, ref: RowLocation, key: SeriesField, fingerprint: SeriesFingerprint, texts: Mapping[AnnotationKey, str]
) -> Record:
    """Build a Record whose one Signal rereads one series from one line.

    Returns:
        The Record, whose values stay in the source file until read.
    """
    signal = Signal.from_loader(
        name=SERIES_NAME,
        spec=fingerprint.dtype.spec,
        time_axis=AXIS,
        n_values=fingerprint.length,
        loader=partial(load_series, ref, key, fingerprint),
    )
    metadata: dict[str, JsonValue] = {"uuid": uuid, "file": ref.file.value, "line": ref.location.line}
    record = Record(sources=(Source(name=SERIES_NAME, signals=(signal,)),), metadata=metadata)
    record.add_annotations(Annotation(key=key, value=text) for key, text in texts.items())
    return record


@dataclass
class RecordIndex:
    """Records keyed by the source identities that tasks reference."""

    series: dict[str, Record]
    rounded_copies: dict[RoundedCopyKey, Record]
    counterfactuals: dict[CounterfactualKey, Record]

    @classmethod
    def build(cls, release: Release) -> "RecordIndex":
        """Return lazy records from the checked release, retaining source order."""
        series = {
            uuid: _record(uuid, row.ref, SeriesField.ORIGINAL, row.fingerprint, row.texts)
            for uuid, row in release.series.items()
        }
        rounded_copies = {
            key: _record(row.uuid, row.ref, SeriesField.ORIGINAL, row.fingerprint, row.texts)
            for key, row in release.rounded_copies.items()
        }
        counterfactuals = {
            key: _record(
                key.uuid,
                row.ref,
                SeriesField.COUNTERFACTUAL,
                row.fingerprint,
                {
                    AnnotationKey.COUNTERFACTUAL: key.counterfactual,
                    AnnotationKey.GENERATOR: row.new_generator,
                },
            )
            for key, row in release.counterfactuals.items()
        }
        return cls(series, rounded_copies, counterfactuals)

    def all(self) -> Iterator[Record]:
        """Yield originals, rounded copies and counterfactuals in source order."""
        yield from self.series.values()
        yield from self.rounded_copies.values()
        yield from self.counterfactuals.values()
