"""Select HARESPOD benchmark windows from complete original recordings."""

from pathlib import Path

import numpy as np
import pyarrow as pa

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Signal, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import iter_cases, moment_us
from timenet_connectors.datasets.yang_ai_lab.hearts_core.derived import (
    derived_signal,
    sample_offsets,
    timestamp_positions,
)
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import REPO, REVISION
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.records import case_records
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.release import TASKS
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.tasks import convert_case
from timenet_connectors.sources.huggingface_hub import hub_snapshot


class HeartsHarespodConnector(BaseConnector[Path]):
    """Four ranking and pairing tasks, with windows owned by the child."""

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch pinned benchmark windows and annotations.

        Returns:
            The frozen release root.
        """
        return [hub_snapshot(REPO, REVISION, cache_dir, ("harespod/*/*.pkl",))]

    def compose(self, raw_refs: list[Path], context: BuildContext) -> TimeFDataset:  # noqa: PLR6301
        """Select timestamped windows or uniquely match windows with reset clocks.

        Returns:
            The four HARESPOD benchmark task types over child-owned windows.

        """
        parents = {record.id: record for record in context.parent("oca-john/harespod").iter_records()}
        dataset = TimeFDataset(metadata=context.metadata)
        for case in iter_cases(raw_refs[0], TASKS):
            original = parents[f"harespod-{case.payload['subject_id']}"]
            original_signals = {signal.name: signal for signal in original.signals}
            records = case_records(case)
            for built in (*records.inputs, *records.candidates):
                for source in built.record.sources:
                    source.signals = tuple(
                        _window(
                            signal,
                            original_signals[signal.name],
                            built.origin_us,
                            moment_us(original.metadata["recording_start_local"]),
                            reset_clock=case.task.endswith("pairing"),
                        )
                        for signal in source.signals
                    )
                built.record.metadata.update(
                    {"parent_dataset": "oca-john/harespod@1.0.0", "parent_record": original.id}
                )
                dataset.add_record(record=built.record)
            dataset.add_tasks(tasks=[convert_case(case, records)])
        return dataset


def _window(expected: Signal, original: Signal, child_origin: int, parent_origin: int, *, reset_clock: bool) -> Signal:
    positions = (
        None
        if reset_clock
        else timestamp_positions(original, sample_offsets(expected) + child_origin, origin_us=parent_origin)
    )

    def load() -> pa.Array:
        values = original.to_arrow()
        if positions is not None:
            return values.take(pa.array(positions))
        reference = expected.to_numpy()
        actual = values.to_numpy()
        if len(reference) > len(actual):
            raise TimeFFormatError(f"{expected.id}: benchmark window is longer than the original recording")
        starts = np.flatnonzero(
            np.isclose(actual[: len(actual) - len(reference) + 1], reference[0], rtol=1e-7, atol=1e-8)
        )
        # Narrow candidates before comparing full windows, especially for piecewise-constant HR.
        for offset in (len(reference) // 2, len(reference) - 1):
            starts = starts[np.isclose(actual[starts + offset], reference[offset], rtol=1e-7, atol=1e-8)]
        matches = [
            int(start)
            for start in starts
            if np.allclose(actual[start : start + len(reference)], reference, rtol=1e-7, atol=1e-8)
        ]
        if len(matches) != 1:
            raise TimeFFormatError(f"{expected.id}: expected one original HARESPOD window, found {len(matches)}")
        start = matches[0]
        offsets = sample_offsets(original)[start : start + len(reference)]
        if not np.array_equal(offsets - offsets[0], sample_offsets(expected) - sample_offsets(expected)[0]):
            raise TimeFFormatError(f"{expected.id}: matched HARESPOD window has a different sample clock")
        return values.slice(start, len(reference))

    signal = derived_signal(expected, original, load, parent="oca-john/harespod@1.0.0")
    signal.metadata["selection"] = "unique matching window; clock reset to zero" if reset_clock else "source timestamps"
    return signal


CONNECTOR = HeartsHarespodConnector
