"""The VerbalTS connector: six caption-paired corpora released with the VerbalTS ICML 2025 paper.

Google Drive holds 60 files: 54 NPY arrays and 6 ``meta.json``. Every component ships three splits,
and every split ships three arrays. ``{split}_ts.npy`` holds ``(n_windows, n_steps, n_signals)``
float64 windows. ``{split}_text_caps.npy`` holds one or three fixed-width UTF-32 captions per
window. ``{split}_attrs_idx.npy`` holds one int64 code per attribute that ``meta.json`` names.

One record is one window. Every signal of a window becomes a :class:`~timenet.dataset.TimeSeries`
with a lazy loader that slices a memory-mapped array. So ``convert`` reads the NPY headers, the
captions and the attribute codes, and not one byte of the 560 MB values plane.

Every caption becomes the prompt of a :class:`~timenet.types.TSGenerationTask` whose
``target_record_id`` is the window. That is the text-to-series direction the release supervises.
Weather ships three captions per window, so a Weather record carries three tasks.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import functools
import json
import logging
import operator
from pathlib import Path

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, TSGenerationTask
from timenet_connectors.datasets.seqml.verbalts.files import (
    COMPONENTS,
    DRIVE_URL,
    FILES,
    SPLITS,
    check_drive_body,
)
from timenet_connectors.datasets.seqml.verbalts.keys import (
    VerbalTsKey,
    attribute_key,
    codebook_id,
    codebook_key,
    record_id,
)
from timenet_connectors.datasets.seqml.verbalts.specs import AXIS_BY_COMPONENT, SPEC_BY_COMPONENT
from timenet_connectors.datasets.seqml.verbalts.tables import VARIABLES_BY_COMPONENT, signal_names
from timenet_connectors.download import Artifact, download_files


_LOG = logging.getLogger(__name__)

_VALUES_DIMS = 3
_PLANE_DIMS = 2


@dataclass(frozen=True)
class VerbalTsComponent:
    """One of the six Drive folders, as paths only.

    ``download`` gives these back instead of arrays, so ``convert`` opens every NPY through a
    memory map and the values plane is never resident.
    """

    name: str
    folder: Path

    def npy(self, split: str, kind: str) -> Path:
        """Give the path of one array.

        Args:
            split: ``train``, ``valid`` or ``test``.
            kind: ``ts``, ``attrs_idx`` or ``text_caps``.

        Returns:
            The path of that split's array.
        """
        return self.folder / f"{split}_{kind}.npy"

    @property
    def meta_path(self) -> Path:
        """Give the path of this component's ``meta.json``.

        Returns:
            The path of ``meta.json`` in this component's folder.
        """
        return self.folder / "meta.json"


@functools.lru_cache(maxsize=24)
def _open_npy(path: str) -> np.ndarray:
    """Memory-map one NPY file.

    The 128-byte header states the shape and the dtype. No page of the body is read until a loader
    slices it. The cache holds the arrays open across the writer's series-ordered walk, so a file is
    mapped one time and not once per window.

    Args:
        path: The file to map, as a string so the cache key stays stable.

    Returns:
        The memory-mapped array.
    """
    return np.load(path, mmap_mode="r")


def _plane(path: Path, n_rows: int, n_columns: int | None = None) -> np.ndarray:
    """Map one split's caption or attribute plane and check its shape.

    Args:
        path: The plane's file.
        n_rows: The number of windows the split's values file states.
        n_columns: The number of columns the release states, where it states one. ``meta.json``
            fixes the attribute count. Nothing states the caption count.

    Returns:
        The memory-mapped plane.

    Raises:
        TimeFFormatError: If the plane is not a rectangle of one row per window, or holds a
            different number of columns from the one ``meta.json`` states.
    """
    plane = _open_npy(str(path))
    if plane.ndim != _PLANE_DIMS or plane.shape[0] != n_rows:
        raise TimeFFormatError(
            f"{path.name} holds an array of shape {plane.shape}, expected ({n_rows}, n) to match its values file"
        )
    if n_columns is not None and plane.shape[1] != n_columns:
        raise TimeFFormatError(
            f"{path.name} holds {plane.shape[1]} columns, but meta.json names {n_columns} attributes"
        )
    return plane


def _values_loader(path: str, row: int, signal: int) -> Callable[[], pa.Array]:
    """Build the lazy values loader for one signal of one window.

    The closure captures a string and two integers, so nothing is decoded and nothing is held. It
    re-slices the mapped array on every call, so the writer can call it more than once.

    Args:
        path: The ``{split}_ts.npy`` file.
        row: The window's index inside that file.
        signal: The signal's index inside that window.

    Returns:
        A callable that gives the signal's values as a float64 Arrow array.
    """

    def load() -> pa.Array:
        window = _open_npy(path)[row, :, signal]
        return pa.array(np.ascontiguousarray(window, dtype=np.float64))

    return load


class VerbalTsConnector(BaseConnector[VerbalTsComponent]):
    """Connector for the VerbalTS corpus: six caption-paired components on Google Drive."""

    async def download_async(self, cache_dir: Path) -> list[VerbalTsComponent]:  # noqa: PLR6301 (override: the file table is a constant)
        """Fetch the 60 pinned Drive files and give back one handle per component.

        Every file is re-read after the download and checked against its pinned byte count, its NPY
        magic number and its SHA-256. Drive serves its virus-scan interstitial under a success
        status, and the download layer checks a digest only for a file it fetched.

        Args:
            cache_dir: The directory that holds the downloaded files.

        Returns:
            One handle per component, in the release's own order.
        """
        await download_files(
            [
                Artifact(url=DRIVE_URL.format(file_id=file_id), dest=cache_dir / component / name, sha256=digest)
                for component, name, file_id, _n_bytes, digest in FILES
            ],
            max_concurrency=4,
        )
        for component, name, _file_id, n_bytes, digest in FILES:
            check_drive_body(cache_dir / component / name, n_bytes, digest)
        return [VerbalTsComponent(name=component, folder=cache_dir / component) for component in COMPONENTS]

    def convert(self, raw_refs: list[VerbalTsComponent]) -> TimeFDataset:
        """Build one record per window, with one generation task per caption.

        Each component's codebook annotations are registered before its records, because every task
        of that component names them.

        Args:
            raw_refs: The component handles from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for component in raw_refs:
            names, counts = _read_meta(component)
            dataset.register_annotations(_codebook_annotations(component.name, names, counts))
            _add_component(dataset, component, names, counts)
        return dataset


def _add_component(
    dataset: TimeFDataset,
    component: VerbalTsComponent,
    attribute_names: tuple[str, ...],
    option_counts: tuple[int, ...],
) -> None:
    """Add every window of one component's three splits.

    Args:
        dataset: The dataset under construction.
        component: The component handle.
        attribute_names: The attribute names ``meta.json`` states for this component.
        option_counts: The number of codes each of those attributes declares.

    Raises:
        TimeFFormatError: If a values file does not hold a three-dimensional array of float64, or
            if the component names its signal from a ``var_id`` that ``meta.json`` does not state
            first.
    """
    spec = SPEC_BY_COMPONENT[component.name]
    axis = AXIS_BY_COMPONENT[component.name]
    input_ids = tuple(codebook_id(component.name, name) for name in attribute_names)
    _check_var_id_is_first(component.name, attribute_names)
    for split in SPLITS:
        values_path = component.npy(split, "ts")
        values = _open_npy(str(values_path))
        if values.ndim != _VALUES_DIMS:
            raise TimeFFormatError(
                f"{values_path.name} holds a {values.ndim}-dimensional array; the release stores "
                f"(n_windows, n_steps, n_signals)"
            )
        if values.dtype != np.float64:
            raise TimeFFormatError(f"{values_path.name} holds {values.dtype} values; the release stores float64")
        n_rows, n_steps, n_signals = values.shape
        attributes = _plane(component.npy(split, "attrs_idx"), n_rows, len(attribute_names))
        _warn_on_undeclared_codes(component.name, split, attributes, attribute_names, option_counts)
        captions = _plane(component.npy(split, "text_caps"), n_rows)
        for row in range(n_rows):
            identifier = record_id(component.name, split, row)
            codes = tuple(int(code) for code in attributes[row])
            names = signal_names(component.name, codes, n_signals)
            record = dataset.add_record(
                record_id=identifier,
                time_series=tuple(
                    TimeSeries(
                        spec=spec,
                        signal=name,
                        time_axis=axis,
                        loader=_values_loader(str(values_path), row, signal),
                        source_id=identifier,
                        time_series_id=f"{identifier}-{name}",
                        n_values=n_steps,
                    )
                    for signal, name in enumerate(names)
                ),
            )
            record.add_annotations(
                (
                    Annotation(key=VerbalTsKey.COMPONENT, value=component.name),
                    Annotation(key=VerbalTsKey.SPLIT, value=split),
                    *(
                        Annotation(key=attribute_key(component.name, name), value=code)
                        for name, code in zip(attribute_names, codes, strict=True)
                    ),
                )
            )
            prompts = tuple(str(caption) for caption in captions[row])
            dataset.add_tasks(record, _generation_tasks(identifier, prompts, input_ids))


def _check_var_id_is_first(component: str, attribute_names: tuple[str, ...]) -> None:
    """Check that a component whose signal name needs ``var_id`` states it first.

    :func:`~timenet_connectors.datasets.seqml.verbalts.tables.signal_names` reads the code from
    attribute 0. A release that moved it would mislabel every window whose code still lands inside
    the column list, and raise for the rest, so the position is checked before any row is read.

    Args:
        component: The component name.
        attribute_names: The attribute names ``meta.json`` states, in column order.

    Raises:
        TimeFFormatError: If this component decodes a signal name and ``var_id`` is not first.
    """
    if component in VARIABLES_BY_COMPONENT and attribute_names[:1] != ("var_id",):
        stated = attribute_names[0] if attribute_names else None
        raise TimeFFormatError(
            f"{component}/meta.json names {stated!r} as its first attribute; this component names "
            f"its signal from the var_id the release states first"
        )


def _warn_on_undeclared_codes(
    component: str,
    split: str,
    attributes: np.ndarray,
    names: Sequence[str],
    counts: Sequence[int],
) -> None:
    """Warn one time for each attribute that holds a code ``meta.json`` does not declare.

    A code outside the declared range is an inconsistency the release ships, not a corrupt file, so
    it is written as the release states it. This is the only warning the build emits.

    Args:
        component: The component name.
        split: The split name.
        attributes: The split's attribute plane.
        names: The attribute names, in column order.
        counts: The number of codes each attribute declares.
    """
    for column, (name, count) in enumerate(zip(names, counts, strict=True)):
        codes = np.asarray(attributes[:, column])
        outside = np.flatnonzero((codes < 0) | (codes >= count))
        if outside.size:
            row = int(outside[0])
            _LOG.warning(
                "%s/%s: %d of %d %s codes fall outside the %d options meta.json declares, first at "
                "row %d with code %d; every code is converted as the release states it",
                component,
                split,
                outside.size,
                codes.size,
                name,
                count,
                row,
                int(codes[row]),
            )


def _read_meta(component: VerbalTsComponent) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Read one component's attribute names and their option counts.

    Args:
        component: The component handle.

    Returns:
        The attribute names, and the number of codes each one takes.

    Raises:
        TimeFFormatError: If the file is not a JSON object, states neither list, states an option
            count that is not a whole number, or holds two lists of different lengths.
    """
    try:
        meta = json.loads(component.meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TimeFFormatError(f"{component.name}/meta.json is not valid JSON: {error}") from error
    if not isinstance(meta, dict):
        raise TimeFFormatError(f"{component.name}/meta.json holds a JSON {type(meta).__name__}, not an object")
    absent = [key for key in ("attr_list", "attr_n_ops") if key not in meta]
    if absent:
        raise TimeFFormatError(f"{component.name}/meta.json states no {' and no '.join(absent)}")
    try:
        names = tuple(str(name) for name in meta["attr_list"])
        counts = tuple(operator.index(count) for count in meta["attr_n_ops"])
    except (TypeError, ValueError) as error:
        raise TimeFFormatError(
            f"{component.name}/meta.json holds an attr_list or an attr_n_ops this connector cannot read: {error}"
        ) from error
    if len(names) != len(counts):
        raise TimeFFormatError(
            f"{component.name}/meta.json lists {len(names)} attributes but {len(counts)} option counts"
        )
    return names, counts


def _codebook_annotations(component: str, names: Sequence[str], counts: Sequence[int]) -> list[Annotation]:
    """Build one component's corpus-level codebook annotations.

    Each attribute gets one annotation that carries the number of codes it takes. ``meta.json``
    states that count and states no label for any code, so the count is what the artifact knows.

    Args:
        component: The component name.
        names: The attribute names.
        counts: The number of codes each attribute takes.

    Returns:
        One annotation per attribute the component declares.
    """
    return [
        Annotation(
            key=codebook_key(component, name),
            value=count,
            description=(
                f"Number of distinct codes the {name!r} attribute of the VerbalTS {component} "
                f"component takes, from its meta.json. The release ships no label for any code."
            ),
            id=codebook_id(component, name),
        )
        for name, count in zip(names, counts, strict=True)
    ]


def _generation_tasks(
    identifier: str, captions: Sequence[str], input_ids: tuple[str, ...]
) -> Iterator[TSGenerationTask]:
    """Build one generation task per caption of one window.

    The caption is the prompt and the window is the answer, so each caption is one specification of
    the series to synthesize. A window with three captions gets three tasks over one target.

    Args:
        identifier: The record id of the window to generate.
        captions: The window's caption texts.
        input_ids: The codebook annotations the task gets as context.

    Yields:
        One task per caption.
    """
    for caption in captions:
        yield TSGenerationTask(prompt=caption, target_record_id=identifier, input_annotation_ids=input_ids)


CONNECTOR = VerbalTsConnector
