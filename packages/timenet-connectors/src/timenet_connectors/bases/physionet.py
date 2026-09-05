"""A reusable base class for connectors that read WFDB records from PhysioNet.

A subclass downloads a PhysioNet database archive with :func:`~timenet_connectors.download.ensure_archive`
and reads the records. Headers are parsed straight from the ``.hea`` text, and format-16 signals are read
directly from the ``.dat`` with NumPy; ``wfdb`` is used only as a fallback for signal formats the direct
reader does not handle. ``wfdb`` is declared in the connector's ``requirements.txt`` and installed into the
environment the build runs in, and the base class imports it lazily, so ``--no-isolation`` stays usable and
a build that never reaches the fallback needs nothing extra.
"""

from abc import ABC
from collections.abc import Callable
import functools
from pathlib import Path
import re
from typing import Any, NamedTuple, TypeVar

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.errors import TimeFFormatError


TRaw = TypeVar("TRaw")

# The gain field of a signal line: ``<gain>[(<baseline>)][/<units>]``, e.g. ``1000.0(0)/mV``.
_GAIN_RE = re.compile(r"(?P<gain>[-\d.eE+]+)(?:\((?P<baseline>-?\d+)\))?")
# A WFDB signal line is <file> <fmt> <gain> <adc_res> <adc_zero> <init> <checksum> <block> <description>;
# the optional description (which may hold spaces) starts at this field index.
_DESCRIPTION_FIELD = 8


class WfdbHeader(NamedTuple):
    """The header fields a PhysioNet connector reads, parsed straight from the ``.hea`` text.

    ``dat_names``/``formats``/``gains``/``baselines`` are per signal, in signal order; the direct
    ``.dat`` reader uses them to turn raw ADC samples into physical units.
    """

    fs: float
    sig_len: int
    sig_name: list[str]
    dat_names: tuple[str, ...]
    formats: tuple[str, ...]
    gains: tuple[float, ...]
    baselines: tuple[int, ...]
    direct_read: bool
    """Whether the whole record can be read by the direct format-16 path (see :func:`_read_dat16`).
    False routes to ``wfdb``, which handles the parts the direct reader does not."""


@functools.lru_cache(maxsize=4)
def _read_dat16(dat_path: str, n_sig: int) -> np.ndarray:
    """Read a whole format-16 ``.dat`` once and return raw int16 samples, shape ``(n_frames, n_sig)``.

    Cached so the leads of one record (written contiguously after the writer's source-grouped sort) read
    the file once rather than once per lead.

    Args:
        dat_path: The ``.dat`` file path.
        n_sig: The number of interleaved signals per frame.

    Returns:
        The raw samples, one column per signal.
    """
    return np.fromfile(dat_path, dtype="<i2").reshape(-1, n_sig)


class BasePhysioNetConnector(BaseConnector[TRaw], ABC):
    """The base class for PhysioNet-backed connectors that read WFDB records."""

    @staticmethod
    def _wfdb() -> Any:
        """Import ``wfdb`` lazily. Raise an error that states the fix if ``wfdb`` is missing.

        Returns:
            The imported ``wfdb`` module.

        Raises:
            ImportError: If ``wfdb``, declared in this connector's requirements, is not installed.
        """
        try:
            import wfdb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "reading PhysioNet records needs wfdb, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc
        return wfdb

    @staticmethod
    def _read_header(record_base: Path) -> WfdbHeader:
        """Parse a WFDB ``.hea`` header directly for the fields a connector needs.

        Reads the header text rather than through ``wfdb.rdheader``, which builds a pandas DataFrame and
        reads it cell by cell — those millions of scalar lookups otherwise dominate a large convert. The
        record line is ``<name> <n_sig> <fs> <sig_len> ...``; each signal line is
        ``<dat> <fmt> <gain>(<baseline>)/<units> <adc_res> <adc_zero> <init> <checksum> <block> <desc>``;
        ``#`` lines are comments.

        Args:
            record_base: The record path without the ``.dat`` and ``.hea`` extensions.

        Returns:
            The parsed header.

        Raises:
            TimeFFormatError: If the header is malformed: an unparseable gain, a signal line with no
                signal name, or fewer signal lines than the record declares.
        """
        text = Path(f"{record_base}.hea").read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
        record = lines[0].split()
        n_sig = int(record[1])
        dat_names, formats, raw_formats, gains, baselines, sig_name = [], [], [], [], [], []
        for line in lines[1 : 1 + n_sig]:
            fields = line.split()
            raw_format = fields[1]  # <fmt>[x<spf>][:<skew>]
            match = _GAIN_RE.match(fields[2])
            if match is None:
                raise TimeFFormatError(f"{record_base}.hea: cannot parse the gain field {fields[2]!r}")
            if len(fields) <= _DESCRIPTION_FIELD:
                raise TimeFFormatError(f"{record_base}.hea: signal line names no signal: {line!r}")
            gain = float(match["gain"])
            dat_names.append(fields[0])
            raw_formats.append(raw_format)
            formats.append(raw_format.partition("x")[0].partition(":")[0])
            # WFDB reads a gain of 0 as "unspecified" and uses 200 adu/mV as the default (WFDB spec).
            gains.append(gain if gain != 0 else 200.0)
            # A missing (baseline) defaults to adc_zero (field index 4 on the signal line).
            baselines.append(int(match["baseline"]) if match["baseline"] is not None else int(fields[4]))
            # The signal description can hold spaces, so keep all of it.
            sig_name.append(" ".join(fields[_DESCRIPTION_FIELD:]))
        if len(sig_name) != n_sig:
            raise TimeFFormatError(f"{record_base}.hea declares {n_sig} signals but has {len(sig_name)} signal lines")
        # The direct reader interleaves one .dat as (-1, n_sig) int16, so it is valid only when every
        # signal is plainly "16" (no "16x2" frames, no "16:5" skew) and they share a single file. Any
        # other shape (multi-.dat, samples-per-frame > 1, skew, byte offset) routes to wfdb instead of
        # being silently misread.
        direct_read = len(set(dat_names)) == 1 and all(fmt == "16" for fmt in raw_formats)
        return WfdbHeader(
            fs=float(record[2].split("/")[0]),  # <fs>[/<counter_freq>]
            sig_len=int(record[3]),
            sig_name=sig_name,
            dat_names=tuple(dat_names),
            formats=tuple(formats),
            gains=tuple(gains),
            baselines=tuple(baselines),
            direct_read=direct_read,
        )

    def _lead_loader(self, record_base: Path, header: WfdbHeader, lead_idx: int) -> Callable[[], pa.Array]:
        """Build a lazy loader for one lead's samples as a float32 Arrow array in physical units.

        For a standard format-16 record the ``.dat`` is read directly with NumPy — all leads of a record
        read the file once through the cache — and converted with the header's gain and baseline. Any
        record the direct path does not cover (``header.direct_read`` is false) falls back to
        ``wfdb.rdsamp``.

        Args:
            record_base: The record path without the ``.dat`` and ``.hea`` extensions.
            header: The record's parsed header.
            lead_idx: The zero-based lead index within the record.

        Returns:
            A no-argument loader that returns the lead's physical signal.
        """
        if not header.direct_read:

            def load_wfdb() -> pa.Array:
                signal, _ = self._wfdb().rdsamp(str(record_base), channels=[lead_idx])
                return pa.array(signal[:, 0].astype("float32"))

            return load_wfdb

        dat_path = str(record_base.parent / header.dat_names[lead_idx])
        n_sig = len(header.sig_name)
        gain, baseline = header.gains[lead_idx], header.baselines[lead_idx]

        def load() -> pa.Array:
            raw = _read_dat16(dat_path, n_sig)[:, lead_idx].astype("float32")
            return pa.array((raw - baseline) / gain)

        return load
