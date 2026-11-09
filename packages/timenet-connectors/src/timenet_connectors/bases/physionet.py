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
import math
from pathlib import Path
import re
from typing import Any, NamedTuple, TypeVar

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Signal, SignalChannel
from timenet.errors import TimeNetConnectorError
from timenet.types import TimeSeriesSpec
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.sources.errors import error_context
from timenet_connectors.sources.signals import check_channel_matrix, numeric_signal


TRaw = TypeVar("TRaw")
_WAVEFORM_SPEC = TimeSeriesSpec(spec_type="waveform", name="Waveform", unit_value=None)

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


def _read_dat16(dat_path: str, n_sig: int) -> np.ndarray:
    """Read a format-16 ``.dat`` as raw int16 samples, shape ``(n_frames, n_sig)``.

    Args:
        dat_path: The ``.dat`` file path.
        n_sig: The number of interleaved signals per frame.

    Returns:
        The raw samples, one column per signal.

    Raises:
        TimeNetConnectorError: If ``n_sig`` is less than 1, the file has an odd number of bytes, or
            its samples do not split evenly into ``n_sig`` signals.
    """
    if n_sig < 1:
        raise TimeNetConnectorError(f"{dat_path}: a format-16 file needs at least 1 signal, got {n_sig}")
    size = Path(dat_path).stat().st_size
    if size % 2:
        raise TimeNetConnectorError(f"{dat_path}: {size} bytes is not a whole number of 2-byte samples")
    raw = np.fromfile(dat_path, dtype="<i2")
    if raw.size % n_sig:
        raise TimeNetConnectorError(f"{dat_path}: {raw.size} samples do not split into {n_sig} signals")
    return raw.reshape(-1, n_sig)


def _parse_record_line(record_base: Path, line: str) -> tuple[int, float, int]:
    """Return the signal count, sampling frequency, and length from a header's record line.

    The record line is ``<name> <n_sig> <fs>[/<counter_freq>] <sig_len> ...``.

    Raises:
        TimeNetConnectorError: If the count or the length is negative, or the frequency is not
            finite and positive.
        IndexError: If the line has too few fields. The caller reports it as a malformed header.
        ValueError: If a field is not a number. The caller reports it as a malformed header.
    """  # ruff: ignore[docstring-extraneous-exception] - int(), float() and indexing raise the last two
    record = line.split()
    n_sig, fs, sig_len = int(record[1]), float(record[2].split("/")[0]), int(record[3])
    if n_sig < 0 or sig_len < 0:
        raise TimeNetConnectorError(
            f"{record_base}.hea: the signal count {n_sig} and the length {sig_len} must not be negative"
        )
    if not math.isfinite(fs) or fs <= 0:
        raise TimeNetConnectorError(f"{record_base}.hea: the sampling frequency {fs} is not finite and positive")
    return n_sig, fs, sig_len


class _SignalHeader(NamedTuple):
    """The parsed fields of one WFDB signal line."""

    dat_name: str
    raw_format: str
    gain: float
    baseline: int
    name: str


def _parse_signal_line(record_base: Path, line: str) -> _SignalHeader:
    """Return one signal's file, encoding, calibration and name.

    Raises:
        TimeNetConnectorError: If its gain or description is invalid.
    """
    fields = line.split()
    raw_format = fields[1]  # <fmt>[x<spf>][:<skew>]
    match = _GAIN_RE.match(fields[2])
    if match is None:
        raise TimeNetConnectorError(f"{record_base}.hea: cannot parse the gain field {fields[2]!r}")
    if len(fields) <= _DESCRIPTION_FIELD:
        raise TimeNetConnectorError(f"{record_base}.hea: signal line names no signal: {line!r}")
    gain = float(match["gain"])
    if not math.isfinite(gain):
        raise TimeNetConnectorError(f"{record_base}.hea: the gain {fields[2]!r} is not finite")
    # WFDB's unspecified gain is 200 adu/mV; a missing baseline uses adc_zero.
    baseline = int(match["baseline"]) if match["baseline"] is not None else int(fields[4])
    return _SignalHeader(
        fields[0], raw_format, gain if gain != 0 else 200.0, baseline, " ".join(fields[_DESCRIPTION_FIELD:])
    )


class _WfdbRead(NamedTuple):
    """A recording and the header fields that determine its decoded channel values."""

    base: Path
    length: int
    dat_names: tuple[str, ...]
    gains: tuple[float, ...]
    baselines: tuple[int, ...]
    direct: bool
    names: tuple[str, ...]


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
            import wfdb  # ruff: ignore[import-outside-top-level]
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
            TimeNetConnectorError: If the header is malformed: text that is not UTF-8, a missing line
                or field, a field that is not a number, a negative signal count or length, a sampling
                frequency that is not finite and positive, a gain that is not finite or cannot be
                parsed, a signal line with no signal name, or fewer signal lines than the record
                declares.
        """
        try:
            # Decode inside the try, so that text that is not UTF-8 is a malformed header too.
            lines = [
                ln
                for ln in Path(f"{record_base}.hea").read_bytes().decode("utf-8").splitlines()
                if ln and not ln.startswith("#")
            ]
            n_sig, fs, sig_len = _parse_record_line(record_base, lines[0])
            signals = [_parse_signal_line(record_base, line) for line in lines[1 : 1 + n_sig]]
            if len(signals) != n_sig:
                raise TimeNetConnectorError(
                    f"{record_base}.hea declares {n_sig} signals but has {len(signals)} signal lines"
                )
            # The direct reader interleaves one .dat as (-1, n_sig) int16, so it is valid only when every
            # signal is plainly "16" (no "16x2" frames, no "16:5" skew) and they share a single file. Any
            # other shape (multi-.dat, samples-per-frame > 1, skew, byte offset) routes to wfdb instead of
            # being silently misread.
            direct_read = len({signal.dat_name for signal in signals}) == 1 and all(
                signal.raw_format == "16" for signal in signals
            )
            return WfdbHeader(
                fs=fs,
                sig_len=sig_len,
                sig_name=[signal.name for signal in signals],
                dat_names=tuple(signal.dat_name for signal in signals),
                formats=tuple(signal.raw_format.partition("x")[0].partition(":")[0] for signal in signals),
                gains=tuple(signal.gain for signal in signals),
                baselines=tuple(signal.baseline for signal in signals),
                direct_read=direct_read,
            )
        except OSError as error:
            raise TimeNetConnectorError(f"{record_base}.hea: cannot read header: {error}") from error
        except (IndexError, ValueError) as error:
            # Text that is not UTF-8, a truncated line, or a field that is not a number.
            raise TimeNetConnectorError(f"{record_base}.hea: malformed header: {error}") from error

    def _shared_leads(self) -> SharedChannelLoader[_WfdbRead]:
        """Return a conversion-wide loader for all leads of a WFDB recording.

        Both the direct format-16 reader and the WFDB fallback decode all leads together into
        float32 physical units. The cache retains only the most recently read recording.
        """

        def read_channels(request: _WfdbRead) -> tuple[pa.Array, ...]:
            n_sig = len(request.gains)
            try:
                if request.direct:
                    samples = _read_dat16(str(request.base.parent / request.dat_names[0]), n_sig)
                else:
                    samples, _ = self._wfdb().rdsamp(str(request.base))
                with error_context(str(request.base)):
                    check_channel_matrix(samples, "waveforms", sample_axis=0, n_channels=n_sig, n_values=request.length)
                    channels = []
                    for index in range(n_sig):
                        values = samples[:, index]
                        if request.direct:
                            # Keep the existing float32 calibration order for digital samples.
                            values = (values.astype("float32") - request.baselines[index]) / request.gains[index]
                        channels.append(
                            numeric_signal(
                                values,
                                request.names[index],
                                spec=_WAVEFORM_SPEC,
                                n_values=request.length,
                                casting="float_rounding",
                            )
                        )
                return tuple(channels)
            except (OSError, ValueError) as error:
                raise TimeNetConnectorError(f"{request.base}: cannot read WFDB samples: {error}") from error

        return SharedChannelLoader(read_channels)

    def _lead_loaders(self) -> Callable[[Path, WfdbHeader, int], Callable[[], pa.Array]]:
        """Return lazy lead callbacks sharing a conversion-wide recording cache."""
        shared_channels = self._shared_leads()

        def channel(record_base: Path, header: WfdbHeader, lead_idx: int) -> Callable[[], pa.Array]:
            request = _WfdbRead(
                record_base,
                header.sig_len,
                header.dat_names,
                header.gains,
                header.baselines,
                header.direct_read,
                tuple(header.sig_name),
            )
            return shared_channels.channel_loader(request, lead_idx)

        return channel

    def _recording_signals(self) -> Callable[[Path, WfdbHeader, tuple[SignalChannel, ...]], tuple[Signal, ...]]:
        """Return a signal factory with one shared cache for this conversion."""
        shared = self._shared_leads()

        def signals(base: Path, header: WfdbHeader, channels: tuple[SignalChannel, ...]) -> tuple[Signal, ...]:
            request = _WfdbRead(
                base,
                header.sig_len,
                header.dat_names,
                header.gains,
                header.baselines,
                header.direct_read,
                tuple(header.sig_name),
            )
            return Signal.from_shared_loader(loader=shared, recording_key=request, channels=channels)

        return signals
