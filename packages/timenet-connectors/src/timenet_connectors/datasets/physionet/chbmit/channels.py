"""Which EDF channels of CHB-MIT become signals, and what each one measures.

The montage is not fixed. Most files hold 23 bipolar EEG derivations of the 10-20 system, but some
cases change it between files: chb12 switches to referential leads, chb15 adds ``-Ref`` leads, and
a few files carry an ECG, EOG or vagus nerve stimulator channel. The connector therefore reads the
channels of each file from that file's own header and maps each label through this module.

The placeholders ``-`` and ``.`` become no signal. They name no electrode, and all 1791 of them in
chb01 to chb23 hold only zeros.

Every file writes ``T8-P8`` twice: once in the right temporal chain of the double banana, and once
to close the chain through FT9 and FT10. Both become signals. In 583 of 633 files the second
is a byte-identical copy of the first, but in chb12_32 to chb12_42 and in every chb15 file with the
pair, it differs in gain and above about 13 Hz, as if a separate amplifier input recorded it. The
first keeps the label, and a repeat is named ``T8-P8 (2)``, the same in every montage.

A label this module does not know stops the build, rather than becoming a signal of a guessed kind.
"""

from collections import Counter
from dataclasses import dataclass
import re

from timenet.errors import TimeFFormatError
from timenet.types import TimeSeriesSpec, ureg


# Labels that name no electrode, written where the recording software leaves a channel empty.
PLACEHOLDER_LABELS = frozenset({"-", "."})

# The unit every channel of the release states.
_UNIT = "uV"

_EEG = TimeSeriesSpec(spec_type="eeg", name="Scalp EEG", unit_value=ureg.microvolt)
_ECG = TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=ureg.microvolt)
_EOG = TimeSeriesSpec(spec_type="eog", name="EOG", unit_value=ureg.microvolt)
_VNS = TimeSeriesSpec(spec_type="vns", name="Vagus nerve stimulator channel", unit_value=ureg.microvolt)
# The release documents neither of these two. Each pairs an electrode on the body with another
# one, so it is a biopotential, and no more can be said about it.
_OTHER = TimeSeriesSpec(spec_type="biopotential", name="Other biopotential", unit_value=ureg.microvolt)

# Every non-EEG label the release writes.
_NON_EEG_SPECS = {
    "ECG": _ECG,
    "EKG1-EKG2": _ECG,
    "LOC-ROC": _EOG,  # left outer canthus to right outer canthus
    "VNS": _VNS,
    "EKG1-CHIN": _OTHER,
    "LUE-RAE": _OTHER,
}

# The scalp electrodes the release records. chb12 writes ``01`` in place of ``O1`` in some
# files. The connector keeps the label as the release writes it, so ``01`` is listed here too.
_ELECTRODES = frozenset(
    {
        "FP1", "FP2", "F3", "F4", "F7", "F8", "FZ",
        "FC1", "FC2", "FC5", "FC6", "FT9", "FT10",
        "C2", "C3", "C4", "C6", "CZ", "T7", "T8",
        "CP1", "CP2", "CP4", "CP5", "CP6",
        "P3", "P4", "P7", "P8", "PZ",
        "O1", "O2", "OZ", "01",
    }
)  # fmt: skip

# An EEG label is an electrode, alone or against a second electrode or a named reference.
_EEG_LABEL = re.compile(r"(?P<first>[A-Z0-9]+)(?:-(?P<second>[A-Z0-9]+|Ref))?")
_REFERENCES = frozenset({"Ref", "CS2"})


@dataclass(frozen=True)
class EdfChannel:
    """One channel of an EDF file that becomes a signal.

    ``index`` is its position in the header and ``label`` the label there. ``name`` is the signal
    name: the label, with the occurrence count added when the file writes the label again.
    """

    index: int
    label: str
    name: str
    spec: TimeSeriesSpec


def select_channels(labels: tuple[str, ...], units: tuple[str, ...], file_name: str) -> tuple[EdfChannel, ...]:
    """Give the channels of one file that become signals, in header order.

    Args:
        labels: The signal labels of the EDF header.
        units: The physical unit of each signal, from the same header.
        file_name: The file, for the error messages.

    Returns:
        One channel for each label that names a measurement. The second occurrence of a label is
        named ``<label> (2)``, the third ``<label> (3)``.

    Raises:
        TimeFFormatError: If a label names no known channel, or a channel states another unit.
    """
    selected: list[EdfChannel] = []
    occurrences: Counter[str] = Counter()
    for index, (label, unit) in enumerate(zip(labels, units, strict=True)):
        if label in PLACEHOLDER_LABELS:
            continue

        if unit != _UNIT:
            raise TimeFFormatError(f"{file_name}: channel {label!r} states the unit {unit!r}, not {_UNIT!r}")

        occurrences[label] += 1
        count = occurrences[label]
        name = label if count == 1 else f"{label} ({count})"
        selected.append(EdfChannel(index=index, label=label, name=name, spec=spec_for_label(label, file_name)))
    return tuple(selected)


def spec_for_label(label: str, file_name: str) -> TimeSeriesSpec:
    """Give the spec of one channel label.

    Args:
        label: The label as the EDF header writes it.
        file_name: The file, for the error message.

    Returns:
        The spec of what the channel measures.

    Raises:
        TimeFFormatError: If the label is neither a known non-EEG channel nor an EEG derivation of
            known electrodes.
    """
    if label in _NON_EEG_SPECS:
        return _NON_EEG_SPECS[label]

    match = _EEG_LABEL.fullmatch(label)
    if match is not None and match["first"] in _ELECTRODES:
        second = match["second"]
        if second is None or second in _ELECTRODES or second in _REFERENCES:
            return _EEG

    raise TimeFFormatError(f"{file_name}: the channel label {label!r} names no channel this release records")
