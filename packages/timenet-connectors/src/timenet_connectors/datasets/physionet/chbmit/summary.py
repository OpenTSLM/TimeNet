"""Read ``chbNN-summary.txt``, the file in which CHB-MIT states its seizures.

Each case directory holds one summary. It lists the channels of the EDF files, and then one entry
for each file::

    Channels in EDF Files:
    **********************
    Channel 1: FP1-F7
    ...

    File Name: chb01_03.edf
    File Start Time: 13:43:04
    File End Time: 14:43:04
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds

A ``Channels changed:`` block can appear between two entries. Its channel list then holds for the
entries that follow it. Later cases number their seizures (``Seizure 1 Start Time:``) and some
pad the number with more than one space; the parser reads all of these forms.

The seizure times count whole seconds from the start of the file. The clock times are read and
not used: the EDF header states the same clock, and the connector times the files from that.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Self

from timenet.errors import TimeFFormatError
from timenet_connectors.bases.seizures import SeizureInterval


_FILE_NAME = re.compile(r"File Name: (?P<name>\S+\.edf)")
_SEIZURE_COUNT = re.compile(r"Number of Seizures in File: (?P<count>\d+)")
_SEIZURE_START = re.compile(r"Seizure(?: \d+)? Start Time:\s+(?P<seconds>\d+) seconds")
_SEIZURE_END = re.compile(r"Seizure(?: \d+)? End Time:\s+(?P<seconds>\d+) seconds")
_CHANNEL = re.compile(r"Channel \d+: (?P<label>.*)")
_CHANNEL_LIST_HEADINGS = ("Channels in EDF Files:", "Channels changed:")

# Lines that state nothing the connector reads: the sampling rate (each EDF header states its
# own), the clock times of a file (the EDF header states them too), and the rules under headings.
_IGNORED = re.compile(r"Data Sampling Rate: .*|File (Start|End) Time: .*|\*+|")


@dataclass(frozen=True)
class SummaryEntry:
    """What the summary states about one EDF file: its channels and its seizures."""

    file_name: str
    channels: tuple[str, ...]
    seizures: tuple[SeizureInterval, ...]

    def check_channels(self, header_labels: tuple[str, ...], path: Path) -> None:
        """Check that the EDF header names the channels the summary lists for this file.

        Args:
            header_labels: The signal labels of the EDF header, in header order.
            path: The EDF file, for the error message.

        Raises:
            TimeFFormatError: If the two lists differ.
        """
        if header_labels != self.channels:
            raise TimeFFormatError(
                f"{path}: the EDF header names the channels {list(header_labels)}, and the case summary lists "
                f"{list(self.channels)}"
            )


@dataclass
class _OpenEntry:
    """An entry while the parser still reads its lines."""

    file_name: str
    channels: tuple[str, ...]
    stated_count: int | None = None
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)

    def read_seizure_line(self, line: str) -> bool:
        """Take one line that states the seizures of this entry.

        Returns:
            Whether the line was a seizure count, onset or end.
        """
        if match := _SEIZURE_COUNT.fullmatch(line):
            self.stated_count = int(match["count"])
        elif match := _SEIZURE_START.fullmatch(line):
            self.starts.append(int(match["seconds"]))
        elif match := _SEIZURE_END.fullmatch(line):
            self.ends.append(int(match["seconds"]))
        else:
            return False

        return True

    def close(self, path: Path) -> SummaryEntry:
        """Check the entry and give its final form.

        Returns:
            The entry.

        Raises:
            TimeFFormatError: If the entry states no seizure count, or a count that differs from
                the onsets and ends it lists.
        """
        starts, ends = self.starts, self.ends
        if self.stated_count is None:
            raise TimeFFormatError(f"{path}: the entry of {self.file_name} states no number of seizures")

        if not self.stated_count == len(starts) == len(ends):
            raise TimeFFormatError(
                f"{path}: the entry of {self.file_name} states {self.stated_count} seizures and lists "
                f"{len(starts)} onsets and {len(ends)} ends"
            )

        seizures = tuple(
            SeizureInterval.from_whole_seconds(start, end) for start, end in zip(starts, ends, strict=True)
        )
        return SummaryEntry(file_name=self.file_name, channels=self.channels, seizures=seizures)


@dataclass(frozen=True)
class CaseSummary:
    """The parsed summary of one case: one entry for each EDF file it lists, in file order."""

    path: Path
    entries: tuple[SummaryEntry, ...]

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Parse a summary file.

        Args:
            path: The ``chbNN-summary.txt`` file.

        Returns:
            The summary.

        Raises:
            TimeFFormatError: If a line has a form the parser does not know, if a file is listed
                before any channel list or twice, or if an entry does not hold together.
        """
        entries: list[SummaryEntry] = []
        channels: list[str] = []
        reading_channels = False
        entry: _OpenEntry | None = None
        for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if line in _CHANNEL_LIST_HEADINGS:
                channels, reading_channels = [], True
            elif match := _CHANNEL.fullmatch(line):
                if not reading_channels:
                    raise TimeFFormatError(f"{path}:{number}: a channel line outside a channel list: {line!r}")

                channels.append(match["label"].strip())
            elif match := _FILE_NAME.fullmatch(line):
                if entry is not None:
                    entries.append(entry.close(path))

                if not channels:
                    raise TimeFFormatError(f"{path}:{number}: {match['name']} is listed before any channel list")

                reading_channels = False
                entry = _OpenEntry(file_name=match["name"], channels=tuple(channels))
            elif entry is not None and entry.read_seizure_line(line):
                continue
            elif not _IGNORED.fullmatch(line):
                raise TimeFFormatError(f"{path}:{number}: a line this parser does not know: {line!r}")

        if entry is not None:
            entries.append(entry.close(path))

        names = [one.file_name for one in entries]
        if len(set(names)) != len(names):
            raise TimeFFormatError(f"{path}: lists a file more than once")

        return cls(path=path, entries=tuple(entries))

    def entry_for(self, file_name: str) -> SummaryEntry:
        """Give the entry of one EDF file.

        Args:
            file_name: The name of the file, such as ``chb01_03.edf``.

        Returns:
            Its entry.

        Raises:
            TimeFFormatError: If the summary does not list the file. A file the summary leaves out
                has no statement about its seizures, so it cannot be given an exhaustive answer.
        """
        for entry in self.entries:
            if entry.file_name == file_name:
                return entry

        raise TimeFFormatError(f"{self.path}: does not list {file_name}, so it states nothing about its seizures")
