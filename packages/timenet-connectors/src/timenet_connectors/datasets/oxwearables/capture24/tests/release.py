"""Write a small fake CAPTURE-24 release for the connector tests."""

from datetime import datetime, timedelta
import gzip
from pathlib import Path

from timenet.types.text import read_value


SLEEP = "7030 sleeping;MET 0.95"
SIT = "sitting;9060 sitting;MET 1.5"
WALK = "walking;17190 walking;MET 3.5"
# The fixture dictionary, as the release writes it: a raw label, then its class in each of the six schemes.
_DICTIONARY = (
    "annotation,label:WillettsSpecific2018,label:WillettsMET2018,label:DohertySpecific2018,"
    "label:Willetts2018,label:Doherty2018,label:Walmsley2020\n"
    f"{SLEEP},sleep,sleep,sleep,sleep,sleep,sleep\n"
    f'"{SIT}",sitting,sitting,sedentary-screen,sit-stand,sedentary,sedentary\n'
    f'"{WALK}",walking,walking,walking,walking,walking,light\n'
)
_START = datetime(2016, 11, 13, 2, 18)
# P101's second segment starts off the first segment's 10 s grid, after a gap of about an hour.
SECOND_SEGMENT_US = 3_615_685_000


def write_recording(path: Path, rows: list[tuple[int, str]]) -> None:
    """Write one ``time,x,y,z,annotation`` row per ``(offset_us, label)``.

    The row index determines x and y. Every row uses the same z value."""
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        handle.write("time,x,y,z,annotation\n")
        for index, (offset_us, label) in enumerate(rows):
            stamp = (_START + timedelta(microseconds=offset_us)).strftime("%Y-%m-%d %H:%M:%S.%f")
            quoted = f'"{label}"' if label else ""
            handle.write(f"{stamp},{index * 0.001:.8f},{-index * 0.001:.8f},0.98765432,{quoted}\n")


def _p001() -> list[tuple[int, str]]:
    """Four full bins and one half bin, sampled every 10 ms with no gap."""
    labels = (
        [SLEEP] * 1000  # bin 0: sleep in every scheme
        + [SIT] * 500
        + [WALK] * 500  # bin 1: a tie, which goes to the class first in string order
        + [""] * 999
        + [WALK]  # bin 2: one labeled sample, whose class the window takes
        + [""] * 1000  # bin 3: unlabeled, so dropped
        + [SLEEP] * 500  # bin 4: 500 samples, so dropped
    )
    return [(index * 10_000, label) for index, label in enumerate(labels)]


def _p101() -> list[tuple[int, str]]:
    """Two segments separated by a gap: 1500 samples, then 1432.

    The last 1000 samples form one full bin."""
    first = [(index * 10_000, SIT) for index in range(1500)]
    second = [(SECOND_SEGMENT_US + index * 10_000, WALK) for index in range(1432)]
    return first + second


def _p100() -> list[tuple[int, str]]:
    """One bin of sleep: the last participant of the train split."""
    return [(index * 10_000, SLEEP) for index in range(1000)]


def release(tmp_path: Path) -> Path:
    """Write the fixture release under ``tmp_path`` and return its directory."""
    root = tmp_path / "capture24"
    root.mkdir()
    (root / "annotation-label-dictionary.csv").write_text(_DICTIONARY, encoding="utf-8")
    (root / "metadata.csv").write_text("pid,age,sex\nP001,38-52,F\nP100,30-37,F\nP101,18-29,M\n", encoding="utf-8")
    write_recording(root / "P001.csv.gz", _p001())
    write_recording(root / "P100.csv.gz", _p100())
    write_recording(root / "P101.csv.gz", _p101())
    return root


def pid_of(record) -> str:
    """The participant id that a Record's ``subject_ids`` annotation holds."""
    (pid,) = next(read_value(a.value) for a in record.annotations if a.key == "subject_ids")
    return pid


def rewrite(path: Path, edit) -> None:
    """Rewrite a gzipped CSV after applying edit to its lines."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(edit(lines)) + "\n")


def with_x(line: str, value: str) -> str:
    """Replace the x-axis cell in a generated recording row with no quoted commas."""
    time, _, *rest = line.split(",", 2)
    return ",".join([time, value, *rest])
