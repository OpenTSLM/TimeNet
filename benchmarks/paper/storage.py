"""The storage rule, applied once for all six rows.

Apparent size: the sum of ``st_size`` over regular files, symlinks skipped and never followed,
extracted tree only, decimal GB. Allocated size buys nothing here, while the errors that matter are
at 100% scale. A Git-LFS clone keeps a second copy of every object; a HuggingFace ``snapshots/``
tree is symlinks into ``blobs/``, so following them counts every byte twice.

No number is ever sourced from ``du``. ``du -sk`` reports KiB of allocated blocks, and reading that
output as decimal GB is how a tree of 8,715,229,580 B came to be printed as 8.10.

Three tiers:

* Tier A is counted on both sides and is what the table prints. Every file the lane's loader opens,
  plus everything needed to address the items: manifest, index, control tables.
* Tier B is reported in the appendix and excluded. Content one side ships that the canonical item
  does not carry.
* Tier C is a derived cache, charged to the side whose loader needs one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from benchmarks.paper.errors import StorageMeasurementError


TIER_A = "tier_a"
TIER_B = "tier_b"
TIER_C = "tier_c"
TIERS = (TIER_A, TIER_B, TIER_C)

BYTES_PER_GB = 1_000_000_000
"""Decimal, matching the storage rule and the block byte budget."""


@dataclass(frozen=True)
class ScannedFile:
    """One regular file found under a tree.

    Attributes:
        relpath: Path relative to the scanned root, with forward slashes.
        size: ``st_size``, the apparent size.
    """

    relpath: str
    size: int


@dataclass(frozen=True)
class TierRule:
    """A rule that moves matching files out of Tier A.

    Attributes:
        tier: The tier matching files land in.
        reason: One short sentence for the appendix. A tier without a reason is not finished.
        matches: Called with each file's relative path. True moves that file.
    """

    tier: str
    reason: str
    matches: Callable[[str], bool]

    def __post_init__(self) -> None:
        """Refuse a rule that has no tier or no reason.

        Raises:
            StorageMeasurementError: If the tier is unknown or the reason is empty.
        """
        if self.tier not in TIERS:
            raise StorageMeasurementError(f"tier must be one of {list(TIERS)}, got {self.tier!r}")
        if not self.reason.strip():
            raise StorageMeasurementError(f"the {self.tier} rule must state why those bytes are not Tier A")


@dataclass(frozen=True)
class TierCCharge:
    """A derived cache charged to one side's storage.

    Attributes:
        bytes: The cache's apparent size.
        reason: Which cache it is and which loader writes it.
    """

    bytes: int
    reason: str

    def __post_init__(self) -> None:
        """Refuse a charge that is negative or has no reason.

        Raises:
            StorageMeasurementError: If the size is negative, or a non-zero charge has no reason.
        """
        if self.bytes < 0:
            raise StorageMeasurementError(f"Tier C bytes must not be negative, got {self.bytes}")
        if self.bytes and not self.reason.strip():
            raise StorageMeasurementError("charging Tier C bytes needs a reason naming the cache and its owner")


@dataclass(frozen=True)
class TierTotal:
    """What one tier holds.

    Attributes:
        tier: Which tier.
        bytes: Apparent bytes.
        files: How many regular files.
        reasons: The distinct reasons that put files here. Empty for Tier A.
    """

    tier: str
    bytes: int
    files: int
    reasons: tuple[str, ...] = ()

    @property
    def gb(self) -> float:
        """Return the tier's size in decimal GB.

        Returns:
            Bytes over 1e9.
        """
        return self.bytes / BYTES_PER_GB

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of every field plus the decimal GB.
        """
        return {
            "tier": self.tier,
            "bytes": self.bytes,
            "files": self.files,
            "gb": self.gb,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StorageLedger:
    """The three tiers of one representation of one dataset.

    Attributes:
        dataset: Dataset id.
        representation: ``original``, ``timef`` or ``control``.
        root: The tree that was scanned.
        totals: One total per tier that holds anything.
    """

    dataset: str
    representation: str
    root: str
    totals: dict[str, TierTotal] = field(default_factory=dict)

    def total(self, tier: str) -> TierTotal:
        """Return one tier's total, empty when nothing landed in it.

        Returns:
            The total.
        """
        return self.totals.get(tier, TierTotal(tier=tier, bytes=0, files=0))

    @property
    def tier_a_bytes(self) -> int:
        """Return the bytes the table prints for this cell.

        Returns:
            Tier A apparent bytes.
        """
        return self.total(TIER_A).bytes

    @property
    def tier_b_bytes(self) -> int:
        """Return the bytes the appendix reports and the table excludes.

        Returns:
            Tier B apparent bytes.
        """
        return self.total(TIER_B).bytes

    @property
    def tier_c_bytes(self) -> int:
        """Return the derived-cache bytes charged to this side.

        Returns:
            Tier C apparent bytes.
        """
        return self.total(TIER_C).bytes

    @property
    def tier_a_gb(self) -> float:
        """Return Tier A in decimal GB.

        Returns:
            Tier A bytes over 1e9.
        """
        return decimal_gb(self.tier_a_bytes)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping with one entry per tier that holds anything.
        """
        return {
            "dataset": self.dataset,
            "representation": self.representation,
            "root": self.root,
            "totals": {tier: total.as_dict() for tier, total in sorted(self.totals.items())},
        }


def decimal_gb(size: int) -> float:
    """Return a byte count in decimal GB.

    Returns:
        Bytes over 1e9. Never GiB: PhysioNet's "8.1 GB" is a GiB number, and mixing the two
        understates one side of the comparison by 7.6%.
    """
    return size / BYTES_PER_GB


def scan_tree(root: Path) -> Iterator[ScannedFile]:
    """Walk a tree and yield every regular file with its apparent size.

    Symlinks are skipped and never followed, in either direction: a link is not counted itself, and
    a link to a directory is not descended into. A HuggingFace ``snapshots/`` tree is entirely
    symlinks into ``blobs/``, so following them counts the corpus twice.

    Args:
        root: The tree to walk.

    Yields:
        One entry per regular file, in sorted path order.

    Raises:
        StorageMeasurementError: If the root does not exist.
    """
    if not root.exists():
        raise StorageMeasurementError(f"storage tree {root} does not exist")
    if root.is_file() and not root.is_symlink():
        yield ScannedFile(relpath=root.name, size=root.stat().st_size)
        return
    if not root.is_dir():
        raise StorageMeasurementError(f"storage tree {root} is neither a regular file nor a directory")
    yield from _walk(root, root)


def apparent_bytes(root: Path) -> int:
    """Return the apparent size of a tree.

    Returns:
        The sum of ``st_size`` over every regular file, symlinks skipped.
    """
    return sum(entry.size for entry in scan_tree(root))


def measure_tree(
    root: Path,
    *,
    dataset: str,
    representation: str,
    rules: Sequence[TierRule] = (),
    tier_c: TierCCharge | None = None,
) -> StorageLedger:
    """Scan a tree and split it into tiers.

    A file lands in the tier of the first rule that matches it, and in Tier A when none does. Tier C
    is passed in rather than scanned, because a derived cache lives outside the artifact tree.

    Args:
        root: The extracted tree of this representation.
        dataset: Dataset id.
        representation: ``original``, ``timef`` or ``control``.
        rules: The Tier B and Tier C rules, in priority order.
        tier_c: The derived-cache charge, or ``None`` when this side needs no cache.

    Returns:
        The ledger.

    Raises:
        StorageMeasurementError: If the tree holds no regular file.
    """
    sizes: dict[str, int] = dict.fromkeys(TIERS, 0)
    counts: dict[str, int] = dict.fromkeys(TIERS, 0)
    reasons: dict[str, list[str]] = {tier: [] for tier in TIERS}
    found = False
    for entry in scan_tree(root):
        found = True
        tier, reason = _tier_of(entry.relpath, rules)
        sizes[tier] += entry.size
        counts[tier] += 1
        if reason and reason not in reasons[tier]:
            reasons[tier].append(reason)
    if not found:
        raise StorageMeasurementError(f"storage tree {root} holds no regular file")
    if tier_c is not None and tier_c.bytes:
        sizes[TIER_C] += tier_c.bytes
        reasons[TIER_C].append(tier_c.reason)
    totals = {
        tier: TierTotal(tier=tier, bytes=sizes[tier], files=counts[tier], reasons=tuple(reasons[tier]))
        for tier in TIERS
        if sizes[tier] or counts[tier]
    }
    return StorageLedger(dataset=dataset, representation=representation, root=str(root), totals=totals)


def suffix_rule(tier: str, reason: str, suffixes: Iterable[str]) -> TierRule:
    """Return a rule that matches files by extension.

    Args:
        tier: Where matching files go.
        reason: Why they are not Tier A.
        suffixes: Extensions to match, with the dot, case-insensitive.

    Returns:
        The rule.
    """
    wanted = tuple(suffix.lower() for suffix in suffixes)
    return TierRule(tier=tier, reason=reason, matches=lambda relpath: relpath.lower().endswith(wanted))


def prefix_rule(tier: str, reason: str, prefixes: Iterable[str]) -> TierRule:
    """Return a rule that matches files by the directory they sit under.

    Args:
        tier: Where matching files go.
        reason: Why they are not Tier A.
        prefixes: Relative directory prefixes, with forward slashes.

    Returns:
        The rule.
    """
    wanted = tuple(prefix.strip("/") + "/" for prefix in prefixes)
    return TierRule(tier=tier, reason=reason, matches=lambda relpath: relpath.startswith(wanted))


def _tier_of(relpath: str, rules: Sequence[TierRule]) -> tuple[str, str]:
    """Return the tier one file lands in, and why.

    Returns:
        The tier and its reason. Tier A carries no reason.
    """
    for rule in rules:
        if rule.matches(relpath):
            return rule.tier, rule.reason
    return TIER_A, ""


def _walk(root: Path, current: Path) -> Iterator[ScannedFile]:
    """Recurse one directory, skipping symlinks in both roles.

    Yields:
        One entry per regular file, in sorted name order.
    """
    with os.scandir(current) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.is_symlink():
                continue
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                yield from _walk(root, path)
            elif entry.is_file(follow_symlinks=False):
                yield ScannedFile(
                    relpath=path.relative_to(root).as_posix(), size=entry.stat(follow_symlinks=False).st_size
                )
