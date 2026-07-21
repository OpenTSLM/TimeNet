"""Semantic version value type."""

from dataclasses import dataclass

from timenet.errors import TimeFValidationError


@dataclass(frozen=True, order=True)
class Version:
    """A ``major.minor.patch`` semantic version with non-negative integer components.

    Frozen and ordered, so versions compare with the usual precedence
    (``Version(1, 2, 0) > Version(1, 1, 9)``).
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        """Reject non-integer or negative components.

        Raises:
            TimeFValidationError: If any component is not a non-negative ``int``.
        """
        for name, value in (("major", self.major), ("minor", self.minor), ("patch", self.patch)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TimeFValidationError(f"Version.{name} must be a non-negative int, got {value!r}")

    @classmethod
    def parse(cls, text: str) -> "Version":
        """Parse a ``major.minor.patch`` string.

        Args:
            text: The version string to parse.

        Returns:
            The parsed :class:`Version`.

        Raises:
            TimeFValidationError: If ``text`` is not exactly three integer components.
        """
        parts = text.split(".")
        if len(parts) != 3:
            raise TimeFValidationError(f"Version must be 'major.minor.patch', got {text!r}")
        try:
            major, minor, patch = (int(part) for part in parts)
        except ValueError:
            raise TimeFValidationError(f"Version components must be integers, got {text!r}") from None
        return cls(major, minor, patch)

    def __str__(self) -> str:
        """Return the canonical ``major.minor.patch`` string."""
        return f"{self.major}.{self.minor}.{self.patch}"
