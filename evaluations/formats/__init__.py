"""One module per format under test. Each writes its own artifact and reads it back."""

from evaluations.formats.base import Format, FormatName


__all__ = ["Format", "FormatName"]
