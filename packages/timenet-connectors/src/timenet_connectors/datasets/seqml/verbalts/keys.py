"""The ids and the annotation keys this connector writes. Pure functions, no I/O.

Every attribute key carries its component as a prefix. Two components can give one attribute name
two meanings: Weather's ``season`` is a calendar season, and ETTm1's is a code for the dominant
periodicity. Schema derivation refuses one key that yields two descriptors, so the prefix is needed.
"""

from enum import StrEnum


_ID_PREFIX = "verbalts"

# Every prefix is distinct, and none is a prefix of another, so no two attribute keys can collide.
_SLUG_BY_COMPONENT: dict[str, str] = {
    "synthetic_u": "synth_u",
    "synthetic_m": "synth_m",
    "Weather": "weather",
    "BlindWays": "blindways",
    "ETTm1": "ettm1",
    "istanbul_traffic": "istanbul",
}


class VerbalTsKey(StrEnum):
    """The keys of the two annotations that say where a window came from."""

    COMPONENT = "component"
    """The component the window was cut from, spelled as the release spells it."""

    SPLIT = "split"
    """The split the window sits in: ``train``, ``valid`` or ``test``."""


def record_id(component: str, split: str, row: int) -> str:
    """Build the id of one window.

    The row index is padded to five digits, so lexicographic order equals numeric order. The writer
    sorts series by ``source_id``, and every series of a window carries the window's id, so the
    writer walks each values file front to back.

    Args:
        component: The component name.
        split: The split name.
        row: The window's index inside its split.

    Returns:
        The record id.
    """
    return f"{_ID_PREFIX}-{component}-{split}-{row:05d}"


def attribute_key(component: str, attribute: str) -> str:
    """Build the annotation key for one attribute code of one component.

    Args:
        component: The component name.
        attribute: The attribute name from ``meta.json``.

    Returns:
        The component-prefixed annotation key.
    """
    return f"{_SLUG_BY_COMPONENT[component]}_{attribute}"


def codebook_key(component: str, attribute: str) -> str:
    """Build the annotation key that states how many codes one attribute takes.

    Args:
        component: The component name.
        attribute: The attribute name from ``meta.json``.

    Returns:
        The corpus-level annotation key.
    """
    return f"{attribute_key(component, attribute)}_options"


def codebook_id(component: str, attribute: str) -> str:
    """Build the id of the corpus-level codebook annotation for one attribute.

    Every generation task of a component lists these ids, so both ends build the id from the same
    two names.

    Args:
        component: The component name.
        attribute: The attribute name from ``meta.json``.

    Returns:
        The registered annotation's id.
    """
    return f"{_ID_PREFIX}-codebook-{_id_segment(component)}-{_id_segment(attribute)}"


def _id_segment(value: str) -> str:
    """Turn one source name into an id segment.

    Args:
        value: A component or attribute name, as the release spells it.

    Returns:
        The name in lowercase, with every underscore turned into a hyphen.
    """
    return value.lower().replace("_", "-")
