"""Name VerbalTS channels only as precisely as the release permits.

The paper names source variables, but the released NPY files and ``meta.json`` do not
map array positions or ``var_id`` codes to those names. The authors' loader keeps
array order without supplying a channel schema:
https://github.com/seqml/VerbalTS/blob/main/data/data.py
"""

from collections.abc import Sequence

from timenet.errors import TimeFFormatError


VARIABLE_COUNTS_BY_COMPONENT: dict[str, int] = {"ETTm1": 7, "istanbul_traffic": 3}
"""Published variable counts for components with one channel per window."""

_CHANNEL_COUNTS: dict[str, int] = {"Weather": 21, "BlindWays": 72}
"""Published widths for components with many channels per window."""


def signal_names(component: str, attributes: Sequence[int], n_signals: int) -> tuple[str, ...]:
    """Return stable positional names for the channels in one released window.

    ``var_id`` stays in the window's annotations. Its code does not prove a
    physical variable name without the missing source codebook.

    Args:
        component: The component name.
        attributes: The window's attribute codes, first one first.
        n_signals: The number of signals the window holds.

    Returns:
        One positional name per signal, in the order the array stores them.

    Raises:
        TimeFFormatError: If a channel count or variable code exceeds the released shape.
    """
    variable_count = VARIABLE_COUNTS_BY_COMPONENT.get(component)
    if variable_count is not None:
        var_id = int(attributes[0])
        if n_signals != 1 or not 0 <= var_id < variable_count:
            raise TimeFFormatError(
                f"{component} window has {n_signals} signal(s) and var_id {var_id}; expected one "
                f"signal and a var_id in [0, {variable_count})"
            )
        return (f"var_{var_id}",)
    expected = _CHANNEL_COUNTS.get(component)
    if expected is not None and n_signals != expected:
        raise TimeFFormatError(f"{component} window has {n_signals} signals, expected {expected}")
    return tuple(f"x{index:02d}" for index in range(n_signals))
