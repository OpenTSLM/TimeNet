"""What the release's array shapes and attribute codes mean. No I/O at all.

The artifact states no signal name anywhere. Every name here comes from the paper that released
the data, or from the position of the signal inside its window. The README quotes the sentences
these names rest on.
"""

from collections.abc import Sequence

from timenet.errors import TimeFFormatError


# The 21 Jena weather columns, in the order the paper lists them. The names keep the Jena CSV
# header's spelling, with the degree-square and the micro sign written in ASCII as "2" and "u".
WEATHER_SIGNALS: tuple[str, ...] = (
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
    "rain (mm)",
    "raining (s)",
    "SWDR (W/m2)",
    "PAR (umol/m2/s)",
    "max. PAR (umol/m2/s)",
    "Tlog (degC)",
    "CO2 (ppm)",
)

# The seven ETTm1 columns and the three istanbul_traffic ones, in the order the paper lists them. A
# window of either component holds one signal, and its own var_id code says which column that
# signal was cut from.
VARIABLES_BY_COMPONENT: dict[str, tuple[str, ...]] = {
    "ETTm1": ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"),
    "istanbul_traffic": ("TI", "TI_An", "TI_Av"),
}

_BLINDWAYS_JOINTS = 24
_BLINDWAYS_AXES: tuple[str, ...] = ("x", "y", "z")


def signal_names(component: str, attributes: Sequence[int], n_signals: int) -> tuple[str, ...]:
    """Name every signal of one window.

    Weather takes the 21 column names the paper lists, in that order. BlindWays names signal ``k``
    as joint ``k // 3`` on axis ``k % 3``. ETTm1 and istanbul_traffic ship one signal per window and
    name it from that window's own ``var_id`` code, which is its first attribute. Everything else is
    named by position.

    Args:
        component: The component name.
        attributes: The window's attribute codes, first one first.
        n_signals: The number of signals the window holds.

    Returns:
        One name per signal, in the order the array stores them.

    Raises:
        TimeFFormatError: If the signal count or the ``var_id`` code disagrees with what the
            component declares. A release that changed shape must fail loudly.
    """
    variables = VARIABLES_BY_COMPONENT.get(component)
    if variables is not None:
        var_id = int(attributes[0])
        if n_signals != 1 or not 0 <= var_id < len(variables):
            raise TimeFFormatError(
                f"{component} window has {n_signals} signal(s) and var_id {var_id}; expected one "
                f"signal and a var_id in [0, {len(variables)})"
            )
        return (variables[var_id],)
    if component == "BlindWays":
        expected = _BLINDWAYS_JOINTS * len(_BLINDWAYS_AXES)
        if n_signals != expected:
            raise TimeFFormatError(f"BlindWays window has {n_signals} signals, expected {expected}")
        return tuple(f"j{index // 3:02d}_{_BLINDWAYS_AXES[index % 3]}" for index in range(n_signals))
    if component == "Weather":
        if n_signals != len(WEATHER_SIGNALS):
            raise TimeFFormatError(f"Weather window has {n_signals} signals, expected {len(WEATHER_SIGNALS)}")
        return WEATHER_SIGNALS
    return tuple(f"x{index}" for index in range(n_signals))
