"""The shared pint unit registry.

TimeNet uses `pint <https://pint.readthedocs.io>`_ for all physical units. This registry owns every
definition and conversion. Units built from another registry will not compare or convert cleanly, so
always reference units through :data:`ureg` (``ureg.hertz``, ``ureg.millivolt``, ...).

The registry is private to TimeNet. Importing this module does not call
``pint.set_application_registry``, so a host application keeps its own registry. Units cross process
and storage boundaries as names, never as registry-bound objects. The manifest codec writes
``str(unit)`` and reads it back through ``ureg.Unit(...)``, and
:class:`~timenet.types.specs.TimeSeriesSpec` pickles the same way.

That covers every unit TimeNet defines. It cannot cover a bare :class:`pint.Unit` or
:class:`pint.Quantity` that you pickle yourself. Those carry only the unit name and resolve it
against pint's *application* registry on unpickle, which does not know ``beat`` or ``bpm``::

    pickle.loads(pickle.dumps(ureg.bpm))  # UndefinedUnitError: 'bpm' is not defined

Pint's application registry is the only hook for that. If you need it, call
:func:`use_as_application_registry` once at application start.
"""

import pint

from timenet.errors import TimeFValidationError


ureg = pint.UnitRegistry()

# Non-physical units pint does not ship. `beat` is a dimensionless count with its own base dimension,
# so heart-rate units stay distinct from plain frequencies.
ureg.define("beat = [beat]")
ureg.define("bpm = beat / minute")


def normalize_unit(unit: "str | pint.Unit | None") -> str | None:
    """Validate a unit against the shared registry, rejecting an unrecognized unit string.

    A :class:`pint.Unit` becomes its canonical name. The function keeps a unit string as written but
    validates it, and raises on an unknown one. ``None`` passes through. The result is always a
    string or ``None``, so serialization is unchanged. Every type that carries a unit uses this
    function, so annotation values and scalar task targets accept and store units the same way.

    Args:
        unit: A :class:`pint.Unit`, a unit string (for example ``"years"``), or ``None``.

    Returns:
        The unit as a string, or ``None``.

    Raises:
        TimeFValidationError: If ``unit`` is a string the shared registry does not recognize.
    """
    if unit is None:
        return None
    if isinstance(unit, pint.Unit):
        return str(unit)
    try:
        ureg.Unit(unit)
    except (pint.UndefinedUnitError, pint.errors.DefinitionSyntaxError, ValueError) as exc:
        raise TimeFValidationError(
            f"unknown unit {unit!r}; pass a pint unit (e.g. ureg.millivolt) or a unit string pint "
            f"recognizes, or omit unit="
        ) from exc
    return unit


def use_as_application_registry() -> None:
    """Make TimeNet's registry pint's process-wide application registry.

    Opt in for two cases. First, when you pickle bare :class:`pint.Unit` or :class:`pint.Quantity`
    objects built from :data:`ureg`, because custom units such as ``bpm`` otherwise fail to unpickle.
    Second, when you want units from :data:`ureg` to compare and convert against units another
    library built.

    Importing this module does not call this function. It replaces the process-wide registry, so a host
    with its own registry can break. Call it once from application startup, where the decision is yours.

    This is a global assignment, not a merge. The last call wins, and any custom units that a
    previously installed registry defined stop resolving.
    """
    pint.set_application_registry(ureg)
