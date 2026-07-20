"""The shared pint unit registry.

TimeNet uses `pint <https://pint.readthedocs.io>`_ for all physical units instead of a hand-rolled
system. This registry owns every definition and conversion; units built from any other registry will
not compare or convert cleanly, so always reference units through :data:`ureg` (``ureg.hertz``,
``ureg.millivolt``, ...).

The registry is private to TimeNet: importing this module does not call
``pint.set_application_registry``, so a host application keeps whatever registry it already had.
Nothing TimeNet owns depends on process-global pint state. Units cross process and storage boundaries
as names, never as registry-bound objects: the manifest codec writes ``str(unit)`` and reads it back
through ``ureg.Unit(...)``, and :class:`~timenet.types.specs.TimeSeriesSpec` pickles the same way.

That covers everything TimeNet defines, but it cannot cover a bare :class:`pint.Unit` or
:class:`pint.Quantity` you pickle yourself. Those carry only the unit's name and resolve it against
pint's *application* registry on unpickle, which does not know ``beat`` or ``bpm``::

    pickle.loads(pickle.dumps(ureg.bpm))  # UndefinedUnitError: 'bpm' is not defined

There is no per-object fix for that; pint's application registry is the only hook. Call
:func:`use_as_application_registry` once at application start if you need it.
"""

import pint


ureg = pint.UnitRegistry()

# Non-physical units pint does not ship. `beat` is a dimensionless count given its own base dimension
# so heart-rate units stay distinct from plain frequencies.
ureg.define("beat = [beat]")
ureg.define("bpm = beat / minute")


def use_as_application_registry() -> None:
    """Make TimeNet's registry pint's process-wide application registry.

    Opt in when you pickle bare :class:`pint.Unit` or :class:`pint.Quantity` objects built from
    :data:`ureg` (custom units such as ``bpm`` otherwise fail to unpickle), or when you want units
    from :data:`ureg` to compare and convert against units another library built.

    Deliberately not called on import. It replaces the process-wide registry, so a library doing it
    behind your back would silently break a host that has its own. Call it from application startup,
    where the decision is yours to make, and only once.

    Note this is a global assignment, not a merge: whichever call runs last wins, and any custom units
    a previously installed registry defined stop resolving.
    """
    pint.set_application_registry(ureg)
