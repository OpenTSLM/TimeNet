"""Parse dataset references of the form ``org/id@version``.

A reference pins a version with ``@`` (``chengsenwang/tsqa@1.0.0``). With no ``@`` (or ``@latest``) it
resolves to the latest committed version.
"""

from timenet.types.references import split_ref


__all__ = ["split_ref"]
