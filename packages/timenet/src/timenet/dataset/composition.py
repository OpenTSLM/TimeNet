"""Copy parent record hierarchies while retaining their lazy value handles."""

from copy import deepcopy

from timenet.dataset.record import Record
from timenet.types import ArrayHandle, TextHandle, TextListHandle


def clone_record(record: Record) -> Record:
    """Copy a record for registration in another dataset, without reading its values.

    Returns:
        A separate hierarchy with fresh annotations and no assigned entity ids.
    """
    handles: list[ArrayHandle | TextHandle | TextListHandle] = []
    for signal in record.signals:
        handles.append(signal._values)
        if signal._time_offsets is not None:
            handles.append(signal._time_offsets)
    handles.extend(
        annotation.value
        for annotation in record.walk_annotations()
        if isinstance(annotation.value, (TextHandle, TextListHandle))
    )
    result = deepcopy(record, {id(handle): handle for handle in handles})
    result._id = None
    for signal in result.signals:
        object.__setattr__(signal, "_id", None)  # noqa: PLC2801 - Signal is frozen
    return result
