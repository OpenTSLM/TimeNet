"""The names this connector writes into a dataset.

An annotation key and a record id both reach a consumer as strings, and several modules state
the same ones. They are named here so that a rename cannot leave one site behind.
"""

from enum import StrEnum


# The prefix of every id this connector writes: records, series, annotations and tasks.
ID_PREFIX = "time-mmd"


class AnnotationKey(StrEnum):
    """The key of an annotation this connector builds.

    A key whose annotation carries a span states something the release wrote about a stretch
    of calendar days. The rest state something about the whole record.
    """

    REPORT_FACT = "report_fact"
    REPORT_PREDICTION = "report_prediction"
    SEARCH_FACT = "search_fact"
    SEARCH_PREDICTION = "search_prediction"
    TARGET_SIGNAL = "target_signal"
    SOURCE_COLUMNS = "source_columns"
    CBSA = "cbsa"
    CBSA_CODE = "cbsa_code"


class TextKind(StrEnum):
    """The two textual sources of the release, as the suffix of its file names spells them.

    A ``report`` row summarizes a report the authors selected for the domain. A ``search`` row
    summarizes the web search results of one week.
    """

    REPORT = "report"
    SEARCH = "search"


def record_id_of(domain: str) -> str:
    """Give the id of the one record a domain becomes.

    Args:
        domain: The lowercase name of the domain, as :mod:`domains` states it.

    Returns:
        The record id, which every id inside the record starts with.
    """
    return f"{ID_PREFIX}-{domain}"
