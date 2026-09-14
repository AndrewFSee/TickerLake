"""Date coercion, in one place because the alternative keeps going wrong.

``pandas.Timestamp`` subclasses ``datetime``, which subclasses ``date``. So the
obvious guard::

    day = value if isinstance(value, date) else pd.Timestamp(value).date()

passes for a Timestamp and leaves it unconverted. A Timestamp is then *not*
equal to the plain ``date`` it represents::

    pd.Timestamp("2026-09-04") == date(2026, 9, 4)   ->  False

which makes it silently unusable as a dict key or join value. DuckDB returns
``date32`` columns as Timestamps, so anything keying a lookup on a stored date
hits this.

It has caused three separate defects in this project -- membership comparisons,
the Finnhub price cross-check, and the Tiingo adjusted-price cross-check, where
it produced zero matched bars while every request succeeded and nothing logged
an error. Hence one helper, used everywhere, rather than the guard rewritten
per module.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def as_date(value: Any) -> date | None:
    """Coerce anything date-like to a plain ``datetime.date``.

    Returns None for null-ish input rather than raising, since these values come
    from vendor payloads where a missing date is normal.
    """
    if value is None:
        return None

    # The null check must come first. ``pd.NaT`` is itself an instance of
    # datetime, so an isinstance check ahead of this returns ``NaT.date()``,
    # which is NaT again -- a null that then travels as though it were a real
    # value. Wrapped because pd.isna raises on list-likes.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # Order matters here too: datetime must be excluded before the date check,
    # because datetime (and therefore Timestamp) satisfies isinstance(x, date).
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    try:
        stamp = pd.Timestamp(value)
    except Exception:
        # pandas raises DateParseError, which is not a ValueError subclass in
        # every version, so this catches broadly rather than guessing.
        return None
    # pd.Timestamp("") yields NaT rather than raising, so re-check the result.
    return None if pd.isna(stamp) else stamp.date()


def as_date_strict(value: Any) -> date:
    """Like :func:`as_date`, but raises when the value cannot be a date."""
    out = as_date(value)
    if out is None:
        raise ValueError(f"cannot interpret {value!r} as a date")
    return out
