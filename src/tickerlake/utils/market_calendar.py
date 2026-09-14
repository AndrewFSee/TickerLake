"""NYSE trading calendar.

Why this exists
---------------
The scheduler fires Monday to Friday, which already excludes weekends -- but
market holidays fall on weekdays, and on those days the sources do not fail
cleanly in the same way:

* OHLCV and intraday bars simply return nothing for the closed date. Harmless.
* SEC and FINRA publish no file and return 403/404. Handled.
* **Yahoo still serves a complete option chain**, carrying the *previous*
  session's quotes. Measured on a closed Saturday against the stored Friday
  snapshot: bid, ask and volume were 100% identical across all 2,412 matched
  contracts.

That last one is the problem. The rows are not duplicates in the key sense -- a
different ``snapshot_date`` makes them distinct -- so nothing would reject them.
They would sit in the dataset as a genuine-looking session whose every quote
happens to equal the previous day's, which reads as a real zero-change day
rather than a non-event. For a model learning from daily changes that is worse
than an absent date, because absence is obvious and this is not.

Why not a calendar library
--------------------------
``pandas_market_calendars`` and ``exchange_calendars`` both do this well, but
NYSE holidays are a closed, algorithmic set and this avoids a dependency for
~60 lines. Note that the federal calendar is *not* a substitute: NYSE closes
for Good Friday, which is not a federal holiday, and stays open for Columbus
Day and Veterans Day, which are.

Not covered: ad-hoc closures (state funerals, 9/11, Hurricane Sandy). Those are
rare, unpredictable, and cost only a wasted run.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def easter_sunday(year: int) -> date:
    """Gregorian Easter (Meeus/Jones/Butcher). Needed only for Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth occurrence of a weekday in a month (Monday == 0)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last occurrence of a weekday in a month."""
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date | None:
    """Shift a fixed-date holiday off the weekend, NYSE-style.

    Sunday moves to Monday. Saturday moves *back* to Friday -- except for New
    Year's Day, where the exchange simply does not observe it (handled by the
    caller returning None).
    """
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset[date]:
    """NYSE full-day closures for a calendar year."""
    days: set[date] = set()

    # New Year's Day. A Saturday Jan 1 is not observed at all: the exchange does
    # not close the preceding 31 December.
    jan1 = date(year, 1, 1)
    if jan1.weekday() != 5:
        days.add(_observed(jan1))

    days.add(_nth_weekday(year, 1, 0, 3))  # MLK Jr Day
    days.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    days.add(easter_sunday(year) - timedelta(days=2))  # Good Friday
    days.add(_last_weekday(year, 5, 0))  # Memorial Day

    if year >= 2022:  # Juneteenth, NYSE from 2022
        days.add(_observed(date(year, 6, 19)))

    days.add(_observed(date(year, 7, 4)))  # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    days.add(_observed(date(year, 12, 25)))  # Christmas

    return frozenset(d for d in days if d is not None and d.year == year)


def is_trading_day(day: date) -> bool:
    """True when the NYSE holds a regular session on ``day``."""
    if day.weekday() >= 5:
        return False
    return day not in nyse_holidays(day.year)


def why_closed(day: date) -> str | None:
    """Human-readable reason a date is not a trading day, or None if it is."""
    if day.weekday() == 5:
        return "Saturday"
    if day.weekday() == 6:
        return "Sunday"
    if day in nyse_holidays(day.year):
        return f"NYSE holiday ({_holiday_name(day)})"
    return None


def _holiday_name(day: date) -> str:
    year = day.year
    named = {
        _observed(date(year, 1, 1)): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Washington's Birthday",
        easter_sunday(year) - timedelta(days=2): "Good Friday",
        _last_weekday(year, 5, 0): "Memorial Day",
        _observed(date(year, 6, 19)): "Juneteenth",
        _observed(date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving",
        _observed(date(year, 12, 25)): "Christmas",
    }
    return named.get(day, "unknown")


def previous_trading_day(day: date) -> date:
    """The most recent trading day strictly before ``day``."""
    cursor = day - timedelta(days=1)
    while not is_trading_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def trading_days_between(start: date, end: date) -> list[date]:
    """Inclusive list of trading days in a range."""
    out, cursor = [], start
    while cursor <= end:
        if is_trading_day(cursor):
            out.append(cursor)
        cursor += timedelta(days=1)
    return out
