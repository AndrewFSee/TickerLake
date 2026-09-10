"""Tests for the survivorship-bias guarantees.

These are the invariants the whole dataset's validity rests on, so they get
tested against explicit adversarial cases rather than happy paths.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter
from tickerlake.universe.membership import MAX_PLAUSIBLE_DAILY_REMOVALS, MembershipTracker
from tickerlake.universe.sources import (
    LiveConstituent,
    from_yahoo_symbol,
    intervals_from_snapshots,
    normalize_symbol,
    to_yahoo_symbol,
)


@pytest.fixture
def tracker(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    return MembershipTracker(
        paths=paths,
        writer=ParquetWriter(),
        index_name="SP500",
        history_start=date(2010, 1, 1),
        silent_delist_threshold=3,
        post_removal_grace_days=30,
    )


def _seed_frame(rows):
    return pd.DataFrame(rows, columns=["symbol", "start_date", "end_date"])


def _live(*symbols):
    return [LiveConstituent(symbol=s, company_name=f"{s} Inc") for s in symbols]


# ------------------------------------------------------------------- symbols


def test_symbol_dialect_roundtrip():
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert from_yahoo_symbol("BRK-B") == "BRK.B"
    assert normalize_symbol(" brk-b ") == "BRK.B"
    assert normalize_symbol("aapl") == "AAPL"


# ---------------------------------------------------------------- point-in-time


def test_members_on_respects_interval_bounds(tracker):
    tracker.seed(
        _seed_frame(
            [
                {"symbol": "OLD", "start_date": date(2010, 1, 1), "end_date": date(2015, 6, 30)},
                {"symbol": "NEW", "start_date": date(2018, 1, 1), "end_date": None},
            ]
        ),
        force=True,
    )
    # Before NEW joined, only OLD was a member.
    assert tracker.members_on(date(2012, 1, 1)) == ["OLD"]
    # Inclusive on the final day of membership.
    assert "OLD" in tracker.members_on(date(2015, 6, 30))
    # The day after removal, OLD is gone.
    assert tracker.members_on(date(2015, 7, 1)) == []
    # And today's list is never applied retroactively.
    assert "NEW" not in tracker.members_on(date(2012, 1, 1))
    assert tracker.members_on(date(2020, 1, 1)) == ["NEW"]


def test_removal_never_deletes_history(tracker):
    tracker.seed(
        _seed_frame([{"symbol": "GONE", "start_date": date(2010, 1, 1), "end_date": None}]),
        force=True,
    )
    tracker.refresh(_live("STAY"), observed_date=date(2024, 5, 2))

    df = tracker.load()
    # The removed symbol is still a row; only its interval closed.
    assert "GONE" in set(df["symbol"]), "removed symbols must never be deleted"
    gone = df[df["symbol"] == "GONE"].iloc[0]
    assert gone["end_date"] is not None
    assert gone["reason_removed"] == "detected:absent_from_live_universe"
    # And it remains queryable for the dates it actually was a member.
    assert "GONE" in tracker.members_on(date(2015, 1, 1))


def test_reentry_creates_a_second_interval(tracker):
    tracker.seed(
        _seed_frame(
            [{"symbol": "BACK", "start_date": date(2010, 1, 1), "end_date": date(2016, 1, 1)}]
        ),
        force=True,
    )
    tracker.refresh(_live("BACK"), observed_date=date(2024, 3, 1))

    rows = tracker.load()
    intervals = rows[rows["symbol"] == "BACK"]
    assert len(intervals) == 2, "re-entry must open a new interval, not mutate the old one"

    # The gap between the two intervals must remain a non-membership period.
    assert "BACK" in tracker.members_on(date(2012, 1, 1))
    assert "BACK" not in tracker.members_on(date(2020, 1, 1))
    assert "BACK" in tracker.members_on(date(2024, 6, 1))


def test_implausible_removal_count_is_rejected(tracker):
    symbols = [f"S{i:03d}" for i in range(60)]
    tracker.seed(
        _seed_frame(
            [{"symbol": s, "start_date": date(2010, 1, 1), "end_date": None} for s in symbols]
        ),
        force=True,
    )
    before = tracker.load().copy()

    # A parse failure that would drop far more names than any real index event.
    diff = tracker.refresh(_live("S000"), observed_date=date(2024, 5, 2))

    assert diff.rejected
    assert len(diff.removed) > MAX_PLAUSIBLE_DAILY_REMOVALS
    # A rejected diff must leave the stored table byte-for-byte untouched.
    pd.testing.assert_frame_equal(
        tracker.load().reset_index(drop=True), before.reset_index(drop=True)
    )
    assert len(tracker.current_members()) == len(symbols)


def test_added_and_removed_are_reported(tracker):
    tracker.seed(
        _seed_frame(
            [
                {"symbol": "KEEP", "start_date": date(2010, 1, 1), "end_date": None},
                {"symbol": "DROP", "start_date": date(2010, 1, 1), "end_date": None},
            ]
        ),
        force=True,
    )
    diff = tracker.refresh(_live("KEEP", "ADD"), observed_date=date(2024, 5, 2))

    assert diff.added == ["ADD"]
    assert diff.removed == ["DROP"]
    assert diff.has_changes
    assert diff.unchanged_count == 1


# ------------------------------------------------------------ silent delisting


def test_silent_delisting_flags_after_threshold(tracker):
    tracker.seed(
        _seed_frame([{"symbol": "QUIET", "start_date": date(2010, 1, 1), "end_date": None}]),
        force=True,
    )
    # Threshold is 3 in the fixture.
    assert tracker.record_data_observations({"QUIET": None}) == []
    assert tracker.record_data_observations({"QUIET": None}) == []
    assert tracker.record_data_observations({"QUIET": None}) == ["QUIET"]
    assert tracker.suspected_delisted() == ["QUIET"]

    # It is flagged, never removed.
    assert "QUIET" in set(tracker.load()["symbol"])


def test_returning_data_clears_the_flag(tracker):
    tracker.seed(
        _seed_frame([{"symbol": "BLIP", "start_date": date(2010, 1, 1), "end_date": None}]),
        force=True,
    )
    for _ in range(3):
        tracker.record_data_observations({"BLIP": None})
    assert tracker.suspected_delisted() == ["BLIP"]

    tracker.record_data_observations({"BLIP": date(2024, 5, 2)})
    assert tracker.suspected_delisted() == []
    row = tracker.load().iloc[0]
    assert row["consecutive_empty_runs"] == 0
    assert row["last_data_date"] == date(2024, 5, 2)


def test_tracked_symbols_retains_delisted_names(tracker):
    tracker.seed(
        _seed_frame(
            [
                {"symbol": "LIVE", "start_date": date(2010, 1, 1), "end_date": None},
                {"symbol": "DEAD", "start_date": date(2010, 1, 1), "end_date": date(2011, 1, 1)},
            ]
        ),
        force=True,
    )
    # A long-removed name is still tracked while it might still return data.
    assert set(tracker.tracked_symbols(as_of=date(2024, 1, 1))) == {"LIVE", "DEAD"}

    # Once flagged as silently delisted and outside the grace window, we stop
    # requesting it -- but its rows and history remain.
    for _ in range(3):
        tracker.record_data_observations({"DEAD": None})
    assert tracker.tracked_symbols(as_of=date(2024, 1, 1)) == ["LIVE"]
    assert "DEAD" in set(tracker.load()["symbol"])
    assert "DEAD" in tracker.members_on(date(2010, 6, 1))


# -------------------------------------------------------- interval rebuilding


def test_intervals_from_snapshots_handles_gaps():
    snapshots = pd.DataFrame(
        [
            {"date": date(2020, 1, 1), "symbol": "A"},
            {"date": date(2020, 1, 1), "symbol": "B"},
            {"date": date(2021, 1, 1), "symbol": "A"},
            {"date": date(2022, 1, 1), "symbol": "A"},
            {"date": date(2022, 1, 1), "symbol": "B"},
        ]
    )
    intervals = intervals_from_snapshots(snapshots)

    a = intervals[intervals["symbol"] == "A"]
    assert len(a) == 1 and a.iloc[0]["end_date"] is None

    # B left after 2020 and came back in 2022: two intervals, not one.
    b = intervals[intervals["symbol"] == "B"].sort_values("start_date")
    assert len(b) == 2
    assert b.iloc[0]["start_date"] == date(2020, 1, 1)
    assert b.iloc[0]["end_date"] == date(2020, 1, 1)
    assert b.iloc[1]["start_date"] == date(2022, 1, 1)
    assert b.iloc[1]["end_date"] is None


def test_seed_refuses_to_overwrite_without_force(tracker):
    tracker.seed(
        _seed_frame([{"symbol": "A", "start_date": date(2010, 1, 1), "end_date": None}]), force=True
    )
    with pytest.raises(RuntimeError, match="discard locally observed history"):
        tracker.seed(
            _seed_frame([{"symbol": "B", "start_date": date(2010, 1, 1), "end_date": None}])
        )


# ----------------------------------------------------- ETFs vs index members


@pytest.fixture
def dual_tracker(tmp_path):
    """A tracker collecting both S&P constituents and ETFs."""
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    return MembershipTracker(
        paths=paths,
        writer=ParquetWriter(),
        index_name="SP500",
        history_start=date(2010, 1, 1),
        collect_indices=["SP500", "ETF"],
    )


def _seed_and_register(tracker):
    tracker.seed(
        _seed_frame(
            [
                {"symbol": "AAPL", "start_date": date(2010, 1, 1), "end_date": None},
                {"symbol": "GONE", "start_date": date(2010, 1, 1), "end_date": date(2015, 6, 30)},
            ]
        ),
        force=True,
    )
    tracker.register_static({"SPY": "SPDR S&P 500", "XLK": "Technology"}, index_name="ETF")


def test_etfs_never_appear_in_point_in_time_membership(dual_tracker):
    """The whole reason ETFs get their own index_name.

    members_on() is the survivorship primitive. An ETF showing up there would
    silently inflate every historical universe and corrupt any backtest built
    on it.
    """
    _seed_and_register(dual_tracker)

    for as_of in (date(2012, 1, 1), date(2020, 1, 1), date(2026, 1, 1)):
        members = dual_tracker.members_on(as_of)
        assert "SPY" not in members
        assert "XLK" not in members

    assert dual_tracker.members_on(date(2012, 1, 1)) == ["AAPL", "GONE"]
    assert dual_tracker.members_on(date(2020, 1, 1)) == ["AAPL"]


def test_etfs_are_included_in_the_collection_universe(dual_tracker):
    """...but every fetcher must still pick them up."""
    _seed_and_register(dual_tracker)

    current = dual_tracker.current_members()
    assert "SPY" in current and "XLK" in current and "AAPL" in current
    assert set(dual_tracker.tracked_symbols()) >= {"AAPL", "GONE", "SPY", "XLK"}


def test_current_members_can_be_scoped_to_one_index(dual_tracker):
    _seed_and_register(dual_tracker)

    assert dual_tracker.current_members("SP500") == ["AAPL"]
    assert dual_tracker.current_members("ETF") == ["SPY", "XLK"]
    assert len(dual_tracker.current_members()) == 3


def test_registering_etfs_is_idempotent(dual_tracker):
    _seed_and_register(dual_tracker)
    before = len(dual_tracker.load())

    added, present = dual_tracker.register_static(
        {"SPY": "SPDR S&P 500", "XLK": "Technology"}, index_name="ETF"
    )
    assert added == []
    assert set(present) == {"SPY", "XLK"}
    assert len(dual_tracker.load()) == before, "re-registering must not duplicate rows"


def test_registering_only_appends_new_etfs(dual_tracker):
    _seed_and_register(dual_tracker)
    added, present = dual_tracker.register_static(
        {"SPY": "SPDR S&P 500", "GLD": "SPDR Gold Shares"}, index_name="ETF"
    )
    assert added == ["GLD"]
    assert present == ["SPY"]
    # XLK was left out of this call but must not be removed.
    assert "XLK" in dual_tracker.current_members("ETF")


def test_index_refresh_does_not_touch_etfs(dual_tracker):
    """A universe diff against Wikipedia must not see ETFs as departed members."""
    _seed_and_register(dual_tracker)

    diff = dual_tracker.refresh(_live("AAPL"), observed_date=date(2026, 3, 2))

    assert "SPY" not in diff.removed
    assert "XLK" not in diff.removed
    assert dual_tracker.current_members("ETF") == ["SPY", "XLK"]


def test_default_tracker_collects_only_its_index(tracker):
    """Without collect_indices, behaviour is unchanged."""
    assert tracker.collect_indices == ["SP500"]
