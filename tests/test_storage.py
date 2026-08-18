"""Tests for the DuckDB + partitioned-Parquet storage backends."""

import pandas as pd
import pytest

from smarthub.core import storage
from smarthub.core.config import StorageSettings


def _frame(ids, won, updated):
    """Return a small lead frame with the given ids, won flags, and times."""
    n = len(ids)
    return pd.DataFrame(
        {
            "id": ids,
            "won": won,
            "updated_at": pd.to_datetime(updated),
            "created_at": pd.to_datetime(["2026-06-20 01:00"] * n),
            "pst_date": pd.to_datetime(["2026-06-20"] * n),
        }
    )


# --- DuckDB ---


def test_duckdb_upsert_updates_in_place(tmp_path):
    """DuckDB upsert updates existing rows in place instead of duplicating."""
    db = tmp_path / "s.duckdb"
    first = _frame([1, 2], ["false", "false"], ["2026-06-20 02:00", "2026-06-20 02:00"])
    assert storage.append_duckdb(first, path=db) == 2

    # re-pull id=1 with a resolved outcome -> should update, not duplicate
    second = _frame([1, 3], ["true", "false"], ["2026-06-20 05:00", "2026-06-20 05:00"])
    assert storage.append_duckdb(second, path=db) == 3  # ids 1,2,3

    out = storage.read_duckdb_table(path=db).set_index("id")
    assert out.loc[1, "won"] == "true"  # updated
    assert len(out) == 3


def test_duckdb_read_projects_columns(tmp_path):
    """read_duckdb_table projects requested columns, ignoring missing ones."""
    db = tmp_path / "s.duckdb"
    storage.append_duckdb(
        _frame([1, 2], ["true", "false"], ["2026-06-20 02:00"] * 2), path=db
    )
    # Ask for a subset (plus a column that doesn't exist -> silently ignored).
    out = storage.read_duckdb_table(path=db, columns=["id", "won", "nope"])
    assert list(out.columns) == ["id", "won"]  # projected; missing col dropped
    assert len(out) == 2


def test_duckdb_read_columns_none_returns_all(tmp_path):
    """columns=None returns every column."""
    db = tmp_path / "s.duckdb"
    storage.append_duckdb(_frame([1], ["true"], ["2026-06-20 02:00"]), path=db)
    out = storage.read_duckdb_table(path=db, columns=None)
    assert {"id", "won", "created_at"}.issubset(out.columns)


def test_duckdb_window_projects_columns(tmp_path):
    """read_duckdb_window projects columns while still filtering created_at."""
    db = tmp_path / "s.duckdb"
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "created_at": pd.to_datetime(["2026-06-01", "2026-06-20"]),
            "won": ["true", "false"],
        }
    )
    storage.append_duckdb(df, path=db)
    # Project to just id; window filter still works on the unselected created_at.
    out = storage.read_duckdb_window(days=5, path=db, columns=["id"])
    assert list(out.columns) == ["id"]
    assert out["id"].tolist() == [2]  # only the recent row


def test_duckdb_window(tmp_path):
    """read_duckdb_window returns rows within N days of the latest created_at."""
    db = tmp_path / "s.duckdb"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "created_at": pd.to_datetime(
                ["2026-06-01 00:00", "2026-06-19 00:00", "2026-06-20 00:00"]
            ),
        }
    )
    storage.append_duckdb(df, path=db)
    recent = storage.read_duckdb_window(7, path=db)  # anchored on max = 06-20
    assert set(recent["id"]) == {2, 3}


def test_duckdb_handles_timestamp_precision_mismatch(tmp_path):
    """append_duckdb aligns timestamp precision so ns pulls append cleanly."""
    db = tmp_path / "s.duckdb"
    first = pd.DataFrame(
        {
            "id": [1],
            "created_at": pd.to_datetime(["2026-06-20 01:00"]),
            "expiration_date": pd.to_datetime(["2026-07-20"]).astype("datetime64[s]"),
        }
    )
    storage.append_duckdb(first, path=db)

    second = pd.DataFrame(
        {
            "id": [2],
            "created_at": pd.to_datetime(["2026-06-21 01:00"]),
            "expiration_date": pd.to_datetime(["2026-07-21"]),  # ns precision
        }
    )
    # Must not raise a ConversionException.
    assert storage.append_duckdb(second, path=db) == 2
    out = storage.read_duckdb_table(path=db).set_index("id")
    assert set(out.index) == {1, 2}


def test_duckdb_adds_new_columns(tmp_path):
    """A later pull with extra columns alters the table and backfills NULL."""
    db = tmp_path / "s.duckdb"
    storage.append_duckdb(_frame([1], ["false"], ["2026-06-20 02:00"]), path=db)
    extra = _frame([2], ["false"], ["2026-06-20 02:00"]).assign(expected_revenue=9.5)
    storage.append_duckdb(extra, path=db)

    out = storage.read_duckdb_table(path=db).set_index("id")
    assert "expected_revenue" in out.columns
    assert out.loc[2, "expected_revenue"] == 9.5
    assert pd.isna(out.loc[1, "expected_revenue"])  # backfilled NULL


# --- Partitioned Parquet ---


def test_parquet_partition_layout_and_dedupe(tmp_path):
    """Parquet writes to a year/month/day path and dedupes on re-pull."""
    root = tmp_path / "leads"
    first = _frame([1, 2], ["false", "false"], ["2026-06-20 02:00", "2026-06-20 02:00"])
    storage.append_parquet(first, root)

    expected = root / "2026" / "06" / "20-06-2026.parquet"
    assert expected.exists()

    # same-day re-pull with id=1 resolved -> merged + deduped (latest wins)
    second = _frame([1], ["true"], ["2026-06-20 06:00"])
    storage.append_parquet(second, root)

    out = pd.read_parquet(expected).set_index("id")
    assert len(out) == 2
    assert out.loc[1, "won"] == "true"


def test_parquet_splits_by_day(tmp_path):
    """Rows spanning two days are written to separate day files."""
    root = tmp_path / "leads"
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "won": ["false", "false"],
            "updated_at": pd.to_datetime(["2026-06-20 02:00", "2026-06-21 02:00"]),
            "created_at": pd.to_datetime(["2026-06-20 02:00", "2026-06-21 02:00"]),
            "pst_date": pd.to_datetime(["2026-06-20", "2026-06-21"]),
        }
    )
    storage.append_parquet(df, root)
    assert (root / "2026" / "06" / "20-06-2026.parquet").exists()
    assert (root / "2026" / "06" / "21-06-2026.parquet").exists()
    assert len(storage.read_parquet_dataset(root)) == 2


def test_parquet_falls_back_to_created_at_when_date_col_missing(tmp_path):
    """Partitioning falls back to created_at when the date column is missing."""
    root = tmp_path / "leads"
    df = pd.DataFrame({"id": [1], "created_at": pd.to_datetime(["2026-06-20 02:00"])})
    storage.append_parquet(df, root, date_col="pst_date")
    assert (root / "2026" / "06" / "20-06-2026.parquet").exists()


# --- Settings facade ---


def test_save_pull_both_backends(tmp_path, monkeypatch):
    """save_pull writes both backends and reports row counts and paths."""
    monkeypatch.setenv("STORAGE_BACKEND", "both")
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "s.duckdb"))
    monkeypatch.setenv("PARQUET_DIR", str(tmp_path / "leads"))
    monkeypatch.setenv("PARTITION_DATE_COL", "pst_date")
    settings = StorageSettings.from_env()

    df = _frame([1, 2], ["false", "true"], ["2026-06-20 02:00", "2026-06-20 02:00"])
    results = storage.save_pull(df, settings)
    assert results["duckdb_rows"] == 2
    assert results["parquet_rows"] == 2
    # storage locations are reported for notifications
    assert results["duckdb_path"].endswith("s.duckdb")
    assert len(results["parquet_paths"]) == 1  # both rows land in one day file
    assert results["parquet_paths"][0].endswith("20-06-2026.parquet")

    loaded = storage.load_leads_raw(settings)
    assert len(loaded) == 2


def test_storage_settings_invalid_backend(monkeypatch):
    """StorageSettings.from_env rejects an unknown backend."""
    monkeypatch.setenv("STORAGE_BACKEND", "mongodb")
    from smarthub.core.config import ConfigError

    with pytest.raises(ConfigError):
        StorageSettings.from_env()


# --- Atomicity regressions (C2 / C3) ---


def test_duckdb_upsert_rolls_back_on_failed_insert(tmp_path):
    """C2: a failure between the DELETE and INSERT must not drop existing rows.

    The upsert is DELETE-then-INSERT; without a transaction, an INSERT failure
    would leave the overlapping window's rows deleted and never re-inserted.
    Force the INSERT to fail (non-castable value) and assert the prior row
    survives intact.
    """
    db = tmp_path / "s.duckdb"
    first = pd.DataFrame(
        {
            "id": [1, 2],
            "age": [30, 40],
            "created_at": pd.to_datetime(["2026-06-20"] * 2),
        }
    )
    assert storage.append_duckdb(first, path=db) == 2

    # id=1 re-pulled but 'age' can't cast to the table's INTEGER -> INSERT fails
    # after the DELETE has already removed id=1.
    bad = pd.DataFrame(
        {"id": [1], "age": ["oops"], "created_at": pd.to_datetime(["2026-06-21"])}
    )
    with pytest.raises(Exception):
        storage.append_duckdb(bad, path=db)

    out = storage.read_duckdb_table(path=db).set_index("id")
    assert len(out) == 2  # nothing lost
    assert out.loc[1, "age"] == 30  # original row rolled back, unchanged


def test_parquet_write_is_atomic_and_leaves_no_temp(tmp_path):
    """C3: a successful partition write leaves a valid file and no .tmp debris."""
    root = tmp_path / "leads"
    df = _frame([1, 2], ["false", "true"], ["2026-06-20 02:00"] * 2)
    storage.append_parquet(df, root)

    part = root / "2026" / "06" / "20-06-2026.parquet"
    assert part.exists()
    assert len(pd.read_parquet(part)) == 2
    # no leftover temp files from the write
    assert not list(part.parent.glob("*.parquet.tmp"))


def test_parquet_write_failure_preserves_existing_partition(tmp_path, monkeypatch):
    """C3: if the atomic rename fails, the previous partition is untouched and no
    temp file is left behind (crash-safety of the read-modify-write)."""
    root = tmp_path / "leads"
    first = _frame([1], ["false"], ["2026-06-20 02:00"])
    storage.append_parquet(first, root)
    part = root / "2026" / "06" / "20-06-2026.parquet"
    before = part.read_bytes()

    # Simulate a crash at the rename step.
    def boom(src, dst):
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(storage.os, "replace", boom)
    second = _frame([2], ["true"], ["2026-06-20 03:00"])
    with pytest.raises(OSError):
        storage.append_parquet(second, root)

    # Old partition intact, new (partial) write discarded, no temp debris.
    assert part.read_bytes() == before
    assert not list(part.parent.glob("*.parquet.tmp"))


def test_fetch_task_forwards_lead_type_ids_list(monkeypatch):
    """C1: the data-pull flow's fetch task must call fetch_leads with the plural
    ``lead_type_ids`` (a list), matching the real signature. Guards against the
    kwarg-name drift that made every scheduled run raise TypeError."""
    import inspect

    # flow imports prefect and data_pull.pull (which imports redshift_connector);
    # skip where either isn't installed (they are on the box / in the pull image,
    # but the CI unit-test env installs neither).
    pytest.importorskip("prefect")
    pytest.importorskip("redshift_connector")
    from smarthub.data_pull import flow, pull

    captured = {}

    def fake_fetch_leads(settings, min_s, max_s, **kwargs):
        captured.update(kwargs)
        # Fails if the forwarded kwargs don't fit the real signature.
        inspect.signature(pull.fetch_leads).bind(settings, min_s, max_s, **kwargs)
        return "DF"

    monkeypatch.setattr(flow, "fetch_leads", fake_fetch_leads)
    monkeypatch.setattr(
        flow,
        "PullSettings",
        type("PS", (), {"from_env": staticmethod(lambda: object())}),
    )

    out = flow.fetch.fn("2026-06-01 00:00:00", "2026-06-02 00:00:00", 6, True, True)
    assert out == "DF"
    assert captured["lead_type_ids"] == [6]
