"""Tests for versioned training-table IO."""

import pandas as pd

from smarthub import io


def test_versioned_training_tables(tmp_path, monkeypatch):
    # Redirect the training dir to a temp location.
    monkeypatch.setattr(io, "TRAINING_DIR", tmp_path / "training")

    df1 = pd.DataFrame({"id": [1], "won_flag": [1]})
    df2 = pd.DataFrame({"id": [1, 2], "won_flag": [1, 0]})

    p1 = io.save_training_table(df1, "auto", version="2026-06-29T000000Z")
    p2 = io.save_training_table(df2, "auto", version="2026-06-30T000000Z")
    assert p1.exists() and p2.exists()
    assert p1 != p2  # both kept, not overwritten

    versions = io.training_versions("auto")
    assert versions == ["2026-06-29T000000Z", "2026-06-30T000000Z"]

    # default load = latest version
    latest = io.load_training_table("auto")
    assert len(latest) == 2

    # explicit older version
    older = io.load_training_table("auto", version="2026-06-29T000000Z")
    assert len(older) == 1


def test_training_manifest_written_and_loadable(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "TRAINING_DIR", tmp_path / "training")
    df = pd.DataFrame({"id": [1, 2], "won_flag": [1, 0]})
    meta = {"training_window_days": 21, "won_rate": 0.5}
    io.save_training_table(df, "auto", version="2026-06-30T000000Z", metadata=meta)

    loaded = io.load_training_metadata("auto")
    assert loaded["version"] == "2026-06-30T000000Z"
    assert loaded["lead_type"] == "auto"
    assert loaded["rows"] == 2
    assert loaded["training_window_days"] == 21
    assert "columns" in loaded


def test_load_training_table_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "TRAINING_DIR", tmp_path / "training")
    try:
        io.load_training_table("home")
    except io.DataNotFoundError:
        pass
    else:
        raise AssertionError("expected DataNotFoundError")
