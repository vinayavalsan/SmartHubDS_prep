"""Tests for the task-config (YAML) loader."""

from smarthub.core import task_config


def test_reads_sections_and_types(tmp_path, monkeypatch):
    """task_config reads typed values from the YAML, defaulting when absent."""
    cfg = tmp_path / "t.yaml"
    cfg.write_text(
        "training:\n"
        "  model_type: logistic_regression\n"
        "  calibrate: false\n"
        "  test_size: 0.3\n"
        "  random_seed: 7\n"
        "prediction:\n"
        "  bid_step: 0.5\n"
    )
    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", str(cfg))
    task_config.reload()
    try:
        assert task_config.get("training", "model_type") == "logistic_regression"
        assert task_config.get_bool("training", "calibrate", True) is False
        assert task_config.get_float("training", "test_size", 0.2) == 0.3
        assert task_config.get_int("training", "random_seed", 42) == 7
        assert task_config.get_float("prediction", "bid_step", 0.25) == 0.5
        # missing key / section -> default
        assert task_config.get_int("training", "nope", 99) == 99
        assert task_config.get("nosec", "k", "d") == "d"
    finally:
        task_config.reload()


def test_native_yaml_bool_is_honored(tmp_path, monkeypatch):
    """A native YAML boolean is returned as-is (not only true/false strings)."""
    cfg = tmp_path / "t.yaml"
    cfg.write_text("training:\n  calibrate: true\n  drop_zero_variance: false\n")
    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", str(cfg))
    task_config.reload()
    try:
        assert task_config.get_bool("training", "calibrate", False) is True
        assert task_config.get_bool("training", "drop_zero_variance", True) is False
    finally:
        task_config.reload()


def test_missing_file_returns_defaults(monkeypatch):
    """A missing config file makes every getter return its default."""
    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", "/no/such/file.yaml")
    task_config.reload()
    try:
        assert task_config.get_bool("training", "calibrate", True) is True
        assert task_config.get_float("prediction", "bid_step", 0.25) == 0.25
        assert task_config.get("training", "model_type", "lightgbm") == "lightgbm"
    finally:
        task_config.reload()
