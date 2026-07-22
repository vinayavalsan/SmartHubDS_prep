"""Tests for configuration loading and validation."""

import pytest

from smarthub.core.config import ConfigError, RedshiftSettings, SSHSettings


def test_redshift_settings_missing_var_raises(monkeypatch):
    """RedshiftSettings.from_env raises when a required env var is missing."""
    for var in (
        "REDSHIFT_HOST",
        "REDSHIFT_DB",
        "REDSHIFT_USER",
        "REDSHIFT_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError):
        RedshiftSettings.from_env()


def test_redshift_settings_from_env(monkeypatch):
    """RedshiftSettings.from_env reads host/port and applies default timeout."""
    monkeypatch.setenv("REDSHIFT_HOST", "redshift.example.com")
    monkeypatch.setenv("REDSHIFT_DB", "analytics")
    monkeypatch.setenv("REDSHIFT_USER", "anton")
    monkeypatch.setenv("REDSHIFT_PASSWORD", "secret")
    monkeypatch.setenv("REDSHIFT_PORT", "5439")

    settings = RedshiftSettings.from_env()
    assert settings.host == "redshift.example.com"
    assert settings.port == 5439
    assert settings.connect_timeout == 10


def test_ssh_settings_missing_key_file_raises(monkeypatch, tmp_path):
    """SSHSettings.from_env raises when the private key file does not exist."""
    monkeypatch.setenv("SSH_HOST", "bastion.example.com")
    monkeypatch.setenv("SSH_USER", "ec2-user")
    monkeypatch.setenv("SSH_PRIVATE_KEY_PATH", str(tmp_path / "does_not_exist"))
    with pytest.raises(ConfigError):
        SSHSettings.from_env()


def test_training_window_days_from_config_and_default(monkeypatch, tmp_path):
    """training_window_days comes from the YAML; defaults to 21 when absent."""
    from smarthub.core import task_config
    from smarthub.core.config import training_window_days

    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", str(tmp_path / "absent.yaml"))
    task_config.reload()
    assert training_window_days() == 21  # default

    cfg = tmp_path / "t.yaml"
    cfg.write_text("feature_engineering:\n  training_window_days: 45\n")
    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", str(cfg))
    task_config.reload()
    assert training_window_days() == 45

    cfg.write_text("feature_engineering:\n  training_window_days: 0\n")  # all data
    task_config.reload()
    assert training_window_days() == 0
    task_config.reload()  # restore real config for other tests


def test_training_config_defaults_to_repo_yaml(monkeypatch):
    """load_training_config falls back to the checked-in repo default YAML."""
    from smarthub.train_and_predict import config as training_config

    monkeypatch.delenv("SMARTHUB_TRAINING_CONFIG", raising=False)
    cfg = training_config.load_training_config()

    assert cfg.model_type in {"logistic_regression", "xgboost", "lightgbm"}
    assert cfg.raw["resolved"]["config_path"].endswith("config/training.yaml")


def test_ssh_settings_ok(monkeypatch, tmp_path):
    """SSHSettings.from_env returns settings with defaults when key exists."""
    key = tmp_path / "id_rsa"
    key.write_text("dummy")
    monkeypatch.setenv("SSH_HOST", "bastion.example.com")
    monkeypatch.setenv("SSH_USER", "ec2-user")
    monkeypatch.setenv("SSH_PRIVATE_KEY_PATH", str(key))
    settings = SSHSettings.from_env()
    assert settings.host == "bastion.example.com"
    assert settings.port == 22
