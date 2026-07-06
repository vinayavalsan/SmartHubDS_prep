"""Tests for configuration loading and validation."""

import pytest

from smarthub.core.config import ConfigError, RedshiftSettings, SSHSettings


def test_redshift_settings_missing_var_raises(monkeypatch):
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
    monkeypatch.setenv("SSH_HOST", "bastion.example.com")
    monkeypatch.setenv("SSH_USER", "ec2-user")
    monkeypatch.setenv("SSH_PRIVATE_KEY_PATH", str(tmp_path / "does_not_exist"))
    with pytest.raises(ConfigError):
        SSHSettings.from_env()


def test_training_window_days_default_and_override(monkeypatch):
    from smarthub.core.config import training_window_days

    monkeypatch.delenv("TRAINING_WINDOW_DAYS", raising=False)
    assert training_window_days() == 21
    monkeypatch.setenv("TRAINING_WINDOW_DAYS", "30")
    assert training_window_days() == 30
    monkeypatch.setenv("TRAINING_WINDOW_DAYS", "0")  # 0 = all data
    assert training_window_days() == 0


def test_ssh_settings_ok(monkeypatch, tmp_path):
    key = tmp_path / "id_rsa"
    key.write_text("dummy")
    monkeypatch.setenv("SSH_HOST", "bastion.example.com")
    monkeypatch.setenv("SSH_USER", "ec2-user")
    monkeypatch.setenv("SSH_PRIVATE_KEY_PATH", str(key))
    settings = SSHSettings.from_env()
    assert settings.host == "bastion.example.com"
    assert settings.port == 22
