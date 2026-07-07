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


def test_training_window_days_from_ini_and_default(monkeypatch, tmp_path):
    from smarthub.core import task_config
    from smarthub.core.config import training_window_days

    # Task knob: comes from config/smarthub.ini; default 21 when absent.
    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", str(tmp_path / "absent.ini"))
    task_config.reload()
    assert training_window_days() == 21               # default

    ini = tmp_path / "t.ini"
    ini.write_text("[feature_engineering]\ntraining_window_days = 45\n")
    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", str(ini))
    task_config.reload()
    assert training_window_days() == 45

    ini.write_text("[feature_engineering]\ntraining_window_days = 0\n")  # all data
    task_config.reload()
    assert training_window_days() == 0
    task_config.reload()  # restore real ini for other tests


def test_ssh_settings_ok(monkeypatch, tmp_path):
    key = tmp_path / "id_rsa"
    key.write_text("dummy")
    monkeypatch.setenv("SSH_HOST", "bastion.example.com")
    monkeypatch.setenv("SSH_USER", "ec2-user")
    monkeypatch.setenv("SSH_PRIVATE_KEY_PATH", str(key))
    settings = SSHSettings.from_env()
    assert settings.host == "bastion.example.com"
    assert settings.port == 22
