"""Tests for configuration loading and validation."""

import pytest

from smarthub.core.config import (
    ConfigError,
    PullSettings,
    RedshiftSettings,
    SSHSettings,
)


def _set_redshift(monkeypatch):
    monkeypatch.setenv("REDSHIFT_HOST", "redshift.example.com")
    monkeypatch.setenv("REDSHIFT_DB", "analytics")
    monkeypatch.setenv("REDSHIFT_USER", "anton")
    monkeypatch.setenv("REDSHIFT_PASSWORD", "secret")


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
    """training_window_days comes from YAML without leaking test config state."""
    from smarthub.core import task_config
    from smarthub.core.config import training_window_days

    # task_config caches the loaded YAML. Use a nested monkeypatch context so
    # SMARTHUB_TASK_CONFIG is restored before the final cache clear; otherwise
    # later prediction tests keep reading this temporary feature-only YAML.
    with monkeypatch.context() as config_patch:
        config_patch.setenv(
            "SMARTHUB_TASK_CONFIG",
            str(tmp_path / "absent.yaml"),
        )
        task_config.reload()
        assert training_window_days() == 21

        cfg = tmp_path / "t.yaml"
        cfg.write_text("feature_engineering:\n  training_window_days: 45\n")
        config_patch.setenv("SMARTHUB_TASK_CONFIG", str(cfg))
        task_config.reload()
        assert training_window_days() == 45

        cfg.write_text("feature_engineering:\n  training_window_days: 0\n")
        task_config.reload()
        assert training_window_days() == 0

    # The nested monkeypatch context has now restored the real config path.
    # Clear the cached temporary YAML so subsequent tests load the repo config.
    task_config.reload()


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


def test_pull_settings_tunnel_default_requires_ssh(monkeypatch, tmp_path):
    """Default (SSH_TUNNEL unset) uses the tunnel and populates ssh settings."""
    _set_redshift(monkeypatch)
    monkeypatch.delenv("SSH_TUNNEL", raising=False)
    key = tmp_path / "id_rsa"
    key.write_text("dummy")
    monkeypatch.setenv("SSH_HOST", "bastion.example.com")
    monkeypatch.setenv("SSH_USER", "ec2-user")
    monkeypatch.setenv("SSH_PRIVATE_KEY_PATH", str(key))

    settings = PullSettings.from_env()
    assert settings.use_ssh_tunnel is True
    assert settings.ssh is not None
    assert settings.ssh.host == "bastion.example.com"


def test_pull_settings_tunnel_off_skips_ssh(monkeypatch):
    """SSH_TUNNEL=false connects directly: no ssh settings, no key required."""
    _set_redshift(monkeypatch)
    monkeypatch.setenv("SSH_TUNNEL", "false")
    # Deliberately leave SSH_* unset -- they must NOT be needed.
    for var in ("SSH_HOST", "SSH_USER", "SSH_PRIVATE_KEY_PATH"):
        monkeypatch.delenv(var, raising=False)

    settings = PullSettings.from_env()
    assert settings.use_ssh_tunnel is False
    assert settings.ssh is None
    assert settings.redshift.host == "redshift.example.com"
