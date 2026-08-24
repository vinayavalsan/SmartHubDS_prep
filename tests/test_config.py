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
    cfg = training_config.load_training_config(6)

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


# --- train_and_predict training/HPO configuration ----------------------------


def _write_yaml(path, payload):
    """Write one temporary YAML config and return its path."""
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _training_payload():
    """Return a minimal valid lead-type-aware training configuration."""
    return {
        "training": {
            "defaults": {
                "random_seed": 42,
                "split": {
                    "strategy": "time",
                    "time": {"test_size": 0.2},
                    "random": {"test_size": 0.25, "stratify": True},
                },
                "calibration": {
                    "enabled": False,
                    "method": "sigmoid",
                    "cv": 3,
                },
                "optimizer": {
                    "target_cm": 0.25,
                    "minimum_bid": 0.25,
                    "bid_step": 0.25,
                    "chunk_size": 500,
                },
                "promotion": {
                    "mode": "automatic",
                    "criteria": {
                        "max_log_loss": 0.55,
                        "min_expected_profit": 0.0,
                        "min_profit_ratio": 0.95,
                        "max_absolute_profit_loss_tolerance": 500.0,
                        "max_log_loss_regression": 0.01,
                        "monotonicity": {
                            "enabled": True,
                            "tolerance": 1e-8,
                            "max_violation_rate": 0.0,
                        },
                    },
                },
                "output": {
                    "report_root": "data/model_evaluations",
                    "model_root": "data/models",
                    "comparison_artifacts": True,
                },
                "mlflow": {
                    "tracking_db_path": "data/mlflow.db",
                    "artifact_root": "data/mlruns",
                    "experiment_name": "anton_win_probability",
                    "registered_model_name": "anton-win-probability-model",
                },
            },
            "lead_types": {
                6: {
                    "name": "auto",
                    "model_type": "lightgbm",
                    "models": {
                        "lightgbm": {
                            "n_estimators": 700,
                            "learning_rate": 0.04,
                        },
                    },
                },
                1: {
                    "name": "home",
                    "model_type": "logistic_regression",
                    "models": {
                        "logistic_regression": {
                            "C": 0.5,
                            "max_iter": 2000,
                        },
                    },
                },
            },
        },
    }


def _hpo_model_config():
    """Return one minimal valid LightGBM search-space definition."""
    return {
        "fixed_parameters": {
            "verbosity": -1,
            "n_jobs": 1,
        },
        "search_space": {
            "n_estimators": {
                "type": "int",
                "low": 100,
                "high": 300,
                "step": 100,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.2,
                "log": True,
            },
        },
    }


def _hpo_payload():
    """Return a minimal valid lead-type-aware HPO configuration."""
    lightgbm = _hpo_model_config()
    logistic = {
        "fixed_parameters": {"max_iter": 2000},
        "search_space": {
            "C": {
                "type": "float",
                "low": 0.01,
                "high": 10.0,
                "log": True,
            },
        },
    }
    return {
        "hyperparameter_search": {
            "defaults": {
                "search": {
                    "scoring": "neg_log_loss",
                    "n_trials": 10,
                    "cv_folds": 3,
                    "timeout_seconds": None,
                    "random_seed": 42,
                    "n_jobs": 1,
                },
                "validation": {"strategy": "time"},
                "split": {
                    "strategy": "time",
                    "time": {"test_size": 0.2},
                    "random": {"test_size": 0.25, "stratify": True},
                },
                "finalists": {
                    "holdout_fraction": 0.2,
                    "probability_shortlist_top_n": 5,
                    "optimizer_top_n": 3,
                    "max_log_loss_regression_from_best": 0.02,
                    "monotonicity": {
                        "enabled": True,
                        "tolerance": 1e-8,
                        "max_violation_rate": 0.0,
                    },
                },
                "optimizer": {
                    "enabled": True,
                    "target_cm": 0.25,
                    "minimum_bid": 0.25,
                    "bid_step": 0.25,
                    "chunk_size": 500,
                },
                "calibration": {
                    "enabled": True,
                    "methods": ["none", "sigmoid", "isotonic"],
                    "cv": 3,
                },
                "output": {"root": "data/hyperparameter_tuning"},
            },
            "lead_types": {
                6: {
                    "name": "auto",
                    "model_type": "lightgbm",
                    "models": {"lightgbm": lightgbm},
                },
                1: {
                    "name": "home",
                    "model_type": "logistic_regression",
                    "models": {"logistic_regression": logistic},
                },
            },
        },
        # config.py currently records the selected model from this root-level
        # mapping when constructing resolved metadata.
        "models": {
            "lightgbm": lightgbm,
            "logistic_regression": logistic,
        },
    }


def test_training_config_deep_merges_lead_type_override(tmp_path):
    """Lead-specific values override defaults without discarding siblings."""
    from smarthub.train_and_predict import config as training_config

    payload = _training_payload()
    payload["training"]["lead_types"][6]["optimizer"] = {"target_cm": 0.30}
    path = _write_yaml(tmp_path / "training.yaml", payload)

    cfg = training_config.load_training_config(6, path)

    assert cfg.optimizer.target_cm == pytest.approx(0.30)
    assert cfg.optimizer.minimum_bid == pytest.approx(0.25)
    assert cfg.optimizer.bid_step == pytest.approx(0.25)
    assert cfg.optimizer.chunk_size == 500


def test_training_config_resolves_each_lead_type_independently(tmp_path):
    """Each lead type resolves its own model family and parameters."""
    from smarthub.train_and_predict import config as training_config

    path = _write_yaml(tmp_path / "training.yaml", _training_payload())

    auto = training_config.load_training_config(6, path)
    home = training_config.load_training_config(1, path)

    assert auto.model_type == "lightgbm"
    assert auto.model_parameters["n_estimators"] == 700
    assert auto.model_parameters["random_state"] == 42
    assert home.model_type == "logistic_regression"
    assert home.model_parameters["C"] == pytest.approx(0.5)
    assert home.model_parameters["random_state"] == 42


def test_training_config_missing_lead_type_raises(tmp_path):
    """A lead type without an explicit config must fail clearly."""
    from smarthub.train_and_predict import config as training_config

    path = _write_yaml(tmp_path / "training.yaml", _training_payload())

    with pytest.raises(ValueError, match="lead_type_id=999"):
        training_config.load_training_config(999, path)


def test_training_config_disabled_calibration_resolves_none(tmp_path):
    """Disabled calibration must not leave an active method or CV."""
    from smarthub.train_and_predict import config as training_config

    path = _write_yaml(tmp_path / "training.yaml", _training_payload())
    cfg = training_config.load_training_config(6, path)

    assert cfg.calibration_enabled is False
    assert cfg.calibration_method is None
    assert cfg.calibration_cv is None


def test_training_config_enabled_calibration_validates_method(tmp_path):
    """Enabled calibration accepts only the supported methods."""
    from smarthub.train_and_predict import config as training_config

    payload = _training_payload()
    calibration = payload["training"]["defaults"]["calibration"]
    calibration["enabled"] = True
    calibration["method"] = "platt"
    path = _write_yaml(tmp_path / "training.yaml", payload)

    with pytest.raises(ValueError, match="calibration.method"):
        training_config.load_training_config(6, path)


def test_training_config_random_stratify_requires_yaml_boolean(tmp_path):
    """Random training splits must use a native YAML boolean."""
    from smarthub.train_and_predict import config as training_config

    payload = _training_payload()
    split = payload["training"]["defaults"]["split"]
    split["strategy"] = "random"
    split["random"]["stratify"] = "true"
    path = _write_yaml(tmp_path / "training.yaml", payload)

    with pytest.raises(TypeError, match="stratify must be a YAML boolean"):
        training_config.load_training_config(6, path)


def test_training_config_rejects_invalid_monotonicity_rate(tmp_path):
    """Promotion monotonicity violation rate is constrained to [0, 1]."""
    from smarthub.train_and_predict import config as training_config

    payload = _training_payload()
    criteria = payload["training"]["defaults"]["promotion"]["criteria"]
    criteria["monotonicity"]["max_violation_rate"] = 1.01
    path = _write_yaml(tmp_path / "training.yaml", payload)

    with pytest.raises(ValueError, match="max_violation_rate"):
        training_config.load_training_config(6, path)


def test_training_config_rejects_nonpositive_optimizer_step(tmp_path):
    """Training bid-step configuration must remain strictly positive."""
    from smarthub.train_and_predict import config as training_config

    payload = _training_payload()
    payload["training"]["defaults"]["optimizer"]["bid_step"] = 0
    path = _write_yaml(tmp_path / "training.yaml", payload)

    with pytest.raises(ValueError, match="optimizer.bid_step"):
        training_config.load_training_config(6, path)


def test_hpo_config_resolves_lead_type_model_and_search_space(tmp_path):
    """HPO selects only the model configured for the requested lead type."""
    from smarthub.train_and_predict import config as training_config

    path = _write_yaml(tmp_path / "hpo.yaml", _hpo_payload())

    auto = training_config.load_hyperparameter_search_config(6, path)
    home = training_config.load_hyperparameter_search_config(1, path)

    assert auto.model_type == "lightgbm"
    assert set(auto.model_configs) == {"lightgbm"}
    assert "n_estimators" in auto.model_config("lightgbm")["search_space"]
    assert home.model_type == "logistic_regression"
    assert set(home.model_configs) == {"logistic_regression"}
    assert "C" in home.model_config("logistic_regression")["search_space"]


def test_hpo_config_missing_lead_type_raises(tmp_path):
    """HPO requires an explicit lead-type configuration."""
    from smarthub.train_and_predict import config as training_config

    path = _write_yaml(tmp_path / "hpo.yaml", _hpo_payload())

    with pytest.raises(ValueError, match="lead_type_id=999"):
        training_config.load_hyperparameter_search_config(999, path)


def test_hpo_random_stratify_requires_yaml_boolean(tmp_path):
    """Random HPO final-test splits must use a native YAML boolean."""
    from smarthub.train_and_predict import config as training_config

    payload = _hpo_payload()
    split = payload["hyperparameter_search"]["defaults"]["split"]
    split["strategy"] = "random"
    split["random"]["stratify"] = "true"
    path = _write_yaml(tmp_path / "hpo.yaml", payload)

    with pytest.raises(TypeError, match="stratify must be a YAML boolean"):
        training_config.load_hyperparameter_search_config(6, path)


def test_hpo_disabled_calibration_and_optimizer_resolve_cleanly(tmp_path):
    """Optional HPO stages can be disabled without carrying active settings."""
    from smarthub.train_and_predict import config as training_config

    payload = _hpo_payload()
    defaults = payload["hyperparameter_search"]["defaults"]
    defaults["calibration"]["enabled"] = False
    defaults["optimizer"]["enabled"] = False
    path = _write_yaml(tmp_path / "hpo.yaml", payload)

    cfg = training_config.load_hyperparameter_search_config(6, path)

    assert cfg.calibration_enabled is False
    assert cfg.calibration_methods == ("none",)
    assert cfg.optimizer_enabled is False
    assert cfg.optimizer is None


def test_hpo_config_rejects_unknown_calibration_method(tmp_path):
    """HPO calibration method list is restricted to supported methods."""
    from smarthub.train_and_predict import config as training_config

    payload = _hpo_payload()
    payload["hyperparameter_search"]["defaults"]["calibration"]["methods"] = [
        "none",
        "unsupported",
    ]
    path = _write_yaml(tmp_path / "hpo.yaml", payload)

    with pytest.raises(ValueError, match="Unsupported calibration methods"):
        training_config.load_hyperparameter_search_config(6, path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target_cm", 1.0, "optimizer.target_cm"),
        ("minimum_bid", -0.01, "optimizer.minimum_bid"),
        ("bid_step", 0.0, "optimizer.bid_step"),
    ],
)
def test_hpo_config_rejects_invalid_optimizer_values(
    tmp_path,
    field,
    value,
    message,
):
    """HPO optimizer values enforce their documented numeric boundaries."""
    from smarthub.train_and_predict import config as training_config

    payload = _hpo_payload()
    payload["hyperparameter_search"]["defaults"]["optimizer"][field] = value
    path = _write_yaml(tmp_path / "hpo.yaml", payload)

    with pytest.raises(ValueError, match=message):
        training_config.load_hyperparameter_search_config(6, path)


def test_hpo_config_rejects_invalid_holdout_fraction(tmp_path):
    """Finalist holdout must remain positive and below one half."""
    from smarthub.train_and_predict import config as training_config

    payload = _hpo_payload()
    finalists = payload["hyperparameter_search"]["defaults"]["finalists"]
    finalists["holdout_fraction"] = 0.5
    path = _write_yaml(tmp_path / "hpo.yaml", payload)

    with pytest.raises(ValueError, match="holdout_fraction"):
        training_config.load_hyperparameter_search_config(6, path)


def test_hpo_config_rejects_invalid_search_space_definition(tmp_path):
    """Malformed parameter ranges fail during config loading, before Optuna."""
    from smarthub.train_and_predict import config as training_config

    payload = _hpo_payload()
    search_space = payload["hyperparameter_search"]["lead_types"][6]["models"][
        "lightgbm"
    ]["search_space"]
    search_space["n_estimators"]["low"] = 400
    search_space["n_estimators"]["high"] = 300
    path = _write_yaml(tmp_path / "hpo.yaml", payload)

    with pytest.raises(ValueError, match="low must be less than high"):
        training_config.load_hyperparameter_search_config(6, path)
