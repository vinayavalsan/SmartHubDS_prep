"""Parameter adoption, exact HPO partitions, and fixed calendar retry behavior."""

from datetime import datetime, timezone
from importlib import import_module
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from smarthub.core import notifications
from smarthub.train_and_predict import config, model_parameters, registry

pytest.importorskip("prefect")
hpo_flow = import_module("smarthub.train_and_predict.hpo_flow")

SETTINGS = {
    "model_type": "lightgbm",
    "model_parameters": {"learning_rate": 0.05},
    "calibration": {"enabled": False},
}


@pytest.fixture(autouse=True)
def slack_payloads(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        notifications,
        "_post",
        lambda payload, category=None: payloads.append(payload) or True,
    )
    return payloads


@pytest.fixture
def policy(tmp_path, monkeypatch):
    bootstrap = config.load_training_config(6)
    cfg = SimpleNamespace(
        raw={
            "parameters": {"current_file": str(tmp_path / "current.yaml")},
            "hpo_schedule": {
                "state_file": str(tmp_path / "state.yaml"),
                "anchor_date": "2026-10-05",
                "timezone": "UTC",
                "interval_days": 14,
                "retry_days": 1,
            },
        },
        model_type=bootstrap.model_type,
        model_parameters=bootstrap.model_parameters,
        calibration_enabled=bootstrap.calibration_enabled,
        calibration_method=bootstrap.calibration_method,
        calibration_cv=bootstrap.calibration_cv,
        production_mlflow_enabled=False,
    )
    monkeypatch.setattr(config, "load_training_config", lambda *_: cfg)
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.setattr(registry, "_production_store", lambda _: None)
    return cfg


def save_candidate(settings, hpo_id="hpo_test", eligible=True):
    return registry.save_version(
        {},
        "auto",
        feature_cols=["bid"],
        metrics={},
        optimizer_summary={},
        lineage={},
        model_params=settings["model_parameters"],
        training_config={
            "model_settings": settings,
            "parameter_provenance": {
                "parameter_version": model_parameters.parameter_version(settings),
                "hpo_run_id": hpo_id,
                "hpo_mlflow_run_id": "mlflow_test",
            },
        },
        promotion_mode="automatic",
        eligibility_status="eligible" if eligible else "not_eligible",
        promotion_status="awaiting_manual_promotion" if eligible else "rejected",
        promotion_decision_reason="test",
    )


def test_rejected_candidate_keeps_current_then_promotion_adopts_it(policy):
    before = model_parameters.resolve_training_parameters(policy, 6)
    path = model_parameters.paths.resolve(policy.raw["parameters"]["current_file"])
    initial_bytes = path.read_bytes()
    rejected = save_candidate(SETTINGS, eligible=False)
    with pytest.raises(ValueError, match="eligible"):
        registry.promote("auto", rejected["training_run_id"])
    assert (
        model_parameters.resolve_training_parameters(policy, 6)["model_settings"]
        == before["model_settings"]
    )
    assert path.read_bytes() == initial_bytes
    candidate = save_candidate(SETTINGS)
    registry.promote("auto", candidate["training_run_id"])
    # Simulate a crash before writing current_params: startup repairs from serving.
    resolved = model_parameters.resolve_training_parameters(policy, 6)
    assert resolved["model_settings"] == SETTINGS
    assert resolved["hpo_run_id"] == "hpo_test"
    assert resolved["approved_training_run_id"] == candidate["training_run_id"]
    assert path.read_bytes() != initial_bytes


def test_explicit_candidate_does_not_modify_current_and_checksum_is_checked(policy):
    model_parameters.resolve_training_parameters(policy, 6)
    path = model_parameters.paths.resolve(policy.raw["parameters"]["current_file"])
    before = path.read_bytes()
    candidate = path.parent / "candidate.yaml"
    artifact = {
        "schema_version": 1,
        "lead_type_id": 6,
        "model_settings": SETTINGS,
        "parameter_version": model_parameters.parameter_version(SETTINGS),
    }
    model_parameters.atomic_write_yaml(candidate, artifact)
    assert (
        model_parameters.resolve_training_parameters(policy, 6, candidate)[
            "model_settings"
        ]
        == SETTINGS
    )
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="lead_type_id"):
        model_parameters.load_artifact(candidate, 1)
    artifact["model_settings"] = {
        **SETTINGS,
        "model_parameters": {"learning_rate": 0.1},
    }
    model_parameters.atomic_write_yaml(candidate, artifact)
    with pytest.raises(ValueError, match="checksum"):
        model_parameters.load_artifact(candidate, 6)


def test_hpo_partition_handover_rejects_dataset_changes_and_overlap():
    frame = pd.DataFrame({"bid": [1, 2, 3, 4], "won_flag": [0, 1, 0, 1]})
    data = {
        "frame_fingerprint": model_parameters.dataset_fingerprint(frame),
        "training_positions": [2, 0, 1],
        "test_positions": [3],
    }
    fit, test = model_parameters.candidate_partitions(frame, data)
    assert fit.index.tolist() == [2, 0, 1]
    assert test.index.tolist() == [3]
    with pytest.raises(ValueError, match="changed"):
        model_parameters.candidate_partitions(frame.assign(bid=10), data)
    with pytest.raises(ValueError, match="overlap"):
        model_parameters.candidate_partitions(frame, {**data, "test_positions": [1]})


def at(day):
    return datetime(2026, 10, day, 5, tzinfo=timezone.utc)


def runners():
    calls = []
    outcome = {"promoted": False}

    def hpo(lead_type_id, version, config_path):
        calls.append(("hpo", lead_type_id))
        return {"hpo_run_id": "new_hpo", "parameters_path": "candidate.yaml"}

    def train(**kwargs):
        calls.append(("training", kwargs))
        return {**outcome, "training_run_id": "new_run", "promotion_reason": "test"}

    return calls, outcome, hpo, train


def test_thursday_retry_promotion_preserves_every_other_monday(policy):
    calls, outcome, hpo, train = runners()
    kwargs = {"hpo_runner": hpo, "training_runner": train}
    for day in (5, 6, 7):
        result = hpo_flow.run_daily_cycle(6, now=at(day), **kwargs)
        assert result["hpo_state"]["retry_pending"]
        assert result["hpo_state"]["next_scheduled_date"] == "2026-10-19"
    outcome["promoted"] = True
    thursday = hpo_flow.run_daily_cycle(6, now=at(8), **kwargs)
    assert not thursday["hpo_state"]["retry_pending"]
    assert thursday["hpo_state"]["next_scheduled_date"] == "2026-10-19"
    assert (
        hpo_flow.run_daily_cycle(6, now=at(9), **kwargs)["action"] == "daily_training"
    )
    assert "parameter_file" not in calls[-1][1]
    assert (
        hpo_flow.run_daily_cycle(6, now=at(12), **kwargs)["action"] == "daily_training"
    )
    next_monday = hpo_flow.run_daily_cycle(6, now=at(19), **kwargs)
    assert next_monday["action"] == "hpo_candidate"
    assert next_monday["hpo_state"]["next_scheduled_date"] == "2026-11-02"
    assert sum(call[0] == "hpo" for call in calls) == 5


@pytest.mark.parametrize("phase", ["hpo", "training"])
def test_error_is_persisted_then_fresh_hpo_is_retried_next_day(policy, phase):
    calls, _, hpo, train = runners()

    def failing(*args, **kwargs):
        raise RuntimeError("worker failed")

    kwargs = {
        "hpo_runner": failing if phase == "hpo" else hpo,
        "training_runner": failing if phase == "training" else train,
    }
    with pytest.raises(RuntimeError):
        hpo_flow.run_daily_cycle(6, now=at(5), **kwargs)
    state = yaml.safe_load(
        model_parameters.paths.resolve(
            policy.raw["hpo_schedule"]["state_file"]
        ).read_text()
    )
    assert state["status"] == f"{phase}_failed"
    assert state["retry_date"] == "2026-10-06"
    assert state["next_scheduled_date"] == "2026-10-19"
    hpo_flow.run_daily_cycle(6, now=at(6), hpo_runner=hpo, training_runner=train)
    assert calls[-2][0] == "hpo"


def test_crash_after_promotion_recovers_retry_journal_without_hpo(policy):
    _, _, hpo, train = runners()
    hpo_flow.run_daily_cycle(6, now=at(5), hpo_runner=hpo, training_runner=train)
    candidate = save_candidate(SETTINGS, hpo_id="new_hpo")
    registry.promote("auto", candidate["training_run_id"])
    result = hpo_flow.run_daily_cycle(
        6, now=at(6), hpo_runner=hpo, training_runner=train
    )
    assert result["action"] == "daily_training"
    assert not result["hpo_state"]["retry_pending"]
    assert result["hpo_state"]["next_scheduled_date"] == "2026-10-19"


def test_schedule_missed_during_downtime_keeps_calendar_anchor(policy):
    _, _, hpo, train = runners()
    result = hpo_flow.run_daily_cycle(
        6, now=at(21), hpo_runner=hpo, training_runner=train
    )
    assert result["hpo_state"]["next_scheduled_date"] == "2026-11-02"


def test_candidate_alerts_include_rejection_retry_and_fixed_schedule(
    policy, slack_payloads
):
    _, outcome, hpo, training = runners()
    hpo_flow.run_daily_cycle(6, now=at(5), hpo_runner=hpo, training_runner=training)
    text = slack_payloads[-1]["text"]
    assert "WARNING" in text
    assert "HPO candidate not promoted" in text
    assert "new_hpo" in text and "new_run" in text
    assert "2026-10-06" in text and "2026-10-19" in text
    assert "Unchanged" in text
    outcome["promoted"] = True
    hpo_flow.run_daily_cycle(6, now=at(6), hpo_runner=hpo, training_runner=training)
    text = slack_payloads[-1]["text"]
    assert "HPO candidate promoted" in text
    assert "Updated" in text and "2026-10-19" in text
    assert "Next HPO retry" not in text
    count = len(slack_payloads)
    hpo_flow.run_daily_cycle(6, now=at(7), hpo_runner=hpo, training_runner=training)
    assert len(slack_payloads) == count


def test_training_error_alert_reports_retry_without_claiming_promotion(
    policy, slack_payloads
):
    _, _, hpo, _ = runners()

    def failing(**kwargs):
        raise RuntimeError("candidate failed")

    with pytest.raises(RuntimeError, match="candidate failed"):
        hpo_flow.run_daily_cycle(6, now=at(5), hpo_runner=hpo, training_runner=failing)
    text = slack_payloads[-1]["text"]
    assert "FAILED" in text and "HPO candidate training failed" in text
    assert "candidate failed" in text and "new_hpo" in text
    assert "2026-10-06" in text and "2026-10-19" in text
    assert "Outcome not confirmed" in text


def test_daily_prefect_entrypoint_uses_local_cycle_without_starting_prefect(
    monkeypatch,
):
    calls = []

    def training(**kwargs):
        calls.append(kwargs)
        return {"promoted": False}

    def cycle(lead_type_id, *, hpo_runner, training_runner):
        assert lead_type_id == 6
        assert hpo_runner is hpo_flow._hpo_task
        return training_runner(lead_type_id=lead_type_id)

    monkeypatch.setattr(hpo_flow, "train_flow", training)
    monkeypatch.setattr(hpo_flow, "run_daily_cycle", cycle)
    assert hpo_flow.daily_model_cycle.fn(6, register_mlflow=False) == {
        "promoted": False
    }
    assert calls == [{"lead_type_id": 6, "register_mlflow": False}]


def test_bootstrap_hpo_and_promoted_current_files_use_identical_schema(policy):
    bootstrap = model_parameters.resolve_training_parameters(policy, 6)
    current_path = model_parameters.paths.resolve(
        policy.raw["parameters"]["current_file"]
    )
    bootstrap_yaml = yaml.safe_load(current_path.read_text())
    hpo_artifact = model_parameters.make_artifact(
        SETTINGS,
        6,
        hpo_run_id="hpo_test",
        hpo_mlflow_run_id="mlflow_test",
        data={"training_table_version": "dataset_v1"},
    )
    candidate_path = current_path.parent / "best_parameters.yaml"
    model_parameters.atomic_write_yaml(candidate_path, hpo_artifact)
    manifest = save_candidate(SETTINGS)
    registry.promote("auto", manifest["training_run_id"])
    current = model_parameters.resolve_training_parameters(policy, 6)
    current_yaml = yaml.safe_load(current_path.read_text())
    candidate_yaml = yaml.safe_load(candidate_path.read_text())
    assert bootstrap_yaml.keys() == candidate_yaml.keys() == current_yaml.keys()
    assert (
        candidate_yaml["model_settings"] == current_yaml["model_settings"] == SETTINGS
    )
    assert "models" not in candidate_yaml and "calibration" not in candidate_yaml
    assert candidate_yaml["approved_training_run_id"] is None
    assert current_yaml["approved_training_run_id"] == manifest["training_run_id"]
    assert current_yaml["hpo_run_id"] == candidate_yaml["hpo_run_id"]
    assert bootstrap["hpo_run_id"] is None
    assert current["parameter_version"] == candidate_yaml["parameter_version"]


def test_manual_and_legacy_artifacts_are_normalized_to_same_schema(policy):
    file = model_parameters.paths.resolve(policy.raw["parameters"]["current_file"])
    manual = {"schema_version": 1, "lead_type_id": 6, "model_settings": SETTINGS}
    model_parameters.atomic_write_yaml(file, manual)
    loaded = model_parameters.load_artifact(file, 6)
    assert loaded["hpo_run_id"] is None
    assert loaded["approved_training_run_id"] is None
    legacy = {**manual, "models": {"unused_duplicate": {}}, "calibration": {}}
    model_parameters.atomic_write_yaml(file, legacy)
    assert model_parameters.load_artifact(file, 6) == loaded
