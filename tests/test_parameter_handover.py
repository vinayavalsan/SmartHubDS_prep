"""Parameter adoption, reproducible splits, and fixed-calendar HPO retries."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split

from smarthub.core import notifications
from smarthub.train_and_predict import config, model_parameters, registry


@pytest.fixture
def hpo_flow():
    """Only scheduling tests require the optional Prefect dependency."""
    pytest.importorskip("prefect")
    return import_module("smarthub.train_and_predict.hpo_flow")


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

    return (calls, outcome, hpo, train)


def test_thursday_retry_promotion_preserves_every_other_monday(policy, hpo_flow):
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
    assert sum((call[0] == "hpo" for call in calls)) == 5


@pytest.mark.parametrize("phase", ["hpo", "training"])
def test_error_is_persisted_then_fresh_hpo_is_retried_next_day(policy, phase, hpo_flow):
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


def test_crash_after_promotion_recovers_retry_journal_without_hpo(policy, hpo_flow):
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


def test_schedule_missed_during_downtime_keeps_calendar_anchor(policy, hpo_flow):
    _, _, hpo, train = runners()
    result = hpo_flow.run_daily_cycle(
        6, now=at(21), hpo_runner=hpo, training_runner=train
    )
    assert result["hpo_state"]["next_scheduled_date"] == "2026-11-02"


def test_candidate_alerts_include_rejection_retry_and_fixed_schedule(
    policy, slack_payloads, hpo_flow
):
    _, outcome, hpo, training = runners()
    hpo_flow.run_daily_cycle(6, now=at(5), hpo_runner=hpo, training_runner=training)
    text = slack_payloads[-1]["text"]
    assert "completed (not promoted)" in text
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
    policy, slack_payloads, hpo_flow
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
    monkeypatch, hpo_flow
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


@pytest.mark.parametrize(
    "strategy,stratify", [("time", False), ("random", False), ("random", True)]
)
def test_candidate_split_matches_original_hpo_algorithm(strategy, stratify):
    frame = (
        pd.DataFrame(
            {
                "created_at": pd.to_datetime(["2026-10-01"] * 10 + ["2026-10-02"] * 10),
                "won_flag": [0, 1] * 10,
                "row_id": range(20),
            }
        )
        .sample(frac=1, random_state=5)
        .reset_index(drop=True)
    )
    settings = {"strategy": strategy, "test_size": 0.25, "stratify": stratify}
    if strategy == "time":
        ordered = frame.sort_values("created_at", kind="stable")
        expected_fit, expected_test = (ordered.iloc[:-5], ordered.iloc[-5:])
    else:
        expected_fit, expected_test = train_test_split(
            frame,
            test_size=0.25,
            random_state=17,
            shuffle=True,
            stratify=frame["won_flag"] if stratify else None,
        )
    data = {
        "frame_fingerprint": model_parameters.dataset_fingerprint(frame),
        "split_version": 1,
        "split_settings": settings,
        "random_seed": 17,
    }
    fit, test = model_parameters.candidate_partitions(frame, data, "won_flag")
    pd.testing.assert_frame_equal(fit, expected_fit)
    pd.testing.assert_frame_equal(test, expected_test)
    daily_fit, daily_test = model_parameters.split_training_data(
        frame, "won_flag", settings, 17
    )
    pd.testing.assert_frame_equal(fit, daily_fit)
    pd.testing.assert_frame_equal(test, daily_test)
    with pytest.raises(ValueError, match="changed"):
        model_parameters.candidate_partitions(frame.iloc[::-1], data, "won_flag")
    with pytest.raises(ValueError, match="split_version"):
        model_parameters.candidate_partitions(
            frame, {**data, "split_version": 99}, "won_flag"
        )


def test_legacy_partition_artifacts_remain_readable():
    frame = pd.DataFrame({"row_id": range(4)})
    data = {
        "frame_fingerprint": model_parameters.dataset_fingerprint(frame),
        "training_positions": [2, 0, 1],
        "test_positions": [3],
    }
    fit, test = model_parameters.candidate_partitions(frame, data)
    assert fit.index.tolist() == [2, 0, 1]
    assert test.index.tolist() == [3]
    with pytest.raises(ValueError, match="overlap"):
        model_parameters.candidate_partitions(frame, {**data, "test_positions": [1]})


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_invalid_split_fraction_is_rejected(test_size):
    with pytest.raises(ValueError, match="test_size"):
        model_parameters.split_training_data(
            pd.DataFrame({"row_id": range(4)}),
            "won_flag",
            {"strategy": "time", "test_size": test_size},
            17,
        )


def test_split_cannot_leave_empty_training_partition():
    frame = pd.DataFrame({"created_at": pd.to_datetime(["2026-10-01"])})
    with pytest.raises(ValueError, match="empty"):
        model_parameters.split_training_data(
            frame, "won_flag", {"strategy": "time", "test_size": 0.2}, 17
        )


@pytest.mark.parametrize(
    "metadata,label",
    [
        ({}, "Manual"),
        ({"hpo_run_id": "hpo_new"}, "HPO candidate"),
        (
            {"hpo_run_id": "hpo_old", "approved_training_run_id": "approved_run"},
            "Current (explicit file)",
        ),
    ],
)
def test_explicit_parameter_source_is_described_without_changing_artifact(
    policy, metadata, label
):
    cfg = policy
    path = Path(cfg.raw["parameters"]["current_file"])
    candidate = path.parent / "candidate.yaml"
    model_parameters.atomic_write_yaml(
        candidate, model_parameters.make_artifact(SETTINGS, 6, **metadata)
    )
    before = candidate.read_bytes()
    info = model_parameters.resolve_training_parameters(cfg, 6, candidate)
    assert info["parameter_source"] == "parameter_file"
    assert info["parameter_source_label"] == label
    assert info["parameter_file"] == str(candidate)
    assert candidate.read_bytes() == before


@pytest.mark.parametrize(
    "bootstrap,label", [(False, "Current"), (True, "Bootstrap (current copy)")]
)
def test_current_parameter_source_does_not_mislabel_inherited_hpo_provenance(
    policy, bootstrap, label
):
    cfg = policy
    path = Path(cfg.raw["parameters"]["current_file"])
    bootstrap_file = path.parent / "bootstrap.yaml"
    cfg.raw["parameters"]["bootstrap_file"] = str(bootstrap_file)
    model_parameters.atomic_write_yaml(
        path,
        model_parameters.make_artifact(
            SETTINGS, 6, hpo_run_id="earlier_hpo", initialized_from_bootstrap=bootstrap
        ),
    )
    info = model_parameters.resolve_training_parameters(cfg, 6)
    assert info["parameter_source"] == "current_file"
    assert info["parameter_source_label"] == label
    assert info["parameter_file"] == str(path)
    assert info["bootstrap_parameter_file"] == (
        str(bootstrap_file) if bootstrap else None
    )


def test_bootstrap_initialization_reports_current_copy_and_origin(policy):
    cfg = policy
    path = Path(cfg.raw["parameters"]["current_file"])
    bootstrap_file = path.parent / "bootstrap.yaml"
    cfg.raw["parameters"]["bootstrap_file"] = str(bootstrap_file)
    info = model_parameters.resolve_training_parameters(cfg, 6)
    assert info["parameter_source_label"] == "Bootstrap (current copy)"
    assert info["parameter_file"] == str(path)
    assert info["bootstrap_parameter_file"] == str(bootstrap_file)
    assert "parameter_source_label" not in yaml.safe_load(path.read_text())


def test_training_alert_includes_parameter_source_and_file(monkeypatch):
    pytest.importorskip("prefect")
    training_flow = import_module("smarthub.train_and_predict.flow")
    alerts = []
    monkeypatch.setattr(
        training_flow.notifications,
        "notify_success_grouped",
        lambda *args, **kwargs: alerts.append(kwargs),
    )
    monkeypatch.setattr(
        training_flow,
        "_feature_breakdown",
        lambda *args: {"total": 2, "n_registered": 2, "n_registered_used": 2},
    )
    training_flow._notify_success(
        "auto",
        6,
        {
            "parameter_source_label": "Bootstrap (current copy)",
            "parameter_file": "/data/model_parameters/auto/current_params.yaml",
            "bootstrap_parameter_file": "/config/model_parameters_auto.yaml",
            "model_path": "/data/models/auto/run/model.pkl",
            "training_run_id": "run_test",
        },
        {},
        {},
    )
    fields = dict(alerts[0]["groups"])["Model"]
    assert fields["Parameters"] == "Bootstrap (current copy)"
    assert fields["Parameter file"] == "/data/model_parameters/auto/current_params.yaml"
    assert fields["Bootstrap file"] == "/config/model_parameters_auto.yaml"


def test_search_reports_started_and_completed_without_claiming_promotion(
    slack_payloads,
    hpo_flow,
):
    result = {
        "hpo_run_id": "hpo_selected",
        "parameter_version": "params_selected",
        "hpo_mlflow_run_id": "mlflow_selected",
        "model_type": "lightgbm",
        "selected_trial": 0,
        "selected_calibration_method": "none",
        "holdout_probability_metrics": {"log_loss": 0.4},
        "parameters_path": "/data/hpo/best_parameters.yaml",
    }

    @hpo_flow._report_search
    def search(lead_type_id, version, config_path):
        assert lead_type_id == 6 and version == "dataset_v1" and config_path is None
        return result

    assert search(6, version="dataset_v1") is result
    assert len(slack_payloads) == 2
    assert "started" in slack_payloads[0]["text"]
    assert "auto (6)" in slack_payloads[0]["text"]
    text = slack_payloads[1]["text"]
    assert "HPO completed; parameters saved" in text
    assert "hpo_selected" in text and "mlflow_selected" in text
    assert "params_selected" in text and "0.4" in text
    assert "promoted" not in text.lower()


def test_search_failure_alert_preserves_original_exception(slack_payloads, hpo_flow):
    error = ValueError("insufficient data")

    @hpo_flow._report_flow_failure
    @hpo_flow._report_search
    def search(*args):
        raise error

    with pytest.raises(ValueError) as caught:
        search(1)
    assert caught.value is error
    assert len(slack_payloads) == 2
    text = slack_payloads[-1]["text"]
    assert "FAILED" in text and "HPO flow failed" in text
    assert "home (1)" in text and "ValueError: insufficient data" in text


def test_slack_delivery_error_never_changes_search_result(monkeypatch, hpo_flow):
    def broken(*args, **kwargs):
        raise OSError("Slack unavailable")

    monkeypatch.setattr(notifications, "_post", broken)
    result = {"hpo_run_id": "completed"}

    @hpo_flow._report_search
    def search(*args):
        return result

    assert search(6) is result


def test_slack_delivery_error_never_masks_search_exception(monkeypatch, hpo_flow):
    def broken(*args, **kwargs):
        raise OSError("Slack unavailable")

    monkeypatch.setattr(notifications, "_post", broken)

    @hpo_flow._report_search
    def search(*args):
        raise ValueError("search failed")

    with pytest.raises(ValueError, match="search failed"):
        search(6)


@pytest.mark.parametrize(
    "status,headline,category,display_status",
    [
        ("started", "HPO started", "updates", "started"),
        ("success", "HPO completed; parameters saved", "updates", "completed"),
        ("success", "HPO candidate promoted", "updates", "promoted"),
        (
            "success",
            "HPO candidate not promoted",
            "updates",
            "completed (not promoted)",
        ),
        ("failure", "HPO failed; retry scheduled", "failures", "FAILED"),
        ("failure", "HPO candidate training failed", "failures", "FAILED"),
    ],
)
def test_hpo_notification_uses_shared_format_and_category(
    monkeypatch, status, headline, category, display_status, hpo_flow
):
    delivered = []

    def capture(payload, channel=None):
        delivered.append((payload, channel))
        return True

    monkeypatch.setattr(notifications, "_post", capture)
    hpo_flow._send_notification(
        status,
        {"Lead type": "auto (6)", "Status": headline, "HPO run": "hpo_test"},
        error="test error" if status == "failure" else None,
    )
    payload, channel = delivered[0]
    assert channel == category
    title = payload["blocks"][0]["text"]["text"].splitlines()[0]
    assert f"· hpo · {display_status} · auto (6)" in title
    assert headline in payload["text"]
    assert "```" in payload["blocks"][0]["text"]["text"]
    assert payload["blocks"][1]["type"] == "context"
    assert ("<!here>" in title) == (status == "failure")
    if status == "failure":
        assert "Error: test error" in payload["text"]


@pytest.mark.parametrize("phase", ["hpo", "training"])
def test_candidate_failure_is_reported_once_by_parent_flow(
    policy, slack_payloads, hpo_flow, phase
):
    _, _, hpo, training = runners()
    error = RuntimeError("candidate failed")

    def fail(*args, **kwargs):
        raise error

    wrapped = hpo_flow._report_flow_failure(hpo_flow.run_daily_cycle)
    with pytest.raises(RuntimeError) as caught:
        wrapped(
            6,
            now=at(5),
            hpo_runner=fail if phase == "hpo" else hpo,
            training_runner=fail if phase == "training" else training,
        )
    assert caught.value is error
    failures = [p for p in slack_payloads if "FAILED" in p["text"]]
    assert len(failures) == 1
    assert "2026-10-06" in failures[0]["text"]
    assert "2026-10-19" in failures[0]["text"]


def test_flow_preflight_failure_is_reported(slack_payloads, hpo_flow):
    error = ValueError("invalid schedule")

    @hpo_flow._report_flow_failure
    def fail(lead_type_id):
        raise error

    with pytest.raises(ValueError) as caught:
        fail(6)
    assert caught.value is error
    assert len(slack_payloads) == 1
    assert "invalid schedule" in slack_payloads[0]["text"]


def test_hpo_candidate_uses_parent_failure_notification(
    monkeypatch, slack_payloads, hpo_flow
):
    result = {"hpo_run_id": "hpo_test", "parameters_path": "/tmp/candidate.yaml"}
    monkeypatch.setattr(hpo_flow, "_hpo_task", lambda *args: result)
    options = []

    def fail(**kwargs):
        raise ValueError("training failed")

    def with_options(**kwargs):
        options.append(kwargs)
        return fail

    monkeypatch.setattr(
        hpo_flow, "train_flow", SimpleNamespace(with_options=with_options)
    )
    with pytest.raises(ValueError, match="training failed"):
        hpo_flow.hpo_flow.fn(6, train_candidate=True)
    assert options == [{"on_failure": []}]
    assert len(slack_payloads) == 1
    assert "HPO candidate training failed" in slack_payloads[0]["text"]
