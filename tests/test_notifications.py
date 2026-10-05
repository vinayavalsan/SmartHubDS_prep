"""Offline Slack payload, routing, and training-title tests; never send alerts."""

import json

import pytest

from smarthub.core import notifications as n


@pytest.fixture(autouse=True)
def disable_slack_delivery(monkeypatch):
    """Disable delivery and fail any accidental HTTP request in this module."""
    for env in (
        "SLACK_WEBHOOK_URL",
        "SLACK_WEBHOOK_UPDATES_URL",
        "SLACK_WEBHOOK_WARNINGS_URL",
        "SLACK_WEBHOOK_CRITICAL_URL",
        "SLACK_WEBHOOK_FAILURES_URL",
        "SLACK_MENTION_ON_FAILURE",
        "SLACK_ENV_LABEL",
    ):
        monkeypatch.delenv(env, raising=False)

    def forbid_http(*args, **kwargs):
        pytest.fail("Notification tests must not make HTTP requests.")

    monkeypatch.setattr(n.urllib.request, "urlopen", forbid_http)
    monkeypatch.setattr(n, "_post", lambda payload, category=None: False)


@pytest.fixture
def capture_slack(monkeypatch):
    """Capture payload and intended route before the HTTP sender is called."""
    captured = {}

    def capture(payload, category=None):
        captured["payload"] = payload
        captured["category"] = category
        captured["url"] = n._category_webhook_url(category)
        return True  # Simulate delivery without executing the sender.

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/T/B/xxx")
    monkeypatch.setattr(n, "_post", capture)
    return captured


def test_disabled_without_webhook(monkeypatch):
    """Slack is disabled and notify_success is a no-op without a webhook URL."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert n.slack_enabled() is False
    # No webhook -> no-op, returns False, does not raise.
    assert n.notify_success("data-pull", {"Rows": 5}) is False


def test_enabled_with_webhook(monkeypatch):
    """slack_enabled is True when SLACK_WEBHOOK_URL is set."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
    assert n.slack_enabled() is True


def test_success_payload(capture_slack):
    """Success payload: a section (title + code block) and a context footer."""
    ok = n.notify_success("data-pull", {"Lead type": "auto (6)", "Rows fetched": 42})
    assert ok is True
    payload = capture_slack["payload"]
    kinds = [b["type"] for b in payload["blocks"]]
    assert kinds[0] == "section" and kinds[-1] == "context"
    body = payload["blocks"][0]["text"]["text"]
    assert ":white_check_mark:" in body
    assert "```" in body  # values render inside a monospace code block
    # values are carried in the fallback text too
    assert "auto (6)" in payload["text"]
    assert "42" in payload["text"]


def test_failure_payload_includes_error_and_mention(monkeypatch, capture_slack):
    """Failure payload includes the error text and configured mention."""
    monkeypatch.setenv("SLACK_MENTION_ON_FAILURE", "<@U123>")
    ok = n.notify_failure("build-features", {"Lead type": "home (1)"}, error="boom")
    assert ok is True
    payload = capture_slack["payload"]
    assert ":x:" in payload["blocks"][0]["text"]["text"]
    assert "boom" in payload["text"]
    # mention appears somewhere in the blocks
    dumped = json.dumps(payload)
    assert "<@U123>" in dumped


def test_empty_fields_are_skipped(capture_slack):
    """Empty or blank field values are omitted from the payload."""
    n.notify_success("data-pull", {"Present": "x", "Empty": None, "Blank": ""})
    payload = capture_slack["payload"]
    assert "Present" in payload["text"]
    assert "Empty" not in payload["text"]
    assert "Blank" not in payload["text"]


def test_long_error_text_is_truncated(capture_slack):
    """Long error details are trimmed when the payload is formatted."""
    n.notify_failure("data-pull", {}, error="x" * 2000)
    payload = capture_slack["payload"]
    assert "x" * 1500 in payload["text"]
    assert "x" * 1501 not in payload["text"]
    assert "(truncated)" in payload["text"]


def test_flow_failure_hook_builds_fields(capture_slack):
    """flow_failure_hook builds a failure payload from Prefect flow objects."""

    class FakeFlow:
        name = "smarthub-data-pull"

    class FakeFlowRun:
        id = "abc-123"
        name = "brave-otter"
        deployment_id = "dep-1"
        parameters = {"lead_type_id": 6, "lead_type_name": "auto"}

    class FakeState:
        message = "Task 'fetch' failed: Redshift timeout"

    # Must not raise; the failure payload is captured locally.
    n.flow_failure_hook(FakeFlow(), FakeFlowRun(), FakeState())
    payload = capture_slack["payload"]
    assert "auto (6)" in payload["text"]
    assert "Redshift timeout" in payload["text"]
    assert ":x:" in payload["blocks"][0]["text"]["text"]


def test_flow_failure_hook_swallows_bad_input(monkeypatch):
    """flow_failure_hook swallows bad input without raising."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
    # Passing junk should not raise (hook must never mask the real error).
    n.flow_failure_hook(None, None, None)


def test_grouped_payload_structure(capture_slack):
    """Grouped payload: title + headline outside, fields flattened into a box."""
    ok = n.notify_success_grouped(
        "train-model",
        subject="auto (6)",
        headline=":white_check_mark: *Promoted to serving* · `v4`",
        groups=[
            ("Model", {"Model": "lightgbm", "Rows trained": 147628}),
            ("Performance (held-out)", {"ROC AUC": "0.883", "Empty": None}),
            ("Features · 25", {"Optional excluded": "military_affiliation"}),
        ],
        footer_extra="model `/app/data/models/auto/v4.pkl`",
    )
    assert ok is True
    payload = capture_slack["payload"]
    kinds = [b["type"] for b in payload["blocks"]]
    # No header/dividers anymore: just one section (title+headline+box) + context
    assert kinds == ["section", "context"]
    body = payload["blocks"][0]["text"]["text"]
    assert "auto (6)" in body  # subject in the title line
    assert "Promoted to serving" in body  # headline rendered outside the box
    assert "```" in body  # fields flattened into a code block
    assert "Empty" not in payload["text"]  # empty value skipped
    assert "military_affiliation" in payload["text"]
    ctx = payload["blocks"][-1]["elements"][0]["text"]
    assert "v4.pkl" in ctx  # footer in the context block


def test_grouped_payload_skips_empty_values(capture_slack):
    """Empty values contribute no rows; group titles are flattened away."""
    n.notify_success_grouped(
        "train-model",
        groups=[
            ("Real", {"a": 1}),
            ("AllEmpty", {"x": None, "y": ""}),
        ],
    )
    payload = capture_slack["payload"]
    assert "a: 1" in payload["text"]  # the real field renders
    assert "x:" not in payload["text"]  # empty fields add no rows
    assert "y:" not in payload["text"]


def test_critical_pings_and_routes(monkeypatch, capture_slack):
    """Critical payload selects the critical route and includes @here."""
    monkeypatch.setenv(
        "SLACK_WEBHOOK_CRITICAL_URL", "https://hooks.slack.test/critical"
    )
    ok = n.notify_critical("bid-api", {"TAT p99 (s)": 1.83}, error="p99 over target")
    assert ok is True
    assert capture_slack["url"] == "https://hooks.slack.test/critical"
    body = capture_slack["payload"]["blocks"][0]["text"]["text"]
    assert ":red_circle:" in body and "<!here>" in body


def test_category_routing_selects_webhook(monkeypatch, capture_slack):
    """Each severity selects its intended webhook without posting."""
    monkeypatch.setenv("SLACK_WEBHOOK_UPDATES_URL", "https://hooks.slack.test/updates")
    monkeypatch.setenv(
        "SLACK_WEBHOOK_FAILURES_URL", "https://hooks.slack.test/failures"
    )
    n.notify_success("data-pull", {"Rows": 5})
    assert capture_slack["url"] == "https://hooks.slack.test/updates"
    n.notify_failure("data-pull", {"Rows": 0}, error="boom")
    assert capture_slack["url"] == "https://hooks.slack.test/failures"


def test_category_falls_back_to_single_webhook(capture_slack):
    """With no per-category URL set, routing selects the legacy webhook."""
    n.notify_success("data-pull", {"Rows": 5})
    assert capture_slack["url"] == "https://hooks.slack.test/T/B/xxx"


def test_env_tag_in_title(monkeypatch, capture_slack):
    """SLACK_ENV_LABEL shows (uppercased) in the title when set."""
    monkeypatch.setenv("SLACK_ENV_LABEL", "prod")
    n.notify_success("data-pull", {"Rows": 5})
    body = capture_slack["payload"]["blocks"][0]["text"]["text"]
    assert "SmartHub PROD · data-pull" in body


def test_slack_enabled_with_only_category_webhook(monkeypatch):
    """slack_enabled is True when only a per-category webhook is set."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    for env in (
        "SLACK_WEBHOOK_UPDATES_URL",
        "SLACK_WEBHOOK_WARNINGS_URL",
        "SLACK_WEBHOOK_CRITICAL_URL",
        "SLACK_WEBHOOK_FAILURES_URL",
    ):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(
        "SLACK_WEBHOOK_CRITICAL_URL", "https://hooks.slack.test/critical"
    )
    assert n.slack_enabled() is True


@pytest.mark.parametrize(
    "status",
    [
        "promoted",
        "awaiting manual promotion",
        "completed (not promoted)",
        "completed (promotion disabled)",
    ],
)
def test_success_display_status_preserves_updates_routing(
    capture_slack, monkeypatch, status
):
    monkeypatch.setenv("SLACK_WEBHOOK_UPDATES_URL", "https://hooks.slack.test/updates")
    assert n.notify_success_grouped("train-model", subject="auto (6)", status=status)
    payload = capture_slack["payload"]
    assert f"· train-model · {status} · auto (6)" in payload["text"]
    assert (
        f"· train-model · {status} · auto (6)" in payload["blocks"][0]["text"]["text"]
    )
    assert capture_slack["url"] == "https://hooks.slack.test/updates"
    assert ":white_check_mark:" in payload["text"]
    assert "<!here>" not in payload["text"]


def test_display_status_does_not_change_failure_routing(capture_slack, monkeypatch):
    monkeypatch.setenv(
        "SLACK_WEBHOOK_FAILURES_URL", "https://hooks.slack.test/failures"
    )
    n.notify_grouped("failure", "train-model", status="FAILED")
    assert capture_slack["url"] == "https://hooks.slack.test/failures"
    assert "<!here>" in capture_slack["payload"]["text"]


@pytest.mark.parametrize(
    "result_flags,status,headline",
    [
        ({"promoted": True}, "promoted", "Promoted to serving"),
        (
            {
                "promotion_status": "awaiting_manual_promotion",
                "eligibility_status": "eligible",
            },
            "awaiting manual promotion",
            "Eligible — awaiting manual promotion",
        ),
        (
            {"eligibility_status": "eligible", "promotion_status": "skipped"},
            "completed (not promoted)",
            "Eligible — promotion execution skipped",
        ),
        (
            {"eligibility_status": "rejected"},
            "completed (not promoted)",
            "Not eligible — serving model unchanged",
        ),
        (
            {"promotion_mode": "disabled"},
            "completed (promotion disabled)",
            "Promotion evaluation disabled",
        ),
    ],
)
def test_training_title_status(monkeypatch, result_flags, status, headline):
    from smarthub.train_and_predict import flow

    monkeypatch.setattr(
        flow,
        "_feature_breakdown",
        lambda *args: {
            "total": 2,
            "n_registered_used": 2,
            "n_registered": 2,
        },
    )
    sent = []
    monkeypatch.setattr(
        flow.notifications,
        "_post",
        lambda payload, category=None: sent.append((payload, category)) or True,
    )
    result = {
        "promoted": False,
        "promotion_mode": "automatic",
        "lineage": {},
        "training_run_id": "run_test",
        "model_path": "data/models/auto/model.pkl",
        "promotion_reason": "test decision",
        **result_flags,
    }
    flow._notify_success("auto", 6, result, {}, {})
    assert len(sent) == 1
    payload, category = sent[0]
    assert category == "updates"
    assert f"· train-model · {status} · auto (6)" in payload["text"]
    assert headline in payload["text"]
    assert "test decision" in payload["text"]
    assert "<!here>" not in payload["text"]
