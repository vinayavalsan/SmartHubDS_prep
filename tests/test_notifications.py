"""Tests for the Slack notifier — payload shape and best-effort safety."""

import json
from contextlib import contextmanager

import pytest

from smarthub.core import notifications as n


@pytest.fixture
def capture_slack(monkeypatch):
    """Capture the JSON payload sent to Slack instead of hitting the network."""
    sent = {}

    @contextmanager
    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["payload"] = json.loads(req.data.decode("utf-8"))

        class _Resp:
            def read(self):
                return b"ok"

        yield _Resp()

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/T/B/xxx")
    monkeypatch.setattr(n.urllib.request, "urlopen", fake_urlopen)
    return sent


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
    assert ":red_circle:" in payload["blocks"][0]["text"]["text"]
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


def test_never_raises_on_network_error(monkeypatch):
    """Network errors are swallowed; notify returns False without raising."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(n.urllib.request, "urlopen", boom)
    # Swallowed -> returns False, no exception.
    assert n.notify_failure("data-pull", {"Rows": 1}, error="x") is False


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

    # Must not raise, and should send a failure payload.
    n.flow_failure_hook(FakeFlow(), FakeFlowRun(), FakeState())
    payload = capture_slack["payload"]
    assert "auto (6)" in payload["text"]
    assert "Redshift timeout" in payload["text"]
    assert ":red_circle:" in payload["blocks"][0]["text"]["text"]


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
