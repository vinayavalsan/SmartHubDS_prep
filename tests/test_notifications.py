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
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert n.slack_enabled() is False
    # No webhook -> no-op, returns False, does not raise.
    assert n.notify_success("data-pull", {"Rows": 5}) is False


def test_enabled_with_webhook(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
    assert n.slack_enabled() is True


def test_success_payload(capture_slack):
    ok = n.notify_success("data-pull", {"Lead type": "auto (6)", "Rows fetched": 42})
    assert ok is True
    payload = capture_slack["payload"]
    # header + at least one field section + context
    kinds = [b["type"] for b in payload["blocks"]]
    assert "header" in kinds and "context" in kinds
    assert ":white_check_mark:" in payload["blocks"][0]["text"]["text"]
    # values are carried in the fallback text too
    assert "auto (6)" in payload["text"]
    assert "42" in payload["text"]


def test_failure_payload_includes_error_and_mention(monkeypatch, capture_slack):
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
    n.notify_success("data-pull", {"Present": "x", "Empty": None, "Blank": ""})
    payload = capture_slack["payload"]
    assert "Present" in payload["text"]
    assert "Empty" not in payload["text"]
    assert "Blank" not in payload["text"]


def test_never_raises_on_network_error(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(n.urllib.request, "urlopen", boom)
    # Swallowed -> returns False, no exception.
    assert n.notify_failure("data-pull", {"Rows": 1}, error="x") is False


def test_flow_failure_hook_builds_fields(capture_slack):
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
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
    # Passing junk should not raise (hook must never mask the real error).
    n.flow_failure_hook(None, None, None)


def test_grouped_payload_structure(capture_slack):
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
    # header, a headline section, 3 dividers (one per non-empty group), context
    assert kinds[0] == "header"
    assert kinds.count("divider") == 3
    assert kinds[-1] == "context"
    # subject shows in the header; empty field skipped; footer in context
    assert "auto (6)" in payload["blocks"][0]["text"]["text"]
    assert "Empty" not in payload["text"]
    assert "military_affiliation" in payload["text"]
    ctx = payload["blocks"][-1]["elements"][0]["text"]
    assert "v4.pkl" in ctx
    # group titles render as bold section text
    dumped = json.dumps(payload)
    assert "*Model*" in dumped and "*Performance (held-out)*" in dumped


def test_grouped_payload_skips_empty_group(capture_slack):
    n.notify_success_grouped(
        "train-model",
        groups=[
            ("Real", {"a": 1}),
            ("AllEmpty", {"x": None, "y": ""}),
        ],
    )
    payload = capture_slack["payload"]
    dumped = json.dumps(payload)
    assert "*Real*" in dumped
    assert "*AllEmpty*" not in dumped  # group with no values is dropped entirely
