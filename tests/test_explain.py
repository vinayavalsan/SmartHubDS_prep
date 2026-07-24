"""Tests for the offline "why did Anton bid $X" explanation feature.

Split into two tiers, same convention as test_train_and_predict.py:
  - pure-logic tests (prompt formatting, Ollama call/fallback, the
    "no viable bid" short-circuit) run in the base env — no shap/lightgbm
    needed since those are lazily imported inside explain.py.
  - SHAP/LightGBM-gated tests (`pytest.importorskip`) fit a tiny real
    LightGBM pipeline and exercise the actual SHAP attribution path.
"""

import numpy as np
import pandas as pd
import pytest

from smarthub.server import predict
from smarthub.train_and_predict import config, explain, llm_explain

# --- _to_native (JSON-safety for numpy scalars) ------------------------------


def test_to_native_converts_numpy_scalars():
    """numpy int64/float64/bool_ become plain Python types FastAPI can encode.

    Regression test: frame.iloc[0][col] returns numpy scalars for numeric
    columns, and some FastAPI/pydantic version combinations can't
    JSON-encode those (seen as "TypeError: 'numpy.int64' object is not
    iterable" from jsonable_encoder when /explain_bid's top_factors carried
    one straight through).
    """
    assert explain._to_native(np.int64(34)) == 34
    assert not isinstance(explain._to_native(np.int64(34)), np.generic)
    assert explain._to_native(np.float64(1.5)) == 1.5
    assert not isinstance(explain._to_native(np.float64(1.5)), np.generic)
    assert explain._to_native(np.bool_(True)) is True
    # non-numpy values pass through unchanged
    assert explain._to_native("TX") == "TX"
    assert explain._to_native(None) is None


# --- format_llm_prompt -------------------------------------------------------


def _facts(**overrides):
    facts = {
        "recommended_bid": 12.5,
        "predicted_win_rate": 0.42,
        "base_win_rate": 0.30,
        "expected_profit": 3.75,
        "top_factors": [
            {"feature": "bid", "value": 12.5, "shap": 0.08, "direction": "increased"},
            {
                "feature": "state",
                "value": "TX",
                "shap": -0.02,
                "direction": "decreased",
            },
        ],
    }
    facts.update(overrides)
    return facts


def test_format_llm_prompt_includes_facts_no_jargon_instruction():
    """The prompt carries every numeric fact and the anti-hallucination rule."""
    prompt = explain.format_llm_prompt(_facts())
    assert "Do not invent numbers" in prompt
    assert "$12.50" in prompt
    assert "42%" in prompt and "30%" in prompt
    assert "$3.75" in prompt
    assert "bid=12.5: increased win likelihood" in prompt
    assert "state=TX: decreased win likelihood" in prompt


def test_format_llm_prompt_includes_monotonic_bid_guardrail():
    """The prompt tells the LLM win rate never falls as bid rises.

    Regression guard: on a real lead, the LLM once reasoned backwards about
    the `bid` factor's SHAP sign, implying a lower bid would win more often
    -- which the model's monotonic constraint rules out.
    """
    prompt = explain.format_llm_prompt(_facts())
    assert "never decreases as the bid rises" in prompt
    assert "never claim a lower bid would win more often" in prompt


def test_format_llm_prompt_includes_bid_curve_when_present():
    """Nearby bid/win-rate points are rendered as facts, not left for the LLM
    to guess at ("the shape of the market" around the chosen bid)."""
    facts = _facts(
        bid_curve=[
            {"bid": 3.25, "predicted_win_rate": 0.50, "expected_profit": 8.0},
            {"bid": 3.75, "predicted_win_rate": 0.58, "expected_profit": 9.5},
        ]
    )
    prompt = explain.format_llm_prompt(facts)
    assert "Nearby bids explored" in prompt
    assert "$3.25 -> 50%" in prompt
    assert "$3.75 -> 58%" in prompt


def test_format_llm_prompt_omits_bid_curve_section_when_absent():
    """No bid_curve fact -> no "Nearby bids explored" section (nothing to fake)."""
    prompt = explain.format_llm_prompt(_facts())
    assert "Nearby bids explored" not in prompt


def test_format_llm_prompt_lists_all_top_factors_in_order():
    """Factors are rendered in the order given (already SHAP-ranked upstream)."""
    facts = _facts(
        top_factors=[
            {"feature": "a", "value": 1, "shap": 0.5, "direction": "increased"},
            {"feature": "b", "value": 2, "shap": 0.3, "direction": "increased"},
            {"feature": "c", "value": 3, "shap": -0.1, "direction": "decreased"},
        ]
    )
    prompt = explain.format_llm_prompt(facts)
    assert prompt.index("a=1") < prompt.index("b=2") < prompt.index("c=3")


# --- call_ollama --------------------------------------------------------------


def test_call_ollama_success(monkeypatch):
    """A reachable Ollama server's response text is returned, stripped."""
    import requests

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "  Anton bid higher because of X.  \n"}

    def _fake_post(url, json, timeout):
        assert url == "http://localhost:11434/api/generate"
        assert json["model"] == explain.LLM_MODEL
        assert json["stream"] is False
        return _Resp()

    monkeypatch.setattr(requests, "post", _fake_post)
    assert explain.call_ollama("some prompt") == "Anton bid higher because of X."


def test_call_ollama_falls_back_when_unreachable(monkeypatch):
    """An unreachable/erroring Ollama server yields a clear fallback, not a raise."""
    import requests

    def _fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", _fake_post)
    result = explain.call_ollama("some prompt", host="http://localhost:11434")
    assert "Explanation unavailable" in result
    assert "http://localhost:11434" in result
    assert "numeric factors above are still accurate" in result


def test_call_ollama_uses_ini_defaults_when_unset(monkeypatch):
    """model/host/timeout fall back to the [explain] ini-backed module constants."""
    import requests

    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok"}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["model"] = json["model"]
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(requests, "post", _fake_post)
    explain.call_ollama("prompt")
    assert captured["url"] == f"{explain.OLLAMA_HOST}/api/generate"
    assert captured["model"] == explain.LLM_MODEL
    assert captured["timeout"] == explain.LLM_TIMEOUT_SECONDS


# --- is_model_pulled / pull_model / ensure_model_pulled_async ----------------
#
# Ollama model-availability check + non-blocking pull (2026-07-23) -- pure
# HTTP calls against Ollama's own API (GET /api/tags, POST /api/pull), no
# shap/lightgbm needed, so these run in the base env like the call_ollama
# tests above.


def test_is_model_pulled_true_when_present(monkeypatch):
    """A model name present in /api/tags's response is reported as pulled."""
    import requests

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "qwen2.5:1.5b-instruct"}, {"name": "llama3"}]}

    monkeypatch.setattr(requests, "get", lambda url, timeout: _Resp())
    assert explain.is_model_pulled("qwen2.5:1.5b-instruct", "http://ollama:11434")


def test_is_model_pulled_false_when_absent(monkeypatch):
    """A model name NOT in /api/tags's response is reported as not pulled."""
    import requests

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "llama3"}]}

    monkeypatch.setattr(requests, "get", lambda url, timeout: _Resp())
    assert not explain.is_model_pulled("qwen2.5:1.5b-instruct", "http://ollama:11434")


def test_is_model_pulled_false_when_ollama_unreachable(monkeypatch):
    """Any failure reaching Ollama is treated as 'not pulled', not raised --
    the caller reacts to either case the same way (attempt a pull)."""
    import requests

    def _boom(url, timeout):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _boom)
    assert not explain.is_model_pulled("qwen2.5:1.5b-instruct", "http://ollama:11434")


def test_pull_model_posts_expected_payload(monkeypatch):
    """pull_model hits POST /api/pull with the model name, no timeout (a
    real pull has no fixed time budget)."""
    import requests

    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(requests, "post", _fake_post)
    assert explain.pull_model("qwen2.5:1.5b-instruct", "http://ollama:11434") is True
    assert captured["url"] == "http://ollama:11434/api/pull"
    assert captured["json"] == {"name": "qwen2.5:1.5b-instruct", "stream": False}
    assert captured["timeout"] is None


def test_pull_model_returns_false_on_failure_without_raising(monkeypatch):
    """A failed pull (Ollama down mid-pull, network blip) returns False --
    never raises into the caller (background thread)."""
    import requests

    def _boom(url, json, timeout):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", _boom)
    assert explain.pull_model("qwen2.5:1.5b-instruct", "http://ollama:11434") is False


def test_ensure_model_pulled_sync_skips_pull_when_already_present(monkeypatch):
    """Already-pulled model -> no pull attempted at all."""
    # Patched on llm_explain (not explain) -- _ensure_model_pulled_sync's
    # internal calls to is_model_pulled/pull_model are bare-name lookups
    # resolved in llm_explain's own module globals, since that's where all
    # three are actually defined (explain.py only re-imports/re-exports the
    # names for its own public surface -- see explain.py's module docstring).
    monkeypatch.setattr(llm_explain, "is_model_pulled", lambda model, host: True)

    def _boom(*args, **kwargs):
        raise AssertionError("pull_model must not be called when already pulled")

    monkeypatch.setattr(llm_explain, "pull_model", _boom)

    explain._ensure_model_pulled_sync(
        "qwen2.5:1.5b-instruct",
        "http://ollama:11434",
        wait_for_host_seconds=1,
        poll_interval_seconds=0.01,
    )


def test_ensure_model_pulled_sync_pulls_when_missing(monkeypatch):
    """Missing model -> pull_model is called with the same model/host."""
    import requests

    class _Resp:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "get", lambda url, timeout: _Resp())
    # Patched on llm_explain -- see comment in
    # test_ensure_model_pulled_sync_skips_pull_when_already_present.
    monkeypatch.setattr(llm_explain, "is_model_pulled", lambda model, host: False)

    called = {}

    def _fake_pull(model, host):
        called["model"] = model
        called["host"] = host
        return True

    monkeypatch.setattr(llm_explain, "pull_model", _fake_pull)

    explain._ensure_model_pulled_sync(
        "qwen2.5:1.5b-instruct",
        "http://ollama:11434",
        wait_for_host_seconds=1,
        poll_interval_seconds=0.01,
    )
    assert called == {"model": "qwen2.5:1.5b-instruct", "host": "http://ollama:11434"}


def test_ensure_model_pulled_sync_gives_up_quietly_if_host_never_reachable(
    monkeypatch,
):
    """Ollama never becomes reachable within the wait budget -> no pull
    attempted, no exception raised (retried automatically next startup)."""
    import requests

    def _boom(url, timeout):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _boom)

    def _fail(*args, **kwargs):
        raise AssertionError("pull_model must not run if Ollama is unreachable")

    # Patched on llm_explain -- see comment in
    # test_ensure_model_pulled_sync_skips_pull_when_already_present.
    monkeypatch.setattr(llm_explain, "pull_model", _fail)

    # Small budget so the test itself stays fast.
    explain._ensure_model_pulled_sync(
        "qwen2.5:1.5b-instruct",
        "http://ollama:11434",
        wait_for_host_seconds=0.05,
        poll_interval_seconds=0.01,
    )


def test_ensure_model_pulled_async_returns_immediately(monkeypatch, tmp_path):
    """The public entry point spawns a background thread and returns right
    away -- doesn't block on the (potentially slow) sync check/pull."""
    import threading
    import time

    # Patched on llm_explain -- ensure_model_pulled_async's background thread
    # target (_ensure_model_pulled_locked) and its internal call to
    # _ensure_model_pulled_sync are both bare-name lookups resolved in
    # llm_explain's own module globals, since that's where all three are
    # actually defined (see comment in
    # test_ensure_model_pulled_sync_skips_pull_when_already_present).
    monkeypatch.setattr(llm_explain, "_OLLAMA_PULL_LOCK_PATH", str(tmp_path / "lock"))

    started = threading.Event()
    finish = threading.Event()

    def _slow_sync(model=None, host=None):
        started.set()
        finish.wait(timeout=5)  # would block the caller for up to 5s if not threaded

    monkeypatch.setattr(llm_explain, "_ensure_model_pulled_sync", _slow_sync)

    t0 = time.monotonic()
    explain.ensure_model_pulled_async()
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0  # returned immediately, did not wait for _slow_sync
    assert started.wait(timeout=2)  # the background thread did start
    finish.set()  # let the background thread exit cleanly


def test_ensure_model_pulled_locked_dedupes_concurrent_workers(monkeypatch, tmp_path):
    """Regression test for the live incident (2026-07-23): with
    SERVE_WORKERS > 1, every uvicorn worker process independently called
    ensure_model_pulled_async at its own startup, so all of them fired a
    simultaneous POST /api/pull for the identical model -- enough
    bandwidth/CPU contention that /docs and /health started timing out.

    Simulates two "workers" (threads standing in for separate worker
    PROCESSES -- the lock is filesystem-based, so it dedupes the same way
    across real processes) racing for the same pull: only the first to grab
    the lock should actually run the check/pull; the second must see it
    held and return immediately, without duplicating the work."""
    import threading

    # Patched on llm_explain -- see comment in
    # test_ensure_model_pulled_async_returns_immediately.
    monkeypatch.setattr(llm_explain, "_OLLAMA_PULL_LOCK_PATH", str(tmp_path / "lock"))

    call_count = {"n": 0}
    first_running = threading.Event()
    release_first = threading.Event()

    def _fake_sync(model=None, host=None):
        call_count["n"] += 1
        first_running.set()
        release_first.wait(timeout=5)

    monkeypatch.setattr(llm_explain, "_ensure_model_pulled_sync", _fake_sync)

    first = threading.Thread(target=explain._ensure_model_pulled_locked)
    first.start()
    assert first_running.wait(timeout=2)  # first worker is now holding the lock

    # Second "worker" arrives while the first still holds the lock -- must
    # not call _ensure_model_pulled_sync at all, and must not block/wait.
    second = threading.Thread(target=explain._ensure_model_pulled_locked)
    second.start()
    second.join(timeout=2)
    assert not second.is_alive()  # returned immediately, didn't wait for the lock

    release_first.set()
    first.join(timeout=2)

    assert call_count["n"] == 1  # the pull logic ran exactly once, not twice


# --- explain_bid: no-viable-bid short circuit (no shap/lightgbm needed) -------


def test_explain_bid_no_viable_bid_skips_shap_and_llm(monkeypatch):
    """When no bid clears the margin floor, explain_bid skips SHAP + the LLM."""

    def _fake_optimize(
        row,
        model,
        manifest,
        expected_revenue,
        target_cm,
        min_bid,
        bid_step,
        created_dayofweek=None,
        created_hour=None,
    ):
        return {
            "recommended_bid": float("nan"),
            "recommended_bid_predicted_win_rate": None,
            "recommended_bid_predicted_profit": None,
            "decision_path": "model",
            "decision_reason": "No viable bid: too little margin.",
        }

    monkeypatch.setattr(predict, "decide_bid", _fake_optimize, raising=False)
    monkeypatch.setattr(
        explain.preprocessing,
        "serving_frame",
        lambda records, lead_type_id: pd.DataFrame([{"bid": 0.25}]),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("SHAP/LLM must not run on the no-viable-bid path")

    monkeypatch.setattr(explain, "explain_row", _boom)
    monkeypatch.setattr(explain, "call_ollama", _boom)

    result = explain.explain_bid(
        model="unused",
        record={"bid": 0.25},
        lead_type_id=6,
        expected_revenue=1.0,
    )
    assert pd.isna(result["recommended_bid"])
    assert result["top_factors"] == []
    assert result["base_win_rate"] is None
    assert "No viable bid" in result["explanation"]


def test_explain_bid_cold_start_skips_shap_and_llm_uses_policy_text(monkeypatch):
    """A true cold-start bid explains itself via the policy reason, no SHAP/LLM."""

    def _fake_decide(
        row,
        model,
        manifest,
        expected_revenue,
        target_cm,
        min_bid,
        bid_step,
        created_dayofweek=None,
        created_hour=None,
    ):
        return {
            "recommended_bid": 5.0,
            "recommended_bid_predicted_win_rate": None,
            "recommended_bid_predicted_profit": None,
            "decision_path": "cold_start_fallback",
            "decision_reason": "No model has ever been trained/promoted yet.",
        }

    monkeypatch.setattr(predict, "decide_bid", _fake_decide, raising=False)
    monkeypatch.setattr(
        explain.preprocessing,
        "serving_frame",
        lambda records, lead_type_id: pd.DataFrame([{"bid": 0.25}]),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("SHAP/LLM must not run on the cold-start path")

    monkeypatch.setattr(explain, "explain_row", _boom)
    monkeypatch.setattr(explain, "call_ollama", _boom)

    result = explain.explain_bid(
        model=None,
        record={"bid": 0.25},
        lead_type_id=6,
        expected_revenue=10.0,
    )
    assert result["recommended_bid"] == 5.0
    assert result["top_factors"] == []
    assert result["base_win_rate"] is None
    assert result["explanation"] == "No model has ever been trained/promoted yet."


def test_explain_bid_swaps_in_recommended_bid_before_explaining(monkeypatch):
    """SHAP explains the model's prediction AT the chosen bid, not the input bid."""

    def _fake_decide(
        row,
        model,
        manifest,
        expected_revenue,
        target_cm,
        min_bid,
        bid_step,
        created_dayofweek=None,
        created_hour=None,
    ):
        return {
            "recommended_bid": 15.0,
            "recommended_bid_predicted_win_rate": 0.5,
            "recommended_bid_predicted_profit": 4.0,
            "max_bid": 18.0,
            "decision_path": "model",
            "decision_reason": "Standard profit-maximizing bid.",
        }

    seen_records = {}

    def _fake_explain_row(model, record, lead_type_id, top_n=None):
        seen_records["record"] = record
        return {"top_factors": [], "base_win_rate": 0.3}

    monkeypatch.setattr(predict, "decide_bid", _fake_decide, raising=False)
    monkeypatch.setattr(
        explain.preprocessing,
        "serving_frame",
        lambda records, lead_type_id: pd.DataFrame([{"bid": 0.25}]),
    )
    monkeypatch.setattr(explain, "explain_row", _fake_explain_row)
    monkeypatch.setattr(
        predict,
        "bid_curve_around",
        lambda **kwargs: [
            {"bid": 15.0, "predicted_win_rate": 0.5, "expected_profit": 4.0}
        ],
        raising=False,
    )
    monkeypatch.setattr(explain, "call_ollama", lambda prompt: "stub explanation")

    result = explain.explain_bid(
        model="unused",
        record={"bid": 0.25},
        lead_type_id=6,
        expected_revenue=10.0,
    )
    assert seen_records["record"]["bid"] == 15.0  # swapped, not the input 0.25
    assert result["recommended_bid"] == 15.0
    assert result["explanation"] == "stub explanation"
    assert result["bid_curve"] == [
        {"bid": 15.0, "predicted_win_rate": 0.5, "expected_profit": 4.0}
    ]


def test_explain_bid_includes_decision_note_for_exploration(monkeypatch):
    """A non-'model' decision path is passed to the LLM as a factual note."""

    def _fake_decide(
        row,
        model,
        manifest,
        expected_revenue,
        target_cm,
        min_bid,
        bid_step,
        created_dayofweek=None,
        created_hour=None,
    ):
        return {
            "recommended_bid": 12.0,
            "recommended_bid_predicted_win_rate": 0.4,
            "recommended_bid_predicted_profit": 3.0,
            "max_bid": 15.0,
            "decision_path": "exploration",
            "decision_reason": "Scheduled exploration probe above the optimum.",
        }

    monkeypatch.setattr(predict, "decide_bid", _fake_decide, raising=False)
    monkeypatch.setattr(
        explain.preprocessing,
        "serving_frame",
        lambda records, lead_type_id: pd.DataFrame([{"bid": 0.25}]),
    )
    monkeypatch.setattr(
        explain,
        "explain_row",
        lambda model, record, lead_type_id, top_n=None: (
            {"top_factors": [], "base_win_rate": 0.3}
        ),
    )
    monkeypatch.setattr(
        predict,
        "bid_curve_around",
        lambda **kwargs: [
            {"bid": 12.0, "predicted_win_rate": 0.4, "expected_profit": 3.0}
        ],
        raising=False,
    )

    seen_prompts = {}

    def _fake_ollama(prompt):
        seen_prompts["prompt"] = prompt
        return "stub"

    monkeypatch.setattr(explain, "call_ollama", _fake_ollama)

    result = explain.explain_bid(
        model="unused",
        record={"bid": 0.25},
        lead_type_id=6,
        expected_revenue=10.0,
    )
    assert result["decision_path"] == "exploration"
    assert "Scheduled exploration probe" in seen_prompts["prompt"]


# --- SHAP / LightGBM-gated: real fitted pipeline ------------------------------


NUMERIC = ["bid", "age"]
CATEGORICAL = ["state"]


def _tiny_lightgbm_pipeline(calibrate=False):
    pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
    from smarthub.train_and_predict import models

    n = 40
    frame = pd.DataFrame(
        {
            "bid": [float(i % 10) for i in range(n)],
            "age": [20 + (i % 40) for i in range(n)],
            "state": ["TX", "CA"] * (n // 2),
        }
    )
    y = [1 if (i % 10) >= 5 else 0 for i in range(n)]  # correlated with bid

    model = models.build_model(
        "lightgbm",
        NUMERIC,
        CATEGORICAL,
        model_params={"n_estimators": 5, "min_child_samples": 1, "num_leaves": 7},
        calibrate=calibrate,
    )
    model.fit(frame, y)
    return model


@pytest.fixture
def small_feature_columns(monkeypatch):
    """Point config.feature_columns at the tiny NUMERIC/CATEGORICAL set above."""
    monkeypatch.setattr(
        config, "feature_columns", lambda lead_type_id: (NUMERIC, CATEGORICAL)
    )


def test_fitted_lgbm_estimators_plain_pipeline(small_feature_columns):
    """A plain (uncalibrated) pipeline yields exactly one (preprocessor, clf) pair."""
    lightgbm_mod = pytest.importorskip("lightgbm")
    model = _tiny_lightgbm_pipeline(calibrate=False)
    pairs = explain._fitted_lgbm_estimators(model)
    assert len(pairs) == 1
    preprocessor, classifier = pairs[0]
    assert isinstance(classifier, lightgbm_mod.LGBMClassifier)


def test_fitted_lgbm_estimators_calibrated_has_one_pair_per_fold(small_feature_columns):
    """A calibrated model yields one fitted pipeline per CV fold (cv=3)."""
    model = _tiny_lightgbm_pipeline(calibrate=True)
    pairs = explain._fitted_lgbm_estimators(model)
    assert len(pairs) == 3


def test_fitted_lgbm_estimators_rejects_non_lightgbm(small_feature_columns):
    """A non-LightGBM classifier raises a clear, actionable ValueError."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
    from smarthub.train_and_predict import models

    model = models.build_model(
        "logistic_regression",
        NUMERIC,
        CATEGORICAL,
        model_params={},
        calibrate=False,
    )
    frame = pd.DataFrame({"bid": [1.0, 2.0], "age": [30, 40], "state": ["TX", "CA"]})
    model.fit(frame, [0, 1])
    with pytest.raises(ValueError, match="model_type='lightgbm'"):
        explain._fitted_lgbm_estimators(model)


def test_explain_row_returns_top_factors_ranked_by_shap(
    small_feature_columns, monkeypatch
):
    """explain_row ranks factors by |SHAP| and reports a base win rate."""
    pytest.importorskip("shap")
    model = _tiny_lightgbm_pipeline(calibrate=False)
    record = {"bid": 7.0, "age": 33, "state": "TX"}

    result = explain.explain_row(model, record, lead_type_id=6, top_n=2)
    assert len(result["top_factors"]) == 2
    shap_abs = [abs(f["shap"]) for f in result["top_factors"]]
    assert shap_abs == sorted(shap_abs, reverse=True)
    for f in result["top_factors"]:
        assert f["direction"] in ("increased", "decreased")
        # regression guard: numpy scalars aren't JSON-safe for FastAPI's
        # encoder (see test_to_native_converts_numpy_scalars).
        assert not isinstance(f["value"], np.generic)
    # base_win_rate must be a genuine probability (TreeExplainer works in
    # log-odds space internally; _shap_for_row must convert it back).
    assert 0.0 <= result["base_win_rate"] <= 1.0


def test_explain_row_respects_top_n_default_from_ini(small_feature_columns):
    """With top_n=None, explain_row falls back to TOP_N_FACTORS (ini-configurable)."""
    model = _tiny_lightgbm_pipeline(calibrate=False)
    record = {"bid": 7.0, "age": 33, "state": "TX"}
    result = explain.explain_row(model, record, lead_type_id=6, top_n=None)
    # only 3 features exist here, so top_n=5 (ini default) is clamped by
    # availability
    expected = min(explain.TOP_N_FACTORS, len(NUMERIC + CATEGORICAL))
    assert len(result["top_factors"]) == expected
