"""Tests for API-key authentication (smarthub.server.auth).

Store-level tests use a temp SQLite DB directly. The dependency/endpoint tests
run against the real FastAPI app via TestClient, toggling
``SMARTHUB_API_AUTH_ENABLED`` per test so the default (auth off) path stays the
behaviour every other test relies on.
"""

from __future__ import annotations

import pytest

from smarthub.server import auth


@pytest.fixture
def store(tmp_path):
    """An ApiKeyStore on a fresh temp SQLite DB, cache TTL 0 (always fresh)."""
    url = f"sqlite:///{tmp_path/'keys.db'}"
    return auth.ApiKeyStore(url=url, cache_ttl_seconds=0.0)


def test_hash_is_sha256_hex_and_stable():
    h1 = auth.hash_key("shk_abc")
    h2 = auth.hash_key("shk_abc")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex
    assert h1 != auth.hash_key("shk_def")


def test_generate_key_shape():
    raw, key_id = auth.generate_key()
    assert raw.startswith(auth.KEY_PREFIX)
    assert len(raw) > 20
    assert len(key_id) == 12  # token_hex(6)
    # Two generations never collide.
    assert auth.generate_key()[0] != raw


def test_create_then_verify_roundtrip(store):
    raw, key_id = store.create_key("anton", note="prod")
    assert store.verify(raw) == "anton"
    # The raw key is never stored — only its hash.
    listed = store.list_keys()
    assert listed[0]["key_id"] == key_id
    assert "key_hash" not in listed[0]
    assert raw not in str(listed)


def test_verify_unknown_or_empty_key_returns_none(store):
    store.create_key("anton")
    assert store.verify("shk_not_a_real_key") is None
    assert store.verify("") is None
    assert store.verify(None) is None


def test_revoke_by_key_id_invalidates(store):
    raw, key_id = store.create_key("anton")
    assert store.verify(raw) == "anton"
    n = store.revoke(key_id=key_id)
    assert n == 1
    assert store.verify(raw) is None


def test_revoke_by_client_name(store):
    raw1, _ = store.create_key("anton")
    raw2, _ = store.create_key("anton")
    other, _ = store.create_key("dashboards")
    assert store.revoke(client_name="anton") == 2
    assert store.verify(raw1) is None and store.verify(raw2) is None
    assert store.verify(other) == "dashboards"  # unaffected


def test_revoke_requires_a_selector(store):
    with pytest.raises(ValueError):
        store.revoke()


def test_key_with_future_expiry_is_valid(store):
    raw, _ = store.create_key("anton", expires_in_days=30)
    assert store.verify(raw) == "anton"


def test_expired_key_is_rejected(store):
    from datetime import datetime, timedelta, timezone

    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    raw, _ = store.create_key("anton", expires_at=past)
    assert store.verify(raw) is None


def test_no_expiry_never_expires(store):
    raw, _ = store.create_key("anton")  # expires_at is None
    assert store.verify(raw) == "anton"
    listed = store.list_keys()[0]
    assert listed["expires_at"] is None


def test_expiry_shows_in_list(store):
    store.create_key("anton", expires_in_days=7)
    assert store.list_keys()[0]["expires_at"] is not None


def test_cache_refreshes_after_ttl(tmp_path):
    """A revoke in another store is picked up once the TTL elapses."""
    url = f"sqlite:///{tmp_path/'keys.db'}"
    reader = auth.ApiKeyStore(url=url, cache_ttl_seconds=0.0)  # always refetch
    writer = auth.ApiKeyStore(url=url, cache_ttl_seconds=0.0)
    raw, key_id = writer.create_key("anton")
    assert reader.verify(raw) == "anton"
    writer.revoke(key_id=key_id)
    # TTL 0 -> reader reloads on the next verify and sees the revocation.
    assert reader.verify(raw) is None


def test_extract_key_parsing():
    assert auth._extract_key("Bearer shk_x", None) == "shk_x"
    assert auth._extract_key("bearer shk_x", None) == "shk_x"  # case-insensitive
    assert auth._extract_key("shk_bare", None) == "shk_bare"  # lenient bare token
    assert auth._extract_key(None, "shk_hdr") == "shk_hdr"  # X-API-Key
    assert auth._extract_key(None, None) is None


def test_auth_enabled_env(monkeypatch):
    monkeypatch.delenv(auth.AUTH_ENABLED_ENV, raising=False)
    assert auth.auth_enabled() is False
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv(auth.AUTH_ENABLED_ENV, truthy)
        assert auth.auth_enabled() is True
    monkeypatch.setenv(auth.AUTH_ENABLED_ENV, "0")
    assert auth.auth_enabled() is False


# --- Endpoint-level: the dependency enforces keys only when enabled ----------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smarthub.server import predict  # noqa: E402
from smarthub.train_and_predict import config  # noqa: E402


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    """Real app with auth ENABLED against a temp key DB; yields (client, store).

    Points every DB the app touches at temp SQLite and stubs the config-store
    lookup, so the test doesn't require the shared Postgres to be reachable.
    """
    monkeypatch.setenv(auth.AUTH_ENABLED_ENV, "1")
    monkeypatch.setenv("SMARTHUB_AUTH_DB_URL", f"sqlite:///{tmp_path/'keys.db'}")
    monkeypatch.setenv(
        "SMARTHUB_PREDICTION_LOG_DB_URL", f"sqlite:///{tmp_path/'log.db'}"
    )
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    predict._prediction_log_store_holder.clear()
    auth.reset_store()
    with TestClient(predict.app, raise_server_exceptions=False) as c:
        yield c, auth.get_store()
    auth.reset_store()


def test_recommend_bid_401_without_key(auth_client):
    client, _ = auth_client
    resp = client.post("/recommend_bid", json={"expected_revenue": 1})
    assert resp.status_code == 401


def test_recommend_bid_401_with_bad_key(auth_client):
    client, _ = auth_client
    resp = client.post(
        "/recommend_bid",
        json={"expected_revenue": 1},
        headers={"Authorization": "Bearer shk_wrong"},
    )
    assert resp.status_code == 401


def test_valid_key_passes_auth(auth_client):
    """A valid key clears the auth gate (request proceeds past 401)."""
    client, store = auth_client
    raw, _ = store.create_key("anton")
    resp = client.post(
        "/recommend_bid",
        json={"expected_revenue": 1},  # intentionally incomplete
        headers={"Authorization": f"Bearer {raw}"},
    )
    # Not 401: auth passed. It's a 422 instead (missing required bid fields),
    # which proves we got past the auth dependency into request validation.
    assert resp.status_code != 401
    assert resp.status_code == 422


def test_health_never_requires_a_key(auth_client, monkeypatch):
    client, _ = auth_client
    # Isolate from model/registry resolution (which needs prod storage config
    # not present here) so the test asserts only the auth claim: /health has no
    # key requirement.
    monkeypatch.setattr(predict, "resolve_model_uri", lambda *a, **k: None)
    monkeypatch.setattr(predict, "is_model_cached", lambda *a, **k: False)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_disabled_auth_needs_no_key(tmp_path, monkeypatch):
    """With auth off (default), the gate is a no-op — no 401."""
    monkeypatch.delenv(auth.AUTH_ENABLED_ENV, raising=False)
    auth.reset_store()
    with TestClient(predict.app, raise_server_exceptions=False) as c:
        resp = c.post("/recommend_bid", json={"expected_revenue": 1})
    assert resp.status_code != 401
