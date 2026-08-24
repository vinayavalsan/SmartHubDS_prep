"""Tests for the model-storage abstraction (filesystem backend + interface).

The S3 backend is exercised in Docker against MinIO; here we cover the
filesystem store and the shared copy/text helpers, which the registry routing
relies on.
"""

from smarthub.train_and_predict.model_storage import FilesystemModelStore


def test_write_read_exists_roundtrip(tmp_path):
    store = FilesystemModelStore(tmp_path)
    assert not store.exists("auto/run_1.pkl")
    store.write_bytes("auto/run_1.pkl", b"\x00\x01model")
    assert store.exists("auto/run_1.pkl")
    assert store.read_bytes("auto/run_1.pkl") == b"\x00\x01model"


def test_text_helpers(tmp_path):
    store = FilesystemModelStore(tmp_path)
    store.write_text("auto/run_1.json", '{"a": 1}')
    assert store.read_text("auto/run_1.json") == '{"a": 1}'


def test_list_returns_relative_keys(tmp_path):
    store = FilesystemModelStore(tmp_path)
    store.write_bytes("auto/run_1.pkl", b"a")
    store.write_bytes("auto/run_1.json", b"b")
    store.write_bytes("home/run_2.pkl", b"c")
    assert store.list("auto") == ["auto/run_1.json", "auto/run_1.pkl"]
    assert "home/run_2.pkl" in store.list()


def test_local_path_points_at_the_file(tmp_path):
    store = FilesystemModelStore(tmp_path)
    store.write_bytes("auto/run_1.pkl", b"payload")
    p = store.local_path("auto/run_1.pkl")
    assert p.read_bytes() == b"payload"


def test_local_path_missing_raises(tmp_path):
    store = FilesystemModelStore(tmp_path)
    try:
        store.local_path("auto/missing.pkl")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_copy_to_publishes_between_stores(tmp_path):
    """copy_to moves a key from one store to another (local -> 'production')."""
    local = FilesystemModelStore(tmp_path / "local")
    prod = FilesystemModelStore(tmp_path / "prod")
    local.write_bytes("auto/run_1.pkl", b"model-bytes")

    local.copy_to(prod, "auto/run_1.pkl")

    assert prod.exists("auto/run_1.pkl")
    assert prod.read_bytes("auto/run_1.pkl") == b"model-bytes"
