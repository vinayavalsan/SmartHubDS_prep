"""Pluggable model-artifact storage for the SmartHub registry.

Two backends behind one small interface:

- :class:`FilesystemModelStore` — local development/training storage
  (e.g. ``data/models``).
- :class:`S3ModelStore` — production storage on S3 or any S3-compatible
  endpoint (MinIO, Ceph, etc.) via ``boto3``, with a configurable
  ``endpoint_url`` so it isn't tied to AWS.

Keys are POSIX-style relative paths (e.g. ``"auto/run_20260803T....pkl"``).
Each backend maps a key onto its own layout — a directory under ``root`` for
the filesystem, or ``<prefix>/<key>`` in a bucket for S3. Callers deal only in
keys, so the registry can route the *same* key to local vs production storage
without knowing which backend is behind it.

``boto3`` is imported lazily inside :class:`S3ModelStore` so environments that
never touch production storage (local dev, CI) don't need it installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ModelStore(ABC):
    """A minimal key/blob store for model artifacts, manifests, and pointers."""

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Return the raw bytes stored at ``key`` (raises if missing)."""

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key``, creating any parent structure."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """Return all keys under ``prefix`` (keys are relative to the store)."""

    @abstractmethod
    def claim(self, key: str) -> bool:
        """Atomically create an empty marker at ``key`` iff it does not exist.

        Returns ``True`` when this call created it, ``False`` when it already
        existed. Must be atomic against concurrent callers — used to reserve a
        production version number so two simultaneous promotions can never be
        assigned the same ``_vN``.
        """

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """Return a local filesystem path for ``key`` suitable for
        ``joblib.load`` — the file itself for a filesystem store, or a
        downloaded cache copy for a remote store. Raises if ``key`` is
        missing."""

    # -- text + copy helpers built on the byte primitives ---------------------

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def copy_to(
        self, dest: "ModelStore", key: str, dest_key: str | None = None
    ) -> None:
        """Copy ``key`` from this store into ``dest`` (used to publish a
        promoted model from local to production storage)."""
        dest.write_bytes(dest_key or key, self.read_bytes(key))


class FilesystemModelStore(ModelStore):
    """Model store backed by a local directory tree rooted at ``root``."""

    backend = "filesystem"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        return sorted(
            str(p.relative_to(self.root).as_posix())
            for p in base.rglob("*")
            if p.is_file()
        )

    def local_path(self, key: str) -> Path:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"No object at key {key!r} under {self.root}.")
        return path

    def claim(self, key: str) -> bool:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" = exclusive create; atomic on POSIX — only one caller wins.
            with open(path, "x"):
                pass
            return True
        except FileExistsError:
            return False


class S3ModelStore(ModelStore):
    """Model store backed by S3 or any S3-compatible endpoint (via boto3).

    ``endpoint_url`` lets this target MinIO or another S3-compatible service
    rather than AWS. Credentials/region are resolved by boto3 the usual way
    (env vars, shared config, instance role). Objects fetched via
    :meth:`local_path` are cached under ``cache_dir`` so repeated serving loads
    don't re-download.
    """

    backend = "s3"

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        import boto3  # lazy: only needed when production storage is used

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)
        self._cache_dir = Path(cache_dir) if cache_dir else None

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def read_bytes(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self.bucket, Key=self._full_key(key))
        return obj["Body"].read()

    def write_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            # Only a genuine "not found" means absent. Any other error (auth,
            # throttling, 5xx, network) is a real failure and must surface --
            # never masked as "doesn't exist", which would let serving silently
            # fall back to a stale local copy.
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                return False
            raise

    def list(self, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        full_prefix = self._full_key(prefix) if prefix else self.prefix
        strip = f"{self.prefix}/" if self.prefix else ""
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for item in page.get("Contents", []):
                name = item["Key"]
                if strip and name.startswith(strip):
                    name = name[len(strip) :]
                keys.append(name)
        return sorted(keys)

    def claim(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            # IfNoneMatch="*" -> the put succeeds only if the object does not
            # already exist; S3 returns 412 PreconditionFailed otherwise. This
            # is an atomic compare-and-create, safe under concurrent promotions.
            self._client.put_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
                Body=b"",
                IfNoneMatch="*",
            )
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("PreconditionFailed", "412") or status == 412:
                return False
            raise

    def local_path(self, key: str) -> Path:
        import tempfile

        cache_root = self._cache_dir or Path(tempfile.gettempdir()) / "smarthub_models"
        dest = cache_root / key
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.read_bytes(key))
        return dest
