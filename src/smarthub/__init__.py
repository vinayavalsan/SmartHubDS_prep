"""SmartHub / Anton data-science toolkit.

Shared library powering the Redshift data pull and the Streamlit dashboards.
See CONTEXT.md for the business domain this package models.

Versioning
----------
``__version__`` is the canonical application/package version, maintained with
semantic versioning (``major.minor.patch``). The single source of truth is the
``version`` field in ``pyproject.toml``: at runtime it is read from the installed
package metadata (``importlib.metadata``). The literal below is only a fallback
for a source checkout that was never installed, and must be kept in sync with
``pyproject.toml``.

Bump policy (semver): backward-incompatible changes to the public/serving
contract -- including the raw ``field_registry`` and the ``feature_registry``
(feature set, ordering, or semantics that change how a stored model scores) --
warrant a MAJOR bump; backward-compatible additions a MINOR bump; fixes a PATCH.
Because every prediction row records this version (see
``train_and_predict.prediction_log_schema``), a bump makes each prediction
traceable to the exact code contract that produced it.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Canonical: the installed package metadata (pyproject.toml [project].version).
    __version__ = _pkg_version("smarthub")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    # Fallback for an un-installed checkout. Keep in sync with pyproject.toml.
    __version__ = "0.1.0"

__all__ = ["__version__"]
