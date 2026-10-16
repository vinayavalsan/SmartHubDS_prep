"""Full-run tracking using explicit run IDs, including threaded Prefect tasks.

The driver owns all MLflow writes. Worker processes never receive a client or
active fluent run, so concurrent Optuna trials cannot mix their run context.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import threading
import time
import traceback
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

import yaml

from smarthub.core import paths

logger = logging.getLogger(__name__)


def tracking_settings(lead_type_id, kind, config_path=None):
    """Read tracking policy before validating the model/data configuration."""
    from . import config

    file = config_path or (
        config._training_config_path()
        if kind == "training"
        else config._hyperparameter_search_config_path()
    )
    with paths.resolve(file).open(encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}
    namespace = "training" if kind == "training" else "hyperparameter_search"
    resolved = config._lead_type_config(root[namespace], lead_type_id, namespace)
    policy = dict(resolved.get("mlflow") or {})
    policy.setdefault("enabled", kind == "training")
    policy.setdefault("required", kind == "hpo")
    return policy


def environment_snapshot():
    from . import model_parameters

    packages = {}
    for name in (
        "smarthub",
        "mlflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "optuna",
        "pyarrow",
        "joblib",
        "prefect",
        "psutil",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_version": model_parameters.code_version(),
        "packages": packages,
    }


class RunTracker:
    """One run lifecycle; tracking failures never mask execution exceptions."""

    def __init__(self, settings, name, kind, lead_type_id, enabled=True):
        self.settings = settings
        self.name = name
        self.kind = kind
        self.lead_type_id = lead_type_id
        self.enabled = enabled and settings.get("enabled", True)
        self.required = settings.get("required", False)
        self.run_id = None
        self.client = None
        self.experiment_id = None
        self.tracking_uri = None
        self.started = time.perf_counter()
        self.stage_name = "initializing"
        self._stage_stack = []
        self.children = {}
        self.child_states = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._degraded = False

    def _write(self, function, *args, **kwargs):
        if not self.run_id:
            return None
        try:
            with self._lock:
                return function(*args, **kwargs)
        except Exception:
            if self.required:
                raise
            if not self._degraded:
                logger.warning(
                    "Optional MLflow tracking failed; run continues.", exc_info=True
                )
            self._degraded = True
            return None

    def __enter__(self):
        logger.info("Starting %s run: %s", self.kind, self.name)
        if not self.enabled:
            return self
        try:
            from mlflow import MlflowClient

            from smarthub.core.lead_types import lead_type_name

            from .mlflow_utils import _configure_tracking

            experiment = self.settings["experiment_name"]
            lead = lead_type_name(self.lead_type_id)
            self.tracking_uri, self.experiment_id = _configure_tracking(
                self.settings["tracking_db_path"],
                self.settings["artifact_root"],
                f"{experiment}_{lead}",
            )
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            run = self.client.create_run(
                self.experiment_id,
                tags={
                    "mlflow.runName": self.name,
                    "run_type": self.kind,
                    "lead_type_id": str(self.lead_type_id),
                    "tracking.stage": "initializing",
                    "resource_scope": "driver + local descendants (sampled RSS sum)",
                },
            )
            self.run_id = run.info.run_id
            self.params(
                {f"{self.kind}_run_id": self.name, "lead_type_id": self.lead_type_id}
            )
            environment = environment_snapshot()
            self.artifact(environment, "config/environment.json")
            self.params(
                {
                    "code_version": environment.get("code_version"),
                    "python_version": environment.get("python"),
                }
            )
            self._start_resources()
        except Exception as exc:
            # An error after create_run must not leave a RUNNING orphan.
            self._finish("FAILED", exc)
            if self.required:
                raise
            logger.warning(
                "Optional MLflow startup failed; run continues.", exc_info=True
            )
            self.run_id = None
        return self

    def params(self, values, run_id=None):
        if not self.run_id:
            return
        target = run_id or self.run_id
        from mlflow.entities import Param

        parameters = []
        for key, value in values.items():
            if value is not None:
                text = (
                    json.dumps(value, sort_keys=True, default=str)
                    if isinstance(value, (list, dict))
                    else str(value)
                )
                parameters.append(Param(key, text))
        if parameters:
            self._write(
                self.client.log_batch, target, params=parameters, synchronous=True
            )

    def tags(self, values, run_id=None):
        if not self.run_id:
            return
        for key, value in values.items():
            if value is not None:
                self._write(self.client.set_tag, run_id or self.run_id, key, str(value))

    def metrics(self, values, step=0, run_id=None):
        if not self.run_id:
            return
        from mlflow.entities import Metric

        samples = []
        for key, value in values.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                samples.append(Metric(key, float(value), int(time.time() * 1000), step))
        if samples:
            self._write(
                self.client.log_batch,
                run_id or self.run_id,
                metrics=samples,
                synchronous=True,
            )

    def artifact(self, payload, file, run_id=None):
        if self.run_id:
            self._write(self.client.log_dict, run_id or self.run_id, payload, file)

    def dataset(self, frame, name, context, source, target=None):
        """Log metadata/digest only; do not upload lead records as datasets."""
        if not self.run_id:
            return
        try:
            import mlflow.data
            from mlflow.entities import DatasetInput, InputTag

            dataset = mlflow.data.from_pandas(
                frame,
                source=source,
                name=name,
                targets=target if target in frame.columns else None,
            )
            record = DatasetInput(
                dataset._to_mlflow_entity(), [InputTag("mlflow.data.context", context)]
            )
            self._write(self.client.log_inputs, self.run_id, [record])
            metadata = {
                "name": name,
                "context": context,
                "source": source,
                "digest": dataset.digest,
                "rows": len(frame),
                "columns": list(frame.columns),
            }
            if "created_at" in frame.columns:
                metadata.update(
                    min_created_at=str(frame["created_at"].min()),
                    max_created_at=str(frame["created_at"].max()),
                )
            self.artifact(metadata, f"data/{name}.json")
        except Exception:
            if self.required:
                raise
            self._degraded = True
            logger.warning(
                "Dataset tracking unavailable; run continues.", exc_info=True
            )

    @contextmanager
    def stage(self, name):
        self.stage_name = name
        self._stage_stack.append(name)
        started = time.perf_counter()
        failed = False
        self.tags({"tracking.stage": name, f"stage.{name}.status": "RUNNING"})
        try:
            yield
        except BaseException:
            failed = True
            # Preserve the operation error if tracking its failure also fails.
            try:
                self.tags({f"stage.{name}.status": "FAILED"})
            except Exception:
                logger.warning("Could not record failed stage.", exc_info=True)
            raise
        else:
            self.tags({f"stage.{name}.status": "FINISHED"})
        finally:
            try:
                self.metrics({f"timing.{name}_seconds": time.perf_counter() - started})
            except Exception:
                if not failed:
                    raise
                logger.warning("Could not record stage duration.", exc_info=True)
            finally:
                self._stage_stack.pop()
                if self._stage_stack and not failed:
                    self.stage_name = self._stage_stack[-1]
                    self.tags({"tracking.stage": self.stage_name})

    def child(self, key, kind, params, metrics, tags=None, status="FINISHED"):
        """Idempotently record a completed trial/finalist from the driver."""
        if not self.run_id:
            return
        with self._lock:
            child_id = self.children.get(key)
            if child_id is None:
                run = self._write(
                    self.client.create_run,
                    self.experiment_id,
                    tags={
                        "mlflow.runName": f"{self.name}/{key}",
                        "mlflow.parentRunId": self.run_id,
                        "run_type": kind,
                        "hpo_run_id": self.name,
                    },
                )
                if run is None:
                    return
                child_id = run.info.run_id
                self.children[key] = child_id
            try:
                self.child_states[key] = status
                self.params(params, child_id)
                self.metrics(metrics, run_id=child_id)
                self.tags(tags or {}, child_id)
            finally:
                self._write(self.client.set_terminated, child_id, status=status)

    def _start_resources(self):
        if not self.settings.get("system_metrics", {}).get("enabled", True):
            self.tags({"resource_monitor": "disabled"})
            return
        try:
            import psutil

            interval = float(
                self.settings.get("system_metrics", {}).get("interval_seconds", 5)
            )
            if not math.isfinite(interval) or interval < 0.1:
                raise ValueError("System metric interval must be >= 0.1 seconds.")
            self._process = psutil.Process(os.getpid())
            self._io_available = hasattr(self._process, "io_counters")
            self._processes = {self._process.pid: self._process}
            self._peak_rss = 0.0
            self.tags(
                {
                    "resource_monitor": "enabled",
                    "resource_io_available": self._io_available,
                    "resource_limitations": (
                        "sampled peaks; shared pages counted per process; "
                        "remote workers excluded; short-lived workers may be missed"
                    ),
                }
            )
            self._sample_resources()

            def monitor():
                while not self._stop.wait(interval):
                    try:
                        self._sample_resources()
                    except Exception:
                        logger.warning("Resource sampling failed.", exc_info=True)

            self._thread = threading.Thread(
                target=monitor, daemon=True, name="smarthub-resource-metrics"
            )
            self._thread.start()
        except Exception:
            # Missing telemetry must not change model execution.
            logger.warning("Resource monitoring unavailable.", exc_info=True)
            self.tags({"resource_monitor": "unavailable"})

    def _sample_resources(self):
        import psutil

        processes = [self._process, *self._process.children(recursive=True)]
        rss, cpu, read, written = 0, 0.0, 0, 0
        for process in processes:
            try:
                cached = self._processes.setdefault(process.pid, process)
                rss += cached.memory_info().rss
                cpu += cached.cpu_percent()
                if self._io_available:
                    io = cached.io_counters()
                    read += io.read_bytes
                    written += io.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        live = {process.pid for process in processes}
        self._processes = {
            pid: process for pid, process in self._processes.items() if pid in live
        }
        self._peak_rss = max(self._peak_rss, rss / 2**20)
        samples = {
            "system/process_tree_rss_mb": rss / 2**20,
            "system/process_tree_cpu_percent": cpu,
            "system/host_memory_percent": psutil.virtual_memory().percent,
            "system/sampled_peak_process_tree_rss_mb": self._peak_rss,
        }
        if self._io_available:
            samples.update(
                {
                    "system/process_tree_read_mb": read / 2**20,
                    "system/process_tree_write_mb": written / 2**20,
                }
            )
        self.metrics(samples, step=int(time.perf_counter() - self.started))

    def _finish(self, status, error=None):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if not self.run_id:
            return
        for key, child_id in self.children.items():
            if self.child_states.get(key) == "RUNNING":
                try:
                    self.client.set_tag(child_id, "error", "Parent ended before result")
                    self.client.set_terminated(child_id, status="FAILED")
                except Exception:
                    logger.warning(
                        "Could not close unfinished child run.", exc_info=True
                    )
        # Finalization never replaces an execution error. On successful work,
        # required tracking errors still propagate after cleanup.
        finish_error = None
        if hasattr(self, "_process"):
            try:
                self._sample_resources()
            except Exception:
                logger.warning("Final resource sample unavailable.", exc_info=True)
        try:
            self.metrics({"timing.total_seconds": time.perf_counter() - self.started})
            if error is not None:
                self.artifact(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "stage": self.stage_name,
                        "traceback": "".join(
                            traceback.format_exception(
                                type(error), error, error.__traceback__
                            )
                        ),
                    },
                    "results/failure.json",
                )
            self.tags(
                {
                    "tracking.stage": "completed" if error is None else self.stage_name,
                    "tracking.degraded": self._degraded,
                }
            )
        except Exception as exc:
            finish_error = exc
            logger.warning("Could not finalize MLflow details.", exc_info=True)
        finally:
            try:
                final_status = "FAILED" if finish_error is not None else status
                self.client.set_terminated(self.run_id, status=final_status)
            except Exception as exc:
                finish_error = finish_error or exc
                logger.warning("Could not terminate MLflow run.", exc_info=True)
        if self.required and error is None and finish_error is not None:
            raise finish_error

    def __exit__(self, error_type, error, tb):
        self._finish("FINISHED" if error is None else "FAILED", error)
        return False
