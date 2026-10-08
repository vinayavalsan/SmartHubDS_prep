"""HPO metadata and interactive plots have separate MLflow destinations."""

from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
mlflow_utils = import_module("smarthub.train_and_predict.mlflow_utils")


def test_hpo_artifact_layout_keeps_html_interactive_and_metadata_separate(
    tmp_path, monkeypatch
):
    metadata = [
        "best_parameters.yaml",
        "summary.json",
        "finalist_results.json",
        "hyperparameter_search.yaml",
    ]
    for name in metadata:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    plots = [
        "optimization_history.html",
        "parameter_importance.html",
        "contour_matrix.html",
    ]
    html = "<html><script>window.interactive = true;</script></html>"
    for name in plots:
        (plots_dir / name).write_text(html, encoding="utf-8")
    # A stray metadata file must not be exposed as a plot.
    (plots_dir / "summary.json").write_text("{}", encoding="utf-8")
    uploads = []

    def capture(file, artifact_path):
        uploads.append((Path(file).name, artifact_path))

    monkeypatch.setattr(mlflow_utils.mlflow, "log_artifact", capture)
    mlflow_utils._log_hpo_artifacts(tmp_path)
    assert set(uploads) == {
        *(
            (name, "results")
            for name in metadata
            if name != "hyperparameter_search.yaml"
        ),
        ("hyperparameter_search.yaml", "config"),
        *((name, "plots") for name in plots),
    }
    for name in plots:
        assert (plots_dir / name).read_text(encoding="utf-8") == html


@pytest.mark.parametrize("empty_plots_folder", [False, True])
def test_missing_hpo_plots_warns_without_discarding_metadata(
    tmp_path, monkeypatch, caplog, empty_plots_folder
):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    if empty_plots_folder:
        (tmp_path / "plots").mkdir()
    uploads = []
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_artifact",
        lambda file, artifact_path: uploads.append((Path(file).name, artifact_path)),
    )
    mlflow_utils._log_hpo_artifacts(tmp_path)
    assert uploads == [("summary.json", "results")]
    assert "No HPO HTML plots found" in caplog.text
