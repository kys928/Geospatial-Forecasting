from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKLEARN_ARTIFACT_VERSION = "scikit-learn==1.7.2"


def test_sklearn_runtime_dependency_matches_persisted_artifact_version() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert SKLEARN_ARTIFACT_VERSION in requirements
    assert SKLEARN_ARTIFACT_VERSION in pyproject["project"]["dependencies"]
