from pathlib import Path

import pytest

from bge_small_onnx import DEFAULT_REVISION, REPO_ID, Artifact


def test_published_identity_is_immutable() -> None:
    assert REPO_ID == "listlessbird/bge-small-en-v1.5-onnx"
    assert len(DEFAULT_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in DEFAULT_REVISION)


def test_load_rejects_an_incomplete_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"model-int8\.onnx"):
        Artifact.load(tmp_path)
