from pathlib import Path

import pytest


def test_load_rejects_an_incomplete_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"model-int8\.onnx"):
        Artifact.load(tmp_path)
