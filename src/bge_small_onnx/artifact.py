from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

REPO_ID: Final = "listlessbird/bge-small-en-v1.5-onnx"
DEFAULT_REVISION: Final = "d46fcc3e67304e574e08e911ce7e50d71bb728cf"
MODEL_FILENAME: Final = "model-int8.onnx"
MODEL_SHA256: Final = "6fb40fbcdf3dcc7a3fed12d56ff2d1324f69d0b7fd6c5afe05f4530a6142fdf8"
TOKENIZER_SHA256: Final = "0d3aef594edd5f9b53e7f814277a9171dc70ff93eb66bda6e01f7aa53997d963"
METADATA_SHA256: Final = "619bafc6963e85a17c13e58b30cd8ac11ad0b5d737d70395bda9e06655883858"
SOURCE_MODEL: Final = "BAAI/bge-small-en-v1.5"
SOURCE_REVISION: Final = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MAX_LENGTH: Final = 256
DIMENSION: Final = 384
QUERY_PREFIX: Final = "Represent this sentence for searching relevant passages: "

_FILES: Final = {
    MODEL_FILENAME: MODEL_SHA256,
    "tokenizer.json": TOKENIZER_SHA256,
    "export_meta.json": METADATA_SHA256,
}
_EXPECTED_METADATA: Final[dict[str, object]] = {
    "source_model": SOURCE_MODEL,
    "source_revision": SOURCE_REVISION,
    "opset": 18,
    "tokenizer_sha256": TOKENIZER_SHA256,
    "max_length": MAX_LENGTH,
    "query_prefix": QUERY_PREFIX,
    "pooling": "cls",
    "normalization": "l2",
    "dimension": DIMENSION,
    "quantization": "int8-dynamic",
    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
    "output_name": "embeddings",
}


@dataclass(frozen=True, slots=True)
class Artifact:
    """A local copy of the pinned files after checksum and contract validation."""

    directory: Path

    @classmethod
    def download(
        cls,
        *,
        revision: str = DEFAULT_REVISION,
        cache_dir: Path | None = None,
    ) -> Artifact:
        directory = Path(
            snapshot_download(
                repo_id=REPO_ID,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=list(_FILES),
            )
        )
        return cls.load(directory)

    @classmethod
    def load(cls, directory: Path) -> Artifact:
        directory = directory.resolve()
        for filename, expected in _FILES.items():
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing BGE artifact file: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"BGE artifact checksum mismatch for {filename}: "
                    f"expected {expected}, got {actual}"
                )

        raw = json.loads((directory / "export_meta.json").read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("BGE export metadata must be a JSON object")
        metadata = cast(dict[str, object], raw)
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in _EXPECTED_METADATA.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"BGE export metadata is incompatible: {mismatches}")
        return cls(directory)

    @property
    def model_path(self) -> Path:
        return self.directory / MODEL_FILENAME

    @property
    def tokenizer_path(self) -> Path:
        return self.directory / "tokenizer.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
