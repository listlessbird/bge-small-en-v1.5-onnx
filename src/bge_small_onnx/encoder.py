from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt

from bge_small_onnx.artifact import (
    DEFAULT_REVISION,
    DIMENSION,
    MAX_LENGTH,
    QUERY_PREFIX,
    Artifact,
)

FloatMatrix = npt.NDArray[np.float32]
_INPUT_NAMES: Final = ("input_ids", "attention_mask", "token_type_ids")


class _Encoding(Protocol):
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class _Tokenizer(Protocol):
    def enable_padding(self, *, pad_id: int, pad_token: str) -> None: ...

    def enable_truncation(self, *, max_length: int) -> None: ...

    def encode_batch(self, inputs: list[str]) -> list[_Encoding]: ...


class _TokenizerFactory(Protocol):
    def from_file(self, path: str) -> _Tokenizer: ...


class _TokenizerModule(Protocol):
    Tokenizer: _TokenizerFactory


class _SessionOptions(Protocol):
    intra_op_num_threads: int
    inter_op_num_threads: int
    execution_mode: object


class _Session(Protocol):
    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, npt.NDArray[np.int64]],
    ) -> Sequence[object]: ...


class _ExecutionMode(Protocol):
    ORT_SEQUENTIAL: object


class _OrtModule(Protocol):
    ExecutionMode: _ExecutionMode

    def SessionOptions(self) -> _SessionOptions: ...

    def InferenceSession(
        self,
        path: str,
        *,
        sess_options: _SessionOptions,
        providers: list[str],
    ) -> _Session: ...


def _module(name: str) -> ModuleType:
    return import_module(name)


class Encoder:
    """Encode queries and documents with the frozen BGE retrieval contract."""

    def __init__(self, artifact: Artifact, *, threads: int = 2) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")

        tokenizers = cast(_TokenizerModule, cast(object, _module("tokenizers")))
        ort = cast(_OrtModule, cast(object, _module("onnxruntime")))
        tokenizer = tokenizers.Tokenizer.from_file(str(artifact.tokenizer_path))
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        tokenizer.enable_truncation(max_length=MAX_LENGTH)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            str(artifact.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    @classmethod
    def from_directory(cls, directory: Path, *, threads: int = 2) -> Encoder:
        return cls(Artifact.load(directory), threads=threads)

    @classmethod
    def from_huggingface(
        cls,
        *,
        revision: str = DEFAULT_REVISION,
        cache_dir: Path | None = None,
        threads: int = 2,
    ) -> Encoder:
        artifact = Artifact.download(revision=revision, cache_dir=cache_dir)
        return cls(artifact, threads=threads)

    def encode_queries(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode([query_text(text) for text in texts])

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.empty((0, DIMENSION), dtype=np.float32)

        encoded = self._tokenizer.encode_batch(list(texts))
        fields = {
            "input_ids": [item.ids for item in encoded],
            "attention_mask": [item.attention_mask for item in encoded],
            "token_type_ids": [item.type_ids for item in encoded],
        }
        inputs = {name: np.asarray(fields[name], dtype=np.int64) for name in _INPUT_NAMES}
        raw = self._session.run(["embeddings"], inputs)[0]
        result = np.asarray(raw, dtype=np.float32)
        if result.shape != (len(texts), DIMENSION):
            raise ValueError(
                f"BGE encoder returned shape {result.shape}, expected {(len(texts), DIMENSION)}"
            )
        norms = np.linalg.norm(result, axis=1)
        if not np.all(np.isfinite(result)) or not np.allclose(norms, 1, atol=1e-3):
            raise ValueError("BGE encoder returned invalid or unnormalized embeddings")
        return result


def query_text(text: str) -> str:
    """Apply the instruction used for retrieval queries, but never documents."""
    return f"{QUERY_PREFIX}{text}"
