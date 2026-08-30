"""Pinned BGE small English v1.5 ONNX encoder."""

from bge_small_onnx.artifact import DEFAULT_REVISION, REPO_ID, Artifact
from bge_small_onnx.encoder import Encoder, query_text

__all__ = ["DEFAULT_REVISION", "REPO_ID", "Artifact", "Encoder", "query_text"]
