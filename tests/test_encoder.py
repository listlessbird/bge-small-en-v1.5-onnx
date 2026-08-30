import os

import numpy as np
import pytest

from bge_small_onnx import Encoder, query_text
from bge_small_onnx.artifact import DIMENSION, QUERY_PREFIX


def test_query_instruction_is_applied_exactly_once() -> None:
    assert query_text("quokka") == f"{QUERY_PREFIX}quokka"


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE") != "1",
    reason="set RUN_MODEL_SMOKE=1 to download and exercise the model",
)
def test_published_model_encodes_queries_and_documents() -> None:
    encoder = Encoder.from_huggingface(threads=1)

    queries = encoder.encode_queries(["quokka", "production outage"])
    documents = encoder.encode_documents(["quokka", "production outage"])

    assert queries.shape == documents.shape == (2, DIMENSION)
    np.testing.assert_allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(documents, axis=1), 1, atol=1e-3)
    assert not np.allclose(queries, documents)
