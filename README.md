---
license: mit
base_model: BAAI/bge-small-en-v1.5
library_name: onnxruntime
pipeline_tag: feature-extraction
tags:
  - onnx
  - embeddings
  - sentence-transformers
---

# BGE small English v1.5 ONNX

A pinned INT8 ONNX export of
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
with a small Python runtime. It produces normalized 384-dimensional embeddings
without PyTorch or Transformers.

The model files live on
[Hugging Face](https://huggingface.co/listlessbird/bge-small-en-v1.5-onnx).
[GitHub](https://github.com/listlessbird/bge-small-en-v1.5-onnx) contains the
runtime, export script, tests, and release history.

This is a project-controlled export, not an upstream BAAI release.

## Use it

Add the package directly from the tagged GitHub source:

```bash
uv add "bge-small-onnx @ git+https://github.com/listlessbird/bge-small-en-v1.5-onnx.git@v1.0.1"
```

Queries and documents have separate methods because BGE applies an instruction
to queries only:

```python
from bge_small_onnx import Encoder

encoder = Encoder.from_huggingface()

queries = encoder.encode_queries(["why did the deploy fail?"])
documents = encoder.encode_documents(
    [
        "The Friday deployment caused an outage.",
        "A quokka is a small marsupial from Western Australia.",
    ]
)
scores = queries @ documents.T
print(scores[0].tolist())
```

`Encoder.from_huggingface()` downloads the exact published revision and checks
the model, tokenizer, and metadata SHA-256 values before ONNX Runtime loads the
model. Pass `threads=N` to set ONNX Runtime's intra-op thread count.

The CLI supports small encoding jobs and artifact verification:

```bash
uvx --from "git+https://github.com/listlessbird/bge-small-en-v1.5-onnx.git@v1.0.1" \
  bge-small-onnx encode "why did the deploy fail?"

uvx --from "git+https://github.com/listlessbird/bge-small-en-v1.5-onnx.git@v1.0.1" \
  bge-small-onnx verify
```

## Frozen model contract

- Source model: `BAAI/bge-small-en-v1.5`
- Source revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- Published artifact revision: `d46fcc3e67304e574e08e911ce7e50d71bb728cf`
- Maximum input length: 256 tokens
- Query instruction: `Represent this sentence for searching relevant passages: `
- Document instruction: none
- Pooling: CLS
- Normalization: L2
- Output: 384 float32 values
- Quantization: per-channel dynamic INT8 for MatMul weights
- ONNX opset: 18

Changing any item in this contract creates incompatible embeddings. Existing
document indexes must use the same contract as their query encoder.

The published model SHA-256 is
`6fb40fbcdf3dcc7a3fed12d56ff2d1324f69d0b7fd6c5afe05f4530a6142fdf8`.
Six query and document fixtures measured cosine similarity from 0.993772 to
0.997500 against the pinned PyTorch model.

On a Raspberry Pi ARM64 container limited to two CPUs and 768 MiB, 100 query
encodes measured 26.97 ms p50 and 28.89 ms p95. Loading the encoder increased
process RSS by 127,766,528 bytes.

## Reproduce the export

Clone the repository and install the export dependencies:

```bash
git clone https://github.com/listlessbird/bge-small-en-v1.5-onnx.git
cd bge-small-en-v1.5-onnx
uv sync --all-groups --extra export
uv run python export_bge_onnx.py ./artifacts/new-export
```

The output directory must not exist. The script downloads the pinned source
revision, exports an fp32 graph, quantizes it, and checks six fixed fixtures.
It deletes the intermediate fp32 graph only after every fixture reaches 0.99
cosine similarity.

The output contains:

- `model-int8.onnx`, the quantized encoder
- `tokenizer.json`, the matching tokenizer
- `export_meta.json`, the machine-readable contract
- `export_bge_onnx.py`, the exact exporter used for the output

## Develop

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Normal tests do not download model weights. Run the published model smoke test
explicitly:

```bash
RUN_MODEL_SMOKE=1 uv run pytest -m model
```

The code in this repository is MIT licensed. See `NOTICE` for the source model
attribution.
