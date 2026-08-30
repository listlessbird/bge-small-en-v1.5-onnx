from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_MODEL = "BAAI/bge-small-en-v1.5"
SOURCE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIMENSION = 384
MAX_LENGTH = 256
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OPSET = 18
MINIMUM_COSINE = 0.99

_FIXTURES = (
    f"{QUERY_PREFIX}when the deploy breaks on friday",
    f"{QUERY_PREFIX}quokka",
    f'{QUERY_PREFIX}the exact phrase "this is fine"',
    "Titles: Distracted boyfriend\nTags: reaction\nDescriptions: A person looks away",
    "OCR: nobody expects the production outage\nCaptions: A dog sits in a burning room",
    "Titles: Success Kid\nOrigins: Flickr\nYears: 2007",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parity(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, ...]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and ONNX outputs have incompatible shapes")
    reference_norms = np.linalg.norm(reference, axis=1)
    candidate_norms = np.linalg.norm(candidate, axis=1)
    cosines = np.sum(reference * candidate, axis=1) / (reference_norms * candidate_norms)
    if not np.all(np.isfinite(cosines)) or np.any(cosines < MINIMUM_COSINE):
        raise ValueError(f"INT8 parity cosine {float(np.min(cosines)):.6f} is too low")
    return tuple(float(value) for value in cosines)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _encoded(tokenizer: Any, texts: tuple[str, ...]) -> dict[str, Any]:
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def main() -> None:
    import onnx
    import onnxruntime as ort
    import torch
    import transformers
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer

    output = _arguments().output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_MODEL,
        revision=SOURCE_REVISION,
    )
    model = AutoModel.from_pretrained(
        SOURCE_MODEL,
        revision=SOURCE_REVISION,
    ).eval()

    class Encoder(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.source = source

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any,
        ) -> Any:
            hidden = self.source(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state
            return torch.nn.functional.normalize(hidden[:, 0], p=2, dim=1)

    wrapper = Encoder(model).eval()
    encoded = _encoded(tokenizer, _FIXTURES)
    full_path = output / "model.onnx"
    int8_path = output / "model-int8.onnx"
    torch.onnx.export(
        wrapper,
        (encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]),
        full_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "tokens"},
            "attention_mask": {0: "batch", 1: "tokens"},
            "token_type_ids": {0: "batch", 1: "tokens"},
            "embeddings": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    quantize_dynamic(
        full_path,
        int8_path,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    tokenizer_path = output / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_path))
    reference = (
        wrapper(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"])
        .detach()
        .cpu()
        .numpy()
    )
    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    candidate = session.run(
        ["embeddings"],
        {name: value.detach().cpu().numpy().astype(np.int64) for name, value in encoded.items()},
    )[0]
    cosines = _validate_parity(reference, candidate)
    metadata = {
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "opset": OPSET,
        "tokenizer_sha256": _sha256(tokenizer_path),
        "max_length": MAX_LENGTH,
        "query_prefix": QUERY_PREFIX,
        "pooling": "cls",
        "normalization": "l2",
        "dimension": DIMENSION,
        "quantization": "int8-dynamic",
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "output_name": "embeddings",
        "minimum_reference_cosine": MINIMUM_COSINE,
        "exporter_sha256": _sha256(Path(__file__).resolve()),
        "versions": {
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    metadata_path = output / "export_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    full_path.unlink()
    shutil.copyfile(
        Path(__file__).resolve(),
        output / "export_bge_onnx.py",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "model_sha256": _sha256(int8_path),
                "tokenizer_sha256": metadata["tokenizer_sha256"],
                "export_meta_sha256": _sha256(metadata_path),
                "reference_cosines": cosines,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
