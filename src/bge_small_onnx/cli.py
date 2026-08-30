from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from bge_small_onnx.artifact import DEFAULT_REVISION, Artifact
from bge_small_onnx.encoder import Encoder


class CliArgs(argparse.Namespace):
    revision: str
    model_dir: Path | None
    threads: int
    command: Literal["verify", "encode"]
    texts: list[str]
    documents: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bge-small-onnx")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="verify artifact checksums and metadata")
    encode = subparsers.add_parser("encode", help="encode one or more texts")
    encode.add_argument("texts", nargs="+")
    encode.add_argument(
        "--documents",
        action="store_true",
        help="encode documents without the BGE query instruction",
    )
    return parser


def _parse_args() -> CliArgs:
    return _parser().parse_args(namespace=CliArgs())


def main() -> None:
    args = _parse_args()
    artifact = (
        Artifact.load(args.model_dir)
        if args.model_dir is not None
        else Artifact.download(revision=args.revision)
    )

    if args.command == "verify":
        print(json.dumps({"directory": str(artifact.directory), "revision": args.revision}))
        return

    encoder = Encoder(artifact, threads=args.threads)
    values = (
        encoder.encode_documents(args.texts)
        if args.documents
        else encoder.encode_queries(args.texts)
    )
    print(json.dumps(values.tolist()))
