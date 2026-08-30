"""Command line interface for verification and small encoding jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn, cast

from bge_small_onnx.artifact import DEFAULT_REVISION, Artifact
from bge_small_onnx.encoder import Encoder


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


def main() -> None:
    args = _parser().parse_args()
    model_dir = _path_value(args, "model_dir")
    revision = _string_value(args, "revision")
    threads = _int_value(args, "threads")
    artifact = (
        Artifact.load(model_dir) if model_dir is not None else Artifact.download(revision=revision)
    )

    if args.command == "verify":
        print(json.dumps({"directory": str(artifact.directory), "revision": revision}))
        return
    if args.command == "encode":
        encoder = Encoder(artifact, threads=threads)
        texts = _string_list(args, "texts")
        values = (
            encoder.encode_documents(texts)
            if _bool_value(args, "documents")
            else encoder.encode_queries(texts)
        )
        print(json.dumps(values.tolist()))
        return
    _unreachable(args.command)


def _path_value(args: argparse.Namespace, name: str) -> Path | None:
    value = cast(object, getattr(args, name, None))
    if value is None or isinstance(value, Path):
        return value
    raise TypeError(f"{name} must be a path")


def _string_value(args: argparse.Namespace, name: str) -> str:
    value = cast(object, getattr(args, name, None))
    if isinstance(value, str):
        return value
    raise TypeError(f"{name} must be a string")


def _int_value(args: argparse.Namespace, name: str) -> int:
    value = cast(object, getattr(args, name, None))
    if isinstance(value, int):
        return value
    raise TypeError(f"{name} must be an integer")


def _bool_value(args: argparse.Namespace, name: str) -> bool:
    value = cast(object, getattr(args, name, None))
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a boolean")


def _string_list(args: argparse.Namespace, name: str) -> list[str]:
    value = cast(object, getattr(args, name, None))
    if isinstance(value, list):
        items = cast(list[object], value)
        if all(isinstance(item, str) for item in items):
            return cast(list[str], items)
    raise TypeError(f"{name} must be a list of strings")


def _unreachable(command: object) -> NoReturn:
    raise RuntimeError(f"unknown command: {command}")
