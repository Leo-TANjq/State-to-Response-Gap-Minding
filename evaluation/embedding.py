"""Compute response/reference cosine similarity with a configurable embedding model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, help="SentenceTransformers model path or ID")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    args = arguments()
    if args.limit < 0 or args.offset < 0 or args.batch_size < 1:
        raise SystemExit("limit and offset must be non-negative; batch-size must be positive")

    from sentence_transformers import SentenceTransformer

    rows = read_jsonl(args.input)
    rows = rows[args.offset : args.offset + args.limit if args.limit else None]
    valid = [row for row in rows if row.get("response") and row.get("ground_truth")]
    if not valid:
        raise SystemExit("No rows contain both a non-empty response and ground_truth")
    model = SentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    generated = model.encode(
        [row["response"] for row in valid],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    references = model.encode(
        [row["ground_truth"] for row in valid],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarities = np.sum(generated * references, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for position, (row, similarity) in enumerate(zip(valid, similarities, strict=True)):
            stream.write(json.dumps({
                "id": str(row.get("id", f"row:{position + args.offset}")),
                "cosine_similarity": float(similarity),
                "cosine_similarity_x100": float(similarity * 100),
                "embedding_model": args.model,
            }, ensure_ascii=False) + "\n")
    print(f"Scored {len(valid)} responses; mean cosine similarity = {similarities.mean():.6f}")


if __name__ == "__main__":
    main()
