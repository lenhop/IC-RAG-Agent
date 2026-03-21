"""
End-to-end rewrite evaluation pipeline.

Pipeline steps:
1) Bulk rewrite from input CSV via gateway rewrite endpoint
2) Compute vector similarity/distance between query and rewritten query
3) Generate a final CSV report with rewrite + metric fields
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import requests


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for rewrite evaluation."""
    parser = argparse.ArgumentParser(
        description="Run bulk rewrite and distance evaluation from CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path containing at least a `query` column.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Default: <input_stem>_eval.csv",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/api/v1/rewrite",
        help="Gateway rewrite endpoint URL.",
    )
    parser.add_argument(
        "--rewrite-backend",
        default="ollama",
        choices=("ollama", "deepseek"),
        help="Rewrite backend sent to gateway endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout (seconds) per rewrite call.",
    )
    parser.add_argument(
        "--embed-model",
        default="models/all-MiniLM-L6-v2",
        help="Embedding model name/path for sentence-transformers.",
    )
    parser.add_argument(
        "--too-similar-threshold",
        type=float,
        default=0.98,
        help="Similarity >= this value is flagged as too_similar.",
    )
    parser.add_argument(
        "--too-different-threshold",
        type=float,
        default=0.75,
        help="Similarity <= this value is flagged as too_different.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str) -> Path:
    """Resolve output CSV path from argument or default naming."""
    if output_arg.strip():
        return Path(output_arg).expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_eval{input_path.suffix}")


def is_blank_query(value: str) -> bool:
    """Return True when query is empty or whitespace-only."""
    return not (value or "").strip()


def call_rewrite(
    endpoint: str,
    query: str,
    rewrite_backend: str,
    timeout: int,
) -> Dict[str, str]:
    """
    Call the rewrite endpoint and normalize result fields.

    Returns:
        Dict with rewrite columns used in final report.
    """
    payload = {
        "query": query,
        "workflow": "auto",
        "rewrite_backend": rewrite_backend,
        "session_id": None,
        "stream": False,
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return {
            "rewritten_query": str(data.get("rewritten_query", "")),
            "rewrite_time_ms": str(data.get("rewrite_time_ms", "")),
            "rewrite_backend_used": str(data.get("rewrite_backend", "")),
            "rewrite_status": "ok",
            "rewrite_error": "",
        }
    except Exception as exc:  # noqa: BLE001 - keep per-row error details.
        return {
            "rewritten_query": "",
            "rewrite_time_ms": "",
            "rewrite_backend_used": "",
            "rewrite_status": "error",
            "rewrite_error": str(exc),
        }


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def classify_distance(
    similarity: float,
    too_similar_threshold: float,
    too_different_threshold: float,
) -> str:
    """Classify distance flag from similarity score and thresholds."""
    if similarity >= too_similar_threshold:
        return "too_similar"
    if similarity <= too_different_threshold:
        return "too_different"
    return "ok"


def load_embedder(model_name: str):
    """
    Load sentence-transformers embedder.

    Raises:
        RuntimeError: if sentence-transformers is unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001 - surface install/runtime issue.
        raise RuntimeError(
            "sentence-transformers is required for distance evaluation."
        ) from exc
    return SentenceTransformer(model_name)


def enrich_with_distance(
    rows: List[Dict[str, str]],
    embed_model: str,
    too_similar_threshold: float,
    too_different_threshold: float,
) -> None:
    """
    Add cosine similarity, distance, and distance flags to rows in-place.

    Rows not eligible for distance computation keep metric columns blank and
    receive distance flags reflecting rewrite status.
    """
    for row in rows:
        row["cosine_similarity"] = ""
        row["cosine_distance"] = ""
        status = row.get("rewrite_status", "")
        if status == "skipped":
            row["distance_flag"] = "skipped"
        elif status == "error":
            row["distance_flag"] = "error"
        else:
            row["distance_flag"] = ""

    candidate_indices: List[int] = []
    texts: List[str] = []
    for idx, row in enumerate(rows):
        if row.get("rewrite_status") != "ok":
            continue
        query = (row.get("query") or "").strip()
        rewritten = (row.get("rewritten_query") or "").strip()
        if not query or not rewritten:
            row["distance_flag"] = "error"
            if not row.get("rewrite_error"):
                row["rewrite_error"] = "Missing query or rewritten_query for distance"
            continue
        candidate_indices.append(idx)
        texts.append(query)
        texts.append(rewritten)

    if not candidate_indices:
        return

    embedder = load_embedder(embed_model)
    vectors = embedder.encode(texts)

    for pos, row_index in enumerate(candidate_indices):
        vec_query = vectors[pos * 2]
        vec_rewrite = vectors[pos * 2 + 1]
        similarity = cosine_similarity(vec_query, vec_rewrite)
        distance = 1.0 - similarity
        rows[row_index]["cosine_similarity"] = f"{similarity:.6f}"
        rows[row_index]["cosine_distance"] = f"{distance:.6f}"
        rows[row_index]["distance_flag"] = classify_distance(
            similarity,
            too_similar_threshold=too_similar_threshold,
            too_different_threshold=too_different_threshold,
        )


def run_pipeline(args: argparse.Namespace) -> Dict[str, int]:
    """Execute rewrite + distance evaluation pipeline and write output CSV."""
    input_path = Path(args.input).expanduser().resolve()
    output_path = resolve_output_path(input_path, args.output)

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        base_fields = list(reader.fieldnames or [])

    if "query" not in base_fields:
        raise ValueError("Input CSV must contain a `query` column.")

    rewrite_success = 0
    rewrite_skip = 0
    rewrite_error = 0

    for index, row in enumerate(rows, start=1):
        query_value = (row.get("query") or "").strip()
        if is_blank_query(query_value):
            rewrite_result = {
                "rewritten_query": "",
                "rewrite_time_ms": "",
                "rewrite_backend_used": "",
                "rewrite_status": "skipped",
                "rewrite_error": "Skipped blank query",
            }
            rewrite_skip += 1
        else:
            rewrite_result = call_rewrite(
                endpoint=args.endpoint,
                query=query_value,
                rewrite_backend=args.rewrite_backend,
                timeout=args.timeout,
            )
            if rewrite_result["rewrite_status"] == "ok":
                rewrite_success += 1
            else:
                rewrite_error += 1

        row.update(rewrite_result)
        print(
            f"[{index}/{len(rows)}] rewrite_status={rewrite_result['rewrite_status']} "
            f"time_ms={rewrite_result['rewrite_time_ms'] or '-'} "
            f"query={query_value[:60]}"
        )

    enrich_with_distance(
        rows=rows,
        embed_model=args.embed_model,
        too_similar_threshold=args.too_similar_threshold,
        too_different_threshold=args.too_different_threshold,
    )

    extra_fields = [
        "rewritten_query",
        "rewrite_time_ms",
        "rewrite_backend_used",
        "rewrite_status",
        "rewrite_error",
        "cosine_similarity",
        "cosine_distance",
        "distance_flag",
    ]
    output_fields = base_fields + [f for f in extra_fields if f not in base_fields]

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    distance_counts = {"ok": 0, "too_similar": 0, "too_different": 0, "skipped": 0, "error": 0}
    for row in rows:
        flag = row.get("distance_flag", "")
        if flag in distance_counts:
            distance_counts[flag] += 1

    print("")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows  : {len(rows)}")
    print(f"Rewrite OK    : {rewrite_success}")
    print(f"Rewrite Skip  : {rewrite_skip}")
    print(f"Rewrite Error : {rewrite_error}")
    print(
        "Distance flags: "
        f"ok={distance_counts['ok']}, "
        f"too_similar={distance_counts['too_similar']}, "
        f"too_different={distance_counts['too_different']}, "
        f"skipped={distance_counts['skipped']}, "
        f"error={distance_counts['error']}"
    )
    return {
        "rows": len(rows),
        "rewrite_ok": rewrite_success,
        "rewrite_skip": rewrite_skip,
        "rewrite_error": rewrite_error,
        "distance_ok": distance_counts["ok"],
        "distance_too_similar": distance_counts["too_similar"],
        "distance_too_different": distance_counts["too_different"],
        "distance_skipped": distance_counts["skipped"],
        "distance_error": distance_counts["error"],
    }


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

