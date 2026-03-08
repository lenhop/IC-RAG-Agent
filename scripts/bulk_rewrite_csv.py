"""
Bulk rewrite tester for CSV query datasets.

This script reads an input CSV containing a `query` column, calls the gateway
rewrite endpoint for each row, and writes an output CSV with additional
rewrite-result columns.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import requests


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for bulk rewrite testing."""
    parser = argparse.ArgumentParser(
        description="Run bulk query rewriting for a CSV dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path containing a `query` column.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Default: <input_stem>_rewritten.csv",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/api/v1/rewrite",
        help="Rewrite API endpoint.",
    )
    parser.add_argument(
        "--rewrite-backend",
        default="ollama",
        choices=("ollama", "deepseek"),
        help="Rewrite backend to request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds for each request.",
    )
    return parser.parse_args()


def build_output_path(input_path: Path, output_arg: str) -> Path:
    """Resolve output path from CLI argument or default suffix naming."""
    if output_arg.strip():
        return Path(output_arg).expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_rewritten{input_path.suffix}")


def call_rewrite(
    endpoint: str,
    query: str,
    rewrite_backend: str,
    timeout: int,
) -> Dict[str, str]:
    """
    Call gateway rewrite endpoint and normalize response fields.

    Returns a dict with these keys:
    - rewritten_query
    - rewrite_time_ms
    - rewrite_backend_used
    - rewrite_status
    - rewrite_error
    """
    payload = {
        "query": query,
        "workflow": "auto",
        "rewrite_enable": True,
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
    except Exception as exc:  # noqa: BLE001 - keep full failure details per row.
        return {
            "rewritten_query": "",
            "rewrite_time_ms": "",
            "rewrite_backend_used": "",
            "rewrite_status": "error",
            "rewrite_error": str(exc),
        }


def main() -> int:
    """Run bulk rewrite processing and write enriched CSV output."""
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = build_output_path(input_path, args.output)

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        rows: List[Dict[str, str]] = list(reader)
        base_fields = list(reader.fieldnames or [])

    if "query" not in base_fields:
        raise ValueError("Input CSV must contain a `query` column.")

    extra_fields = [
        "rewritten_query",
        "rewrite_time_ms",
        "rewrite_backend_used",
        "rewrite_status",
        "rewrite_error",
    ]
    output_fields = base_fields + [f for f in extra_fields if f not in base_fields]

    success_count = 0
    skip_count = 0
    for index, row in enumerate(rows, start=1):
        raw_query = row.get("query") or ""
        query_value = raw_query.strip()
        if not query_value:
            # Skip blank queries explicitly to avoid unnecessary API calls.
            rewrite_result = {
                "rewritten_query": "",
                "rewrite_time_ms": "",
                "rewrite_backend_used": "",
                "rewrite_status": "skipped",
                "rewrite_error": "Skipped blank query",
            }
            skip_count += 1
        else:
            rewrite_result = call_rewrite(
                endpoint=args.endpoint,
                query=query_value,
                rewrite_backend=args.rewrite_backend,
                timeout=args.timeout,
            )
        row.update(rewrite_result)
        if rewrite_result["rewrite_status"] == "ok":
            success_count += 1
        print(
            f"[{index}/{len(rows)}] status={rewrite_result['rewrite_status']} "
            f"time_ms={rewrite_result['rewrite_time_ms'] or '-'} "
            f"query={query_value[:60]}"
        )

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows  : {len(rows)}")
    print(f"OK    : {success_count}")
    print(f"Skip  : {skip_count}")
    print(f"Error : {len(rows) - success_count - skip_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

