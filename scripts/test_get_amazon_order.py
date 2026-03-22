#!/usr/bin/env python3
"""
One-off script: call Amazon SP-API Orders v0 getOrder using snb_na refresh token from .env.

Prerequisites (repo root .env):
    SP_API_SNB_NA_REFRESH_TOKEN — LWA refresh token for the snb_na seller (NA).
    SP_API_CLIENT_ID, SP_API_CLIENT_SECRET — LWA application (shared app in typical setups).

Optional:
    SP_API_MARKETPLACE_ID (default ATVPDKIKX0DER), SP_API_ENDPOINT (NA host),
    SNB_NA_SELLER_ID (stored on credentials; getOrder path does not require it).

Usage (from repository root):
    python scripts/test_get_amazon_order.py

Output:
    Pretty-printed JSON from SP-API, or error details on failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Repository root (parent of scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError as exc:  # pragma: no cover
    print("Missing dependency: pip install python-dotenv", file=sys.stderr)
    raise SystemExit(1) from exc

import httpx

from src.agent.sp_api.order import get_order
from src.agent.sp_api.sp_api_client import SPAPIClient, SPAPICredentials

# Target order for this smoke test (Amazon order id format).
_DEFAULT_ORDER_ID = "112-2632701-5204214"

# Env key for snb_na-specific refresh token (see .env comments in this repo).
_SNB_NA_REFRESH_ENV = "SP_API_SNB_NA_REFRESH_TOKEN"


def _load_env() -> None:
    """Load variables from .env at project root so SP_API_* are available."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _build_credentials_from_snb_na() -> SPAPICredentials:
    """
    Build SPAPICredentials using snb_na refresh token and shared LWA keys.

    Returns:
        Validated SPAPICredentials instance.

    Raises:
        ValueError: When required variables are missing or empty.
    """
    refresh = (os.environ.get(_SNB_NA_REFRESH_ENV) or "").strip()
    if not refresh:
        raise ValueError(
            f"Set {_SNB_NA_REFRESH_ENV} in .env (snb_na LWA refresh token)."
        )
    client_id = (os.environ.get("SP_API_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("SP_API_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        raise ValueError("SP_API_CLIENT_ID and SP_API_CLIENT_SECRET are required in .env.")

    marketplace_id = (os.environ.get("SP_API_MARKETPLACE_ID") or "").strip() or "ATVPDKIKX0DER"
    seller_id = (os.environ.get("SNB_NA_SELLER_ID") or "").strip()

    return SPAPICredentials(
        refresh_token=refresh,
        client_id=client_id,
        client_secret=client_secret,
        marketplace_id=marketplace_id,
        role_arn=(os.environ.get("SP_API_ROLE_ARN") or "").strip(),
        aws_access_key=(os.environ.get("SP_API_AWS_ACCESS_KEY") or "").strip(),
        aws_secret_key=(os.environ.get("SP_API_AWS_SECRET_KEY") or "").strip(),
        region=(os.environ.get("SP_API_REGION") or "").strip() or "us-east-1",
        app_id=(os.environ.get("SP_API_APP_ID") or "").strip(),
        seller_id=seller_id,
    )


def main() -> int:
    """Run getOrder for the configured order id and print JSON to stdout."""
    _load_env()
    order_id = (os.environ.get("TEST_SP_API_ORDER_ID") or _DEFAULT_ORDER_ID).strip()
    if not order_id:
        print("TEST_SP_API_ORDER_ID is empty.", file=sys.stderr)
        return 1

    try:
        creds = _build_credentials_from_snb_na()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        # SPAPIClient reads SP_API_ENDPOINT from os.environ for regional host.
        with SPAPIClient(creds) as client:
            payload = get_order(client, order_id)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        body = ""
        try:
            if exc.response is not None:
                body = exc.response.text
        except Exception:
            body = ""
        print(
            json.dumps(
                {
                    "ok": False,
                    "order_id": order_id,
                    "http_status": status,
                    "error": str(exc),
                    "response_body_preview": body[:4000],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1
    except Exception as exc:  # pragma: no cover - network/runtime
        print(
            json.dumps(
                {"ok": False, "order_id": order_id, "error": str(exc)},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1

    # Success: SP-API returns an envelope (often includes "payload" with order fields).
    print(
        json.dumps(
            {"ok": True, "order_id": order_id, "sp_api_response": payload},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
