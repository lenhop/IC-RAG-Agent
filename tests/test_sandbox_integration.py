"""
Sandbox integration tests - real SP-API calls with live tokens.
Run with: SP_API_SANDBOX=true pytest tests/test_sandbox_integration.py -v
"""
import os
from datetime import datetime, timedelta

import pytest

from src.sp_api.sp_api_client import SPAPIClient, SPAPICredentials
from src.sp_api.tools.orders import ListOrdersTool
from src.sp_api.tools.inventory import InventoryTool
from src.sp_api.tools.catalog import ProductCatalogTool


def _creds_for_account(account: str) -> SPAPICredentials:
    """Load credentials with account-specific refresh token."""
    rt = os.environ.get(f"SP_API_{account.upper()}_REFRESH_TOKEN")
    if not rt:
        raise ValueError(f"SP_API_{account.upper()}_REFRESH_TOKEN not set")
    return SPAPICredentials(
        refresh_token=rt,
        client_id=os.environ["SP_API_CLIENT_ID"],
        client_secret=os.environ["SP_API_CLIENT_SECRET"],
        marketplace_id=os.environ.get("SP_API_MARKETPLACE_ID", "ATVPDKIKX0DER"),
        role_arn=os.environ.get("SP_API_ROLE_ARN", ""),
        aws_access_key=os.environ.get("SP_API_AWS_ACCESS_KEY", ""),
        aws_secret_key=os.environ.get("SP_API_AWS_SECRET_KEY", ""),
        region=os.environ.get("SP_API_REGION", "us-east-1"),
        app_id=os.environ.get("SP_API_APP_ID", ""),
    )


@pytest.mark.skipif(os.environ.get("SP_API_SANDBOX") != "true", reason="sandbox only")
def test_list_orders_snb_na():
    creds = _creds_for_account("SNB_NA")
    client = SPAPIClient(creds, redis_client=None)
    tool = ListOrdersTool(client)
    created_after = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = tool.execute(created_after=created_after)
    assert isinstance(result, dict)
    assert "orders" in result
    assert isinstance(result["orders"], list)


@pytest.mark.skipif(os.environ.get("SP_API_SANDBOX") != "true", reason="sandbox only")
def test_list_orders_juvo_na():
    creds = _creds_for_account("JUVO_NA")
    client = SPAPIClient(creds, redis_client=None)
    tool = ListOrdersTool(client)
    created_after = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = tool.execute(created_after=created_after)
    assert isinstance(result, dict)
    assert "orders" in result
    assert isinstance(result["orders"], list)


@pytest.mark.skipif(os.environ.get("SP_API_SANDBOX") != "true", reason="sandbox only")
def test_list_orders_juvo_eu():
    creds = _creds_for_account("JUVO_EU")
    client = SPAPIClient(creds, redis_client=None)
    tool = ListOrdersTool(client)
    created_after = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = tool.execute(created_after=created_after)
    assert isinstance(result, dict)
    assert "orders" in result


@pytest.mark.skipif(os.environ.get("SP_API_SANDBOX") != "true", reason="sandbox only")
def test_inventory_snb_na():
    creds = _creds_for_account("SNB_NA")
    client = SPAPIClient(creds, redis_client=None)
    tool = InventoryTool(client)
    result = tool.execute()
    assert isinstance(result, dict)
    assert "items" in result
    assert isinstance(result["items"], list)


@pytest.mark.skipif(os.environ.get("SP_API_SANDBOX") != "true", reason="sandbox only")
def test_product_catalog_asin():
    creds = _creds_for_account("SNB_NA")
    client = SPAPIClient(creds, redis_client=None)
    tool = ProductCatalogTool(client)
    result = tool.execute(identifier="B07XJ8C8F5", identifier_type="asin")
    assert isinstance(result, dict)
    assert "title" in result or "status" in result
    if result.get("title"):
        assert len(str(result["title"])) > 0
