"""Tests for GATEWAY_UDS_AGENT_DISABLED UDS stub path."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.gateway.dispatcher.clients.uds_client import UdsWorkflowClient
from src.gateway.dispatcher.clients.worker_profile import (
    is_uds_agent_disabled,
    uds_agent_disabled_stub_payload,
)


class TestUdsAgentDisabled(unittest.TestCase):
    def test_default_unset_not_agent_disabled_flag(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GATEWAY_UDS_AGENT_DISABLED", None)
            self.assertFalse(is_uds_agent_disabled())

    def test_false_disables_agent_disabled_flag(self) -> None:
        with patch.dict(os.environ, {"GATEWAY_UDS_AGENT_DISABLED": "false"}, clear=False):
            self.assertFalse(is_uds_agent_disabled())

    def test_true_enables_stub_flag(self) -> None:
        with patch.dict(os.environ, {"GATEWAY_UDS_AGENT_DISABLED": "true"}, clear=False):
            self.assertTrue(is_uds_agent_disabled())

    def test_stub_payload_default_message(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GATEWAY_UDS_STUB_MESSAGE", None)
            p = uds_agent_disabled_stub_payload()
        self.assertEqual(p["answer"], "UDS available")
        self.assertEqual(p["sources"], [])

    @patch.dict(os.environ, {"GATEWAY_UDS_STUB_MESSAGE": "custom"}, clear=False)
    def test_stub_payload_custom_message(self) -> None:
        p = uds_agent_disabled_stub_payload()
        self.assertEqual(p["answer"], "custom")

    def test_call_uds_returns_stub_when_disabled(self) -> None:
        with patch.dict(
            os.environ,
            {"GATEWAY_UDS_AGENT_DISABLED": "true"},
            clear=False,
        ):
            os.environ.pop("GATEWAY_UDS_STUB_MESSAGE", None)
            out = UdsWorkflowClient.call_uds("any query", "sid")
        self.assertEqual(out.get("answer"), "UDS available")
        self.assertEqual(out.get("sources"), [])


if __name__ == "__main__":
    unittest.main()
