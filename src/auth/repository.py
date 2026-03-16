"""
User repository for ClickHouse ic_rag_agent_user table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from src.agent.uds.config import UDSConfig
from src.agent.uds.uds_client import UDSClient

logger = logging.getLogger(__name__)

TABLE_NAME = "ic_rag_agent_user"
VALID_ROLES = frozenset({"general", "supervisor", "admin"})
VALID_STATUS = frozenset({"active", "inactive", "suspended"})


class UserRepository:
    """
    CRUD operations for ic_rag_agent_user in ClickHouse.

    Uses UDSConfig for connection; table lives in CH_DATABASE.
    """

    def __init__(self, client: Optional[UDSClient] = None):
        """
        Initialize repository.

        Args:
            client: Optional UDSClient. If None, creates one from UDSConfig.
        """
        self._client = client or UDSClient(
            host=UDSConfig.CH_HOST,
            port=UDSConfig.CH_PORT,
            user=UDSConfig.CH_USER,
            password=UDSConfig.CH_PASSWORD,
            database=UDSConfig.CH_DATABASE,
        )
        self._db = UDSConfig.CH_DATABASE
        self._table = f"{self._db}.{TABLE_NAME}"

    def get_by_user_name(self, user_name: str) -> Optional[dict[str, Any]]:
        """
        Fetch user by user_name. Uses FINAL for ReplacingMergeTree deduplication.

        Args:
            user_name: Display name.

        Returns:
            User row as dict, or None if not found.
        """
        try:
            sql = f"""
                SELECT user_id, user_name, email, password_hash, role, status,
                       created_time, updated_time, last_login_time, last_login_ip, metadata
                FROM {self._table} FINAL
                WHERE user_name = {{user_name:String}}
                LIMIT 1
            """
            df = self._client.query(sql, params={"user_name": user_name}, as_dataframe=True)
            if df.empty:
                return None
            row = df.iloc[0]
            return self._row_to_dict(row)
        except Exception as e:
            logger.exception("get_by_user_name failed: %s", e)
            raise

    def get_by_email(self, email: str) -> Optional[dict[str, Any]]:
        """
        Fetch user by email. Uses FINAL for deduplication.

        Args:
            email: User email.

        Returns:
            User row as dict, or None if not found.
        """
        if not email or not str(email).strip():
            return None
        try:
            sql = f"""
                SELECT user_id, user_name, email, password_hash, role, status,
                       created_time, updated_time, last_login_time, last_login_ip, metadata
                FROM {self._table} FINAL
                WHERE email = {{email:String}}
                LIMIT 1
            """
            df = self._client.query(sql, params={"email": str(email).strip()}, as_dataframe=True)
            if df.empty:
                return None
            row = df.iloc[0]
            return self._row_to_dict(row)
        except Exception as e:
            logger.exception("get_by_email failed: %s", e)
            raise

    def create(
        self,
        user_id: UUID,
        user_name: str,
        password_hash: str,
        email: str = "",
        role: str = "general",
    ) -> None:
        """
        Insert a new user.

        Args:
            user_id: UUID for the user.
            user_name: Display name.
            password_hash: Bcrypt hash of password.
            email: Optional email.
            role: User role (general, supervisor, admin).
        """
        now = datetime.now(timezone.utc)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            sql = f"""
                INSERT INTO {self._table}
                (user_id, user_name, email, password_hash, role, status, created_time, updated_time, metadata)
                VALUES
                ({{user_id:UUID}}, {{user_name:String}}, {{email:String}}, {{password_hash:String}},
                 {{role:String}}, 'active', {{created_time:DateTime64(3)}}, {{updated_time:DateTime64(3)}}, '{{}}')
            """
            self._client.execute(
                sql,
                params={
                    "user_id": str(user_id),
                    "user_name": user_name.strip(),
                    "email": (email or "").strip(),
                    "password_hash": password_hash,
                    "role": role if role in VALID_ROLES else "general",
                    "created_time": now_str,
                    "updated_time": now_str,
                },
            )
        except Exception as e:
            logger.exception("create user failed: %s", e)
            raise

    def update_last_login(self, user_id: UUID, ip: Optional[str] = None) -> None:
        """
        Update last_login_time and last_login_ip by inserting a new version.
        ReplacingMergeTree keeps the row with max(updated_time).

        Args:
            user_id: User UUID.
            ip: Optional client IP.
        """
        try:
            ip_val = (ip or "").strip() if ip else ""
            sql = f"""
                INSERT INTO {self._table}
                (user_id, user_name, email, password_hash, role, status, created_time, updated_time,
                 last_login_time, last_login_ip, metadata)
                SELECT user_id, user_name, email, password_hash, role, status, created_time,
                       now64(3), now64(3), {{ip:String}}, metadata
                FROM {self._table} FINAL
                WHERE user_id = {{user_id:UUID}}
            """
            self._client.execute(
                sql,
                params={"user_id": str(user_id), "ip": ip_val},
            )
        except Exception as e:
            logger.exception("update_last_login failed: %s", e)
            raise

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """Convert DataFrame row to dict with string user_id."""
        d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        if "user_id" in d and d["user_id"] is not None:
            d["user_id"] = str(d["user_id"])
        return d
