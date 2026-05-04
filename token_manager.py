"""
Token Manager — Codaze-compatible session management.
Uses refresh_token via /api/auth/session, persistent storage, installation_id.
"""

import os
import time
import json
import base64
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from curl_cffi import requests
from dotenv import load_dotenv

load_dotenv()

SESSION_FILE = Path(os.getenv("SESSION_FILE", "sessions.json"))
_INSTALLATION_ID_NAMESPACE = _uuid.UUID("6d0ab975-7f88-4ef4-9466-3f9047d5064d")


class TokenManager:
    def __init__(self):
        self.access_token: str | None = None
        self.session_token: str | None = None
        self.account_id_value: str | None = None
        self.expires_at: float = 0
        self.device_id = os.getenv("OAI_DEVICE_ID", "46600ebf-c112-4824-9fa7-bd0636febef8")
        self.raw_session: dict | None = None
        self._load_from_file()

    def _load_from_file(self):
        if not SESSION_FILE.exists():
            return
        try:
            data = json.loads(SESSION_FILE.read_text())
            self._apply_session(data)
            acct = self.account_id_value[:8] if self.account_id_value else "N/A"
            print(f"[TokenMgr] loaded session from {SESSION_FILE} (account={acct}...)")
        except Exception as e:
            print(f"[TokenMgr] failed to load session: {e}")

    def _save_to_file(self):
        if not self.raw_session:
            return
        try:
            SESSION_FILE.write_text(json.dumps(self.raw_session, indent=2, ensure_ascii=False))
            print(f"[TokenMgr] session saved to {SESSION_FILE}")
        except Exception as e:
            print(f"[TokenMgr] failed to save session: {e}")

    def _apply_session(self, data: dict):
        self.raw_session = data
        self.access_token = data.get("accessToken", "")
        self.session_token = data.get("sessionToken", "")
        account = data.get("account", {})
        self.account_id_value = account.get("id", "")
        expires_str = data.get("expires", "")
        if expires_str:
            try:
                dt = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
                self.expires_at = dt.timestamp()
            except Exception:
                self.expires_at = time.time() + 3600
        else:
            payload = self._jwt_payload(self.access_token)
            self.expires_at = payload.get("exp", time.time() + 3600)

    def load_session_from_json(self, raw_json: str | dict) -> dict:
        if isinstance(raw_json, str):
            data = json.loads(raw_json)
        else:
            data = raw_json
        self._apply_session(data)
        self.expires_at = 0  # force refresh on next request
        self._save_to_file()
        acct = self.account_id_value[:8] if self.account_id_value else "N/A"
        return {"status": "ok", "account_id": self.account_id_value, "message": f"Session saved, will refresh on next request (account={acct}...)"}

    def _jwt_payload(self, token: str) -> dict:
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return {}
            payload_b64 = parts[1]
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            return json.loads(base64.urlsafe_b64decode(payload_b64))
        except Exception:
            return {}

    async def get_valid_token(self) -> str:
        if self.access_token and time.time() < self.expires_at - 120:
            return self.access_token
        if not self.session_token:
            raise Exception("No sessionToken available. Paste session JSON at /")
        print("[Auth] Token expired or missing, refreshing via sessionToken...")
        cookies = {"__Secure-next-auth.session-token": self.session_token}
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "oai-device-id": self.device_id,
        }
        async with requests.AsyncSession(impersonate="chrome110") as session:
            try:
                response = await session.get("https://chatgpt.com/api/auth/session", headers=headers, cookies=cookies)
                if response.status_code != 200:
                    raise Exception(f"Refresh failed: {response.status_code}")
                data = response.json()
                if "accessToken" not in data:
                    raise Exception("sessionToken may be invalid, no accessToken in response")
                self._apply_session(data)
                self._save_to_file()
                acct = self.account_id_value[:8] if self.account_id_value else "N/A"
                print(f"[Auth] Token refreshed! account_id={acct}...")
                return self.access_token
            except Exception as e:
                print(f"[Auth Error] {e}")
                raise e

    @property
    def account_id(self) -> str:
        return self.account_id_value or ""

    @property
    def installation_id(self) -> str:
        """UUID v5 from account_id, matching Codaze stable_installation_id()"""
        if not self.account_id_value:
            return ""
        return str(_uuid.uuid5(_INSTALLATION_ID_NAMESPACE, self.account_id_value))


token_manager = TokenManager()
