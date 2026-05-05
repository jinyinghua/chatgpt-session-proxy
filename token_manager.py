"""
Token Manager — Multi-session ChatGPT session pool with round-robin rotation.
Supports: add/remove sessions, auto-refresh, health tracking, persistent storage.
"""

import os
import time
import json
import base64
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from curl_cffi import requests
from dotenv import load_dotenv

load_dotenv()

# Persistent storage
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE = Path(os.getenv("SESSION_FILE", str(DATA_DIR / "sessions.json")))
_INSTALLATION_ID_NAMESPACE = _uuid.UUID("6d0ab975-7f88-4ef4-9466-3f9047d5064d")
MAX_ERROR_COUNT = 5  # disable session after this many consecutive errors


@dataclass
class SessionSlot:
    """One ChatGPT session in the pool."""
    sid: str                          # short id for UI (first 8 chars of account_id)
    access_token: str = ""
    session_token: str = ""
    account_id: str = ""
    expires_at: float = 0
    raw_session: dict = field(default_factory=dict)
    error_count: int = 0
    last_error: str = ""
    disabled: bool = False

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at - 120

    @property
    def is_healthy(self) -> bool:
        return not self.disabled and self.error_count < MAX_ERROR_COUNT

    def to_dict(self) -> dict:
        return {
            "sid": self.sid,
            "account_id": self.account_id,
            "expires_at": self.expires_at,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "disabled": self.disabled,
            "is_expired": self.is_expired,
            "is_healthy": self.is_healthy,
        }


def _jwt_payload(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return {}


def _apply_session_data(data: dict) -> dict:
    """Extract fields from a raw session JSON dict."""
    access_token = data.get("accessToken", "")
    session_token = data.get("sessionToken", "")
    account = data.get("account", {})
    account_id = account.get("id", "")
    expires_str = data.get("expires", "")
    expires_at = 0
    if expires_str:
        try:
            dt = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
            expires_at = dt.timestamp()
        except Exception:
            expires_at = time.time() + 3600
    else:
        payload = _jwt_payload(access_token)
        expires_at = payload.get("exp", time.time() + 3600)
    return {
        "access_token": access_token,
        "session_token": session_token,
        "account_id": account_id,
        "expires_at": expires_at,
    }


class TokenManager:
    def __init__(self):
        self.sessions: list[SessionSlot] = []
        self.device_id = os.getenv("OAI_DEVICE_ID", "46600ebf-c112-4824-9fa7-bd0636febef8")
        self._current_idx = 0
        self._current: SessionSlot | None = None
        self._load_from_file()
        self._load_from_env()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load_from_file(self):
        if not SESSION_FILE.exists():
            return
        try:
            data = json.loads(SESSION_FILE.read_text())
            items = data if isinstance(data, list) else [data]
            for item in items:
                self._add_slot(item)
            print(f"[TokenMgr] loaded {len(items)} session(s) from {SESSION_FILE}")
        except Exception as e:
            print(f"[TokenMgr] failed to load sessions: {e}")

    def _load_from_env(self):
        """Load SESSION_TOKEN_0..9 from env vars."""
        for i in range(10):
            token = os.getenv(f"SESSION_TOKEN_{i}", "").strip()
            if not token:
                continue
            # Skip if we already have this session_token
            if any(s.session_token == token for s in self.sessions):
                continue
            data = {"sessionToken": token, "accessToken": "", "account": {"id": ""}}
            self._add_slot(data)
            print(f"[TokenMgr] loaded SESSION_TOKEN_{i} from env")

    def _save_to_file(self):
        try:
            items = [s.raw_session for s in self.sessions if s.raw_session]
            SESSION_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False))
            print(f"[TokenMgr] saved {len(items)} session(s) to {SESSION_FILE}")
        except Exception as e:
            print(f"[TokenMgr] failed to save: {e}")

    # ── Session management ───────────────────────────────────────────────

    def _add_slot(self, data: dict) -> SessionSlot:
        fields = _apply_session_data(data)
        sid = fields["account_id"][:8] if fields["account_id"] else _uuid.uuid4().hex[:8]
        slot = SessionSlot(
            sid=sid,
            access_token=fields["access_token"],
            session_token=fields["session_token"],
            account_id=fields["account_id"],
            expires_at=fields["expires_at"],
            raw_session=data,
        )
        self.sessions.append(slot)
        return slot

    def load_session_from_json(self, raw_json: str | dict) -> dict:
        """Add or update a session. If account_id already exists, update it."""
        if isinstance(raw_json, str):
            data = json.loads(raw_json)
        else:
            data = raw_json

        fields = _apply_session_data(data)
        account_id = fields["account_id"]

        # Try to find existing session with same account_id
        existing = None
        if account_id:
            for s in self.sessions:
                if s.account_id == account_id:
                    existing = s
                    break

        if existing:
            existing.access_token = fields["access_token"]
            existing.session_token = fields["session_token"]
            existing.expires_at = 0  # force refresh
            existing.raw_session = data
            existing.error_count = 0
            existing.last_error = ""
            existing.disabled = False
            slot = existing
            action = "updated"
        else:
            slot = self._add_slot(data)
            slot.expires_at = 0  # force refresh
            action = "added"

        self._save_to_file()
        return {
            "status": "ok",
            "action": action,
            "sid": slot.sid,
            "account_id": slot.account_id,
            "total_sessions": len(self.sessions),
            "message": f"Session {action}: {slot.sid} (account={slot.account_id[:8]}..., total={len(self.sessions)})",
        }

    def remove_session(self, sid: str) -> bool:
        for i, s in enumerate(self.sessions):
            if s.sid == sid:
                self.sessions.pop(i)
                if self._current_idx >= len(self.sessions):
                    self._current_idx = 0
                if self._current and self._current.sid == sid:
                    self._current = None
                self._save_to_file()
                return True
        return False

    def toggle_session(self, sid: str, disabled: bool) -> bool:
        for s in self.sessions:
            if s.sid == sid:
                s.disabled = disabled
                if disabled and s.error_count > 0:
                    s.error_count = 0  # reset errors when manually toggling
                self._save_to_file()
                return True
        return False

    def get_all_status(self) -> list[dict]:
        return [s.to_dict() for s in self.sessions]

    # ── Token rotation ───────────────────────────────────────────────────

    async def get_valid_token(self) -> str:
        """Pick a healthy session, refresh if needed, return access_token.
        Also sets self._current so .account_id / .installation_id work."""
        if not self.sessions:
            raise Exception("No sessions available. Add one via / or POST /auth/session")

        n = len(self.sessions)
        for _ in range(n):
            slot = self.sessions[self._current_idx % n]
            self._current_idx += 1

            if not slot.is_healthy:
                continue

            # Refresh if expired
            if slot.is_expired or not slot.access_token:
                ok = await self._refresh_slot(slot)
                if not ok:
                    continue

            self._current = slot
            acct = slot.account_id[:8] if slot.account_id else "?"
            print(f"[Auth] using session {slot.sid} (account={acct}...)")
            return slot.access_token

        # All sessions failed — try to refresh all once more
        print("[Auth] all sessions unhealthy, attempting full refresh...")
        for slot in self.sessions:
            if slot.disabled:
                continue
            slot.error_count = 0
            ok = await self._refresh_slot(slot)
            if ok:
                self._current = slot
                acct = slot.account_id[:8] if slot.account_id else "?"
                print(f"[Auth] recovered session {slot.sid} (account={acct}...)")
                return slot.access_token

        raise Exception("All sessions failed. Check logs and re-add sessions via /")

    async def _refresh_slot(self, slot: SessionSlot) -> bool:
        """Refresh a single slot's access_token via its sessionToken."""
        if not slot.session_token:
            slot.error_count += 1
            slot.last_error = "no sessionToken"
            return False

        acct = slot.account_id[:8] if slot.account_id else "?"
        print(f"[Auth] refreshing session {slot.sid} (account={acct}...)")
        cookies = {"__Secure-next-auth.session-token": slot.session_token}
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "oai-device-id": self.device_id,
        }
        try:
            async with requests.AsyncSession(impersonate="chrome110") as sess:
                resp = await sess.get("https://chatgpt.com/api/auth/session", headers=headers, cookies=cookies)
                if resp.status_code != 200:
                    slot.error_count += 1
                    slot.last_error = f"refresh HTTP {resp.status_code}"
                    print(f"[Auth] session {slot.sid} refresh failed: {resp.status_code}")
                    return False
                data = resp.json()
                if "accessToken" not in data:
                    slot.error_count += 1
                    slot.last_error = "no accessToken in response"
                    print(f"[Auth] session {slot.sid} invalid sessionToken")
                    return False
                # Apply new data
                fields = _apply_session_data(data)
                slot.access_token = fields["access_token"]
                slot.session_token = fields["session_token"] or slot.session_token
                if fields["account_id"]:
                    slot.account_id = fields["account_id"]
                    slot.sid = fields["account_id"][:8]
                slot.expires_at = fields["expires_at"]
                slot.raw_session = data
                slot.error_count = 0
                slot.last_error = ""
                self._save_to_file()
                print(f"[Auth] session {slot.sid} refreshed, expires at {fields['expires_at']}")
                return True
        except Exception as e:
            slot.error_count += 1
            slot.last_error = str(e)[:200]
            print(f"[Auth] session {slot.sid} refresh error: {e}")
            return False

    # ── Compat properties (reflect currently selected session) ───────────

    @property
    def account_id(self) -> str:
        if self._current:
            return self._current.account_id
        return ""

    @property
    def installation_id(self) -> str:
        aid = self.account_id
        if not aid:
            return ""
        return str(_uuid.uuid5(_INSTALLATION_ID_NAMESPACE, aid))

    @property
    def access_token(self) -> str:
        if self._current:
            return self._current.access_token
        return ""

    @property
    def session_token(self) -> str:
        if self._current:
            return self._current.session_token
        return ""

    @property
    def expires_at(self) -> float:
        if self._current:
            return self._current.expires_at
        return 0


token_manager = TokenManager()
