"""
Token Manager — 支持从 JSON 文件加载/保存 session，自动刷新 accessToken。

Session JSON 格式（直接粘贴 /api/auth/session 的返回值）:
{
  "accessToken": "eyJ...",
  "sessionToken": "eyJ...",
  "account": {"id": "...", ...},
  "expires": "2026-08-01T12:44:19.668Z",
  ...
}
"""

import os
import time
import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from curl_cffi import requests
from dotenv import load_dotenv

load_dotenv()

SESSION_FILE = Path(os.getenv("SESSION_FILE", "sessions.json"))


class TokenManager:
    def __init__(self):
        self.access_token: str | None = None
        self.session_token: str | None = None
        self.account_id_value: str | None = None
        self.expires_at: float = 0
        self.device_id = os.getenv("OAI_DEVICE_ID", "46600ebf-c112-4824-9fa7-bd0636febef8")
        self.raw_session: dict | None = None  # 保存原始 JSON

        self._load_from_file()

    # ── 持久化 ────────────────────────────────────────────────────────
    def _load_from_file(self):
        """从文件加载 session"""
        if not SESSION_FILE.exists():
            return
        try:
            data = json.loads(SESSION_FILE.read_text())
            self._apply_session(data)
            print(f"[TokenMgr] 从 {SESSION_FILE} 加载了 session (account={self.account_id_value[:8] if self.account_id_value else 'N/A'}...)")
        except Exception as e:
            print(f"[TokenMgr] 加载 session 文件失败: {e}")

    def _save_to_file(self):
        """保存 session 到文件"""
        if not self.raw_session:
            return
        try:
            SESSION_FILE.write_text(json.dumps(self.raw_session, indent=2, ensure_ascii=False))
            print(f"[TokenMgr] Session 已保存到 {SESSION_FILE}")
        except Exception as e:
            print(f"[TokenMgr] 保存 session 文件失败: {e}")

    # ── 应用 session ──────────────────────────────────────────────────
    def _apply_session(self, data: dict):
        """从 JSON dict 中提取并应用 session 信息"""
        self.raw_session = data

        # 提取 accessToken
        self.access_token = data.get("accessToken", "")

        # 提取 sessionToken（用于刷新）
        self.session_token = data.get("sessionToken", "")

        # 提取 account_id
        account = data.get("account", {})
        self.account_id_value = account.get("id", "")

        # 解析过期时间
        expires_str = data.get("expires", "")
        if expires_str:
            try:
                # ISO 8601 格式: "2026-08-01T12:44:19.668Z"
                dt = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
                self.expires_at = dt.timestamp()
            except Exception:
                # 如果解析失败，假设 1 小时后过期
                self.expires_at = time.time() + 3600
        else:
            # 从 JWT 中解析 exp
            payload = self._jwt_payload(self.access_token)
            self.expires_at = payload.get("exp", time.time() + 3600)

def load_session_from_json(self, raw_json: str | dict) -> dict:
        """从 JSON 字符串或 dict 加载新 session"""
        if isinstance(raw_json, str):
            data = json.loads(raw_json)
        else:
            data = raw_json

        self._apply_session(data)
        
        # 强制过期时间为0，使得下一次请求必须刷新
        self.expires_at = 0 
        
        self._save_to_file()

        return {
            "status": "ok",
            "account_id": self.account_id_value,
            "message": f"Session 已保存，下次请求将自动刷新 (account={self.account_id_value[:8] if self.account_id_value else 'N/A'}...)"
        }...)"
        }

    # ── JWT 解析 ──────────────────────────────────────────────────────
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

    # ── 获取有效的 Access Token ────────────────────────────────────────
    async def get_valid_token(self) -> str:
        """获取有效的 Access Token，过期则用 sessionToken 刷新"""
        if self.access_token and time.time() < self.expires_at - 120:
            return self.access_token

        if not self.session_token:
            raise Exception("没有可用的 sessionToken，请在前端页面 (/) 粘贴 session JSON")

        print("[Auth] Access Token 不存在或即将过期，正在使用 sessionToken 刷新...")

        # 设置 cookie: __Secure-next-auth.session-token
        cookies = {
            "__Secure-next-auth.session-token": self.session_token,
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "oai-device-id": self.device_id,
        }

        async with requests.AsyncSession(impersonate="chrome110") as session:
            try:
                response = await session.get(
                    "https://chatgpt.com/api/auth/session",
                    headers=headers,
                    cookies=cookies,
                )

                if response.status_code != 200:
                    raise Exception(f"刷新失败，状态码: {response.status_code}")

                data = response.json()
                if "accessToken" not in data:
                    raise Exception("sessionToken 可能已失效，返回数据中没有 accessToken")

                # 更新并保存
                self._apply_session(data)
                self._save_to_file()

                print(f"[Auth] 成功刷新 Token! account_id={self.account_id_value[:8] if self.account_id_value else 'N/A'}...")
                return self.access_token

            except Exception as e:
                print(f"[Auth Error] {e}")
                raise e

    @property
    def account_id(self) -> str:
        return self.account_id_value or ""


token_manager = TokenManager()
