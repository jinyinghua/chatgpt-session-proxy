import os
import time
import json
import base64
from curl_cffi import requests
from dotenv import load_dotenv

load_dotenv()

class TokenManager:
    def __init__(self):
        self.access_token = None
        self.expires_at = 0
        self._account_id = None
        
        self.session_0 = os.getenv("SESSION_TOKEN_0", "")
        self.session_1 = os.getenv("SESSION_TOKEN_1", "")
        self.device_id = os.getenv("OAI_DEVICE_ID", "46600ebf-c112-4824-9fa7-bd0636febef8")
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
            "Accept": "*/*",
            "oai-device-id": self.device_id
        }

    def _jwt_payload(self, token: str) -> dict:
        """解析 JWT payload（不验证签名）"""
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return {}
            payload_b64 = parts[1]
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64)
            return json.loads(payload_json)
        except Exception:
            return {}

    @property
    def account_id(self) -> str:
        """从 access_token JWT 中提取 chatgpt_account_id"""
        if self._account_id:
            return self._account_id
        if not self.access_token:
            return ""
        payload = self._jwt_payload(self.access_token)
        auth = payload.get("https://api.openai.com/auth", {})
        self._account_id = auth.get("chatgpt_account_id", "")
        return self._account_id

    async def get_valid_token(self) -> str:
        """获取有效的 Access Token，如果过期则自动去官网刷新"""
        if self.access_token and time.time() < self.expires_at - 60:
            return self.access_token

        print("[Auth] Access Token 不存在或即将过期，正在使用 Cookie 刷新...")
        
        cookies = {
            "__Secure-next-auth.session-token.0": self.session_0,
            "__Secure-next-auth.session-token.1": self.session_1
        }

        async with requests.AsyncSession(impersonate="chrome120") as session:
            try:
                response = await session.get(
                    "https://chatgpt.com/api/auth/session",
                    headers=self.headers,
                    cookies=cookies
                )
                
                if response.status_code != 200:
                    raise Exception(f"刷新 Token 失败，状态码: {response.status_code}, 内容: {response.text}")
                
                data = response.json()
                if "accessToken" not in data:
                    raise Exception("Cookie 可能已失效，返回数据中没有 accessToken")
                
                self.access_token = data["accessToken"]
                self._account_id = None  # 重置，下次重新解析
                self.expires_at = time.time() + 3600
                print(f"[Auth] 成功刷新 Token! account_id={self.account_id[:8]}...")
                
                return self.access_token
            except Exception as e:
                print(f"[Auth Error] {e}")
                raise e

token_manager = TokenManager()
