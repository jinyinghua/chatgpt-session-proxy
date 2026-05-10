"""
ChatGPT Web2API Proxy — Single-file FastAPI server.

Routes:
  POST /v1/chat/completions  → conversation node-tree (supports text + gpt-image-2)
  POST /v1/responses         → codex/responses (standard OpenAI Responses format)
  POST /v1/images/generations → same as /v1/chat/completions image path
  GET  /v1/models             → list supported models
  GET  /ping                  → health check (no auth)

Requires env vars: SESSION_TOKEN_0, SESSION_TOKEN_1, OAI_DEVICE_ID, API_KEY
"""

import os
import base64
import json
import uuid
import asyncio
import logging
import time
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from curl_cffi import requests as curl_requests
from dotenv import load_dotenv

from token_manager import token_manager
from pow_solver import generate_requirements_token, solve_pow

load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("proxy")

# ── Constants ───────────────────────────────────────────────────────────
BASE_URL = "https://chatgpt.com/backend-api"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_USER_AGENT = "codex-tui/0.118.0 (Mac OS 26.3.1; arm64) iTerm.app/3.6.9 (codex-tui; 0.118.0)"
CODEX_ORIGINATOR = "codex-tui"
WEB_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)
DEFAULT_MODEL = "gpt-5.4-mini"
IMAGE_MODELS = {"gpt-image-1", "gpt-image-2", "auto"}

# ── API Key 鉴权 ───────────────────────────────────────────────────────
# Multi-key support: comma-separated keys
_raw_api_keys = os.getenv("API_KEY", "")
API_KEYS: set[str] = {k.strip() for k in _raw_api_keys.split(",") if k.strip()}
API_KEY = next(iter(API_KEYS), "")  # backwards compat

# 不需要鉴权的白名单路径
AUTH_WHITELIST = {"/ping", "/health", "/healthz", "/docs", "/openapi.json", "/", "/favicon.ico"}


# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(title="ChatGPT Web2API Proxy")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    API Key 鉴权中间件。
    支持三种方式传入密钥（兼容 OpenAI SDK 和各类客户端）：
      1. Authorization: Bearer sk-xxxx
      2. X-API-Key: sk-xxxx
      3. 查询参数 ?key=sk-xxxx  （仅 SSE/WebSocket 不方便设 header 时使用）
    """
    path = request.url.path

    # 白名单路径免鉴权
    if path in AUTH_WHITELIST or path.startswith("/docs") or path.startswith("/openapi"):
        return await call_next(request)

    # /auth/login-check 自行验证 key（前端登录用，请求时不带 Bearer）
    if path == "/auth/login-check" and request.method == "POST":
        return await call_next(request)

    # 如果没有配置 API_KEY，跳过鉴权（开发模式）
    if not API_KEYS:
        return await call_next(request)

    # 提取客户端传来的 key
    client_key = ""

    # 方式 1: Authorization: Bearer <key>
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        client_key = auth_header[7:].strip()

    # 方式 2: X-API-Key
    if not client_key:
        client_key = request.headers.get("x-api-key", "").strip()

    # 方式 3: 查询参数 ?key=
    if not client_key:
        client_key = request.query_params.get("key", "").strip()

    # 验证
    if not client_key or client_key not in API_KEYS:
        log.warning(f"[auth] rejected {request.method} {path} — invalid or missing API key")
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Invalid API key. Provide it via 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
        )

    return await call_next(request)


# ══════════════════════════════════════════════════════════════════════════
#  Codaze-compatible header & request normalization
# ══════════════════════════════════════════════════════════════════════════

def build_codex_headers(access_token: str, account_id: str, installation_id: str) -> dict:
    """Build Codex-compatible headers (ported from codexProapi proxy.js)"""
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://chatgpt.com/",
        "Origin": "https://chatgpt.com",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "DNT": "1",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
        "chatgpt-account-id": account_id,
        "Connection": "keep-alive",
        "x-codex-installation-id": installation_id,
    }


def normalize_codex_request(payload: dict) -> dict:
    """Normalize request body to match codexProapi buildResponsesRequest()"""
    # Remove fields codexProapi doesn't send
    for key in ["max_output_tokens", "max_completion_tokens", "temperature",
                "top_p", "truncation", "user", "service_tier"]:
        payload.pop(key, None)

    # Tools: default to empty array
    if "tools" not in payload:
        payload["tools"] = []
    # tool_choice: 'none' if no tools
    if not payload["tools"] and "tool_choice" not in payload:
        payload["tool_choice"] = "none"

    # Required fields matching codexProapi exactly
    payload.setdefault("instructions", "You are a helpful AI assistant. Provide clear, accurate, and concise responses.")
    payload.setdefault("store", False)
    payload.setdefault("stream", True)
    payload.setdefault("parallel_tool_calls", False)
    payload.setdefault("reasoning", None)
    payload.setdefault("include", [])

    return payload


# ══════════════════════════════════════════════════════════════════════════
#  Token & PoW helpers
# ══════════════════════════════════════════════════════════════════════════

async def get_sentinel_tokens(access_token: str, device_id: str) -> tuple[str, str]:
    """Fetch chat-requirements token and solve PoW if needed."""
    req_token = generate_requirements_token()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": WEB_USER_AGENT,
        "oai-device-id": device_id,
    }

    async with curl_requests.AsyncSession(impersonate="chrome110") as session:
        resp = await session.post(
            f"{BASE_URL}/sentinel/chat-requirements",
            json={"p": req_token},
            headers=headers,
        )
        if resp.status_code != 200:
            raise Exception(f"chat-requirements failed: {resp.status_code} {resp.text}")

        data = resp.json()
        # log.info(f"[sentinel] requirements: {json.dumps(data)}")
        chat_token = data.get("token", "")
        pow_info = data.get("proofofwork", {})
        proof_token = ""

        if pow_info.get("required"):
            seed = pow_info["seed"]
            difficulty = pow_info["difficulty"]
            log.info(f"[PoW] required, seed={seed[:16]}... difficulty={difficulty}")
            proof_token = await asyncio.to_thread(solve_pow, seed, difficulty)
            log.info(f"[PoW] solved, prefix={proof_token[:24]}...")

    return chat_token, proof_token


# ══════════════════════════════════════════════════════════════════════════
#  Conversation node-tree builder (for image generation via Free account)

def _parse_data_uri(uri: str) -> tuple[bytes, str]:
    if not uri.startswith("data:"):
        return None, ""
    try:
        header, data = uri.split(",", 1)
        mime = header.split(":", 1)[1].split(";", 1)[0]
        return base64.b64decode(data), mime
    except Exception:
        return None, ""


async def _upload_file(access_token: str, device_id: str, data: bytes, mime_type: str) -> str:
    """Upload file to ChatGPT and return file_id."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": WEB_USER_AGENT,
        "OAI-Device-Id": device_id,
    }
    
    filename = f"input_image_{int(time.time())}.png"
    
    async with curl_requests.AsyncSession(impersonate="chrome110") as session:
        # Step 1: Pre-upload
        pre_payload = {
            "file_name": filename,
            "file_size": len(data),
            "use_case": "multimodal",
            "mime_type": mime_type or "image/png"
        }
        resp = await session.post(f"{BASE_URL}/files", json=pre_payload, headers=headers)
        if resp.status_code != 200:
            log.error(f"[upload] pre-upload failed: {resp.status_code} {resp.text}")
            return ""
        
        pre_data = resp.json()
        upload_url = pre_data.get("upload_url")
        file_id = pre_data.get("file_id")
        
        if not upload_url or not file_id:
            return ""
            
        # Step 2: Upload to blob
        # Note: impersonate might interfere with direct blob upload, using plain headers
        blob_headers = {"x-ms-blob-type": "BlockBlob", "Content-Type": mime_type or "image/png"}
        # Use a new session without impersonation for the PUT request to avoid TLS issues with Azure
        async with curl_requests.AsyncSession() as blob_session:
            put_resp = await blob_session.put(upload_url, data=data, headers=blob_headers)
            if put_resp.status_code not in (200, 201):
                log.error(f"[upload] blob upload failed: {put_resp.status_code}")
                return ""
        
        # Step 3: Confirm
        conf_resp = await session.post(f"{BASE_URL}/files/{file_id}/uploaded", json={}, headers=headers)
        if conf_resp.status_code != 200:
            log.error(f"[upload] confirm failed: {conf_resp.status_code}")
            return ""
            
        log.info(f"[upload] success: {file_id}")
        return file_id


def build_multimodal_body(prompt: str, model: str, file_ids: list, encodings: list = None) -> dict:
    """Build multimodal body for image editing/input."""
    if model in IMAGE_MODELS:
        model = "auto"
    msg_id = str(uuid.uuid4())
    
    parts = [prompt]
    attachments = []
    
    for i, fid in enumerate(file_ids):
        parts.append({
            "content_type": "image_asset_pointer",
            "asset_pointer": f"file-service://{fid}",
            "size_bytes": 0, # optional
        })
        attachments.append({
            "id": fid,
            "name": f"image_{i}.png",
            "mimeType": "image/png",
        })

    return {
        "action": "next",
        "messages": [
            {
                "id": msg_id,
                "author": {"role": "user"},
                "content": {"content_type": "multimodal_text", "parts": parts},
                "metadata": {
                    "attachments": attachments,
                    "system_hints": ["picture_v2"],
                },
            }
        ],
        "parent_message_id": "client-created-root",
        "model": model,
        "timezone_offset_min": 420,
        "timezone": "America/Los_Angeles",
        "conversation_mode": {"kind": "primary_assistant"},
        "enable_message_followups": True,
        "system_hints": ["picture_v2"],
        "supports_buffering": True,
        "supported_encodings": encodings or [],
        "paragen_cot_summary_display_override": "allow",
        "force_parallel_switch": "auto",
    }

# ══════════════════════════════════════════════════════════════════════════

def build_conversation_body(prompt: str, model: str = DEFAULT_MODEL, encodings: list = None) -> dict:
    # Map image model names to "auto" for upstream (ChatGpt-Image-Studio convention)
    if model in IMAGE_MODELS:
        model = "auto"
    """Build the node-tree body for /backend-api/conversation."""
    msg_id = str(uuid.uuid4())
    return {
        "action": "next",
        "messages": [
            {
                "id": msg_id,
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [prompt]},
                "metadata": {
                    "system_hints": ["picture_v2"],
                    "serialization_metadata": {"custom_symbol_offsets": []},
                },
            }
        ],
        "parent_message_id": "client-created-root",
        "model": model,
        "timezone_offset_min": 420,
        "timezone": "America/Los_Angeles",
        "conversation_mode": {"kind": "primary_assistant"},
        "enable_message_followups": True,
        "client_prepare_state": "none",
        "system_hints": ["picture_v2"],
        "supports_buffering": True,
        "supported_encodings": encodings or [],
        "client_contextual_info": {
            "is_dark_mode": True,
            "time_since_loaded": 1000,
            "page_height": 717,
            "page_width": 1200,
            "pixel_ratio": 2,
            "screen_height": 878,
            "screen_width": 1352,
            "app_name": "chatgpt.com",
        },
        "paragen_cot_summary_display_override": "allow",
        "force_parallel_switch": "auto",
    }


def build_text_conversation_body(messages: list, model: str = DEFAULT_MODEL) -> dict:
    """Build node-tree body for text conversation via /backend-api/conversation."""
    # Flatten messages into a single prompt (system + user context)
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(texts)
        if content:
            if role == "system":
                parts.append(f"[System] {content}")
            elif role == "assistant":
                parts.append(f"[Assistant] {content}")
            elif role == "user":
                parts.append(content)
    
    prompt = "\n".join(parts) if parts else "hello"
    msg_id = str(uuid.uuid4())
    return {
        "action": "next",
        "messages": [
            {
                "id": msg_id,
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [prompt]},
            }
        ],
        "parent_message_id": "client-created-root",
        "model": model,
        "timezone_offset_min": 420,
        "timezone": "America/Los_Angeles",
        "conversation_mode": {"kind": "primary_assistant"},
        "enable_message_followups": True,
        "supports_buffering": True,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Conversation SSE parser — extracts image URLs
# ══════════════════════════════════════════════════════════════════════════

def _extract_file_id(asset_pointer: str) -> str:
    for prefix in ("file-service://", "sediment://"):
        if asset_pointer.startswith(prefix):
            return asset_pointer[len(prefix):].split("?")[0]
    return ""

def _is_sediment(asset_pointer: str) -> bool:
    return asset_pointer.startswith("sediment://")


async def _resolve_image_url(access_token: str, device_id: str,
                              file_id: str, conversation_id: str, is_sediment: bool = False) -> str:
    """Resolve file_id → download the actual image bytes → return as data URI.
    This ensures clients don't need auth to access images."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": WEB_USER_AGENT,
        "OAI-Device-Id": device_id,
    }
    async with curl_requests.AsyncSession(impersonate="chrome110") as session:
        # Step 1: resolve the download URL
        download_url = ""
        if is_sediment:
            url = f"{BASE_URL}/conversation/{conversation_id}/attachment/{file_id}/download"
        else:
            url = f"{BASE_URL}/files/download/{file_id}?conversation_id={conversation_id}&inline=false"

        resp = await session.get(url, headers=headers, allow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308):
            download_url = resp.headers.get("Location", "")
        elif resp.status_code == 200:
            try:
                download_url = resp.json().get("download_url", "")
            except Exception:
                pass

        if not download_url:
            # Fallback
            if is_sediment:
                fallback_url = f"{BASE_URL}/files/{file_id}/download"
            else:
                fallback_url = f"{BASE_URL}/attachments/{file_id}"
            resp2 = await session.get(fallback_url, headers=headers, allow_redirects=False)
            if resp2.status_code in (301, 302, 303, 307, 308):
                download_url = resp2.headers.get("Location", "")
            elif resp2.status_code == 200:
                try:
                    download_url = resp2.json().get("download_url", "")
                except Exception:
                    pass

        # Also handle the estuary content URL pattern from asset_pointer
        if not download_url:
            # Check if the asset_pointer itself contains an estuary URL we can try
            log.warning(f"[conv] could not resolve download URL for file_id={file_id[:20]}...")
            return ""

        # Step 2: download the actual image bytes
        log.info(f"[conv] downloading image from: {download_url[:80]}...")
        try:
            # For estuary/chatgpt.com URLs, we need auth headers
            # For CDN URLs (S3 etc), no auth needed
            dl_headers = {}
            if "chatgpt.com" in download_url or "openai" in download_url:
                dl_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "User-Agent": WEB_USER_AGENT,
                    "OAI-Device-Id": device_id,
                }

            img_resp = await session.get(download_url, headers=dl_headers)
            if img_resp.status_code == 200:
                content_type = img_resp.headers.get("content-type", "image/png")
                # Normalize content type for data URI
                if "jpeg" in content_type or "jpg" in content_type:
                    mime = "image/jpeg"
                elif "webp" in content_type:
                    mime = "image/webp"
                else:
                    mime = "image/png"

                img_bytes = img_resp.content
                b64 = base64.b64encode(img_bytes).decode("ascii")
                data_uri = f"data:{mime};base64,{b64}"
                log.info(f"[conv] image downloaded: {len(img_bytes)} bytes, mime={mime}")
                return data_uri
            else:
                log.warning(f"[conv] image download failed: status={img_resp.status_code}")
                # Return the URL anyway as last resort - at least it has auth info
                return download_url
        except Exception as e:
            log.warning(f"[conv] image download error: {e}")
            return download_url if download_url else ""


def _message_signature(msg: dict) -> str:
    author_obj = msg.get("author") or {}
    author = author_obj.get("role", "?")
    status = msg.get("status", "?")
    content_obj = msg.get("content") or {}
    content_type = content_obj.get("content_type", "?")
    return f"{author}/{status}/{content_type}"


async def _extract_images_from_message(
    access_token: str,
    device_id: str,
    msg: dict,
    conversation_id: str,
    seen_ids: set[str],
) -> list[dict]:
    content = msg.get("content", {}) or {}
    content_type = content.get("content_type", "")
    if content_type not in ("multimodal", "multimodal_text"):
        return []

    images = []
    for raw_part in content.get("parts", []) or []:
        if not isinstance(raw_part, dict):
            continue
        if raw_part.get("content_type") != "image_asset_pointer":
            continue

        asset = raw_part.get("asset_pointer", "")
        file_id = _extract_file_id(asset)
        if not file_id or file_id in seen_ids:
            continue
        if not conversation_id:
            log.info(f"[conv] saw image asset before conversation_id was known: {asset[:80]}...")
            continue

        url = await _resolve_image_url(
            access_token,
            device_id,
            file_id,
            conversation_id,
            _is_sediment(asset),
        )
        if not url:
            log.warning(
                f"[conv] failed to resolve image url: file_id={file_id[:16]}... "
                f"conversation_id={conversation_id[:8]}..."
            )
            continue

        seen_ids.add(file_id)
        dalle_meta = raw_part.get("metadata", {}).get("dalle", {})
        images.append(
            {
                "url": url,
                "revised_prompt": dalle_meta.get("prompt", ""),
                "file_id": file_id,
                "gen_id": dalle_meta.get("gen_id", ""),
            }
        )

    return images


def _merge_images(base: list[dict], extra: list[dict]) -> list[dict]:
    seen = {(img.get("file_id"), img.get("url")) for img in base}
    for img in extra:
        key = (img.get("file_id"), img.get("url"))
        if key in seen:
            continue
        base.append(img)
        seen.add(key)
    return base


async def parse_conversation_sse(
    access_token: str, device_id: str,
    chunks: list[str], parent_msg_id: str = "",
) -> list[dict]:
    images = []
    seen_ids = set()
    conversation_id = ""
    events = []

    log.info(f"[conv-sse] parsing {len(chunks)} chunks")
    for chunk in chunks:
        if not chunk.startswith("data: "):
            continue
        data = chunk[6:].strip()
        if data == "[DONE]":
            break
        if not data.startswith("{"):
            continue

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        events.append(event)
        cid = event.get("conversation_id", "")
        if cid:
            conversation_id = cid

    for event in events:
        msg = event.get("message")
        if not msg:
            continue
        if msg.get("id") == parent_msg_id:
            continue

        author_role = msg.get("author", {}).get("role", "")
        if author_role in ("user", "system"):
            continue

        event_conversation_id = event.get("conversation_id") or conversation_id
        found = await _extract_images_from_message(
            access_token,
            device_id,
            msg,
            event_conversation_id,
            seen_ids,
        )
        if found:
            _merge_images(images, found)

    log.info(f"[conv-sse] found {len(images)} images")
    return images
async def _poll_conversation_for_images(access_token: str, device_id: str, conversation_id: str, parent_msg_id: str = "") -> list[dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": WEB_USER_AGENT,
        "OAI-Device-Id": device_id,
    }

    poll_max_wait = 120
    deadline = time.time() + poll_max_wait
    poll_attempt = 0
    seen_ids = set()

    async with curl_requests.AsyncSession(impersonate="chrome110") as session:
        while time.time() < deadline:
            poll_attempt += 1
            wait = 1 if poll_attempt == 1 else 3
            await asyncio.sleep(wait)
            log.info(f"[poll] attempt {poll_attempt} checking conversation {conversation_id}")
            resp = await session.get(f"{BASE_URL}/conversation/{conversation_id}", headers=headers)
            if resp.status_code != 200:
                err_text = resp.content.decode()[:200]
                log.warning(f"[poll] GET conversation returned {resp.status_code}: {err_text}")
                if resp.status_code in (401, 403):
                    raise Exception(f"Poll auth error: {resp.status_code}")
                continue

            try:
                conv = resp.json()
            except Exception as e:
                log.warning(f"[poll] decode error: {e}")
                continue

            mapping = conv.get("mapping", {})
            images = []
            refusal_text = ""
            error_found = False

            for node_id, node in mapping.items():
                msg = node.get("message")
                if not msg:
                    continue
                if parent_msg_id and msg.get("id") == parent_msg_id:
                    continue

                author_role = msg.get("author", {}).get("role", "")
                if author_role in ("user", "system"):
                    continue

                # Detect policy refusal in assistant messages
                if author_role == "assistant":
                    content = msg.get("content") or {}
                    if msg.get("status") == "finished_successfully" and content.get("content_type") == "text":
                        parts = content.get("parts", [])
                        if parts and isinstance(parts[0], str):
                            text_content = parts[0]
                            lower_text = text_content.lower()
                            refusal_keywords = ["content polic", "violat", "got it wrong", "sorry", "can't create", "cannot create", "unable to generate", "inappropriate"]
                            if any(kw in lower_text for kw in refusal_keywords):
                                refusal_text = text_content
                                log.warning(f"[poll] policy refusal detected: {text_content[:200]}")
                    msg_meta = msg.get("metadata") or {}
                    if msg_meta.get("is_blocked") or msg_meta.get("flagged"):
                        error_found = True
                        log.warning(f"[poll] message flagged/blocked in metadata")

                # Still try to extract images
                found = await _extract_images_from_message(
                    access_token,
                    device_id,
                    msg,
                    conversation_id,
                    seen_ids,
                )
                if found:
                    _merge_images(images, found)

            if images:
                log.info(f"[poll] found {len(images)} images after {poll_attempt} attempts")
                return images

            if refusal_text:
                raise Exception(f"Content policy refusal: {refusal_text[:300]}")

            if error_found:
                raise Exception("Image generation blocked by content policy")

            log.info(f"[poll] attempt {poll_attempt}: no images yet, continuing...")

    raise Exception("Timed out waiting for async image generation (120s)")

# ══════════════════════════════════════════════════════════════════════════
#  Image generation core

# ══════════════════════════════════════════════════════════════════════════

async def _handle_image_via_conversation(
    prompt: str, model: str, n: int,
    size: str, quality: str, background: str, response_format: str,
    input_images: list = None,
) -> dict:
    full_prompt = prompt
    if size and size not in ("auto", "1024x1024"):
        full_prompt = f"Generate an image with size {size}. {prompt}"
    if quality in ("hd", "high"):
        full_prompt = f"Generate a high-quality, detailed image: {full_prompt}"
    if background == "transparent":
        full_prompt += " The image must have a transparent background (PNG with alpha channel)."

    access_token = await token_manager.get_valid_token()
    device_id = token_manager.device_id
    chat_token, proof_token = await get_sentinel_tokens(access_token, device_id)

    # Force gpt-4o for images if auto, as it has better DALL-E 3 support
    if model == "auto":
        # Use gpt-5.4-mini as it is the current default for paid in image-studio
        # but for free accounts "auto" is usually better. 
        # Let is stay "auto" for now to match image-studio free route.
        pass
    # Initial body with default encodings
    body = build_conversation_body(full_prompt, model=model)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": WEB_USER_AGENT,
        "OAI-Device-Id": device_id,
        "OAI-Language": "en-US",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://chatgpt.com",
        "Referer": "https://chatgpt.com/",
        "Priority": "u=1, i",
        "Sec-CH-UA": '"Chromium";v="146", "Google Chrome";v="146", "Not?A_Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"macOS"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "openai-sentinel-chat-requirements-token": chat_token,
    }
    if proof_token:
        headers["openai-sentinel-proof-token"] = proof_token

    msg_id = body["messages"][0]["id"]
    for path in ("/f/conversation", "/conversation"):
        route_label = path.split("/")[-1]
        
        # Match ChatGpt-Image-Studio behavior for /f/conversation
        if path == "/f/conversation":
            body["client_prepare_state"] = "none"
            body["supported_encodings"] = ["v1"]
        else:
            body.pop("client_prepare_state", None)
            body["supported_encodings"] = []

        log.info(f"[conv] POST {BASE_URL}{path}")
        try:
            async with curl_requests.AsyncSession(impersonate="chrome110") as session:
                resp = await session.post(
                    f"{BASE_URL}{path}",
                    json=body, headers=headers, stream=True, timeout=300,
                )
                if resp.status_code != 200:
                    err_body = resp.content
                    log.warning(f"[conv] {route_label} returned {resp.status_code}: {err_body[:512]}")
                    if resp.status_code in (403, 404) and path == "/f/conversation":
                        continue
                    raise Exception(f"{route_label} returned {resp.status_code}")

                images = []
                seen_ids = set()
                chunks = []
                chunk_count = 0
                conversation_id = ""
                observed_signatures = []
                async_mode = False

                async for line in resp.aiter_lines():
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    if decoded:
                        chunks.append(decoded)
                    if not decoded.startswith("data: "):
                        continue

                    data_str = decoded[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if not data_str.startswith("{"):
                        continue

                    chunk_count += 1
                    try:
                        event = json.loads(data_str)
                    except Exception as e:
                        log.warning(f"[conv] json decode error: {e}, data: {data_str[:200]}")
                        continue

                    cid = event.get("conversation_id", "")
                    if cid:
                        conversation_id = cid

                    # Detect top-level error (e.g. rate limits, ban, policy)
                    err_val = event.get("error")
                    if not err_val and "v" in event and isinstance(event["v"], dict):
                        err_val = event["v"].get("error")
                    if err_val:
                        err_msg = err_val if isinstance(err_val, str) else json.dumps(err_val)
                        log.warning(f"[conv] Upstream error in stream: {err_msg}")
                        raise Exception(f"Upstream error: {err_msg}")

                    async_status = event.get("async_status")
                    if async_status and isinstance(async_status, int) and async_status > 0:
                        async_mode = True
                        log.info(f"[conv] async_status={async_status}, will poll after stream")

                    # Detect moderation blocking during stream
                    if event.get("moderation_state") == "blocked":
                        log.warning("[conv] moderation blocked in stream")
                        raise Exception("Content policy violation: moderation blocked")

                    msg = event.get("message")
                    if not msg and "v" in event and isinstance(event["v"], dict):
                        msg = event["v"].get("message")
                    if not msg:
                        continue

                    signature = _message_signature(msg)
                    if signature not in observed_signatures and len(observed_signatures) < 12:
                        observed_signatures.append(signature)

                    if chunk_count <= 3 or chunk_count % 10 == 0:
                        log.info(f"[conv] chunk={chunk_count} sig={signature} images={len(images)}")

                    if msg.get("author", {}).get("role", "") in ("user", "system"):
                        continue

                    found = await _extract_images_from_message(
                        access_token,
                        device_id,
                        msg,
                        cid or conversation_id,
                        seen_ids,
                    )
                    if found:
                        _merge_images(images, found)
                        log.info(f"[conv] extracted {len(found)} image(s) from live stream, total={len(images)}")

                if len(images) < n:
                    reparsed = await parse_conversation_sse(access_token, device_id, chunks)
                    if reparsed:
                        before = len(images)
                        _merge_images(images, reparsed)
                        if len(images) > before:
                            log.info(f"[conv] recovered {len(images) - before} image(s) via batch reparse")

                log.info(f"[conv] total: {chunk_count} chunks, {len(images)} images")
                
                if images:
                    return _build_images_response(images[:n], response_format, text=assistant_text)
                
                if async_mode and conversation_id:
                    log.info(f"[conv] async polling for conversation {conversation_id}")
                    images = await _poll_conversation_for_images(access_token, device_id, conversation_id, parent_msg_id=msg_id)
                    if images:
                        return _build_images_response(images[:n], response_format, text=assistant_text)

                log.warning(f"[conv] {route_label} no images found; observed signatures={observed_signatures}")
                
                # When no images found, try to extract useful info from the stream
                # 1) Look for assistant text (rate limit message, refusal, etc.)
                # 2) Look for message events of ANY role to understand stream format
                # 3) Dump raw events for debugging if nothing matched
                assistant_text = ""
                all_msg_signatures = []
                raw_event_types = set()
                raw_event_keys = set()
                error_events = []
                
                for chunk in chunks:
                    if not chunk.startswith("data: "):
                        continue
                    data_str = chunk[6:].strip()
                    if not data_str.startswith("{"):
                        continue
                    try:
                        evt = json.loads(data_str)
                    except Exception:
                        continue
                    
                    # Collect top-level keys and types for diagnostics
                    evt_type = evt.get("type", "")
                    if evt_type:
                        raw_event_types.add(evt_type)
                    for k in evt.keys():
                        raw_event_keys.add(k)
                    
                    # Check for top-level error
                    if evt.get("error"):
                        error_events.append(evt["error"] if isinstance(evt["error"], str) else json.dumps(evt["error"]))
                    
                    # Check for moderation_state / policy rejection
                    if evt.get("moderation_state") == "blocked":
                        error_events.append("moderation_blocked")
                    
                    m = evt.get("message")
                    if not m and "v" in evt and isinstance(evt["v"], dict):
                        m = evt["v"].get("message")
                    if not m:
                        continue
                    
                    # Record ALL message signatures (not just non-user ones)
                    sig = _message_signature(m)
                    all_msg_signatures.append(f"{sig}(role={m.get('author',{}).get('role','?')})")
                    
                    # Extract assistant text
                    if (m.get("author") or {}).get("role") == "assistant":
                        c = m.get("content") or {}
                        parts = c.get("parts", [])
                        for part in parts:
                            if isinstance(part, str):
                                assistant_text += part
                            elif isinstance(part, dict) and "text" in part:
                                assistant_text += str(part["text"])
                            elif isinstance(part, dict):
                                assistant_text += f"[{part.get('content_type', 'unknown')}]"
                
                # Report what we found
                if all_msg_signatures:
                    log.warning(f"[conv] ALL msg signatures: {all_msg_signatures[:20]}")
                else:
                    log.warning(f"[conv] ZERO message events in stream; top-level keys={raw_event_keys}")
                    if raw_event_types:
                        log.warning(f"[conv] event types seen: {raw_event_types}")
                    # Dump first 3 raw data events for diagnosis
                    data_events = [c[6:].strip() for c in chunks if c.startswith("data: ") and c[6:].strip().startswith("{")]
                    for i, de in enumerate(data_events[:3]):
                        log.warning(f"[conv] raw event[{i}]: {de[:500]}")
                
                if error_events:
                    err_summary = "; ".join(str(e)[:200] for e in error_events[:3])
                    log.warning(f"[conv] upstream errors: {err_summary}")
                    raise Exception(f"Upstream error: {err_summary}")
                
                if assistant_text:
                    log.warning(f"[conv] assistant text: {assistant_text[:200]}")
                    raise Exception(f"Assistant response: {assistant_text[:500]}")

                if path == "/conversation":
                    raise Exception("No images in response")
                raise Exception("No images in response")

        except Exception as e:
            if path == "/conversation":
                raise
            log.warning(f"[conv] {route_label} failed: {e}, trying /conversation")
            continue



async def _stream_image_via_conversation(
    prompt: str, model: str, n: int,
    size: str, quality: str, background: str,
    input_images: list = None,
) -> StreamingResponse:
    """Call image generation and stream text + images as OpenAI format."""
    
    async def generate():
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        # We'll just call the non-streaming handler for now but wrap its result in a stream
        # To do true streaming of images, we'd need to refactor _handle_image_via_conversation
        # but let's at least return the combined result as a stream to satisfy the client.
        try:
            img_resp = await _handle_image_via_conversation(
                prompt=prompt, model=model, n=n,
                size=size, quality=quality,
                background=background, response_format="url",
                input_images=input_images,
            )
            
            content = img_resp.get("text", "").strip()
            markdown_parts = []
            for img in img_resp.get("data", []):
                url = img.get("url", "")
                if url:
                    markdown_parts.append(f"![image]({url})")
            
            img_markdown = "\n\n".join(markdown_parts)
            if img_markdown:
                content = (content + "\n\n" + img_markdown).strip()
            
            if content:
                # Yield the full content as one chunk (or we could split it)
                chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield "data: " + json.dumps(chunk) + "\n\n"
            
            # Send stop
            stop_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield "data: " + json.dumps(stop_chunk) + "\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            log.error(f"[stream-img] error: {e}")
            yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _build_images_response(images: list[dict], response_format: str, text: str = "") -> dict:
    data = []
    for img in images:
        item = {"revised_prompt": img.get("revised_prompt", "")}
        url = img.get("url", "")
        if response_format == "b64_json":
            # Extract base64 from data URI, or return empty if it's a regular URL
            if url.startswith("data:"):
                # data:image/png;base64,iVBOR...
                b64_part = url.split(",", 1)[1] if "," in url else ""
                item["b64_json"] = b64_part
                item["url"] = ""
            else:
                item["b64_json"] = ""
                item["url"] = url
        else:
            item["url"] = url
        data.append(item)
    return {"created": int(time.time()), "data": data, "text": text}


# ══════════════════════════════════════════════════════════════════════════
#  Route: POST /v1/images/generations
# ══════════════════════════════════════════════════════════════════════════

class ImageGenRequest(BaseModel):
    model: str = "gpt-image-2"
    prompt: str
    n: int = 1
    size: str = "auto"
    quality: str = "auto"
    background: str = "auto"
    response_format: str = "url"


@app.post("/v1/images/generations")
async def images_generations(req: ImageGenRequest):
    log.info(f"[images] model={req.model} size={req.size} prompt={req.prompt[:80]}...")
    try:
        return await _handle_image_via_conversation(
            prompt=req.prompt, model=req.model, n=req.n,
            size=req.size, quality=req.quality,
            background=req.background, response_format=req.response_format,
        )
    except Exception as e:
        log.error(f"[images] error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════
#  Route: POST /v1/chat/completions
# ══════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: list[ChatMessage]
    stream: bool = False
    n: int = 1
    size: str = "auto"
    quality: str = "auto"
    background: str = "auto"


def _extract_prompt_and_images(messages: list[ChatMessage]) -> tuple[str, list]:
    parts = []
    images = []
    for msg in messages:
        if msg.role in ("system", "assistant", "tool"):
            continue
        content = msg.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                t = item.get("type")
                if t == "text":
                    parts.append(item.get("text", ""))
                elif t == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url:
                        images.append({"url": url})
    return "\n\n".join(parts), images


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model = req.model.lower()

    # ── Image generation mode ────────────────────────────────────────────
    if model in IMAGE_MODELS:
        prompt, input_images = _extract_prompt_and_images(req.messages)
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")

        log.info(f"[chat] image mode, model={model}, stream={req.stream}")
        
        if req.stream:
            return await _stream_image_via_conversation(
                prompt=prompt, model=model, n=req.n,
                size=req.size, quality=req.quality,
                background=req.background,
                input_images=input_images,
            )
        try:
            img_resp = await _handle_image_via_conversation(
                prompt=prompt, model=model, n=req.n,
                size=req.size, quality=req.quality,
                background=req.background, response_format="url",
                input_images=input_images,
            )
            markdown_parts = []
            for img in img_resp.get("data", []):
                url = img.get("url", "")
                if url:
                    markdown_parts.append(f"![image]({url})")
            
            assistant_text = img_resp.get("text", "").strip()
            img_markdown = "\n\n".join(markdown_parts)
            
            content = assistant_text
            if img_markdown:
                content = (content + "\n\n" + img_markdown).strip()
            
            if not content:
                content = "Image generation failed."

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "images": img_resp.get("data", [])},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        except Exception as e:
            log.error(f"[chat] image error: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail=str(e))

    # ── Standard text conversation → /conversation (web endpoint) ──────
    log.info(f"[chat] text mode via /conversation, model={model}, stream={req.stream}")
    
    if req.stream:
        return await _stream_text_via_conversation(payload={}, headers={}, model=model, messages=[{"role": m.role, "content": m.content} for m in req.messages])
    else:
        # Non-streaming: collect full text from stream
        response_stream = await _stream_text_via_conversation(payload={}, headers={}, model=model, messages=[{"role": m.role, "content": m.content} for m in req.messages])
        # TODO: implement non-streaming properly
        raise HTTPException(status_code=501, detail="Non-streaming not yet supported for text mode")


async def _stream_text_via_conversation(payload: dict, headers: dict, model: str, messages: list) -> StreamingResponse:
    """Call /conversation endpoint for text chat and stream as OpenAI format."""
    access_token = await token_manager.get_valid_token()
    device_id = token_manager.device_id
    chat_token, proof_token = await get_sentinel_tokens(access_token, device_id)
    
    body = build_text_conversation_body(messages, model)
    
    req_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": WEB_USER_AGENT,
        "oai-device-id": device_id,
        "Accept": "text/event-stream",
        "openai-sentinel-chat-requirements-token": chat_token,
    }
    if proof_token:
        req_headers["openai-sentinel-proof-token"] = proof_token

    async def generate():
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        for path in ("/f/conversation", "/conversation"):
            route = path.split("/")[-1]
            log.info(f"[text-conv] trying {path}")
            try:
                async with curl_requests.AsyncSession(impersonate="chrome110") as session:
                    resp = await session.post(
                        f"{BASE_URL}{path}",
                        json=body, headers=req_headers, stream=True, timeout=300,
                    )
                    if resp.status_code != 200:
                        log.warning(f"[text-conv] {route} returned {resp.status_code}")
                        if resp.status_code in (403, 404) and path == "/f/conversation":
                            continue
                        yield f'data: {json.dumps({"error": {"message": f"Backend {{resp.status_code}}"}})}' + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    buffer = ""
                    async for line in resp.aiter_lines():
                        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                        if not decoded.startswith("data: "):
                            continue
                        data_str = decoded[6:].strip()
                        if data_str == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        if not data_str.startswith("{"):
                            continue
                        try:
                            event = json.loads(data_str)
                        except:
                            continue
                        
                        msg = event.get("message")
                        if not msg:
                            continue
                        if msg.get("author", {}).get("role") in ("user", "system"):
                            continue
                        
                        content = msg.get("content", {})
                        if content.get("content_type") != "text":
                            continue
                        
                        text_parts = content.get("parts", [])
                        for text in text_parts:
                            if text and isinstance(text, str):
                                chunk = {
                                    "id": cmpl_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                }
                                yield "data: " + json.dumps(chunk) + "\n\n"
                    
                    # If we got here without [DONE], send it
                    yield "data: [DONE]\n\n"
                    return
            except Exception as e:
                log.error(f"[text-conv] {route} error: {e}")
                if path == "/conversation":
                    yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                continue
        
        yield "data: " + json.dumps({"error": {"message": "All conversation endpoints failed"}}) + "\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


async def _stream_codex_response_for_chat_completions(payload: dict, headers: dict, model: str) -> StreamingResponse:
    # 专门为标准 API (chat/completions) 转译 Codex 流
    async def generate():
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        log.info("[chat/completions] Starting stream to codex/responses...")
        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            try:
                log.info(f'[chat/completions] POST {CODEX_BASE_URL}/responses')
                log.info(f'[chat/completions] Headers: { {k: (v[:60]+"..." if isinstance(v,str) and len(v)>60 else v) for k,v in headers.items()} }')
                resp = await session.post(
                    f"{CODEX_BASE_URL}/responses",
                    json=payload, headers=headers, stream=True, timeout=600,
                )
                resp_headers = dict(resp.headers)
                log.info(f"[chat/completions] codex/responses status={resp.status_code}, headers={resp_headers}")
                
                if resp.status_code != 200:
                    err_text = "(could not read body)"
                    try:
                        err_text = resp.text if hasattr(resp, 'text') else str(resp.content)
                    except Exception:
                        try:
                            err_text = resp.content.decode("utf-8", errors="replace")
                        except Exception:
                            pass
                    log.error(f"[chat/completions] Error {resp.status_code}: {err_text}")
                    yield f"data: {json.dumps({'error': {'message': f'Backend {resp.status_code}: {err_text}'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                    yield f"data: {json.dumps({'error': {'message': f'Backend {resp.status_code}: {err_text}'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                chunk_count = 0
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    
                    if not decoded.startswith("data: "):
                        if decoded.strip():
                            log.info(f"[chat/completions] Ignoring non-data line: {decoded[:50]}...")
                        continue
                        
                    data_str = decoded[6:].strip()
                    if data_str == "[DONE]":
                        log.info(f"[chat/completions] Received [DONE] from backend after {chunk_count} chunks.")
                        yield "data: [DONE]\n\n"
                        break
                        
                    try:
                        event = json.loads(data_str)
                    except Exception as e:
                        log.warning(f"[chat/completions] Failed to parse JSON chunk: {e}. Raw: {data_str[:50]}...")
                        continue
                    
                    # 从 Codex 的 event 中提取 output_text
                    outputs = event.get("output", [])
                    has_output = False
                    for item in outputs:
                        if item.get("type") == "message":
                            for part in item.get("content", []):
                                if part.get("type") == "output_text":
                                    text = part.get("text", "")
                                    if text:
                                        has_output = True
                                        chunk_count += 1
                                        if chunk_count <= 3:
                                            log.info(f"[chat/completions] Emitting delta text: {text[:20]}...")
                                        chunk = {
                                            "id": cmpl_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model,
                                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                                        }
                                        yield f"data: {json.dumps(chunk)}" + "\n\n"
                    
                    if not has_output and chunk_count < 2:
                        log.info(f"[chat/completions] Ignored event (no output_text): {data_str[:80]}...")
                        
            except Exception as e:
                log.error(f"[chat/completions] Streaming error: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': {'message': f'Proxy Stream Error: {str(e)}'}})}\n\n"
                yield "data: [DONE]\n\n"
                                    
    return StreamingResponse(generate(), media_type="text/event-stream")

async def _stream_codex_response(payload: dict, headers: dict) -> StreamingResponse:
    async def generate():
        log.info("[responses] Starting direct proxy stream to codex/responses...")
        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            try:
                log.info(f'[responses] POST {CODEX_BASE_URL}/responses')
                log.info(f'[responses] Headers: { {k: (v[:60]+"..." if isinstance(v,str) and len(v)>60 else v) for k,v in headers.items()} }')
                resp = await session.post(
                    f"{CODEX_BASE_URL}/responses",
                    json=payload, headers=headers, stream=True, timeout=600,
                )
                resp_headers = dict(resp.headers)
                log.info(f"[responses] codex/responses status={resp.status_code}, headers={resp_headers}")
                
                if resp.status_code != 200:
                    err_text = "(could not read body)"
                    try:
                        err_text = resp.text if hasattr(resp, 'text') else str(resp.content)
                    except Exception:
                        try:
                            err_text = resp.content.decode("utf-8", errors="replace")
                        except Exception:
                            pass
                    log.error(f"[responses] Error {resp.status_code}: {err_text}")
                    yield f"data: {json.dumps({'error': {'message': f'Backend {resp.status_code}: {err_text}'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                    yield f"data: {json.dumps({'error': {'message': f'Backend {resp.status_code}: {err_text}'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                chunk_count = 0
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    
                    if decoded.strip():
                        chunk_count += 1
                        if chunk_count <= 3:
                            log.info(f"[responses] Got chunk #{chunk_count}: {decoded[:80]}...")
                            
                    yield f"{decoded}\n\n"
                    if decoded.strip() == "data: [DONE]":
                        log.info(f"[responses] Received [DONE] from backend after {chunk_count} chunks.")
                        break
            except Exception as e:
                log.error(f"[responses] Streaming error: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': {'message': f'Proxy Stream Error: {str(e)}'}})}\n\n"
                yield "data: [DONE]\n\n"
                
    return StreamingResponse(generate(), media_type="text/event-stream")


async def _non_stream_codex_response(payload: dict, headers: dict, model: str) -> dict:
    async with curl_requests.AsyncSession(impersonate="chrome110") as session:
        resp = await session.post(
            f"{CODEX_BASE_URL}/responses",
            json=payload, headers=headers, timeout=600,
        )
        if resp.status_code != 200:
            err = resp.content
            raise Exception(f"Codex returned {resp.status_code}: {err.decode()[:500]}")
        data = resp.json()

    output_text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    output_text += part.get("text", "")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ══════════════════════════════════════════════════════════════════════════
#  Route: POST /v1/responses  (Codex passthrough)
# ══════════════════════════════════════════════════════════════════════════


def normalize_codex_payload(payload: dict) -> dict:
    """
    借鉴 Codaze 的请求清洗逻辑:
    防止第三方客户端发送的非标/内部不支持的字段触发封号或报错
    """
    # 1. 移除 Codex 不支持或容易触发风控的生成参数
    for key in ["max_output_tokens", "max_completion_tokens", "temperature", 
                "top_p", "truncation", "user", "presence_penalty", "frequency_penalty"]:
        payload.pop(key, None)
        
    # 2. service_tier 仅允许 priority
    if payload.get("service_tier") != "priority":
        payload.pop("service_tier", None)
        
    # 3. 强制基础配置
    payload["store"] = False
    if payload.get("instructions") is None:
        payload["instructions"] = ""
        
    # 4. 工具兼容与别名转换
    tools = payload.get("tools", [])
    for tool in tools:
        t_type = tool.get("type", "")
        # 将过时的联网工具名清洗为最新标准
        if t_type in ["web_search_preview", "web_search_preview_2025_03_11"]:
            tool["type"] = "web_search"
            
    return payload

@app.post("/v1/responses")

async def proxy_codex_responses(request: Request):
    payload = await request.json()
    access_token = await token_manager.get_valid_token()

    tools = payload.get("tools", [])
    has_image_tool = any(t.get("type", "").lower() == "image_generation" for t in tools)

    headers = build_codex_headers(access_token, token_manager.account_id, token_manager.installation_id)
    # UA now set to Chrome in build_codex_headers
    # Originator now set in build_codex_headers
    headers["session_id"] = str(uuid.uuid4())

    payload = normalize_codex_request(payload)
    log.info(f"[responses] has_image_tool={has_image_tool}, stream={payload.get('stream', False)}")

    if payload.get("stream"):
        return await _stream_codex_response(payload, headers)
    else:
        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            resp = await session.post(
                f"{CODEX_BASE_URL}/responses",
                json=payload, headers=headers, timeout=600,
            )
            if resp.status_code != 200:
                err_text = resp.content.decode()[:500]
                return JSONResponse(
                    status_code=resp.status_code,
                    content={
                        "error": {
                            "message": f"Upstream Codex API Error: {err_text}",
                            "type": "upstream_error",
                            "code": resp.status_code
                        }
                    }
                )
            return resp.json()


# ══════════════════════════════════════════════════════════════════════════
#  前端管理页面 & Session 管理 API
# ══════════════════════════════════════════════════════════════════════════

MANAGER_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChatGPT Session Proxy</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#1a1a2e;color:#e0e0e0;min-height:100vh;padding:1.5rem}
.wrap{max-width:800px;margin:0 auto}
h1{font-size:1.6rem;color:#00d4ff;margin-bottom:.3rem}
.sub{color:#888;font-size:.85rem;margin-bottom:1.5rem}
.card{background:#16213e;border:1px solid #0f3460;border-radius:10px;padding:1.2rem;margin-bottom:1rem}
.card h2{font-size:1rem;color:#e94560;margin-bottom:.8rem}
input,textarea{width:100%;background:#0a0a1a;color:#e0e0e0;border:1px solid #333;border-radius:6px;padding:.65rem .8rem;font-family:'Fira Code',monospace;font-size:.85rem}
input:focus,textarea:focus{outline:none;border-color:#00d4ff}
textarea{height:160px;resize:vertical}
button{background:#e94560;color:#fff;border:none;border-radius:6px;padding:.6rem 1.5rem;font-size:.9rem;cursor:pointer;transition:.2s}
button:hover{background:#ff6b6b}
button.sm{padding:.4rem .9rem;font-size:.8rem;margin:0}
button.ghost{background:transparent;border:1px solid #444;color:#aaa}
button.ghost:hover{border-color:#888;color:#fff}
.row{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}
.msg{padding:.8rem;border-radius:6px;margin-top:.8rem;font-size:.85rem}
.msg.ok{background:#0d3320;border:1px solid #00c853;color:#69f0ae}
.msg.err{background:#3d0000;border:1px solid #ff1744;color:#ff8a80}
.msg.info{background:#0d1b3e;border:1px solid #2979ff;color:#82b1ff}
.tbl{width:100%;border-collapse:collapse;margin-top:.5rem;font-size:.85rem}
.tbl th{text-align:left;color:#888;font-weight:600;padding:.5rem .6rem;border-bottom:1px solid #222}
.tbl td{padding:.5rem .6rem;border-bottom:1px solid #111;vertical-align:middle}
.tbl tr:hover{background:rgba(255,255,255,.03)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot.green{background:#00c853}.dot.red{background:#ff1744}
.dot.yellow{background:#ffd600}.dot.gray{background:#555}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.75rem}
.badge.ok{background:#0d3320;color:#69f0ae}
.badge.err{background:#3d0000;color:#ff8a80}
.badge.off{background:#222;color:#888}
#loginBox{display:flex;align-items:center;justify-content:center;min-height:70vh}
#loginBox .card{max-width:360px;width:100%;text-align:center}
#mainUI{display:none}
.hdr{position:relative}
.logout{position:absolute;top:0;right:0;background:transparent;border:1px solid #333;color:#888;font-size:.8rem;padding:.3rem .8rem;cursor:pointer;border-radius:4px}
.logout:hover{color:#ff6b6b;border-color:#e94560}
.grid2{display:grid;grid-template-columns:auto 1fr;gap:.3rem .8rem;font-size:.85rem}
.grid2 span:first-child{color:#888;font-weight:600}
.hint{color:#666;font-size:.8rem;margin-top:.3rem}
a{color:#00d4ff}
.actions button{margin-right:.4rem}
@media(max-width:600px){.tbl{font-size:.78rem}.tbl td,.tbl th{padding:.4rem}}
</style>
</head>
<body>
<div class="wrap">
<div id="loginBox">
<div class="card">
<h2>LOCK Management Panel</h2>
<p style="color:#888;font-size:.85rem;margin-bottom:1rem">Enter API Key to login</p>
<input type="password" id="loginKey" placeholder="API Key" onkeydown="if(event.key==='Enter')doLogin()">
<button onclick="doLogin()" style="width:100%;margin-top:.8rem">Login</button>
<div id="loginMsg"></div>
</div>
</div>
<div id="mainUI">
<div class="hdr">
<h1>ChatGPT Session Proxy</h1>
<p class="sub">Session Pool / Round-Robin / Auto-Refresh</p>
<button class="logout" onclick="doLogout()">Logout</button>
</div>
<div class="card">
<div class="grid2">
<span>Device ID</span><span id="devId" style="word-break:break-all">-</span>
<span>Sessions</span><span id="sessTotal">-</span>
<span>Healthy</span><span id="sessHealthy">-</span>
</div>
</div>
<div class="card">
<h2>Session List</h2>
<div id="sessTable"><p style="color:#888">Loading...</p></div>
<div class="row" style="margin-top:.8rem">
<button class="sm ghost" onclick="loadStatus()">Refresh</button>
</div>
</div>
<div class="card">
<h2>Add Session</h2>
<p class="hint" style="margin-bottom:.6rem">On <a href="https://chatgpt.com" target="_blank">chatgpt.com</a> console run
<code style="background:#1a1a3e;padding:1px 5px;border-radius:3px;color:#00d4ff;font-size:.8rem">await fetch('/api/auth/session').then(r=>r.json()).then(j=>copy(JSON.stringify(j)))</code></p>
<textarea id="newSess" placeholder='{"accessToken":"eyJ...","sessionToken":"eyJ...","account":{"id":"..."},...}'></textarea>
<div class="row" style="margin-top:.6rem">
<button class="sm" onclick="addSession()">Add</button>
</div>
<div id="addMsg"></div>
</div>
</div>
</div>
<script>
(function() {
  const K = '_pkey';
  const getKey = () => sessionStorage.getItem(K) || '';
  const setKey = (v) => sessionStorage.setItem(K, v);
  const clearKey = () => sessionStorage.removeItem(K);
  const hdrs = () => ({
    'Authorization': 'Bearer ' + getKey(),
    'Content-Type': 'application/json'
  });

  window.doLogin = async function() {
    const k = document.getElementById('loginKey').value.trim();
    const el = document.getElementById('loginMsg');
    if (!k) {
      el.innerHTML = '<div class="msg err">Please enter key</div>';
      return;
    }
    el.innerHTML = '<div class="msg info">Logging in...</div>';
    try {
      const r = await fetch('/auth/login-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: k })
      });
      if (r.status === 401) {
        el.innerHTML = '<div class="msg err">Invalid API Key</div>';
        return;
      }
      if (!r.ok) {
        const txt = await r.text();
        el.innerHTML = `<div class="msg err">Error: ${r.status} ${txt}</div>`;
        return;
      }
      setKey(k);
      showMain();
    } catch (e) {
      el.innerHTML = `<div class="msg err">Connection failed: ${e.message}</div>`;
      console.error(e);
    }
  };

  window.doLogout = function() {
    clearKey();
    location.reload();
  };

  function showMain() {
    document.getElementById('loginBox').style.display = 'none';
    document.getElementById('mainUI').style.display = 'block';
    loadStatus();
  }

  window.loadStatus = async function() {
    const el = document.getElementById('sessTable');
    try {
      const r = await fetch('/auth/status', { headers: hdrs() });
      if (r.status === 401) { doLogout(); return; }
      const d = await r.json();
      document.getElementById('devId').textContent = d.device_id || '-';
      document.getElementById('sessTotal').textContent = d.total || 0;
      document.getElementById('sessHealthy').textContent = d.healthy || 0;
      
      if (!d.sessions || !d.sessions.length) {
        el.innerHTML = '<p style="color:#888;padding:1rem">No sessions in pool. Add one below.</p>';
        return;
      }

      let h = '<table class="tbl"><tr><th>Status</th><th>SID</th><th>Account / Email</th><th>Expires</th><th>Error</th><th>Actions</th></tr>';
      for (const s of d.sessions) {
        const dis = s.disabled, exp = s.is_expired, hlt = s.is_healthy && !dis;
        let dot, txt, bCls;
        if (dis) { dot = 'gray'; txt = 'Disabled'; bCls = 'off'; }
        else if (!hlt) { dot = 'red'; txt = 'Error'; bCls = 'err'; }
        else if (exp) { dot = 'yellow'; txt = 'Expired'; bCls = 'err'; }
        else { dot = 'green'; txt = 'OK'; bCls = 'ok'; }

        const ex = s.expires_at ? new Date(s.expires_at * 1000).toLocaleTimeString() : '-';
        const er = s.last_error ? s.last_error.substring(0, 30) : '-';
        const tb = dis 
          ? `<button class="sm ghost" onclick="togS('${s.sid}',false)">Enable</button>`
          : `<button class="sm ghost" onclick="togS('${s.sid}',true)">Disable</button>`;
        
        h += `<tr>
          <td><span class="dot ${dot}"></span><span class="badge ${bCls}">${txt}</span></td>
          <td><code>${s.sid}</code></td>
          <td><div>${s.account_id ? s.account_id.substring(0, 8) + "..." : "-"}</div><div style="font-size:0.75rem;color:#888">${s.email || ""}</div></td>
          <td>${ex}</td>
          <td title="${s.last_error || ''}">${er}</td>
          <td class="actions">${tb}<button class="sm ghost" onclick="dlS('${s.sid}')">DL</button><button class="sm ghost" onclick="rmS('${s.sid}')">Del</button></td>
        </tr>`;
      }
      h += '</table>';
      h += '<div style="margin-top:12px;text-align:right"><button class="sm ghost" onclick="dlAll()">📦 Download All (ZIP)</button></div>';
      el.innerHTML = h;
    } catch (e) {
      el.innerHTML = `<div class="msg err">Failed to load status: ${e.message}</div>`;
    }
  };

  window.addSession = async function() {
    const v = document.getElementById('newSess').value.trim();
    const el = document.getElementById('addMsg');
    if (!v) { el.innerHTML = '<div class="msg err">Please paste JSON</div>'; return; }
    try {
      const r = await fetch('/auth/session', { method: 'POST', headers: hdrs(), body: v });
      if (r.status === 401) { doLogout(); return; }
      const d = await r.json();
      if (r.ok) {
        el.innerHTML = `<div class="msg ok">${d.message}</div>`;
        document.getElementById('newSess').value = '';
        loadStatus();
      } else {
        el.innerHTML = `<div class="msg err">${d.detail || JSON.stringify(d)}</div>`;
      }
    } catch (e) { el.innerHTML = `<div class="msg err">${e.message}</div>`; }
  };

  window.rmS = async function(sid) {
    if (!confirm(`Delete session ${sid}?`)) return;
    await fetch(`/auth/session/${sid}/remove`, { method: 'POST', headers: hdrs() });
    loadStatus();
  };

  
  window.dlS = function(sid) {
    window.open(`/auth/session/${sid}/download`, '_blank');
  };

  window.dlAll = function() {
    window.open('/auth/sessions/download', '_blank');
  };

  window.togS = async function(sid, dis) {
    await fetch(`/auth/session/${sid}/toggle`, { 
      method: 'POST', 
      headers: hdrs(), 
      body: JSON.stringify({ disabled: dis }) 
    });
    loadStatus();
  };

  // Init
  if (getKey()) showMain();
})();
</script>
</body>
</html>
"""


@app.get("/")
async def manager_page():
    return HTMLResponse(MANAGER_HTML)


@app.post("/auth/session")
async def update_session(request: Request):
    """接收并保存 session JSON"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON")

    # 验证必须字段
    if "accessToken" not in body and "sessionToken" not in body:
        raise HTTPException(status_code=400, detail="JSON 中缺少 accessToken 和 sessionToken")

    result = token_manager.load_session_from_json(body)
    log.info(f"[auth] Session 已更新: account_id={token_manager.account_id[:8]}...")
    return result


@app.post("/auth/login-check")
async def login_check(request: Request):
    """Validate API key without exposing session data. Used by frontend login."""
    body = await request.json()
    key = body.get("key", "")
    # If no API_KEY configured, any non-empty key is accepted (dev mode)
    if not API_KEYS:
        if not key:
            return JSONResponse(status_code=401, content={"error": "invalid_key"})
        return {"status": "ok", "mode": "dev"}
    # Production: key must match one in the pool
    if key not in API_KEYS:
        return JSONResponse(status_code=401, content={"error": "invalid_key"})
    return {"status": "ok"}


@app.get("/auth/status")
async def auth_status():
    """返回所有 session 状态"""
    sessions = token_manager.get_all_status()
    healthy = sum(1 for s in sessions if s.get("is_healthy"))
    return {
        "status": "ok" if sessions else "no_session",
        "device_id": token_manager.device_id,
        "total": len(sessions),
        "healthy": healthy,
        "sessions": sessions,
    }


@app.post("/auth/session/{sid}/remove")
async def remove_session(sid: str):
    """删除指定 session"""
    if token_manager.remove_session(sid):
        return {"status": "ok", "message": f"Session {sid} removed"}
    raise HTTPException(status_code=404, detail=f"Session {sid} not found")


@app.post("/auth/session/{sid}/toggle")
async def toggle_session(sid: str, request: Request):
    """启用/禁用指定 session"""
    body = await request.json()
    disabled = body.get("disabled", False)
    if token_manager.toggle_session(sid, disabled):
        state = "disabled" if disabled else "enabled"
        return {"status": "ok", "message": f"Session {sid} {state}"}
    raise HTTPException(status_code=404, detail=f"Session {sid} not found")


@app.get("/auth/session/{sid}/download")
async def download_session(sid: str):
    """下载单个 session 的 JSON"""
    import io
    for s in token_manager.sessions:
        if s.sid == sid:
            if not s.raw_session:
                raise HTTPException(status_code=404, detail=f"Session {sid} has no raw data")
            data = json.dumps(s.raw_session, indent=2, ensure_ascii=False)
            return Response(
                content=data,
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="session_{sid}.json"'}
            )
    raise HTTPException(status_code=404, detail=f"Session {sid} not found")


@app.get("/auth/sessions/download")
async def download_all_sessions():
    """打包所有 session 为 ZIP 下载"""
    import io, zipfile
    buf = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for s in token_manager.sessions:
            if s.raw_session:
                fname = f"session_{s.sid}.json"
                zf.writestr(fname, json.dumps(s.raw_session, indent=2, ensure_ascii=False))
                count += 1
        # Also add a combined file
        all_data = [s.raw_session for s in token_manager.sessions if s.raw_session]
        zf.writestr("all_sessions.json", json.dumps(all_data, indent=2, ensure_ascii=False))
    buf.seek(0)
    if count == 0:
        raise HTTPException(status_code=404, detail="No sessions with raw data to export")
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="chatgpt_sessions.zip"'}
    )


# ══════════════════════════════════════════════════════════════════════════
#  Health & models endpoints (no auth required)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/ping")
async def health_check():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-image-2", "object": "model", "owned_by": "chatgpt"},
            {"id": "gpt-image-1", "object": "model", "owned_by": "chatgpt"},
            {"id": "gpt-4o", "object": "model", "owned_by": "chatgpt"},
            {"id": "gpt-5.4-mini", "object": "model", "owned_by": "chatgpt"},
            {"id": "gpt-5.5", "object": "model", "owned_by": "chatgpt"},
            {"id": "auto", "object": "model", "owned_by": "chatgpt"},
        ],
    }


# ══════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
