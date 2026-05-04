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
import json
import uuid
import asyncio
import logging
import time
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
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
IMAGE_MODELS = {"gpt-image-1", "gpt-image-2"}

# ── API Key 鉴权 ───────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "")

# 不需要鉴权的白名单路径
AUTH_WHITELIST = {"/ping", "/health", "/healthz", "/docs", "/openapi.json", "/", "/auth/session", "/auth/status"}


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

    # 如果没有配置 API_KEY，跳过鉴权（开发模式）
    if not API_KEY:
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
    if not client_key or client_key != API_KEY:
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
    """Build Codex-compatible headers (ported from Codaze headers.rs)"""
    return {
        "Authorization": f"Bearer {access_token}",
        "ChatGPT-Account-ID": account_id,
        "x-codex-installation-id": installation_id,
        "x-codex-turn-metadata": json.dumps({"thread_source": "user"}),
        "x-openai-subagent": "user",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def normalize_codex_request(payload: dict) -> dict:
    """Normalize request body like Codaze request_normalization.rs"""
    for key in ["max_output_tokens", "max_completion_tokens", "temperature",
                "top_p", "truncation", "user"]:
        payload.pop(key, None)

    st = payload.get("service_tier")
    if st != "priority":
        payload.pop("service_tier", None)

    tools = payload.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                t = tool.get("type", "")
                if t in ("web_search_preview", "web_search_preview_2025_03_11"):
                    tool["type"] = "web_search"

    tc = payload.get("tool_choice")
    if isinstance(tc, str) and tc in ("web_search_preview", "web_search_preview_2025_03_11"):
        payload["tool_choice"] = "web_search"
    elif isinstance(tc, dict):
        t = tc.get("type", "")
        if t in ("web_search_preview", "web_search_preview_2025_03_11"):
            tc["type"] = "web_search"

    inst = payload.get("instructions")
    if inst is None:
        payload["instructions"] = ""
    if "store" not in payload:
        payload["store"] = False
    if "parallel_tool_calls" not in payload:
        payload["parallel_tool_calls"] = True

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
        "User-Agent": CODEX_USER_AGENT,
        "Originator": CODEX_ORIGINATOR,
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
# ══════════════════════════════════════════════════════════════════════════

def build_conversation_body(prompt: str, model: str = DEFAULT_MODEL) -> dict:
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
        "system_hints": ["picture_v2"],
        "supports_buffering": True,
        "supported_encodings": [],
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


# ══════════════════════════════════════════════════════════════════════════
#  Conversation SSE parser — extracts image URLs
# ══════════════════════════════════════════════════════════════════════════

def _extract_file_id(asset_pointer: str) -> str:
    if "file-service://" in asset_pointer:
        return asset_pointer.split("file-service://", 1)[1].split("?")[0]
    if "sediment://" in asset_pointer:
        return asset_pointer.split("sediment://", 1)[1].split("?")[0]
    return ""


async def _resolve_image_url(access_token: str, device_id: str,
                              file_id: str, conversation_id: str) -> str:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": WEB_USER_AGENT,
        "oai-device-id": device_id,
    }
    async with curl_requests.AsyncSession(impersonate="chrome110") as session:
        resp = await session.get(
            f"{BASE_URL}/files/{file_id}/download",
            headers=headers,
            allow_redirects=False,
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            return resp.headers.get("Location", "")
        if resp.status_code == 200:
            try:
                return resp.json().get("download_url", "")
            except Exception:
                pass

        resp2 = await session.get(
            f"{BASE_URL}/attachments/{file_id}",
            headers=headers,
            allow_redirects=False,
        )
        if resp2.status_code in (301, 302, 303, 307, 308):
            return resp2.headers.get("Location", "")
        if resp2.status_code == 200:
            try:
                return resp2.json().get("download_url", "")
            except Exception:
                pass

    return ""


async def parse_conversation_sse(
    access_token: str, device_id: str,
    chunks: list[str], parent_msg_id: str = "",
) -> list[dict]:
    images = []
    conversation_id = ""
    seen_ids = set()

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

        cid = event.get("conversation_id", "")
        if cid:
            conversation_id = cid

        msg = event.get("message")
        if not msg:
            continue
        if msg.get("id") == parent_msg_id:
            continue

        author_role = msg.get("author", {}).get("role", "")
        if author_role in ("user", "system"):
            continue
        if msg.get("content", {}).get("content_type") != "multimodal_text":
            continue
        if msg.get("status") != "finished_successfully":
            continue

        for raw_part in msg.get("content", {}).get("parts", []):
            if isinstance(raw_part, str):
                continue
            if raw_part.get("content_type") != "image_asset_pointer":
                continue
            asset = raw_part.get("asset_pointer", "")
            file_id = _extract_file_id(asset)
            if not file_id or file_id in seen_ids:
                continue
            seen_ids.add(file_id)

            dalle_meta = raw_part.get("metadata", {}).get("dalle", {})
            gen_id = dalle_meta.get("gen_id", "")
            revised = dalle_meta.get("prompt", "")

            url = await _resolve_image_url(access_token, device_id, file_id, conversation_id)
            if url:
                images.append({"url": url, "revised_prompt": revised, "file_id": file_id, "gen_id": gen_id})

    return images


# ══════════════════════════════════════════════════════════════════════════
#  Image generation core
# ══════════════════════════════════════════════════════════════════════════

async def _handle_image_via_conversation(
    prompt: str, model: str, n: int,
    size: str, quality: str, background: str, response_format: str,
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

    body = build_conversation_body(full_prompt, model=model)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": WEB_USER_AGENT,
        "oai-device-id": device_id,
        "Accept": "text/event-stream",
        "openai-sentinel-chat-requirements-token": chat_token,
    }
    if proof_token:
        headers["openai-sentinel-proof-token"] = proof_token

    for path in ("/f/conversation", "/conversation"):
        route_label = path.split("/")[-1]
        log.info(f"[conv] trying {path}")
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

                chunks = []
                async for line in resp.aiter_lines():
                    if line:
                        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                        chunks.append(decoded)

                images = await parse_conversation_sse(access_token, device_id, chunks)
                if images:
                    return _build_images_response(images[:n], response_format)

                log.warning(f"[conv] {route_label} returned no images, trying next path")
                if path == "/conversation":
                    raise Exception("No images generated from either endpoint")
        except Exception as e:
            if path == "/conversation":
                raise
            log.warning(f"[conv] {route_label} failed: {e}, trying /conversation")
            continue

    raise Exception("Image generation failed")


def _build_images_response(images: list[dict], response_format: str) -> dict:
    data = []
    for img in images:
        item = {"revised_prompt": img.get("revised_prompt", "")}
        if response_format == "b64_json":
            item["url"] = img["url"]
            item["b64_json"] = ""
        else:
            item["url"] = img["url"]
        data.append(item)
    return {"created": int(time.time()), "data": data}


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


def _extract_prompt_from_messages(messages: list[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        if msg.role in ("system", "assistant", "tool"):
            continue
        if isinstance(msg.content, str):
            parts.append(msg.content)
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
    return "\n\n".join(parts)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model = req.model.lower()

    # ── Image generation mode ────────────────────────────────────────────
    if model in IMAGE_MODELS:
        prompt = _extract_prompt_from_messages(req.messages)
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")

        log.info(f"[chat] image mode, model={model}")
        try:
            img_resp = await _handle_image_via_conversation(
                prompt=prompt, model=model, n=req.n,
                size=req.size, quality=req.quality,
                background=req.background, response_format="url",
            )
            markdown_parts = []
            for img in img_resp.get("data", []):
                url = img.get("url", "")
                if url:
                    markdown_parts.append(f"![image]({url})")
            content = "\n\n".join(markdown_parts) if markdown_parts else "Image generation failed."

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

    # ── Standard text conversation → codex/responses ─────────────────────
    prompt = _extract_prompt_from_messages(req.messages)
    access_token = await token_manager.get_valid_token()
    device_id = token_manager.device_id
    chat_token, proof_token = await get_sentinel_tokens(access_token, device_id)

    payload = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "stream": req.stream,
        "store": False,
    }

    headers = build_codex_headers(access_token, token_manager.account_id, token_manager.installation_id)
    headers["User-Agent"] = CODEX_USER_AGENT
    headers["Originator"] = CODEX_ORIGINATOR
    headers["session_id"] = str(uuid.uuid4())
    headers["x-client-request-id"] = headers["session_id"]
    headers["openai-sentinel-chat-requirements-token"] = chat_token
    if proof_token:
        headers["openai-sentinel-proof-token"] = proof_token

    payload = normalize_codex_request(payload)
    log.info(f"[chat] text mode, forwarding to codex/responses, stream={req.stream}")

    try:
        if req.stream:
            return await _stream_codex_response_for_chat_completions(payload, headers, req.model)
        else:
            return await _non_stream_codex_response(payload, headers, req.model)
    except Exception as e:
        log.error(f"[chat] codex error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=str(e))


async def _stream_codex_response_for_chat_completions(payload: dict, headers: dict, model: str) -> StreamingResponse:
    # 专门为标准 API (chat/completions) 转译 Codex 流
    async def generate():
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        log.info("[chat/completions] Starting stream to codex/responses...")
        async with curl_requests.AsyncSession(impersonate="chrome110") as session:
            try:
                resp = await session.post(
                    f"{CODEX_BASE_URL}/responses",
                    json=payload, headers=headers, stream=True, timeout=600,
                )
                log.info(f"[chat/completions] codex/responses returned status_code={resp.status_code}")
                
                if resp.status_code != 200:
                    err = resp.content
                    log.error(f"[chat/completions] Error from backend: {err.decode()[:500]}")
                    yield f"data: {json.dumps({'error': {'message': f'Backend returned {resp.status_code}: Auth failed or blocked. {err.decode()[:500]}'}})}\n\n"
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
                                        yield f"data: {json.dumps(chunk)}\n\n"
                    
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
                resp = await session.post(
                    f"{CODEX_BASE_URL}/responses",
                    json=payload, headers=headers, stream=True, timeout=600,
                )
                log.info(f"[responses] codex/responses returned status_code={resp.status_code}")
                
                if resp.status_code != 200:
                    err = resp.content
                    log.error(f"[responses] Error from backend: {err.decode()[:500]}")
                    yield f"data: {json.dumps({'error': {'message': f'Backend returned {resp.status_code}: Auth failed or blocked. {err.decode()[:500]}'}})}\n\n"
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
    payload = normalize_codex_payload(payload)  # 注入清洗器
    access_token = await token_manager.get_valid_token()
    device_id = token_manager.device_id
    chat_token, proof_token = await get_sentinel_tokens(access_token, device_id)

    tools = payload.get("tools", [])
    has_image_tool = any(t.get("type", "").lower() == "image_generation" for t in tools)

    headers = build_codex_headers(access_token, token_manager.account_id, token_manager.installation_id)
    headers["User-Agent"] = CODEX_USER_AGENT
    headers["Originator"] = CODEX_ORIGINATOR
    headers["session_id"] = str(uuid.uuid4())
    headers["x-client-request-id"] = headers["session_id"]
    headers["openai-sentinel-chat-requirements-token"] = chat_token
    if proof_token:
        headers["openai-sentinel-proof-token"] = proof_token

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

MANAGER_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChatGPT Session Proxy - 管理</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; min-height: 100vh;
         display: flex; justify-content: center; align-items: flex-start; padding: 2rem; }
  .container { max-width: 720px; width: 100%; }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #00d4ff; }
  .subtitle { color: #888; margin-bottom: 2rem; font-size: 0.9rem; }
  .card { background: #16213e; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
          border: 1px solid #0f3460; }
  .card h2 { font-size: 1.1rem; color: #e94560; margin-bottom: 1rem; }
  textarea { width: 100%; height: 200px; background: #0a0a1a; color: #e0e0e0;
             border: 1px solid #333; border-radius: 8px; padding: 1rem;
             font-family: 'Fira Code', monospace; font-size: 0.85rem; resize: vertical; }
  textarea:focus { outline: none; border-color: #00d4ff; }
  button { background: #e94560; color: white; border: none; border-radius: 8px;
           padding: 0.8rem 2rem; font-size: 1rem; cursor: pointer; margin-top: 1rem;
           transition: background 0.2s; }
  button:hover { background: #ff6b6b; }
  button.secondary { background: #0f3460; }
  button.secondary:hover { background: #1a4a8a; }
  .status { padding: 1rem; border-radius: 8px; margin-top: 1rem; font-size: 0.9rem; }
  .status.ok { background: #0d3320; border: 1px solid #00c853; color: #69f0ae; }
  .status.error { background: #3d0000; border: 1px solid #ff1744; color: #ff8a80; }
  .status.info { background: #0d1b3e; border: 1px solid #2979ff; color: #82b1ff; }
  .info-grid { display: grid; grid-template-columns: auto 1fr; gap: 0.5rem 1rem; margin-top: 0.5rem; }
  .info-label { color: #888; font-weight: 600; }
  .info-value { color: #e0e0e0; word-break: break-all; }
  .instructions { background: #0a0a1a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;
                  font-size: 0.85rem; line-height: 1.6; color: #aaa; }
  .instructions ol { padding-left: 1.2rem; }
  .instructions li { margin-bottom: 0.3rem; }
  .instructions code { background: #1a1a3e; padding: 0.1rem 0.4rem; border-radius: 4px; color: #00d4ff; }
  .refresh-btn { font-size: 0.85rem; padding: 0.5rem 1rem; margin-left: 0.5rem; }
  .flex-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
</style>
</head>
<body>
<div class="container">
  <h1>ChatGPT Session Proxy</h1>
  <p class="subtitle">管理 Session Token — 粘贴 /api/auth/session 的 JSON 即可</p>

  <div class="card">
    <h2>使用说明</h2>
    <div class="instructions">
      <ol>
        <li>在浏览器中打开 <a href="https://chatgpt.com" target="_blank" style="color:#00d4ff">chatgpt.com</a> 并登录</li>
        <li>按 <code>F12</code> 打开开发者工具 → <code>Console</code> 标签页</li>
        <li>输入 <code>await fetch('/api/auth/session').then(r=>r.json()).then(j=>copy(JSON.stringify(j)))</code></li>
        <li>JSON 已复制到剪贴板，粘贴到下方即可</li>
      </ol>
    </div>
  </div>

  <div class="card">
    <h2>粘贴 Session JSON</h2>
    <textarea id="sessionInput" placeholder='{"accessToken":"eyJ...","sessionToken":"eyJ...","account":{"id":"..."},...}'></textarea>
    <div class="flex-row">
      <button onclick="submitSession()">保存 Session</button>
      <button class="secondary" onclick="refreshStatus()">刷新状态</button>
    </div>
    <div id="submitResult"></div>
  </div>

  <div class="card">
    <h2>当前状态</h2>
    <div id="statusArea">
      <p style="color:#888">正在加载...</p>
    </div>
  </div>
</div>

<script>
async function refreshStatus() {
  const el = document.getElementById('statusArea');
  try {
    const r = await fetch('/auth/status');
    const d = await r.json();
    if (d.status === 'no_session') {
      el.innerHTML = '<div class="status error">未配置 Session，请在上方粘贴 JSON</div>';
      return;
    }
    const expDate = new Date(d.expires * 1000);
    const isExpired = expDate < new Date();
    const statusClass = isExpired ? 'error' : 'ok';
    const statusText = isExpired ? '已过期' : '有效';
    el.innerHTML = `
      <div class="status ${statusClass}">Token 状态: ${statusText}</div>
      <div class="info-grid" style="margin-top:1rem">
        <span class="info-label">Account ID:</span>
        <span class="info-value">${d.account_id || 'N/A'}</span>
        <span class="info-label">过期时间:</span>
        <span class="info-value">${expDate.toLocaleString()} ${isExpired ? '(已过期，下次请求时自动刷新)' : ''}</span>
        <span class="info-label">设备 ID:</span>
        <span class="info-value">${d.device_id || 'N/A'}</span>
      </div>
    `;
  } catch(e) {
    el.innerHTML = `<div class="status error">获取状态失败: ${e.message}</div>`;
  }
}

async function submitSession() {
  const input = document.getElementById('sessionInput').value.trim();
  const result = document.getElementById('submitResult');
  if (!input) {
    result.innerHTML = '<div class="status error">请粘贴 Session JSON</div>';
    return;
  }
  try {
    const r = await fetch('/auth/session', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: input
    });
    const d = await r.json();
    if (r.ok) {
      result.innerHTML = `<div class="status ok">${d.message}</div>`;
      document.getElementById('sessionInput').value = '';
      refreshStatus();
    } else {
      result.innerHTML = `<div class="status error">错误: ${d.detail || JSON.stringify(d)}</div>`;
    }
  } catch(e) {
    result.innerHTML = `<div class="status error">请求失败: ${e.message}</div>`;
  }
}

// 页面加载时刷新状态
refreshStatus();
</script>
</body>
</html>"""


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


@app.get("/auth/status")
async def auth_status():
    """返回当前 session 状态"""
    if not token_manager.access_token and not token_manager.session_token:
        return {"status": "no_session"}

    return {
        "status": "ok",
        "account_id": token_manager.account_id,
        "expires": token_manager.expires_at,
        "device_id": token_manager.device_id,
        "has_session_token": bool(token_manager.session_token),
    }


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
        ],
    }


# ══════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
