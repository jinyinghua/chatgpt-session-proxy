"""Run this INSIDE the Docker container to diagnose the 401 body"""
import asyncio
import json
import uuid
from curl_cffi import requests
from token_manager import token_manager
from main import get_sentinel_tokens, build_codex_headers, normalize_codex_request, CODEX_BASE_URL, CODEX_USER_AGENT, CODEX_ORIGINATOR

async def test():
    access_token = await token_manager.get_valid_token()
    device_id = token_manager.device_id
    chat_token, proof_token = await get_sentinel_tokens(access_token, device_id)

    headers = build_codex_headers(access_token, token_manager.account_id, token_manager.installation_id)
    headers["User-Agent"] = CODEX_USER_AGENT
    headers["Originator"] = CODEX_ORIGINATOR
    headers["session_id"] = str(uuid.uuid4())
    headers["x-client-request-id"] = headers["session_id"]
    headers["openai-sentinel-chat-requirements-token"] = chat_token
    if proof_token:
        headers["openai-sentinel-proof-token"] = proof_token

    payload = {
        "model": "gpt-5.4-mini",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        "stream": True,
        "store": False,
    }
    payload = normalize_codex_request(payload)

    print("=== Test 1: stream=False ===")
    async with requests.AsyncSession(impersonate="chrome110") as session:
        resp = await session.post(f"{CODEX_BASE_URL}/responses", json=payload, headers=headers, stream=False, timeout=30)
        print(f"Status: {resp.status_code}")
        print(f"Body: [{len(resp.content)} bytes] {resp.text[:500]}")

    print("\n=== Test 2: stream=True + aiter_bytes ===")
    async with requests.AsyncSession(impersonate="chrome110") as session:
        resp = await session.post(f"{CODEX_BASE_URL}/responses", json=payload, headers=headers, stream=True, timeout=30)
        print(f"Status: {resp.status_code}")
        chunks = []
        async for chunk in resp.aiter_bytes():
            chunks.append(chunk)
        body = b"".join(chunks)
        print(f"Body: [{len(body)} bytes] {body.decode('utf-8', errors='replace')[:500]}")

    print("\n=== Test 3: stream=True + read() ===")
    async with requests.AsyncSession(impersonate="chrome110") as session:
        resp = await session.post(f"{CODEX_BASE_URL}/responses", json=payload, headers=headers, stream=True, timeout=30)
        print(f"Status: {resp.status_code}")
        try:
            body = resp.content
            print(f"resp.content: [{len(body)} bytes] {body.decode('utf-8', errors='replace')[:500]}")
        except Exception as e:
            print(f"resp.content error: {e}")

asyncio.run(test())
