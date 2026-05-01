import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from curl_cffi import requests

from token_manager import token_manager

app = FastAPI(title="ChatGPT Proxy (PaaS Edition)")

@app.get("/ping")
async def health_check():
    return {"status": "ok", "message": "PaaS deployment is running"}

async def forward_request(url: str, payload: dict, token: str, device_id: str):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "oai-device-id": device_id
    }

    async def generate_stream():
        async with requests.AsyncSession(impersonate="chrome120") as session:
            response = await session.post(
                url,
                json=payload,
                headers=headers,
                stream=True
            )
            
            if response.status_code != 200:
                error_msg = await response.aread()
                yield f"data: {{\"error\": \"Backend error {response.status_code}: {error_msg.decode('utf-8')}\"}}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for chunk in response.aiter_lines():
                if chunk:
                    yield chunk.decode('utf-8') + "\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

# ===============================
# 🚀 路由 1：Codex 专属接口路由
# ===============================
@app.post("/v1/responses")
async def proxy_codex_responses(request: Request):
    print("[Route] 路由至 Codex 后端...")
    try:
        payload = await request.json()
        token = await token_manager.get_valid_token()
        codex_url = "https://chatgpt.com/backend-api/codex/responses"
        return await forward_request(codex_url, payload, token, token_manager.device_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# 💬 路由 2：标准对话接口路由
# ===============================
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    print("[Route] 路由至标准对话后端...")
    try:
        payload = await request.json()
        token = await token_manager.get_valid_token()
        chat_url = "https://chatgpt.com/backend-api/conversation"
        return await forward_request(chat_url, payload, token, token_manager.device_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 绑定 0.0.0.0 和系统分配的 PORT，本地默认 8080
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)