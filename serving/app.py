"""MiniMind 推理服务（阶段 A 定版实现）

- FastAPI + 原生 PyTorch CPU / FP32 / 单 worker
- 模型包目录由环境变量 MODEL_DIR 指定（应指向 versions/<版本> 或 current 符号链接）
- 令牌由环境变量 SERVICE_TOKEN 提供（只存 InstaBox secrets.env，不入代码）
- minimind 仓库路径由 MINIMIND_REPO 指定（默认 /data/home/minimind）
"""
import asyncio
import json
import os
import secrets
import sys
import time

import torch

# 线程限制必须在模型加载与推理前设置（4 vCPU：生成 2 线程，留头寸给 HTTP I/O）
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
try:
    torch.set_num_interop_threads(int(os.environ.get("TORCH_INTEROP_THREADS", "1")))
except RuntimeError:
    pass

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get("MODEL_DIR", "/data/home/minimind-system/current")
MINIMIND_REPO = os.environ.get("MINIMIND_REPO", "/data/home/minimind")
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "")

MAX_INPUT_TOKENS = 384
MAX_TOTAL_WINDOW = 512
DEFAULT_MAX_NEW = 128
MAX_NEW_CAP = 256
QUEUE_MAX_WAITING = 5
QUEUE_TIMEOUT_SEC = 15

sys.path.insert(0, MINIMIND_REPO)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

app = FastAPI(title="minimind-service", version="1.0.0")

_state = {"model": None, "tokenizer": None, "version": "unknown", "ready": False}
_gen_semaphore = asyncio.Semaphore(1)
_waiting = 0


def _load_model():
    with open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    config = MiniMindConfig(
        hidden_size=cfg.get("hidden_size", 768),
        num_hidden_layers=cfg.get("num_hidden_layers", 8),
        use_moe=False,
        vocab_size=cfg.get("vocab_size", 6400),
        num_attention_heads=cfg.get("num_attention_heads", 8),
        num_key_value_heads=cfg.get("num_key_value_heads", 4),
    )
    model = MiniMindForCausalLM(config)
    state_dict = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    manifest_path = os.path.join(MODEL_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            _state["version"] = json.load(f).get("version", "unknown")
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["ready"] = True


@app.on_event("startup")
def _startup():
    _load_model()


def _check_auth(authorization: str | None):
    if not SERVICE_TOKEN:
        raise HTTPException(status_code=500, detail="service token not configured")
    expected = f"Bearer {SERVICE_TOKEN}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW, ge=1, le=MAX_NEW_CAP)
    temperature: float = 0.7
    top_p: float = 0.85
    repetition_penalty: float = 1.1
    greedy: bool = False


def _generate_sync(req: GenerateRequest) -> dict:
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    ids = tokenizer.encode(req.prompt)
    if len(ids) > MAX_INPUT_TOKENS:
        ids = ids[-MAX_INPUT_TOKENS:]
    max_new = min(req.max_new_tokens, MAX_TOTAL_WINDOW - len(ids))
    if max_new < 1:
        raise HTTPException(status_code=400, detail="input too long")
    input_ids = torch.tensor([ids], dtype=torch.long)
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=not req.greedy,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    if not req.greedy:
        gen_kwargs["temperature"] = req.temperature
        gen_kwargs["top_p"] = req.top_p
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kwargs)
    elapsed = time.time() - t0
    new_ids = out[0][len(ids):].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {
        "text": text,
        "version": _state["version"],
        "input_tokens": len(ids),
        "output_tokens": len(new_ids),
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(len(new_ids) / elapsed, 2) if elapsed > 0 else None,
    }


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "version": _state["version"]}


@app.get("/v1/models")
def models():
    return {"current": _state["version"], "data": [{"id": _state["version"], "object": "model"}]}


@app.post("/v1/generate")
async def generate(req: GenerateRequest, request: Request, authorization: str | None = Header(default=None)):
    global _waiting
    _check_auth(authorization)
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="model not loaded")
    if _waiting >= QUEUE_MAX_WAITING:
        raise HTTPException(status_code=429, detail="queue full")
    _waiting += 1
    try:
        await asyncio.wait_for(_gen_semaphore.acquire(), timeout=QUEUE_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="queue timeout")
    finally:
        _waiting -= 1
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _generate_sync, req)
    finally:
        _gen_semaphore.release()
