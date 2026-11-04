"""API 管理（P4）：模型端点连通性探针。

**凭据**：优先用页面配置在 `endpoints.api_key` 的 key；没有才回退 `key_ref` 指向的环境变量。
接口不回显明文。探针 = 对 OpenAI 兼容网关 `GET {base_url}/models`（带 Bearer）；对 opencode go
网关再补它要求的 `x-opencode-session` 头（见 `ops/dsh/README.md` 第 1 条）。只看「通不通 + 有哪些模型」，
不产生费用。
"""

from __future__ import annotations

import os
import time

import httpx


def _headers(base_url: str | None, key: str | None) -> dict:
    h = {"User-Agent": "op-agent-console/0.1 (dsh)"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    if "opencode" in (base_url or "").lower():
        # opencode go 要求带稳定的 x-opencode-session（缺了 400 MissingSessionID）
        h["x-opencode-session"] = "dsh-op-agent-0001"
    return h


def probe(base_url: str | None, key_ref: str | None, *, api_key: str | None = None,
          timeout: float = 15.0) -> dict:
    """探一次端点，返回结果字典（不抛异常）。

    key 优先用页面配置的 `api_key`；没有才回退 `key_ref` 环境变量（兼容旧配置）。
    """
    if not base_url:
        return {"ok": False, "error": "没有 base_url"}
    key = (api_key or "").strip() or (os.environ.get(key_ref) if key_ref else None)
    if key_ref and not key:
        return {"ok": False, "error": f"没有配置 key（页面 key 与环境变量 {key_ref} 都为空）"}
    # base_url 正常不该带 query/fragment（路由已校验）；历史行兜底裁掉，避免拼成 `...?x=1/models`。
    base = (base_url or "").strip().split("?")[0].split("#")[0].rstrip("/")
    url = base + "/models"
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=_headers(base_url, key))
    except httpx.HTTPError as e:
        return {"ok": False, "url": url, "error": f"{type(e).__name__}: {e}"}
    latency_ms = int((time.monotonic() - t0) * 1000)
    out: dict = {"url": url, "status_code": resp.status_code, "latency_ms": latency_ms,
                 "key_present": bool(key)}
    if resp.status_code != 200:
        out["ok"] = False
        out["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return out
    try:
        data = resp.json()
    except ValueError:
        out["ok"] = False
        out["error"] = "返回不是 JSON"
        return out
    models = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        models = [m.get("id") for m in data["data"] if isinstance(m, dict) and m.get("id")]
    out["ok"] = True
    out["models"] = models
    out["model_count"] = len(models)
    return out
