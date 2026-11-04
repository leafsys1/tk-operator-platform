"""当前 API 端点 → 进程环境变量 + dsh 模型路由 patch。

dsh 的模型路由（provider / baseURL / model）由 `--patch` 决定。这里按「API 管理」页的
**当前端点动态生成**一份 patch（写到 `settings.dsh_generated_patch`），并把该端点的 key 同步到
`OPX_LLM_KEY`（dsh 的 `apiKeyEnv`）——这样页面配的 DeepSeek 官方端点/网关，就是 agent 实际
调用的那个端点，不再写死 opencode go。

> 2026-10-21 实测：静态 patch 写死 opencode 网关 + 页面配 DeepSeek 官方端点时，页面 key 被灌给
> dsh，dsh 拿 DeepSeek key 请求 opencode 网关 → `401 Invalid credential`，任务开始页报
> 「本轮没有得到回答」。所以路由与 key 必须来自同一个端点。

优先级：
1. `endpoints` 里 `is_active=1`（且未软删）端点的 key；
2. 页面没配 key（或显式清空）→ **还原启动时环境里的 OPX_LLM_KEY**（`capture_baseline()` 只记一次）；
3. 启动环境里也没有 → 保持未设置。

端点缺 `base_url` / `model`（或根本没有端点）时，生成文件会被删除，退回静态
`settings.dsh_profile_patch`（默认 opencode go；仅老部署/无端点时的兜底）。
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Endpoint
from . import secret_store

_LOCK = threading.Lock()
#: 进程启动时的 OPX_LLM_KEY（页面 key 的最终回退；只记录一次）
_BASELINE_KEY: str | None = None
_BASELINE_CAPTURED = False


def _remember_baseline_locked() -> None:
    global _BASELINE_KEY, _BASELINE_CAPTURED
    if _BASELINE_CAPTURED:
        return
    value = (os.environ.get(settings.llm_key_env) or "").strip()
    _BASELINE_KEY = value or None
    _BASELINE_CAPTURED = True


def capture_baseline() -> None:
    """记录启动时的环境 key（lifespan 在 `apply_active_key` 之前调用）。"""
    with _LOCK:
        _remember_baseline_locked()


def baseline_key() -> str | None:
    with _LOCK:
        _remember_baseline_locked()
        return _BASELINE_KEY


def reset_baseline_for_tests(value: str | None = None) -> None:
    """测试辅助：重置启动基线（生产代码不要在运行中调用）。"""
    global _BASELINE_KEY, _BASELINE_CAPTURED
    with _LOCK:
        _BASELINE_KEY = value
        _BASELINE_CAPTURED = True


def active_endpoint(db: Session) -> Endpoint | None:
    """当前端点（未软删）；没有显式 active 时取 id 最小的未删端点（与页面/计价同口径）。"""
    row = db.scalar(
        select(Endpoint)
        .where(Endpoint.is_active.is_(True), Endpoint.deleted_at.is_(None))
        .order_by(Endpoint.id)
        .limit(1)
    )
    if row is None:
        row = db.scalar(
            select(Endpoint).where(Endpoint.deleted_at.is_(None)).order_by(Endpoint.id).limit(1)
        )
    return row


def endpoint_key(row: Endpoint | None) -> str | None:
    """端点的 key 本体：优先本机密钥库；旧库 `api_key` 列仅在未迁移时兜底。"""
    if row is None:
        return None
    stored = secret_store.get_secret(row.secret_ref) if row.secret_ref else None
    if stored:
        return stored
    legacy = (row.api_key or "").strip()
    return legacy or None


def migrate_legacy_secrets(db: Session, *, dry_run: bool = False) -> dict:
    """把旧库里明文 `endpoints.api_key` 搬到本机密钥库，然后清空该列。

    启动时调用；某个端点迁移失败（如 SSH 无 Keychain 且文件后端不可写）时保留旧值并
    记入 `failed`，运行链路的 `endpoint_key()` 仍能回退旧列，不会把功能弄挂。
    """
    rows = list(db.scalars(select(Endpoint).where(Endpoint.api_key.is_not(None))))
    migrated: list[int] = []
    failed: list[dict] = []
    for row in rows:
        value = (row.api_key or "").strip()
        if not value:
            row.api_key = None
            continue
        ref = row.secret_ref or f"endpoint-{row.id}"
        if dry_run:
            migrated.append(row.id)
            continue
        try:
            secret_store.set_secret(ref, value)
        except secret_store.SecretStoreError as e:
            failed.append({"endpoint_id": row.id, "error": str(e)[:300]})
            continue
        row.secret_ref = ref
        row.api_key = None
        migrated.append(row.id)
    if not dry_run:
        db.commit()
    return {"migrated": migrated, "failed": failed}


#: dsh patch 里模型 provider 的默认上下文窗口（端点没配 max_model_len 时用）。
_DEFAULT_MODEL_CTX = 131072


def provider_slug(row: Endpoint) -> str:
    """端点 provider 名 → patch 里的 provider id（`dsh-llm-deepseek` → `deepseek`）。"""
    raw = (row.provider or "").strip()
    name = raw[len("dsh-llm-"):] if raw.startswith("dsh-llm-") else raw
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-_").lower()
    return name or f"endpoint-{row.id}"


def build_dsh_patch(row: Endpoint | None) -> list[dict] | None:
    """按端点生成 dsh 的 LLM 路由 patch（`llm-pi-ai` + `agent-default-model` 两条）。

    统一走 pi-ai 的 `api: openai-completions`：DeepSeek 官方 API 与 opencode go 网关都是
    OpenAI 兼容的 `/chat/completions`（2026-10-21 用 dsh + DeepSeek 官方 key 实测跑通）。
    端点信息不全（缺 base_url / model）时返回 None —— 调用方回退静态 patch。
    """
    if row is None:
        return None
    base = (row.base_url or "").strip().rstrip("/")
    model = (row.model or "").strip()
    if not base or not model:
        return None
    provider = provider_slug(row)
    ctx = row.max_model_len if row.max_model_len and 1024 <= row.max_model_len <= 2_000_000 else _DEFAULT_MODEL_CTX
    headers: dict[str, str] = {"User-Agent": "op-agent-console/0.1 (dsh)"}
    if "opencode" in base.lower():
        # opencode go 要求带稳定的 x-opencode-session（缺了 400 MissingSessionID）。
        headers["x-opencode-session"] = "dsh-op-agent-0001"
    return [
        {
            "id": "llm-pi-ai",
            "config": {
                "providers": {
                    provider: {
                        "displayName": row.name or provider,
                        "apiKeyEnv": settings.llm_key_env,
                        "api": "openai-completions",
                        "baseURL": base,
                        "compat": {"thinkingFormat": "deepseek"},
                        "headers": headers,
                        "models": [{"id": model, "name": model, "contextWindow": int(ctx)}],
                    }
                }
            },
        },
        {"id": "agent-default-model", "config": {"provider": provider, "model": model}},
    ]


def render_dsh_patch(row: Endpoint | None) -> str | None:
    data = build_dsh_patch(row)
    if data is None:
        return None
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def generated_patch_path() -> Path:
    return Path(settings.repo_root) / settings.dsh_generated_patch


def _write_generated_patch(row: Endpoint | None) -> Path | None:
    """原子写/删生成 patch；失败只返回 None（不阻塞建端点，dsh 走静态兜底）。"""
    path = generated_patch_path()
    text = render_dsh_patch(row)
    try:
        if text is None:
            path.unlink(missing_ok=True)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        return path
    except OSError:
        return None


def sync_dsh_patch(db: Session) -> Path | None:
    """按当前端点刷新生成 patch。没有端点/信息不全 → 删生成文件（回退静态 patch）。"""
    return _write_generated_patch(active_endpoint(db))


def dsh_patch_for_run() -> Path:
    """agent 链路实际传给 `dsh --patch` 的文件：有生成 patch 用生成的，否则用静态兜底。"""
    path = generated_patch_path()
    if path.is_file():
        return path
    return Path(settings.repo_root) / settings.dsh_profile_patch


def reset_generated_patch_for_tests() -> None:
    """测试辅助：删掉生成 patch，避免用例之间串状态。"""
    generated_patch_path().unlink(missing_ok=True)


def apply_active_key(db: Session, *, clear: bool = False) -> Endpoint | None:
    """同步当前端点的 key 到进程环境变量，并按端点重建 dsh 路由 patch；返回当前端点。

    `clear=True`：调用方明确要「重新同步」（activate / 删端点 / 清空 key）。此时若当前端点
    没有页面 key，则**还原启动基线**而不是移除环境变量。`clear=False`（如新建端点）只在端点
    有 page key 时覆盖，避免误清启动时注入的 key。
    """
    with _LOCK:
        _remember_baseline_locked()
        row = active_endpoint(db)
        key = endpoint_key(row) or ""
        if key:
            os.environ[settings.llm_key_env] = key
        elif clear:
            if _BASELINE_KEY:
                os.environ[settings.llm_key_env] = _BASELINE_KEY
            else:
                os.environ.pop(settings.llm_key_env, None)
        # 路由与 key 必须同源：每次同步都重建 patch（端点增删改/切换/启动都会走到这里）。
        _write_generated_patch(row)
        return row
