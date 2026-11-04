"""API 管理（P4）：模型端点的登记、key 配置、**单价**与探针。

**边界（方案 §7 边界②）**：API 管理是**配置态**（用哪个模型 / 哪个 key / 什么价），不是消耗
（运行态，见 `routers/usage.py`）。key 可以存本机 SQLite（页面配置），接口只返回 `has_key` 不回显明文；
没配 key 时仍回退 `key_ref` 指向的环境变量 / 启动时的 `OPX_LLM_KEY`。探针只验证连通性、不产生生成费用。

**单价**（人民币元 / 每百万 token）：四项 `price_in` / `price_out` /
`price_cache_read` / `price_cache_write`；cache 两项留空 = 不单独计价（按 0 计）。
`price_in + price_out` 都配了 = 已定价。校验：非有限值（NaN/Inf）、负数、以及超过
`MAX_PRICE` 的离谱值一律 400（任务外页面审查 H3：Infinity 会让金额溢出、消耗页白屏）。

**软删**（任务外页面审查 H2）：端点不物理删除，置 `deleted_at`。历史 `agent_jobs.endpoint_id`
仍能解析到原端点与当时价目；同时 SQLite 主键不会被新端点复用。列表 / 当前端点 / 重名检查
过滤已删行，删除时把 name 改成 `<原名> #deleted-<id>` 以释放唯一名。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Endpoint
from ..schemas import EndpointCreate, EndpointOut, EndpointUpdate
from ..services import endpoint_probe, llm_config, secret_store

router = APIRouter(prefix="/api", tags=["endpoints"])

#: 允许的 provider（对应 dsh 的适配器；见 ops/dsh/README.md）
PROVIDERS = ("dsh-llm-pi-ai", "dsh-llm-deepseek")

PRICE_FIELDS = ("price_in", "price_out", "price_cache_read", "price_cache_write")

#: 单价上限（人民币元 / 每百万 token）。防止 1e308 这类值在金额计算里溢出成 inf。
MAX_PRICE = 1_000_000.0


def _check_prices(data: dict) -> None:
    """单价必须是有限、非负、且不超过上限；异常直接 400（别带进聚合）。"""
    for k in PRICE_FIELDS:
        v = data.get(k)
        if v is None:
            continue
        try:
            value = float(v)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"{k} 必须是数字") from e
        if not math.isfinite(value) or value < 0:
            raise HTTPException(400, f"{k} 必须是非负有限数")
        if value > MAX_PRICE:
            raise HTTPException(400, f"{k} 不能超过 {MAX_PRICE:g} 元 / 百万 token")
        data[k] = value


def _check_fields(data: dict) -> None:
    """provider / base_url / model / key_ref / max_model_len 的基本校验。"""
    if "provider" in data and data.get("provider") is not None:
        if data["provider"] not in PROVIDERS:
            raise HTTPException(400, f"provider 必须是 {PROVIDERS} 之一")
    if "base_url" in data:
        url = (data.get("base_url") or "").strip() or None
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise HTTPException(400, "base_url 必须是 http(s)://host[/path]")
            if parsed.query or parsed.fragment:
                raise HTTPException(400, "base_url 不要带 query / fragment（探针会拼 /models）")
        data["base_url"] = url
    for key in ("model", "key_ref"):
        if key in data:
            data[key] = (data.get(key) or "").strip() or None
    if "max_model_len" in data and data.get("max_model_len") is not None:
        try:
            value = int(data["max_model_len"])
        except (TypeError, ValueError) as e:
            raise HTTPException(400, "max_model_len 必须是整数") from e
        if value <= 0:
            raise HTTPException(400, "max_model_len 必须大于 0")
        data["max_model_len"] = value


def _out(row: Endpoint) -> dict:
    data = EndpointOut.model_validate(row).model_dump()
    stored = secret_store.get_secret(row.secret_ref) if row.secret_ref else None
    legacy = bool((row.api_key or "").strip())
    data["has_key"] = bool(stored or legacy)
    if stored:
        data["key_backend"] = secret_store.backend_for(row.secret_ref)
    elif legacy:
        data["key_backend"] = "legacy"  # 旧库明文列，启动迁移后应消失
    else:
        data["key_backend"] = None
    data["priced"] = row.price_in is not None and row.price_out is not None
    return data


def _active_rows(db: Session, *, include_deleted: bool = False):
    q = select(Endpoint)
    if not include_deleted:
        q = q.where(Endpoint.deleted_at.is_(None))
    return q.order_by(Endpoint.id)


def _activate(db: Session, row: Endpoint) -> None:
    for x in db.scalars(_active_rows(db)):
        x.is_active = x.id == row.id


def _mask_secret(value, secret: str | None):
    """把目标响应里可能回显的 key 替换掉（探针错误信息会展示前 200 字符）。"""
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "***")
    if isinstance(value, list):
        return [_mask_secret(v, secret) for v in value]
    if isinstance(value, dict):
        return {k: _mask_secret(v, secret) for k, v in value.items()}
    return value


@router.get("/endpoints", response_model=list[EndpointOut])
def list_endpoints(db: Session = Depends(get_db)) -> list[dict]:
    return [_out(r) for r in db.scalars(_active_rows(db))]


@router.post("/endpoints", response_model=EndpointOut, status_code=201)
def create_endpoint(payload: EndpointCreate, db: Session = Depends(get_db)) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "name 不能为空")
    data = payload.model_dump()
    _check_prices(data)
    _check_fields(data)
    # 端点名唯一性按不区分大小写（`cheap` / `CHEAP` 不能并存）。
    if db.scalar(select(Endpoint).where(func.lower(Endpoint.name) == name.lower(), Endpoint.deleted_at.is_(None))):
        raise HTTPException(400, f"端点名 {name!r} 已存在")
    raw_key = (data.pop("api_key", None) or "").strip() or None
    want_active = bool(data.pop("is_active", False))
    row = Endpoint(**{**data, "name": name, "probe_state": "unknown"})
    db.add(row)
    try:
        db.flush()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(400, f"端点名 {name!r} 已存在") from e
    # key 本体进本机密钥库（Keychain/secret-tool/0600 文件），DB 只留引用。
    if raw_key:
        ref = f"endpoint-{row.id}"
        try:
            secret_store.set_secret(ref, raw_key)
        except secret_store.SecretStoreError as e:
            db.rollback()
            raise HTTPException(503, f"密钥库不可用，端点未保存：{e}") from e
        row.secret_ref = ref
    # 第一个端点自动成为当前端点；显式勾选则抢过 active。
    # 注意：只统计**未软删**的 active 行（生产 Session 是 autoflush=False，
    # 这里先 flush，避免拿旧状态判断）。
    db.flush()
    if want_active or not db.scalar(
        select(Endpoint.id).where(Endpoint.is_active.is_(True), Endpoint.deleted_at.is_(None)).limit(1)
    ):
        _activate(db, row)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        if row.secret_ref:
            secret_store.delete_secret(row.secret_ref)
        raise HTTPException(400, f"端点名 {name!r} 已存在") from e
    db.refresh(row)
    llm_config.apply_active_key(db)
    return _out(row)


@router.patch("/endpoints/{endpoint_id}", response_model=EndpointOut)
def update_endpoint(endpoint_id: int, payload: EndpointUpdate, db: Session = Depends(get_db)) -> dict:
    row = db.get(Endpoint, endpoint_id)
    if not row or row.deleted_at is not None:
        raise HTTPException(404, "端点不存在")
    data = payload.model_dump(exclude_unset=True)
    _check_prices(data)
    _check_fields(data)
    if "name" in data:
        name = (data["name"] or "").strip()
        if not name:
            raise HTTPException(400, "name 不能为空")
        dup = db.scalar(
            select(Endpoint).where(
                func.lower(Endpoint.name) == name.lower(),
                Endpoint.id != endpoint_id,
                Endpoint.deleted_at.is_(None),
            )
        )
        if dup:
            raise HTTPException(400, f"端点名 {name!r} 已存在")
        data["name"] = name
    key_changed = "api_key" in data
    if key_changed:
        # 空字符串 = 清空（之后回退启动环境里的 key）；不传 = 不改。
        # 非空 = 写本机密钥库（DB 不落明文）；写失败时保留旧 key 并报 503。
        raw_key = (data.pop("api_key") or "").strip()
        if raw_key:
            ref = row.secret_ref or f"endpoint-{row.id}"
            try:
                secret_store.set_secret(ref, raw_key)
            except secret_store.SecretStoreError as e:
                raise HTTPException(503, f"密钥库不可用，key 未更新：{e}") from e
            row.secret_ref = ref
            row.api_key = None
        else:
            secret_store.delete_secret(row.secret_ref)
            row.secret_ref = None
            row.api_key = None
    if "is_active" in data:
        want_active = data.pop("is_active")
        if want_active is True:
            _activate(db, row)
        elif want_active is False:
            # 不允许把「当前端点」直接置空：运行态/计价会回退 MIN(id)，页面与后端就不一致了。
            raise HTTPException(400, "不能直接把当前端点取消；请用另一个端点的「设为当前」切换")
    for k, v in data.items():
        setattr(row, k, v)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(400, "端点名重复") from e
    db.refresh(row)
    # 只有「当前端点」的 key 变更才需要动进程环境；改非当前端点不能误清当前 key（审查 M6）。
    if row.is_active:
        llm_config.apply_active_key(db, clear=key_changed)
    return _out(row)


@router.post("/endpoints/{endpoint_id}/activate", response_model=EndpointOut)
def activate_endpoint(endpoint_id: int, db: Session = Depends(get_db)) -> dict:
    """把某个端点设为当前 API：agent 调用优先用它的 key。"""
    row = db.get(Endpoint, endpoint_id)
    if not row or row.deleted_at is not None:
        raise HTTPException(404, "端点不存在")
    _activate(db, row)
    db.commit()
    db.refresh(row)
    # 新当前端点若没配页面 key，回退启动时的环境 key（而不是把 env 清掉）。
    llm_config.apply_active_key(db, clear=True)
    return _out(row)


@router.delete("/endpoints/{endpoint_id}", status_code=204)
def delete_endpoint(endpoint_id: int, db: Session = Depends(get_db)) -> Response:
    """软删端点：保留行/id 供历史调用计价；释放唯一名；当前端点顺延到下一台未删端点。"""
    row = db.get(Endpoint, endpoint_id)
    if not row or row.deleted_at is not None:
        raise HTTPException(404, "端点不存在")
    was_active = row.is_active
    secret_ref = row.secret_ref
    row.deleted_at = datetime.now(timezone.utc)
    row.is_active = False
    row.api_key = None  # 旧明文列一并清掉
    row.secret_ref = None
    suffix = f" #deleted-{row.id}"
    row.name = (row.name[: max(0, 200 - len(suffix))] + suffix)
    # 生产 SessionLocal 是 autoflush=False：必须先 flush，否则下面按 deleted_at 查顺延对象
    # 仍会读到「本行未删」的旧状态，把刚删掉的行又设回 active（复核发现的回归）。
    db.flush()
    if was_active:
        replacement = db.scalar(
            select(Endpoint)
            .where(Endpoint.deleted_at.is_(None), Endpoint.id != endpoint_id)
            .order_by(Endpoint.id)
            .limit(1)
        )
        if replacement is not None:
            replacement.is_active = True
    db.commit()
    # 库先提交再删密钥（删失败也不影响端点删除）；失败只记录，不阻塞。
    try:
        secret_store.delete_secret(secret_ref)
    except Exception:  # noqa: BLE001 —— 密钥库清理失败不该让删除接口 500
        pass
    if was_active:
        llm_config.apply_active_key(db, clear=True)
    return Response(status_code=204)


def _run_probe(row: Endpoint, db: Session) -> dict:
    stored = secret_store.get_secret(row.secret_ref) if row.secret_ref else None
    if row.secret_ref and not stored:
        # 换机器/手删 Keychain 后 DB 引用还在：明确报错，不要拿环境 key 冒充这个端点。
        result = {"ok": False, "error": "本机密钥库中找不到该端点的 key（可能换机器或已被删），请重新配置"}
    else:
        result = endpoint_probe.probe(
            row.base_url, row.key_ref,
            api_key=stored or ((row.api_key or "").strip() or None),
        )
    secret = stored or ((row.api_key or "").strip() or None) or _env_key_for(row)
    result = _mask_secret(result, secret)
    row.probe_state = "ok" if result.get("ok") else "fail"
    row.last_probe_at = datetime.now(timezone.utc)
    db.commit()
    return {"endpoint_id": row.id, "probe_state": row.probe_state,
            "last_probe_at": row.last_probe_at.isoformat() if row.last_probe_at else None,
            "probe": result}


def _env_key_for(row: Endpoint) -> str | None:
    """探针回退 key_ref 环境变量时，也把该值纳入掩码范围。"""
    import os

    ref = (row.key_ref or "").strip()
    return os.environ.get(ref) if ref else None


@router.post("/endpoints/{endpoint_id}/probe")
def probe_endpoint(endpoint_id: int, db: Session = Depends(get_db)) -> dict:
    """探一次端点连通性（`GET {base_url}/models`）；结果落 `probe_state` / `last_probe_at`。"""
    row = db.get(Endpoint, endpoint_id)
    if not row or row.deleted_at is not None:
        raise HTTPException(404, "端点不存在")
    return _run_probe(row, db)


@router.post("/endpoints/probe")
def probe_all_endpoints(db: Session = Depends(get_db)) -> dict:
    """全部探一次（页面「全部探针」用）。"""
    return {"results": [_run_probe(r, db) for r in db.scalars(_active_rows(db))]}
