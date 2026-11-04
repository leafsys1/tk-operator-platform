"""服务器与镜像：登记 + 探针（SSH 只读）+ 标记算子开发镜像。"""

import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Host, HostImage, Project, Run, Task
from ..schemas import HostCreate, HostImageOut, HostImagePatch, HostOut, HostUpdate
from ..services.probe import probe_host
from ..services.ssh import ssh_key, ssh_port, ssh_target, validate_ssh_target

router = APIRouter(prefix="/api", tags=["hosts"])

#: host_id → 探针锁（进程内；单进程部署足够，多进程需 DB 状态/分布式锁）
_PROBE_LOCKS: dict[int, threading.Lock] = {}
_PROBE_LOCKS_GUARD = threading.Lock()


def _probe_lock(host_id: int) -> threading.Lock:
    with _PROBE_LOCKS_GUARD:
        return _PROBE_LOCKS.setdefault(host_id, threading.Lock())


def _clean(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def _validated_target(value: str | None) -> str | None:
    """SSH 目标校验：非法值（`-` 开头/空白/控制字符）转 400。

    不校验就是选项注入面：`ssh_alias="-oProxyCommand=..."` 会在控制台主机执行命令（审查 M1）。
    """
    try:
        return validate_ssh_target(value)
    except ValueError as e:
        raise HTTPException(400, f"SSH 连接目标非法：{e}") from e


def _host_out(db: Session, host: Host) -> HostOut:
    total = db.scalar(select(func.count()).select_from(HostImage).where(HostImage.host_id == host.id)) or 0
    opdev = (
        db.scalar(
            select(func.count()).select_from(HostImage).where(HostImage.host_id == host.id, HostImage.is_op_dev.is_(True))
        )
        or 0
    )
    out = HostOut.model_validate(host)
    out.image_count = int(total)
    out.op_dev_image_count = int(opdev)
    # 探针失败/部分失败原因从 npu_state 回读（失败时这里是错误摘要；成功但有告警时是 [探针告警] 段）
    raw = host.npu_state or ""
    if host.probe_state == "fail":
        out.probe_error = raw[:500] or None
    elif "[探针告警]" in raw:
        out.probe_error = ("[探针告警] " + raw.split("[探针告警]", 1)[1])[:500]
    return out


@router.get("/hosts", response_model=list[HostOut])
def list_hosts(db: Session = Depends(get_db)) -> list[HostOut]:
    hosts = db.scalars(select(Host).order_by(Host.is_test_server.desc(), Host.id)).all()
    return [_host_out(db, h) for h in hosts]


@router.post("/hosts", response_model=HostOut, status_code=201)
def create_host(payload: HostCreate, db: Session = Depends(get_db)) -> HostOut:
    """登记一台服务器：**SSH 连接目标是唯一必填项**，名称缺省时直接取目标。

    `cred_ref` 是私钥路径引用（可空 = 用全局 `OPX_SSH_KEY`），不是私钥内容。
    """
    ssh_alias = _validated_target(payload.ssh_alias)
    address = _validated_target(payload.address)
    if not (ssh_alias or address):
        raise HTTPException(400, "至少填一个 SSH 连接目标：ssh_alias（~/.ssh/config alias 或 user@host）或 address")
    name = _clean(payload.name) or ssh_alias or address
    if db.scalar(select(Host).where(Host.name == name)):
        raise HTTPException(409, f"服务器 {name} 已存在")
    host = Host(name=name, address=address, ssh_alias=ssh_alias,
                cred_ref=_clean(payload.cred_ref), note=_clean(payload.note))
    db.add(host)
    db.flush()
    # 第一台服务器自动成为当前测试服务器，省一次点击
    if not db.scalar(select(Host).where(Host.is_test_server.is_(True)).limit(1)):
        host.is_test_server = True
    db.commit()
    return _host_out(db, host)


@router.post("/hosts/ssh-key")
async def upload_ssh_key(file: UploadFile = File(...)) -> dict[str, str]:
    """接收浏览器文件选择框选中的私钥，落到本机 key 目录（0600），返回路径。

    - **私钥不进数据库、不回显内容**：`hosts.cred_ref` 只存这个路径；
    - 复制而不是引用原路径：浏览器安全模型拿不到用户选择的绝对路径；
    - 文件按时间戳 + 随机后缀命名，避免覆盖；原文件名只回给页面做提示。
    """
    # 分块读 + 边读边限长：不要把一个可能很大的上传整个读进内存（审查 hosts #11）。
    max_bytes = 64 * 1024
    buf = bytearray()
    while True:
        chunk = await file.read(8192)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(400, "私钥文件太大（>64 KB），不像一个私钥")
    raw = bytes(buf)
    if not raw:
        raise HTTPException(400, "文件为空")
    # 收紧到 PEM/OpenSSH 私钥头（此前只查子串，任意文本都能过）。
    if not re.search(rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----", raw):
        raise HTTPException(400, "看起来不是私钥文件（缺少 PEM/OpenSSH 私钥头）；请选择 PEM/OpenSSH 私钥")

    key_dir = Path(settings.ssh_key_dir).expanduser()
    try:
        key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        key_dir.chmod(0o700)
    except OSError as e:
        raise HTTPException(500, f"创建私钥目录失败：{e}") from e

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pem", ".key"}:
        suffix = ".pem"
    name = f"ssh-key-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}{suffix}"
    path = key_dir / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(raw)
    return {"path": str(path), "name": file.filename or name}


@router.patch("/hosts/{host_id}", response_model=HostOut)
def update_host(host_id: int, payload: HostUpdate, db: Session = Depends(get_db)) -> HostOut:
    """改登记信息（名称 / SSH 目标 / 私钥引用 / 备注）；探针结果不在这里改。"""
    host = db.get(Host, host_id)
    if not host:
        raise HTTPException(404, "服务器不存在")
    data = payload.model_dump(exclude_unset=True)
    ssh_alias = _validated_target(data["ssh_alias"]) if "ssh_alias" in data else host.ssh_alias
    address = _validated_target(data["address"]) if "address" in data else host.address
    if not (ssh_alias or address):
        raise HTTPException(400, "SSH 连接目标不能为空（ssh_alias 或 address 至少留一个）")
    if "name" in data:
        name = _clean(data["name"])
        if not name:
            raise HTTPException(400, "服务器名称不能为空")
        if name != host.name and db.scalar(select(Host).where(Host.name == name)):
            raise HTTPException(409, f"服务器 {name} 已存在")
        host.name = name
    host.ssh_alias = ssh_alias
    host.address = address
    if "cred_ref" in data:
        host.cred_ref = _clean(data["cred_ref"])
    if "note" in data:
        host.note = _clean(data["note"])
    db.commit()
    db.refresh(host)
    return _host_out(db, host)


@router.delete("/hosts/{host_id}", status_code=204)
def delete_host(host_id: int, db: Session = Depends(get_db)) -> Response:
    """删除服务器（连带其镜像清单）。

    有引用就拒绝：项目/任务/执行绑定过这台机或它的镜像时，直接删会留下悬空引用，
    所以先让用户改绑/删任务。删当前测试服务器时，自动把第一台剩下的服务器顺延为测试服务器。
    """
    host = db.get(Host, host_id)
    if not host:
        raise HTTPException(404, "服务器不存在")

    image_ids = select(HostImage.id).where(HostImage.host_id == host.id)
    refs = {
        "项目": db.scalar(
            select(func.count()).select_from(Project).where(
                or_(Project.host_id == host.id, Project.image_id.in_(image_ids))
            )
        ) or 0,
        "任务": db.scalar(
            select(func.count()).select_from(Task).where(
                or_(Task.host_id == host.id, Task.image_id.in_(image_ids))
            )
        ) or 0,
        "执行": db.scalar(
            select(func.count()).select_from(Run).where(
                or_(Run.host_id == host.id, Run.image_id.in_(image_ids))
            )
        ) or 0,
    }
    used = " / ".join(f"{k} {v}" for k, v in refs.items() if v)
    if used:
        raise HTTPException(
            409,
            f"服务器「{host.name}」仍被引用（{used}）：先把项目/任务的执行机与镜像改绑到别的服务器"
            f"（或删除对应任务）再删。",
        )

    was_test = host.is_test_server
    db.delete(host)  # Host.images 是 cascade=all,delete-orphan，镜像清单一并删
    if was_test:
        replacement = db.scalar(
            select(Host).where(Host.id != host.id).order_by(Host.id).limit(1)
        )
        if replacement is not None:
            replacement.is_test_server = True
    db.commit()
    return Response(status_code=204)


@router.post("/hosts/{host_id}/set-test", response_model=HostOut)
def set_test_server(host_id: int, db: Session = Depends(get_db)) -> HostOut:
    """把某台服务器设为「当前测试服务器」（全局唯一）。

    任务未显式指定 `host_id` 时，`exec_actions.build_plan()` 优先用它；
    项目默认执行机作为次级回退。
    """
    host = db.get(Host, host_id)
    if not host:
        raise HTTPException(404, "服务器不存在")
    for row in db.scalars(select(Host)):
        row.is_test_server = row.id == host_id
    db.commit()
    db.refresh(host)
    return _host_out(db, host)


#: 镜像元数据字段 → 列长度上限（超长截断；与 HostImage 的 String 长度一致）
_IMAGE_META_LIMITS = {"digest": 200, "cann_version": 120, "python_version": 60, "torch_version": 60}


def _image_in_use(db: Session, image_id: int) -> bool:
    """镜像是否被项目/任务/执行引用。被引用的行不能硬删（外键悬空），只能标 missing。"""
    for model in (Project, Task, Run):
        if db.scalar(select(func.count()).select_from(model).where(model.image_id == image_id)):
            return True
    return False


def _reconcile_images(db: Session, host: Host, images: list[dict], now: datetime) -> None:
    """一次探针结果对镜像清单做整批对账（审查 M12）。

    - 键：`image_id` 优先（能区分多张 `<none>:<none>` 悬空镜像，它们没有 repo/tag），
      没有 ID 才退回 `(image_ref, tag)`；
    - 本次扫到的行 → `scan_state="scanned"`；
    - 本次没扫到的行：有 `is_test`/`is_op_dev` 标记或被项目/任务/执行引用 → `scan_state="missing"`，
      否则直接删除（远端已不存在的镜像不该继续出现在清单里）。
    """
    existing = list(db.scalars(select(HostImage).where(HostImage.host_id == host.id)))
    by_id = {row.image_id: row for row in existing if row.image_id}
    by_ref = {(row.image_ref, row.tag): row for row in existing}
    seen: set[int] = set()

    for img in images:
        row = None
        if img.get("image_id"):
            cand = by_id.get(img["image_id"])
            if cand is not None and cand.id not in seen:
                row = cand
        if row is None:
            cand = by_ref.get((img.get("image_ref"), img.get("tag")))
            if cand is not None and cand.id not in seen:
                row = cand
        if row is None:
            row = HostImage(host_id=host.id, **img)
            db.add(row)
            db.flush()
        else:
            row.image_ref = img.get("image_ref") or row.image_ref
            row.tag = img.get("tag")
            row.image_id = img.get("image_id") or row.image_id
            row.created_raw = img.get("created_raw") or row.created_raw
            row.size_bytes = img.get("size_bytes") or row.size_bytes
            row.dangling = bool(img.get("dangling"))
        row.scan_state = "scanned"
        row.last_scanned_at = now
        seen.add(row.id)
        if row.image_id:
            by_id[row.image_id] = row
        by_ref[(row.image_ref, row.tag)] = row

    for row in existing:
        if row.id in seen:
            continue
        if row.is_test or row.is_op_dev or _image_in_use(db, row.id):
            row.scan_state = "missing"
        else:
            db.delete(row)


@router.post("/hosts/{host_id}/probe", response_model=HostOut)
def probe(host_id: int, db: Session = Depends(get_db)) -> HostOut:
    host = db.get(Host, host_id)
    if not host:
        raise HTTPException(404, "服务器不存在")
    if not (host.ssh_alias or host.address):
        raise HTTPException(400, "该服务器没有 ssh_alias / address，无法探针")

    # per-host 探针锁（审查 hosts #7）：保存后的后台探针与手动「重新探针」可能同时到；
    # 并发探针会重复 SSH/重复插镜像行，这里直接 409 让后到者重试。进程内锁足够单进程部署。
    lock = _probe_lock(host_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "该服务器正在探针中，请稍后再试")
    try:
        return _do_probe(host, db)
    finally:
        lock.release()


def _do_probe(host: Host, db: Session) -> HostOut:
    # 探针与真执行同源：目标 / 端口 / 私钥都按 host 解析（私钥空则回退全局）。
    res = probe_host(ssh_target(host), ssh_key(host), settings.ssh_opts, settings.ssh_timeout,
                     port=ssh_port(host))
    now = datetime.now(timezone.utc)
    existing_images = list(db.scalars(select(HostImage).where(HostImage.host_id == host.id)))
    host.last_probe_at = now

    if res.ok:
        # 成功：用本机真实结果覆盖，不再 `or` 回填旧值（换目标后会把 A 机信息挂到 B 目标下，审查 #4）
        host.cards = res.cards
        host.cann_version = res.cann_version
        host.docker_ok = res.docker_ok
        raw = (res.npu_raw or "")
        if res.errors:
            # 部分成功（如卡拿到了、docker 命令失败）也要留痕，页面用 probe_error 展示（审查 hosts #9）
            raw += "\n[探针告警] " + "；".join(res.errors)
        host.npu_state = raw[:4000]
        host.probe_state = "ok"
        if res.hostname:
            host.address = host.address or res.hostname
        if res.docker_ok:
            _reconcile_images(db, host, res.images, now)
        else:
            # docker 命令没成功 ≠ 「0 个镜像」：本轮清单不可信，保留旧行但标 stale
            for row in existing_images:
                row.scan_state = "stale"
    else:
        # 失败：目标可能已换/私钥错，旧机的派生信息必须清掉，只留错误摘要给页面排错（审查 M11）
        host.cards = None
        host.cann_version = None
        host.docker_ok = False
        host.npu_state = ("；".join(res.errors) or "探针失败：ssh 无输出")[:4000]
        host.probe_state = "fail"
        for row in existing_images:
            row.scan_state = "stale"

    db.commit()
    if not res.ok:
        raise HTTPException(502, ("；".join(res.errors) or "探针失败")[:500])
    return _host_out(db, host)


@router.get("/hosts/{host_id}/images", response_model=list[HostImageOut])
def list_host_images(host_id: int, db: Session = Depends(get_db)) -> list[HostImage]:
    return list(
        db.scalars(
            select(HostImage)
            .where(HostImage.host_id == host_id)
            .order_by(HostImage.is_test.desc(), HostImage.is_op_dev.desc(),
                      HostImage.dangling, HostImage.size_bytes.desc().nullslast())
        )
    )


@router.patch("/host-images/{image_id}", response_model=HostImageOut)
def patch_image(image_id: int, payload: HostImagePatch, db: Session = Depends(get_db)) -> HostImage:
    """改镜像标记 / 元数据（只改显式提交的字段，未提交的字段保持原值）。

    - 元数据（digest/CANN/Python/torch）strip 后按列长度截断（审查 M10：此前 schema 不收这些字段，
      页面标记的新镜像永远缺 torch_version，test/benchmark/profile 的 plan 永久缺项）；
    - `is_op_dev=true` 要求最终 `env_recipe` 非空（审查 M9：不能标记成算子开发镜像却没有配方）；
    - 悬空镜像（`<none>:<none>`）禁止标记为测试/算子开发镜像（审查 #13）。
    """
    row = db.get(HostImage, image_id)
    if not row:
        raise HTTPException(404, "镜像不存在")
    data = payload.model_dump(exclude_unset=True)

    for key, limit in _IMAGE_META_LIMITS.items():
        if key in data:
            text = _clean(data[key])
            data[key] = text[:limit] if text else None
    for key in ("env_recipe", "preflight_cmd", "note"):
        if key in data:
            data[key] = _clean(data[key])
    # 非空列：显式 null 当作「不改」
    for key in ("is_test", "is_op_dev"):
        if key in data and data[key] is None:
            data.pop(key)

    final_op_dev = data.get("is_op_dev", row.is_op_dev)
    final_recipe = data.get("env_recipe", row.env_recipe)
    if row.dangling and (data.get("is_test") or data.get("is_op_dev")):
        raise HTTPException(400, "悬空镜像（<none>:<none>）不能标记为测试镜像/算子开发镜像")
    if final_op_dev and not (final_recipe or "").strip():
        raise HTTPException(400, "算子开发镜像必须带 env_recipe（怎么把它变成可跑环境），请同时提交 env_recipe")

    for key, value in data.items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return row
