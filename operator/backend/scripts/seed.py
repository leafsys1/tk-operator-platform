#!/usr/bin/env python
"""种子数据：项目/服务器/首个样例任务（elementwise_add）+ 规格 v1 + 一轮 Loop。

用法（在 backend/ 下）：
    .venv/bin/python scripts/seed.py                # 建项目/服务器/示例任务与规格
    .venv/bin/python scripts/seed.py --probe        # 顺带跑一次真实探针（拉 npu-smi 与镜像清单）

**不造假数据**（方案 v1.1 §12）：真实执行（runs）必须来自容器里真跑，seed 不写假的 run/产物。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.models import Host, HostImage, Loop, Project, SpecVersion, Task  # noqa: E402

WORKSPACE = Path.home() / "Projects/personal/op-agent/workspaces/cann-ops"

#: 首个贯通样例算子的规格（方案 v1.1 §4.3 的字段表；缺失事实标 UNKNOWN）
SPEC_ELEMENTWISE_ADD = {
    "spec_version": 1,
    "operator": "elementwise_add",
    "source": {"input_form": "natural_language", "description": "对两个同形状张量逐元素相加"},
    "semantics": {"formula": "C[i] = A[i] + B[i]", "broadcasting": False, "attributes": []},
    "inputs": [
        {"name": "A", "shape": ["N"], "dtype": "float32", "layout": "ND", "contiguous": True},
        {"name": "B", "shape": ["N"], "dtype": "float32", "layout": "ND", "contiguous": True},
    ],
    "outputs": [{"name": "C", "shape": ["N"], "dtype": "float32"}],
    "constraints": {
        "N_range": [1, 1048576],
        "invalid_shape_behavior": "reject",
        "unsupported_dtype_behavior": "reject",
    },
    "target": {"hardware": "Ascend_910B1", "implementation_path": "Ascend_C"},
    "reference": {"implementation": "numpy 参考实现（独立于候选实现）", "path": "UNKNOWN"},
    "validation": {
        "tolerance_profile": "ops-precision-standard（官方档，按 dtype）",
        "required_case_groups": ["normal", "boundary", "invalid"],
        "empty_input_valid": False,
        "nan_inf_policy": "UNKNOWN",
    },
    "performance": {"scope": "operator", "baseline": "UNKNOWN", "target_latency_us": "UNKNOWN"},
    "delivery": {"required": ["kernel", "invocation", "build_script", "test_script", "reports"]},
    "unknowns": ["参考实现路径", "NaN/Inf 是否属于定义域", "性能基线"],
}

SPEC_ASSUMPTIONS = "同 shape（不做广播）；float32；N 为动态维度；非对齐尺寸必须覆盖（尾块）"

TASK_NEXT_ACTION = "P1.5-a：在容器里真跑一次 test 动作（四动作契约的第一次落地）"


def ensure_project(db, host: Host) -> Project:
    img = db.scalar(select(HostImage).where(HostImage.is_op_dev.is_(True)).limit(1))
    row = db.scalar(select(Project).where(Project.name == "CANN 算子开发"))
    if row:
        if row.host_id is None or row.image_id is None or row.workspace_path is None:
            row.host_id = row.host_id or host.id
            row.image_id = row.image_id or (img.id if img else None)
            row.workspace_path = row.workspace_path or str(WORKSPACE)
            db.commit()
            print(f"项目补齐绑定：执行机 #{row.host_id} / 镜像 #{row.image_id} / 工作区 {row.workspace_path}")
        return row
    row = Project(
        name="CANN 算子开发",
        workspace_path=str(WORKSPACE),
        host_id=host.id,
        image_id=img.id if img else None,
        dsh_profile="headless",
        dsh_home=str(Path.home() / ".dsh-op-agent"),
    )
    db.add(row)
    db.commit()
    return row


def ensure_host(db) -> Host:
    row = db.scalar(select(Host).where(Host.name == "tk36"))
    if row:
        return row
    row = Host(
        name="tk36",
        kind="baremetal",
        address="bms-fae",
        ssh_alias="tk36",
        cred_ref="~/work/pem/KeyPair-fae2027.pem",
        note="8×910B1 裸金属；执行委托目标（rsync + ssh docker run）",
    )
    db.add(row)
    db.commit()
    return row


def run_probe(db, host: Host) -> None:
    from app.config import settings
    from app.services.probe import probe_host

    from datetime import datetime, timezone

    res = probe_host(host.ssh_alias or host.address or "", settings.ssh_key, settings.ssh_opts, settings.ssh_timeout)
    now = datetime.now(timezone.utc)
    host.cards = res.cards or host.cards
    host.cann_version = res.cann_version or host.cann_version
    host.docker_ok = res.docker_ok
    host.npu_state = (res.npu_raw or "")[:4000]
    host.probe_state = "ok" if res.ok else "fail"
    host.last_probe_at = now
    existing = {(i.image_ref, i.tag): i for i in db.scalars(select(HostImage).where(HostImage.host_id == host.id))}
    for img in res.images:
        key = (img["image_ref"], img["tag"])
        row = existing.get(key)
        if row is None:
            db.add(HostImage(host_id=host.id, last_scanned_at=now, **img))
        else:
            row.size_bytes = img["size_bytes"] or row.size_bytes
            row.dangling = img["dangling"]
            if img.get("digest"):
                row.digest = img["digest"]
            row.last_scanned_at = now
    db.commit()
    print(f"探针：ok={res.ok} cards={res.cards} cann={res.cann_version} images={len(res.images)}")
    for e in res.errors:
        print("  !", e)


def ensure_first_task(db, project: Project) -> Task:
    """首个贯通样例：elementwise_add（D6 已定；complex64 mm 留作后续代表验证）。"""
    row = db.scalar(select(Task).where(Task.op_name == "elementwise_add"))
    if row:
        print(f"样例任务已存在：#{row.id}")
        return row
    row = Task(
        project_id=project.id,
        title="elementwise_add · 逐元素加（首个贯通样例）",
        op_name="elementwise_add",
        target_chip="910B1",
        input_form="自然语言 + 形状/dtype",
        stage="需求与设计",
        next_action=TASK_NEXT_ACTION,
    )
    db.add(row)
    db.commit()
    return row


def ensure_spec(db, task: Task) -> SpecVersion:
    row = db.scalar(select(SpecVersion).where(SpecVersion.task_id == task.id).order_by(SpecVersion.version))
    if row:
        print(f"规格已存在：v{row.version}")
        return row
    body = json.dumps(SPEC_ELEMENTWISE_ADD, ensure_ascii=False, sort_keys=True)
    row = SpecVersion(
        task_id=task.id,
        version=1,
        op_name="elementwise_add",
        input_form="自然语言 + 形状/dtype",
        raw_input="把两个同形状的 float32 张量逐元素相加，输出一个张量",
        spec_json=json.dumps(SPEC_ELEMENTWISE_ADD, ensure_ascii=False, indent=2),
        assumptions=SPEC_ASSUMPTIONS,
        unknowns=json.dumps(SPEC_ELEMENTWISE_ADD["unknowns"], ensure_ascii=False),
        change_reason="首版规格（D6 定的首条流程算子）",
        sha256=hashlib.sha256(body.encode()).hexdigest(),
    )
    db.add(row)
    db.commit()
    task.current_spec_version_id = row.id
    db.commit()
    print(f"规格 v{row.version} 已落（sha256 {row.sha256[:12]}…），并设为当前版本")
    return row


def ensure_loop(db, task: Task) -> Loop:
    row = db.scalar(select(Loop).where(Loop.task_id == task.id).limit(1))
    if row:
        return row
    row = Loop(
        task_id=task.id,
        kind="design",
        scope="按规格生成实现/接口/构建与测试脚本，并在容器里真跑一次 test",
        hypothesis="Ascend_C 直调路径足够跑通首条流程（不需要自定义算子注册）",
        judge_condition="test 动作的必需用例全部实际执行且达标（报告与当前代码快照版本一致）",
        status="open",
    )
    db.add(row)
    db.commit()
    print(f"Loop #{row.id}（design）已开")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="跑一次真实 SSH 探针")
    args = ap.parse_args()

    with SessionLocal() as db:
        host = ensure_host(db)
        project = ensure_project(db, host)
        print(f"项目 #{project.id} {project.name}；服务器 #{host.id} {host.name}")
        if args.probe:
            run_probe(db, host)
        task = ensure_first_task(db, project)
        ensure_spec(db, task)
        ensure_loop(db, task)
        print(f"任务 #{task.id} {task.title}（阶段：{task.stage}）")


if __name__ == "__main__":
    main()
