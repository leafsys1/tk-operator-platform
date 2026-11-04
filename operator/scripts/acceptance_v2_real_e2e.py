#!/usr/bin/env python3
"""任务生命周期 v2 · 真机端到端验收（elementwise_add 样例）。

流程：建任务 → 定版 v1.0 → 用例组确认 → build → smoke（新动作，编译+最小用例）
     → test（全量精度）→ benchmark（性能）→ 采纳优化（小版本+1，证据不重跑）
     → 交付包生成/下载 → 校验。
前置：workspace 已恢复 elementwise_add 样例（含 cases/ops/python/tests/docs）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8077/api"
DB = ROOT / "backend" / "opx.db"
WS = ROOT / "workspaces" / "cann-ops"
OP = "elementwise_add"
CRITERIA = "ops-precision-standard@1"


def req(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode()[:800]}")


def wait_job(job_id: int, timeout_s: int) -> dict:
    t0 = time.time()
    last = time.time()
    while time.time() - t0 < timeout_s:
        job = req("GET", f"/action-jobs/{job_id}")
        if not job.get("running"):
            return job
        if time.time() - last > 20:
            print(f"    … job #{job_id} 运行中（{int(time.time()-t0)}s）")
            last = time.time()
        time.sleep(5)
    raise SystemExit(f"job #{job_id} 超时（{timeout_s}s）")


def run_action(tid: int, action: str, timeout_s: int) -> dict:
    plan = req("POST", f"/tasks/{tid}/actions", {"action": action, "dry_run": True})
    print(f"[plan] {action}: ready={plan['ready']} device={plan.get('device_id')} "
          f"missing={plan['missing']} warnings={plan['warnings']}")
    if not plan["ready"]:
        raise SystemExit(f"{action} plan 不 ready：{plan['missing']}")
    started = req("POST", f"/tasks/{tid}/actions", {"action": action, "dry_run": False})
    job_id = started["action_job_id"]
    print(f"[run ] {action}: action_job #{job_id} run_ref={started['run_ref']}")
    job = wait_job(job_id, timeout_s)
    log_tail = "\n".join((job.get("log") or "").splitlines()[-8:])
    print(f"[done] {action}: status={job.get('status')} run_id={job.get('run_id')} "
          f"exit={job.get('exit_code')} gate={job.get('gate_verdict')}")
    if job.get("status") != "ok" or job.get("gate_verdict") != "pass":
        print("---- log tail ----\n" + log_tail + "\n------------------")
        raise SystemExit(f"{action} 失败：status={job.get('status')} gate={job.get('gate_verdict')}")
    return job


def main() -> int:
    db = sqlite3.connect(DB)
    draft = req("POST", "/tasks", {"project_id": 1, "title": "端到端验收 · elementwise_add（v2 真机）",
                                   "op_name": OP, "target_chip": "910B1", "is_draft": True,
                                   "raw_input": "v2 真机端到端验收（elementwise_add 样例）"})
    tid = draft["id"]
    raw = json.dumps({"fields": {
        "op_name": OP, "math_def": "y = a + b", "inputs": "a,b float32 同 shape N∈[1,1048576]",
        "outputs": "y float32 同 shape", "oracle": "torch CPU add", "tolerance": CRITERIA,
        "consistency": "shape/dtype 完全一致，NaN/Inf 逐元素精确比对",
        "performance_baseline": "torch.add (torch_npu)", "performance_target": "仅记录",
    }, "questions": [], "ready": True}, ensure_ascii=False)
    db.execute("update tasks set spec_draft=? where id=?", (raw, tid))
    db.commit()
    db.close()
    fin = req("POST", f"/tasks/{tid}/finalize")
    print(f"[1] task #{tid} finalize → spec v{fin['spec_version']['version']}")

    cases = (WS / "cases" / OP / "cases.json").read_bytes()
    bench = (WS / "cases" / OP / "benchmark.json").read_bytes()
    req("POST", f"/tasks/{tid}/cases/confirm", {
        "cases_sha256": hashlib.sha256(cases).hexdigest(),
        "benchmark_sha256": hashlib.sha256(bench).hexdigest(),
        "by": "acceptance-real",
    })
    print("[2] cases confirmed")

    run_action(tid, "build", 600)
    run_action(tid, "smoke", 600)
    run_action(tid, "test", 900)
    run_action(tid, "benchmark", 1200)

    print("[3] accept synthetic-scope perf Loop（验证采纳 = 小版本+1，候选证据不重跑）")
    loop = req("POST", f"/tasks/{tid}/loops", {
        "kind": "perf", "scope": "端到端验收：版本模型验证（不引入代码改动）",
        "hypothesis": "version bump only", "judge_condition": "synthetic accept for version model",
    })
    req("POST", f"/loops/{loop['id']}/close?conclusion=accepted")
    ver = req("GET", f"/tasks/{tid}/versions")
    print(f"    version={ver['current']['label']} source={ver['current']['source']} "
          f"minor={ver['current']['minor']}")
    assert ver["current"]["minor"] == 1
    ev = ver["evidence"]
    print(f"    evidence deliverable={ev['deliverable']} missing={ev['missing']}")
    assert ev["deliverable"], ev["missing"]

    print("[4] delivery")
    pack = req("POST", f"/tasks/{tid}/deliveries")
    print(f"    delivery run #{pack['run_id']} {pack['version']} {pack['size_bytes']}B")
    with urllib.request.urlopen(API + f"/deliveries/{pack['run_id']}/download", timeout=120) as r:
        data = r.read()
    assert data[:2] == b"PK"
    with zipfile.ZipFile(pack["zip_path"]) as zf:
        names = zf.namelist()
    checks = {
        "manifest": "manifest.json" in names,
        "code": any(n.startswith(f"ops/{OP}/op_host/") for n in names),
        "python_iface": any(n.startswith(f"ops/{OP}/python/") for n in names),
        "pytest": f"tests/test_{OP}.py" in names,
        "readme": "docs/算子说明文档.md" in names,
        "design": "docs/规划文档-补全.md" in names,
        "cases": "tests/cases.json" in names and "tests/benchmark.json" in names,
    }
    print("    zip entries:", len(names), checks)
    assert all(checks.values()), checks
    hist = req("GET", f"/tasks/{tid}/deliveries")["deliveries"]
    print(f"    delivery history={len(hist)} gate={hist[0].get('gate_verdict')}")
    stage = req("GET", f"/tasks/{tid}/stage")
    print(f"[5] stage derived={stage['derived_stage']} version v{stage.get('spec_version')}.{stage.get('version_minor')}")
    print(f"REAL_E2E PASS task_id={tid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
