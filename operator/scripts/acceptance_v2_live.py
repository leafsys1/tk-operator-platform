#!/usr/bin/env python3
"""v2 真机验收（本机 live API）：建任务 → 定版 → 确认用例 → build（真机）→ 门禁 → 设备检查。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import time
import urllib.request

API = "http://127.0.0.1:8077/api"
DB = "backend/opx.db"
WS = "workspaces/cann-ops"


def req(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode()[:800]}")


# 1) 建草稿任务
task = req("POST", "/tasks", {
    "project_id": 1, "title": "v2 真机验收 · elementwise_add", "op_name": "elementwise_add",
    "target_chip": "910B1", "is_draft": True, "raw_input": "a+b（验收用，来自备份样例）",
})
tid = task["id"]
print(f"[1] task #{tid} created")

# 2) 直接写 spec_draft（真流程由任务开始对话写入），然后 finalize
spec = {
    "op_name": "elementwise_add", "math_def": "y = a + b", "target_chip": "910B1",
    "inputs": "a,b float32 同 shape N∈[1,1048576]", "outputs": "y float32 同 shape",
    "oracle": "torch CPU add", "tolerance": "ops-precision-standard@1",
    "consistency": "shape/dtype 完全一致，NaN/Inf 逐元素精确比对",
    "performance_baseline": "torch.add (torch_npu)", "performance_target": "仅记录",
}
raw = json.dumps({"fields": spec, "questions": [], "ready": True}, ensure_ascii=False)
db = sqlite3.connect(DB)
db.execute("update tasks set spec_draft=? where id=?", (raw, tid))
db.commit()
db.close()
fin = req("POST", f"/tasks/{tid}/finalize")
ver = req("GET", f"/tasks/{tid}/versions")
print(f"[2] finalized spec v{fin['spec_version']['version']}；版本 {ver['current']['label']}")

# 3) 确认用例组（精度 + 性能两文件 sha256）
cases = open(f"{WS}/cases/elementwise_add/cases.json", "rb").read()
bench = open(f"{WS}/cases/elementwise_add/benchmark.json", "rb").read()
conf = req("POST", f"/tasks/{tid}/cases/confirm", {
    "cases_sha256": hashlib.sha256(cases).hexdigest(),
    "benchmark_sha256": hashlib.sha256(bench).hexdigest(),
    "by": "acceptance",
})
print(f"[3] cases confirmed；in_sync="
      f"{req('GET', f'/tasks/{tid}/cases/confirmation')['in_sync']}")

# 4) build：先 plan 再真执行（真机、不需要空闲设备）
plan = req("POST", f"/tasks/{tid}/actions", {"action": "build", "dry_run": True})
print(f"[4] build plan ready={plan['ready']} missing={plan['missing']} warnings={plan['warnings']}")
if not plan["ready"]:
    raise SystemExit("build plan 不 ready，停止")
started = req("POST", f"/tasks/{tid}/actions", {"action": "build", "dry_run": False})
job_id = started["action_job_id"]
print(f"[5] build 已起 action_job #{job_id} run_ref={started['run_ref']}，等待…")
job = {}
for i in range(90):
    time.sleep(5)
    job = req("GET", f"/action-jobs/{job_id}")
    if not job.get("running"):
        break
print(f"[6] build 结束 status={job.get('status')} run_id={job.get('run_id')} exit={job.get('exit_code')} "
      f"gate={job.get('gate_verdict')} err={str(job.get('error'))[:300]}")
print("    日志尾部：\n" + "\n".join((job.get("log") or "").splitlines()[-12:]))

# 7) run + 门禁
runs = req("GET", f"/tasks/{tid}")["runs"]
for r in runs:
    print(f"[7] run #{r['id']} action={r['action']} status={r['status']} exit={r['exit_code']}")

# 8) test plan（应被设备占用拦住）
test_plan = req("POST", f"/tasks/{tid}/actions", {"action": "test", "dry_run": True})
print(f"[8] test plan ready={test_plan['ready']}")
print(f"    missing={json.dumps(test_plan['missing'], ensure_ascii=False)}")
print(f"    warnings={json.dumps(test_plan['warnings'], ensure_ascii=False)}")
print(f"    device={test_plan.get('device_id')}")

# 9) 交付门禁（live）：应列出 test/benchmark 缺证据
prev = req("GET", f"/tasks/{tid}/deliveries/preview")
print(f"[9] delivery preview ok={prev.get('ok')} missing={json.dumps(prev.get('missing'), ensure_ascii=False)[:800]}")
try:
    req("POST", f"/tasks/{tid}/deliveries")
    print("    !! deliveries 竟然成功（证据应不齐）")
except SystemExit as e:
    print(f"    delivery 409 符合预期：{str(e)[:400]}")

print(f"TASK_ID={tid}")
