#!/usr/bin/env python3
"""任务生命周期 v2 · 端到端验收（**合成证据**，用于验证 live 控制台的版本/交付链路）。

背景：真机 test/benchmark 需要空闲 NPU 卡，当前共享机 8 卡长期被占。本脚本用**明确标注为合成的**
run 证据，在真实控制台/数据库上把「定版 → 用例确认 → 四动作台账 → 版本证据 → 采纳小版本 → 交付包 →
下载」整条链路跑一遍，验证 v2 的接线（不代表真机正确性/性能结论）。

产物：临时任务 + 临时算子目录（e2e_synth）；结束时全部删除（DB 0 / 工作区恢复）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8077/api"
DB = ROOT / "backend" / "opx.db"
WS = ROOT / "workspaces" / "cann-ops"
RUNS = ROOT / "runs"
OP = "e2e_synth"
CRITERIA = "ops-precision-standard@1"
DIGEST = "sha256:deadbeef-synthetic"

sys.path.insert(0, str(ROOT / "backend"))
from app.services import case_confirm, exec_actions  # noqa: E402


def req(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode()[:600]}")


def write_synth_files() -> None:
    cases_dir = WS / "cases" / OP
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "cases.json").write_text(json.dumps({
        "schema": "opx-cases/1", "op": OP, "dtype": "float32",
        "criteria_ref": CRITERIA, "case_groups": ["常规"],
        "cases": [{"case_id": "e2e_norm_001", "group": "常规", "required": True,
                   "n_a": 64, "n_b": 64, "dtype": "float32",
                   "gen_a": {"dist": "zeros"}, "gen_b": {"dist": "zeros"},
                   "seed": 7, "expected": "match", "note": "合成验收用例（不跑真机）"}],
    }, ensure_ascii=False), encoding="utf-8")
    (cases_dir / "benchmark.json").write_text(json.dumps({
        "schema": "opx-benchmark/1", "op": OP, "dtype": "float32",
        "primary_case": "e2e_bench_64",
        "cases": [{"case_id": "e2e_bench_64", "required": True, "n": 64, "dtype": "float32",
                   "gen_a": {"dist": "zeros"}, "gen_b": {"dist": "zeros"},
                   "seed": 8, "note": "合成验收性能用例（不跑真机）"}],
    }, ensure_ascii=False), encoding="utf-8")
    op_dir = WS / "ops" / OP
    (op_dir / "op_host").mkdir(parents=True, exist_ok=True)
    (op_dir / "op_host" / f"{OP}.asc").write_text("// synthetic acceptance operator\n", encoding="utf-8")
    (op_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (op_dir / "scripts" / "golden.py").write_text(
        "import numpy as np\ndef compute_golden(a, b):\n    return a + b\n", encoding="utf-8")
    (op_dir / "python").mkdir(parents=True, exist_ok=True)
    (op_dir / "python" / f"{OP}.py").write_text("# synthetic python iface\n", encoding="utf-8")
    (op_dir / "README.md").write_text("# e2e_synth（合成验收）\n", encoding="utf-8")
    (WS / "tests").mkdir(exist_ok=True)
    (WS / "tests" / f"test_{OP}.py").write_text("# synthetic pytest\n", encoding="utf-8")
    (WS / "docs" / OP).mkdir(parents=True, exist_ok=True)
    (WS / "docs" / OP / "design.md").write_text("# e2e_synth 规划文档（合成验收）\n", encoding="utf-8")


def write_run_dir(name: str, action: str, *, snapshot: str, spec_id: int, suite: str) -> Path:
    rd = RUNS / name
    (rd / "out").mkdir(parents=True, exist_ok=True)
    arts = []

    def add(rel: str, body: bytes, atype: str) -> None:
        p = rd / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        arts.append({"type": atype, "path": rel,
                     "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)})

    add("out/run.log", b"synthetic acceptance run\n", "log")
    if action == "smoke":
        add("out/smoke.json", json.dumps({
            "schema": "opx-smoke/1", "verdict": "PASS",
            "spec_version_id": spec_id, "code_snapshot_sha256": snapshot,
            "criteria_ref": CRITERIA, "suite_ref": suite,
            "case": {"case_id": "e2e_norm_001", "status": "PASS", "shape": [64]},
        }).encode(), "smoke")
    if action == "test":
        add("out/correctness.json", json.dumps({
            "spec_version_id": spec_id, "code_snapshot_sha256": snapshot,
            "criteria_ref": CRITERIA, "suite_ref": suite,
            "cases": [{"case_id": "e2e_norm_001", "status": "PASS", "required": True}],
        }).encode(), "correctness")
    if action == "benchmark":
        add("out/performance.json", json.dumps({
            "spec_version_id": spec_id, "code_snapshot_sha256": snapshot,
            "criteria_ref": CRITERIA, "suite_ref": suite,
            "summary": {"primary_case": "e2e_bench_64"},
            "cases": [{"case_id": "e2e_bench_64", "primary": True, "required": True}],
        }).encode(), "performance")
    man = {"schema": "opx-manifest/1", "run_id": name, "exit_code": 0,
           "code_snapshot_sha256": snapshot, "spec_version_id": spec_id,
           "criteria_ref": CRITERIA, "suite_ref": suite, "image_digest": DIGEST,
           "image": "synthetic", "device": "0", "lock_mode": "exclusive",
           "snapshot": {"files": 1, "truncated": False}, "artifacts": arts}
    (rd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return rd / "manifest.json"


def main() -> int:
    # 0) 清理上次残留 + 写合成工作区
    for p in [WS / "cases" / OP, WS / "ops" / OP, WS / "docs" / OP,
              WS / "tests" / f"test_{OP}.py"]:
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    write_synth_files()

    # 1) 建任务 → 写 spec_draft → finalize
    task = req("POST", "/tasks", {"project_id": 1, "title": "v2 端到端验收（合成证据）",
                                  "op_name": OP, "target_chip": "910B1", "is_draft": True,
                                  "raw_input": "合成验收：验证 v2 版本/交付链路，不跑真机"})
    tid = task["id"]
    raw = json.dumps({"fields": {"op_name": OP, "math_def": "y=a+b", "inputs": "a,b float32",
                                 "outputs": "y float32", "oracle": "torch CPU",
                                 "tolerance": CRITERIA, "performance_target": "仅记录"},
                      "questions": [], "ready": True}, ensure_ascii=False)
    db = sqlite3.connect(DB)
    db.execute("update tasks set spec_draft=? where id=?", (raw, tid))
    db.commit()
    db.close()
    fin = req("POST", f"/tasks/{tid}/finalize")
    spec_id = fin["spec_version"]["id"]
    print(f"[1] task #{tid} finalize → spec v{fin['spec_version']['version']}")

    # 2) 用例确认
    cases = (WS / "cases" / OP / "cases.json").read_bytes()
    bench = (WS / "cases" / OP / "benchmark.json").read_bytes()
    req("POST", f"/tasks/{tid}/cases/confirm", {
        "cases_sha256": hashlib.sha256(cases).hexdigest(),
        "benchmark_sha256": hashlib.sha256(bench).hexdigest(),
        "by": "acceptance-synthetic",
    })
    suite = case_confirm.cases_ref(WS)
    snap = exec_actions.snapshot(WS).sha256
    print(f"[2] cases confirmed；suite={suite} snapshot={snap[:12]}…")

    # 3) 登记 build/smoke/test/benchmark 合成 run（明确 synthetic）
    run_ids = {}
    for action in ("build", "smoke", "test", "benchmark"):
        man = write_run_dir(f"e2e-{action}", action, snapshot=snap, spec_id=spec_id, suite=suite)
        body = {"action": action, "status": "ok", "exit_code": 0,
                "spec_version_id": spec_id, "code_snapshot_sha256": snap,
                "suite_ref": suite, "criteria_ref": CRITERIA, "image_digest": DIGEST,
                "device_id": "0", "remote_workdir": f"e2e-{action}",
                "manifest_path": str(man), "note": "合成验收证据（不跑真机）"}
        row = req("POST", f"/tasks/{tid}/runs", body)
        run_ids[action] = row["id"]
    print(f"[3] synthetic runs registered: {run_ids}")

    # 4) 版本证据 + 阶段
    ev = req("GET", f"/tasks/{tid}/versions")["evidence"]
    print(f"[4] versions evidence: deliverable={ev['deliverable']} missing={ev['missing']}")
    assert ev["deliverable"], ev["missing"]
    stage = req("GET", f"/tasks/{tid}/stage")
    print(f"[4] derived stage={stage['derived_stage']}")

    # 5) 采纳一个 perf Loop → 小版本 +1（合成）
    loop = req("POST", f"/tasks/{tid}/loops", {
        "kind": "perf", "scope": "合成验收用 Loop（不跑真机）",
        "hypothesis": "synthetic", "judge_condition": "synthetic",
    })
    req("POST", f"/loops/{loop['id']}/close?conclusion=accepted")
    ver = req("GET", f"/tasks/{tid}/versions")
    print(f"[5] accept optimization → version {ver['current']['label']} source={ver['current']['source']}")
    assert ver["current"]["minor"] == 1 and ver["current"]["source"] == "optimization"

    # 6) 交付包 + 下载
    pack = req("POST", f"/tasks/{tid}/deliveries")
    print(f"[6] delivery run #{pack['run_id']} {pack['version']} size={pack['size_bytes']}B")
    with urllib.request.urlopen(API.rstrip("/") + f"/deliveries/{pack['run_id']}/download", timeout=60) as r:
        data = r.read()
    assert data[:2] == b"PK"
    zip_path = Path(pack["zip_path"])
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    print("[6] zip entries:", len(names), "| manifest:", "manifest.json" in names,
          "| python iface:", any("ops/%s/python" % OP in n for n in names))
    hist = req("GET", f"/tasks/{tid}/deliveries")["deliveries"]
    print(f"[6] delivery history={len(hist)} gate_verdict={hist[0].get('gate_verdict')}")

    # 7) 清理（任务 + 合成工作区 + 本地 run 目录）
    req("DELETE", f"/tasks/{tid}?delete_files=true")
    import shutil
    for p in [WS / "cases" / OP, WS / "ops" / OP, WS / "docs" / OP,
              WS / "tests" / f"test_{OP}.py"]:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    for action in ("build", "smoke", "test", "benchmark"):
        shutil.rmtree(RUNS / f"e2e-{action}", ignore_errors=True)
    print("[7] cleaned; tasks =", sqlite3.connect(DB).execute("select count(*) from tasks").fetchone()[0])
    print("E2E_SYNTH PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
