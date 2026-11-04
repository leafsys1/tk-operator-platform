"""交付（任务生命周期 v2 的第五个动作，2026-10-23）。

**本地动作**：不连远端容器、不需要设备。做三件事：

1. **交付门禁**：版本证据有效（build/test/benchmark 对当前版本 pass）+ 用例组已确认且未变
   + 5 件产物齐备；
2. **打包**：生成 zip（代码 / Python 调用接口 / 测试脚本 / 说明文档 / 规划文档 / manifest / 证据）；
3. **登记**：落一条 `runs.action="delivery"` + `artifacts`（delivery/manifest），并把当前版本
   标记 `delivered_at`（交付后冻结：再改代码必须新小版本 + 重跑 + 重新交付）。

包结构见 `docs/plan-task-lifecycle-v2.md` §7.2。失败一律抛 `DeliveryError`（带 fail codes），
路由层转 409，页面直接展示缺什么。
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Artifact, Project, Run, Task, TaskDocument, TaskVersion
from . import exec_actions, gate, versions

DELIVERY_ACTION = "delivery"
DELIVERY_ROOT = "archives/deliveries"

#: 交付包必需产物（缺一不可），(判断函数, fail code, 人话)
def _delivery_root() -> Path:
    return Path(settings.repo_root) / DELIVERY_ROOT


class DeliveryError(RuntimeError):
    def __init__(self, message: str, *, codes: list[str] | None = None, missing: list[dict] | None = None):
        super().__init__(message)
        self.codes = codes or []
        self.missing = missing or []


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _workspace(db: Session, task: Task) -> Path:
    project = db.get(Project, task.project_id)
    if not project or not project.workspace_path:
        raise DeliveryError("项目工作区未配置", codes=["E_DELIVERY_WORKSPACE"])
    ws = Path(project.workspace_path).expanduser()
    if not ws.is_dir():
        raise DeliveryError(f"工作区不存在：{ws}", codes=["E_DELIVERY_WORKSPACE"])
    return ws


def _rel(ws: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(ws.resolve()).as_posix()
    except ValueError:
        return str(path)


def _path_issue(ws: Path, path: Path) -> str | None:
    """工作区交付源必须是普通的工作区内路径，禁止符号链接和越界解析。"""
    if path.is_symlink():
        return "不允许符号链接"
    try:
        path.resolve().relative_to(ws.resolve())
    except (OSError, ValueError):
        return "路径解析到工作区之外"
    return None


def _require(ws: Path, rel: str, *, code: str, label: str, missing: list[dict]) -> Path | None:
    path = ws / rel
    issue = _path_issue(ws, path)
    if issue:
        missing.append({"code": "E_DELIVERY_PATH_UNSAFE", "item": label, "path": rel,
                        "reason": issue})
        return None
    if path.exists():
        return path
    missing.append({"code": code, "item": label, "path": rel, "reason": "文件/目录不存在"})
    return None


def _planning_doc(db: Session, ws: Path, task: Task, op: str) -> tuple[list[tuple[str, Path]], list[dict]]:
    """规划文档：用户原文（若有）+ agent 补全稿（若有）。返回 [(包内路径, 磁盘路径)] 与缺项。"""
    out: list[tuple[str, Path]] = []
    missing: list[dict] = []
    docs = list(db.scalars(select(TaskDocument).where(TaskDocument.task_id == task.id,
                                                     TaskDocument.kind == "design")))
    original = None
    for doc in docs:
        path = ws / doc.path
        if not (path.exists() or path.is_symlink()):
            continue
        issue = _path_issue(ws, path)
        if issue:
            missing.append({"code": "E_DELIVERY_PATH_UNSAFE", "item": "规划文档",
                            "path": doc.path, "reason": issue})
            continue
        if path.is_file():
            original = doc
            break
    if original is not None:
        suffix = Path(original.filename).suffix or ".md"
        out.append((f"docs/规划文档-原始{suffix}", ws / original.path))
    completed = ws / "docs" / op / "design.md"
    completed_issue = _path_issue(ws, completed)
    if completed_issue:
        missing.append({"code": "E_DELIVERY_PATH_UNSAFE", "item": "规划文档",
                        "path": f"docs/{op}/design.md", "reason": completed_issue})
    elif completed.is_file():
        # 用与源文件一致的清晰路径；不再把设计文档命名为「规划文档-补全.md」。
        out.append((f"docs/{op}/design.md", completed))
    if original is None and not completed.is_file():
        missing.append({"code": "E_DELIVERY_DOCS_MISSING", "item": "规划文档",
                        "path": f"docs/{op}/design.md",
                        "reason": "既没有上传的设计文档，也没有 agent 补全稿"})
    return out, missing


def _zip_add(zf: zipfile.ZipFile, arcname: str, path: Path, files: list[dict],
             *, allowed_root: Path | None = None) -> None:
    if path.is_symlink():
        raise DeliveryError(f"交付源不允许符号链接：{path}", codes=["E_DELIVERY_PATH_UNSAFE"])
    if allowed_root is not None and _path_issue(allowed_root, path):
        raise DeliveryError(f"交付源路径越界：{path}", codes=["E_DELIVERY_PATH_UNSAFE"])
    if not path.is_file():
        return
    data = path.read_bytes()
    zf.writestr(arcname, data)
    files.append({"path": arcname, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})


def _zip_tree(zf: zipfile.ZipFile, ws: Path, rel_root: str, arc_root: str, files: list[dict],
              *, skip_dirs: set[str] | None = None) -> None:
    root = ws / rel_root
    if not root.is_dir():
        return
    skip = skip_dirs or set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DeliveryError(f"交付目录包含符号链接：{path}", codes=["E_DELIVERY_PATH_UNSAFE"])
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(p in skip for p in parts):
            continue
        arc = f"{arc_root}/{path.relative_to(root).as_posix()}"
        _zip_add(zf, arc, path, files, allowed_root=ws)


def _artifact_evidence(db: Session, task: Task, evidence: dict) -> list[tuple[str, Path]]:
    """把交付门禁**实际选中**的 test/benchmark 报告带进 evidence/。

    不能再自己取“最新一条”：最新 run 可能是后续调试或失败记录，与门禁选中的
    同一规格/快照/用例证据组不是一组。
    """
    out: list[tuple[str, Path]] = []
    for action, name in (("test", "correctness.json"), ("benchmark", "performance.json")):
        view = (evidence.get("runs") or {}).get(action) or {}
        run_id = view.get("run_id")
        run = db.get(Run, run_id) if run_id else None
        if run is None:
            continue
        paths = gate.artifact_paths(run)
        art = paths.get("correctness" if action == "test" else "performance")
        if art is not None and art.is_file():
            out.append((f"evidence/{name}", art))
    return out


def check(db: Session, task: Task) -> dict:
    """交付门禁预检（不写盘、不落库）：返回 {ok, missing[], evidence, version}。"""
    ws = _workspace(db, task)
    op = (task.op_name or "").strip()
    if not op:
        raise DeliveryError("任务没有算子名（op_name），无法交付", codes=["E_DELIVERY_OP_NAME"])
    missing: list[dict] = []
    ev = versions.evidence(db, task)
    if not ev["deliverable"]:
        for m in ev["missing"]:
            missing.append({"code": "E_VERSION_EVIDENCE_STALE", "item": m.get("label") or m.get("action"),
                            "action": m.get("action"), "reason": m.get("reason")})
    if not ev["cases"].get("confirmed_at"):
        missing.append({"code": "E_CASES_UNCONFIRMED", "item": "用例组",
                        "reason": "精度/性能用例组还没在对话里确认"})
    elif not ev["cases"].get("in_sync"):
        missing.append({"code": "E_CASES_UNCONFIRMED", "item": "用例组",
                        "reason": "用例集已变更，需要重新确认并重跑 test/benchmark"})

    op_root = _require(ws, f"ops/{op}", code="E_DELIVERY_CODE_MISSING", label="算子代码",
                       missing=missing)
    if op_root is not None and op_root.is_dir():
        for path in op_root.rglob("*"):
            issue = _path_issue(ws, path)
            if issue:
                missing.append({"code": "E_DELIVERY_PATH_UNSAFE", "item": "算子代码",
                                "path": _rel(ws, path), "reason": issue})
                break
    py_dir = ws / "ops" / op / "python"
    if not (py_dir.is_dir() and any(p.suffix == ".py" for p in py_dir.rglob("*.py"))):
        missing.append({"code": "E_DELIVERY_PYTHON_IFACE_MISSING", "item": "Python 调用接口",
                        "path": f"ops/{op}/python/", "reason": "缺 Python 调用接口（.py）"})
    _require(ws, f"tests/test_{op}.py", code="E_DELIVERY_TESTS_MISSING", label="pytest 测试脚本",
             missing=missing)
    _require(ws, f"cases/{op}/cases.json", code="E_DELIVERY_CASES_MISSING", label="精度用例",
             missing=missing)
    _require(ws, f"cases/{op}/benchmark.json", code="E_DELIVERY_CASES_MISSING", label="性能用例",
             missing=missing)
    _require(ws, f"ops/{op}/README.md", code="E_DELIVERY_DOCS_MISSING", label="算子说明文档",
             missing=missing)
    _require(ws, f"examples/{op}_demo.py", code="E_DELIVERY_EXAMPLE_MISSING", label="独立示例",
             missing=missing)
    _docs, doc_missing = _planning_doc(db, ws, task, op)
    missing.extend(doc_missing)
    return {"ok": not missing, "missing": missing, "evidence": ev, "op": op, "version": ev["version"]}


def package(db: Session, task: Task) -> dict:
    """跑交付：门禁 → 打包 zip → 落 run/artifacts → 版本标记 delivered。"""
    pre = check(db, task)
    if not pre["ok"]:
        raise DeliveryError("交付门禁未通过：" + "；".join(
            f"{m['item']}（{m['reason']}）" for m in pre["missing"][:6]),
            codes=sorted({m["code"] for m in pre["missing"]}), missing=pre["missing"])

    ws = _workspace(db, task)
    op = pre["op"]
    ev = pre["evidence"]
    major, minor = ev["major"], ev["minor"]
    project = db.get(Project, task.project_id)
    criteria_ref, _src = exec_actions.resolve_criteria(task, project)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    version_dir = _delivery_root() / f"task-{task.id}" / f"v{major}.{minor}"
    version_dir.mkdir(parents=True, exist_ok=True)
    # 先落 run 行拿 run_id（manifest 要写它）
    run = Run(
        task_id=task.id, action=DELIVERY_ACTION, status="ok", exit_code=0,
        spec_version_id=task.current_spec_version_id,
        code_snapshot_sha256=ev.get("expected_snapshot_sha256"),
        suite_ref=ev.get("current_suite_ref"), criteria_ref=criteria_ref,
        command=f"delivery package v{major}.{minor}",
        started_at=datetime.now(timezone.utc), ended_at=datetime.now(timezone.utc),
        note=f"交付包 v{major}.{minor}",
    )
    db.add(run)
    db.flush()
    # 同一版本允许重复交付。文件名带 run id，避免覆盖旧包后历史 Artifact 指向被篡改的内容。
    zip_name = f"{op}_v{major}.{minor}_{stamp}_r{run.id}.zip"
    zip_path = version_dir / zip_name
    run.note = f"交付包 {zip_name}"

    files: list[dict] = []
    docs_from_db, _ = _planning_doc(db, ws, task, op)
    evidence_files = _artifact_evidence(db, task, ev)
    package_manifest = {
        "schema": "opx-delivery-manifest/1",
        "task_id": task.id,
        "op_name": op,
        "version": {"label": ev["version"], "major": major, "minor": minor},
        "spec_version_id": task.current_spec_version_id,
        "code_snapshot_sha256": ev.get("expected_snapshot_sha256"),
        "suite_ref": ev.get("current_suite_ref"),
        "cases_sha256": ev["cases"].get("current_suite_sha256"),
        "criteria_ref": criteria_ref,
        "waivers": ev.get("waived_actions") or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [],
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _zip_tree(zf, ws, f"ops/{op}", f"ops/{op}", files,
                  skip_dirs={"build", "__pycache__", ".git"})
        _zip_add(zf, f"tests/test_{op}.py", ws / f"tests/test_{op}.py", files, allowed_root=ws)
        _zip_add(zf, "tests/cases.json", ws / f"cases/{op}/cases.json", files, allowed_root=ws)
        _zip_add(zf, "tests/benchmark.json", ws / f"cases/{op}/benchmark.json", files, allowed_root=ws)
        _zip_add(zf, "tests/golden.py", ws / f"ops/{op}/scripts/golden.py", files, allowed_root=ws)
        _zip_add(zf, "docs/算子说明文档.md", ws / f"ops/{op}/README.md", files, allowed_root=ws)
        _zip_add(zf, f"examples/{op}_demo.py", ws / f"examples/{op}_demo.py", files, allowed_root=ws)
        for arc, src in docs_from_db:
            _zip_add(zf, arc, src, files, allowed_root=ws)
        for arc, src in evidence_files:
            _zip_add(zf, arc, src, files)
        package_manifest["artifacts"] = files
        zf.writestr("manifest.json", json.dumps(package_manifest, ensure_ascii=False, indent=2))

    zip_bytes = zip_path.stat().st_size
    zip_sha = _sha256_file(zip_path)

    # 外部 manifest：给门禁/下载用的稳定入口（含 zip 自身哈希）
    external = dict(package_manifest)
    external["run_id"] = run.id
    external["exit_code"] = 0
    external["artifacts"] = [{"path": zip_name, "type": "delivery",
                              "sha256": zip_sha, "size_bytes": zip_bytes}]
    external["code_snapshot_sha256"] = run.code_snapshot_sha256
    external["spec_version_id"] = run.spec_version_id
    external["criteria_ref"] = criteria_ref
    external["image_digest"] = ev.get("runs", {}).get("benchmark", {}).get("image_digest") if isinstance(ev.get("runs", {}).get("benchmark"), dict) else None
    external["suite_ref"] = run.suite_ref
    manifest_path = version_dir / f"manifest-run{run.id}.json"
    manifest_path.write_text(json.dumps(external, ensure_ascii=False, indent=2), encoding="utf-8")
    run.manifest_path = str(manifest_path)

    db.add(Artifact(run_id=run.id, type="delivery", path=str(zip_path),
                    sha256=zip_sha, size_bytes=zip_bytes))
    db.add(Artifact(run_id=run.id, type="manifest", path=str(manifest_path),
                    sha256=_sha256_file(manifest_path), size_bytes=manifest_path.stat().st_size))
    # 历史任务可能早于 task_versions 表；交付不能只产 zip 而不留下版本冻结记录。
    cur = versions.ensure_current_for_delivery(db, task)
    cur.delivered_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return {
        "run_id": run.id,
        "version": ev["version"],
        "download_url": f"/api/deliveries/{run.id}/download",
        "sha256": zip_sha,
        "size_bytes": zip_bytes,
        "zip_path": str(zip_path),
        "missing": [],
    }


def view(db: Session, task: Task) -> list[dict]:
    runs = list(db.scalars(
        select(Run).where(Run.task_id == task.id, Run.action == DELIVERY_ACTION)
        .order_by(Run.id.desc())
    ))
    out = []
    for run in runs:
        arts = list(db.scalars(select(Artifact).where(Artifact.run_id == run.id)))
        main = next((a for a in arts if a.type == "delivery"), None)
        out.append({
            "run_id": run.id,
            "action": run.action,
            "status": run.status,
            "gate_verdict": gate.evaluate(run).get("verdict") if run.status == "ok" else None,
            "version": run.note or "",
            "spec_version_id": run.spec_version_id,
            "code_snapshot_sha256": run.code_snapshot_sha256,
            "suite_ref": run.suite_ref,
            "sha256": main.sha256 if main else None,
            "size_bytes": main.size_bytes if main else None,
            "created_at": (run.ended_at or run.started_at).isoformat() if (run.ended_at or run.started_at) else None,
            "download_url": f"/api/deliveries/{run.id}/download" if main else None,
        })
    return out


def delivery_zip_path(db: Session, run_id: int) -> tuple[Run, Path]:
    run = db.get(Run, run_id)
    if run is None or run.action != DELIVERY_ACTION:
        raise DeliveryError("交付记录不存在", codes=["E_DELIVERY_NOT_FOUND"])
    art = db.scalar(select(Artifact).where(Artifact.run_id == run.id,
                                           Artifact.type == "delivery").limit(1))
    if art is None:
        raise DeliveryError("这次交付没有交付包产物", codes=["E_DELIVERY_NOT_FOUND"])
    path = Path(str(art.path)).expanduser()
    if not path.is_file():
        raise DeliveryError(f"交付包不在磁盘上：{path}", codes=["E_DELIVERY_FILE_MISSING"])
    return run, path
