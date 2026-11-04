"""append-only 优化报告仓库（自动优化 Step 3）。

契约见 `docs/plan-optimize-auto-v1.md` §17。核心：报告不再是「run 目录里一个会被重跑覆盖的
JSON」，而是**内容可校验、原始产物自包含、只追加**的目录：

    <report_root>/loops/<loop_id>/<seq>-<report_sha12>/
        report.json     规范化 JSON（sha256 = report_sha256）
        manifest.json   report.json/repro.md 的 sha + raw 副本清单（含原路径与 sha）
        raw/            两侧 run 的关键产物副本（manifest/performance/correctness/gate/log）
        repro.md        校验与复现命令

设计取舍：
- **文件系统是真相源**（报告是产物，不是业务行）；不新增 DB 表，列表/校验都扫目录。
- 校验分两档：列表 `valid` = 快速校验（report.json sha + 文件存在）；`POST .../verify` 与
  采纳门禁 = 深度校验（manifest.files + raw 副本逐个 sha）。
- 幂等：同 loop 同内容（忽略 generated_at）重复生成 → 复用最新一条，不新增 seq；
  「内容」包含 raw 副本清单（raw 变了必须新归档）。
- raw 复制有上限（单文件 8MB / 每侧 32MB，after 优先），超限/源缺失只记 `copied=false`；
  run 目录外的路径一律拒收（远端拉回的 manifest 按不可信输入处理）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Loop, Project, Run
from . import gate, optimization

logger = logging.getLogger(__name__)

#: manifest schema（对外契约）。
MANIFEST_SCHEMA = "opx-report-manifest/1"
#: raw 上限：单文件 8MB；两侧各 32MB（after 优先，避免 before 侧饿死候选证据）。
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_SIDE_BYTES = 32 * 1024 * 1024
#: 列表最多返回多少条（最新在前）。
MAX_LIST = 50
#: 已发布目录名：`<seq>-<sha12>`；`.tmp-*` 是写到一半的残留（崩溃），不参与列表/采纳。
_DIR_RE = re.compile(r"^(\d{4,})-([0-9a-f]{12})$")
#: 崩溃残留 tmp 的清理阈值（秒）：只清明显过期的，避免误删别的进程正在写的。
TMP_TTL = 3600

#: 写盘串行化：单进程部署前提（与 loop_budget 同口径）；多 worker 需换文件锁。
_LOCK = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def root() -> Path:
    """报告仓库根目录：`settings.report_root` 或 `<repo_root>/reports`。"""
    configured = (settings.report_root or "").strip()
    return Path(configured).expanduser() if configured else Path(settings.repo_root) / "reports"


def loop_root(loop_id: int) -> Path:
    return root() / "loops" / str(int(loop_id))


def _canonical(data: dict) -> bytes:
    """规范化 JSON：sort_keys + 固定缩进 + 末尾换行（同一个 payload 永远同一个 sha）。"""
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _safe_sha(value: str | None) -> str:
    return "".join(ch for ch in str(value or "") if ch.isalnum() or ch in "-_")[:64]


def _safe_rel(rel: str) -> str | None:
    """相对路径校验：拒绝绝对路径 / `..` / 空（远端 run manifest 是不可信输入）。"""
    text = str(rel or "").strip().replace("\\", "/").lstrip("/")
    if not text:
        return None
    parts = Path(text).parts
    if any(part in ("..", "") for part in parts) or text.startswith("/"):
        return None
    return text


def _safe_child(report_dir: Path, rel: str) -> Path | None:
    """把归档内相对路径限制在报告目录内（防篡改 manifest 逃逸读任意文件）。"""
    text = str(rel or "").strip()
    if not text or text.startswith("/") or ".." in Path(text).parts:
        return None
    target = (report_dir / text).resolve()
    root_dir = report_dir.resolve()
    if target != root_dir and root_dir not in target.parents:
        return None
    return target


def _seq_of(manifest: dict) -> int:
    try:
        return int(manifest.get("seq") or 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------- 原始产物收集（计划 + 复制）


def _run_dir_of(run: Run | None) -> Path | None:
    if run is None or not run.manifest_path:
        return None
    path = Path(str(run.manifest_path)).expanduser()
    return path.parent if path.name == "manifest.json" else path


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _ensure_gate_reports(rows: list[Run]) -> None:
    """尽量把两侧 run 的 gate_report.json 落盘（best-effort），归档才能独立复核门禁结论。"""
    for run in rows:
        try:
            gate.evaluate(run, write=True)
        except Exception:  # noqa: BLE001 —— 落不了就用已有产物/evidence，不影响归档
            logger.debug("门禁报告落盘失败（run=%s）", getattr(run, "id", None), exc_info=True)


def _collect_runs(db: Session, payload: dict) -> tuple[list[dict], list[Run]]:
    """两侧 run 的元数据（报告里已有 id，这里补 manifest/产物绝对路径与 gate verdict）。"""
    runs: list[dict] = []
    rows: list[Run] = []
    sides = payload.get("runs") if isinstance(payload.get("runs"), dict) else {}
    for side in ("before", "after"):
        per_action = sides.get(side) if isinstance(sides.get(side), dict) else {}
        for action in ("test", "benchmark"):
            rid = per_action.get(action)
            if not isinstance(rid, int):
                continue
            run = db.get(Run, rid)
            if run is None:
                runs.append({"side": side, "action": action, "run_id": rid, "missing": True})
                continue
            rows.append(run)
            run_dir = _run_dir_of(run)
            manifest_path = run_dir / "manifest.json" if run_dir is not None else None
            manifest = _read_json(manifest_path) if manifest_path is not None else None
            artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
            gate_verdict = None
            try:
                gate_verdict = gate.evaluate(run).get("verdict")
            except Exception:  # noqa: BLE001
                gate_verdict = None
            runs.append({
                "side": side,
                "action": action,
                "run_id": run.id,
                "manifest_path": str(manifest_path) if manifest_path is not None else None,
                "run_dir": str(run_dir) if run_dir is not None else None,
                "code_snapshot_sha256": run.code_snapshot_sha256,
                "spec_version_id": run.spec_version_id,
                "criteria_ref": run.criteria_ref,
                "suite_ref": run.suite_ref,
                "image_digest": run.image_digest,
                "device_id": run.device_id,
                "host_id": run.host_id,
                "gate_verdict": gate_verdict,
                "artifacts": [
                    {"type": str(a.get("type") or ""), "path": str(a.get("path") or "")}
                    for a in (artifacts or []) if isinstance(a, dict)
                ],
            })
    return runs, rows


def _oracle_fingerprint(runs: list[dict]) -> dict:
    """从两侧 test run 的 correctness.json 里取 oracle.*_sha256（Step 0 写入）。"""
    out: dict = {"before": None, "after": None}
    for entry in runs:
        if entry.get("action") != "test" or entry.get("side") not in out:
            continue
        run_dir = entry.get("run_dir")
        if not run_dir:
            continue
        corr = _read_json(Path(str(run_dir)) / "out" / "correctness.json")
        oracle = corr.get("oracle") if isinstance(corr, dict) else None
        out[entry["side"]] = oracle if isinstance(oracle, dict) else None
    return out


def _plan_raw(runs: list[dict]) -> list[dict]:
    """只**计划** raw 清单（不复制）：按 after→before 顺序分配每侧 32MB 预算。"""
    raw: list[dict] = []
    side_used = {"before": 0, "after": 0}
    ordered = sorted(runs, key=lambda e: 0 if e.get("side") == "after" else 1)
    for entry in ordered:
        side, action = entry.get("side"), entry.get("action")
        run_dir_text = str(entry.get("run_dir") or "")
        if not run_dir_text or entry.get("missing"):
            raw.append({"kind": f"{side}-{action}-run-dir", "path": None,
                        "source": run_dir_text or None, "sha256": None, "bytes": None,
                        "copied": False, "why": "run 缺少 manifest_path/目录不可用"})
            continue
        run_dir = Path(run_dir_text)
        if not run_dir.is_dir():
            raw.append({"kind": f"{side}-{action}-run-dir", "path": None,
                        "source": run_dir_text, "sha256": None, "bytes": None,
                        "copied": False, "why": "run 目录不存在"})
            continue
        sources: list[tuple[str, str]] = []  # (kind, 相对 run_dir 的路径)
        if (run_dir / "manifest.json").is_file():
            sources.append(("manifest", "manifest.json"))
            manifest = _read_json(run_dir / "manifest.json") or {}
            for art in (manifest.get("artifacts") or []):
                if not isinstance(art, dict):
                    continue
                rel = _safe_rel(str(art.get("path") or ""))
                if rel is None:
                    raw.append({"kind": f"{side}-{action}-artifact", "path": str(art.get("path")),
                                "source": str(run_dir / str(art.get("path"))), "sha256": None,
                                "bytes": None, "copied": False,
                                "why": "产物路径非法（绝对路径/含 ..）"})
                    continue
                src = (run_dir / rel).resolve()
                if run_dir.resolve() not in src.parents and src != run_dir.resolve():
                    raw.append({"kind": f"{side}-{action}-artifact", "path": rel,
                                "source": str(run_dir / rel), "sha256": None, "bytes": None,
                                "copied": False, "why": "产物路径越出 run 目录"})
                    continue
                if src.is_file():
                    sources.append((str(art.get("type") or "artifact"), rel))
                else:
                    raw.append({"kind": f"{side}-{action}-{art.get('type') or 'artifact'}",
                                "path": rel, "source": str(src), "sha256": None,
                                "bytes": None, "copied": False, "why": "源文件不存在"})
        if (run_dir / "gate_report.json").is_file():
            sources.append(("gate", "gate_report.json"))
        seen: set[str] = set()
        for kind, rel in sources:
            if rel in seen:
                continue
            seen.add(rel)
            src = run_dir / rel
            try:
                size = src.stat().st_size
            except OSError:
                raw.append({"kind": f"{side}-{action}-{kind}", "path": rel, "source": str(src),
                            "sha256": None, "bytes": None, "copied": False,
                            "why": "源文件不可读"})
                continue
            sha = _sha_file(src)
            copied = False
            copy_rel = None
            why = None
            if size > MAX_FILE_BYTES:
                why = f"超过单文件上限（{size} > {MAX_FILE_BYTES}）"
            elif side_used.get(side, 0) + size > MAX_SIDE_BYTES:
                why = f"超过 {side} 侧总量预算（{MAX_SIDE_BYTES}）"
            else:
                side_used[side] = side_used.get(side, 0) + size
                copied = True
                copy_rel = f"raw/{side}-{action}/{rel}"
            raw.append({"kind": f"{side}-{action}-{kind}", "path": rel, "source": str(src),
                        "sha256": sha, "bytes": size, "copied": copied,
                        "copy": copy_rel, **({"why": why} if why else {})})
    return raw


def _copy_raw(entries: list[dict], dest_root: Path) -> None:
    """按计划复制 `copied=true` 的 raw 副本（dest 必须在 dest_root 内）。"""
    root_resolved = dest_root.resolve()
    for item in entries:
        if not item.get("copied"):
            continue
        rel = _safe_rel(str(item.get("copy") or ""))
        src_text = item.get("source")
        if rel is None or not src_text:
            item["copied"] = False
            item["why"] = "复制路径非法"
            item["copy"] = None
            continue
        dest = (dest_root / rel).resolve()
        if root_resolved not in dest.parents and dest != root_resolved:
            item["copied"] = False
            item["why"] = "归档目标路径越界"
            item["copy"] = None
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(src_text), str(dest))


# ---------------------------------------------------------------- 校验 / 列表


def _verify(manifest: dict, report_dir: Path, *, deep: bool = True) -> dict:
    """校验目录：report.json sha + manifest 列出的文件 + raw 副本（deep 时逐个算 sha）。"""
    result = {"valid": False, "files": [], "raw": [], "errors": []}
    report_sha = str(manifest.get("report_sha256") or "")
    report_path = report_dir / "report.json"
    actual = _sha_file(report_path) if report_path.is_file() else None
    result["files"].append({"path": "report.json", "ok": bool(report_sha) and actual == report_sha,
                            "expected": report_sha or None, "actual": actual})
    if actual != report_sha:
        result["errors"].append("report.json 的 sha256 与 manifest.report_sha256 不一致")
    for item in manifest.get("files") or []:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path") or "")
        if not rel or rel == "report.json":
            continue
        target = _safe_child(report_dir, rel)
        ok = target is not None and target.is_file()
        got = _sha_file(target) if ok and deep and target is not None else None
        if ok and deep and got != item.get("sha256"):
            ok = False
        result["files"].append({"path": rel, "ok": ok,
                                "expected": item.get("sha256"), "actual": got})
        if not ok:
            result["errors"].append(f"文件校验失败：{rel}")
    for item in manifest.get("raw") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("copied"):
            continue
        rel = str(item.get("copy") or "")
        target = _safe_child(report_dir, rel)
        ok = target is not None and target.is_file()
        got = _sha_file(target) if ok and deep and target is not None else None
        if ok and deep and got != item.get("sha256"):
            ok = False
        result["raw"].append({"kind": item.get("kind"), "path": rel, "ok": ok,
                              "copied": True, "expected": item.get("sha256"), "actual": got})
        if not ok:
            result["errors"].append(f"raw 副本校验失败：{rel}")
    result["valid"] = not result["errors"]
    return result


def _load_manifests(loop_id: int) -> list[tuple[Path, dict]]:
    """返回已发布归档（`seq-sha12`）的 (目录, manifest)；manifest 坏了也保留占位（标 broken）。"""
    base = loop_root(loop_id)
    out: list[tuple[Path, dict]] = []
    if not base.is_dir():
        return out
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = _DIR_RE.match(child.name)
        if match is None:
            continue  # `.tmp-*` / 手放目录：不参与列表与采纳
        manifest = _read_json(child / "manifest.json")
        if not isinstance(manifest, dict):
            manifest = {"schema": MANIFEST_SCHEMA, "loop_id": loop_id,
                        "seq": int(match.group(1)), "report_sha256": None, "broken": True}
        manifest.setdefault("seq", int(match.group(1)))
        out.append((child, manifest))
    out.sort(key=lambda pair: _seq_of(pair[1]), reverse=True)
    return out


def _manifest_summary(report_dir: Path, manifest: dict, *, deep: bool = False) -> dict:
    verify = _verify(manifest, report_dir, deep=deep)
    report = _read_json(report_dir / "report.json") or {}
    summary = report.get("objective_summary") if isinstance(report.get("objective_summary"), dict) else {}
    return {
        "report_sha256": manifest.get("report_sha256"),
        "content_sha256": manifest.get("content_sha256"),
        "seq": _seq_of(manifest),
        "dir": str(report_dir),
        "generated_at": manifest.get("generated_at"),
        "verdict": summary.get("verdict"),
        "conclusion": summary.get("conclusion"),
        "valid": verify["valid"],
        "broken": bool(manifest.get("broken")),
        "size_bytes": manifest.get("size_bytes"),
        "raw_count": sum(1 for item in (manifest.get("raw") or [])
                         if isinstance(item, dict) and item.get("copied")),
    }


def latest(loop_id: int) -> dict | None:
    """最新一条报告的摘要（快速校验）；没有则 None。"""
    pairs = _load_manifests(loop_id)
    if not pairs:
        return None
    report_dir, manifest = pairs[0]
    return _manifest_summary(report_dir, manifest, deep=False)


def list_reports(loop_id: int, *, limit: int = MAX_LIST) -> list[dict]:
    return [_manifest_summary(report_dir, manifest, deep=False)
            for report_dir, manifest in _load_manifests(loop_id)[:max(1, limit)]]


def _find(loop_id: int, report_sha256: str) -> tuple[Path, dict] | None:
    want = _safe_sha(report_sha256)
    if len(want) < 8:
        return None
    for report_dir, manifest in _load_manifests(loop_id):
        if str(manifest.get("report_sha256") or "").startswith(want):
            return report_dir, manifest
    return None


def load(loop_id: int, report_sha256: str) -> dict | None:
    found = _find(loop_id, report_sha256)
    if found is None:
        return None
    report_dir, manifest = found
    report = _read_json(report_dir / "report.json")
    if report is None:
        return {"manifest": manifest, "report": None,
                "valid": False, "dir": str(report_dir),
                "errors": ["report.json 缺失或不可解析"]}
    return {"manifest": manifest, "report": report,
            "valid": _verify(manifest, report_dir, deep=True)["valid"],
            "dir": str(report_dir)}


def verify(loop_id: int, report_sha256: str) -> dict | None:
    """深度校验；归档目录/manifest 都不存在才返回 None（损坏也返回 valid=False + errors）。"""
    found = _find(loop_id, report_sha256)
    if found is None:
        return None
    report_dir, manifest = found
    result = _verify(manifest, report_dir, deep=True)
    if not (report_dir / "manifest.json").is_file():
        result["errors"].append("manifest.json 缺失或不可解析")
        result["valid"] = False
    if report_sha256 and manifest.get("report_sha256") is None:
        result["errors"].append("manifest 缺少 report_sha256（归档损坏）")
        result["valid"] = False
    return {
        "report_sha256": manifest.get("report_sha256"),
        "valid": result["valid"],
        "checked_at": _iso(_now()),
        "files": result["files"],
        "raw": result["raw"],
        "errors": result["errors"],
    }


# ---------------------------------------------------------------- 写入


def _clean_tmp(base: Path) -> None:
    """清理明显过期的 `.tmp-*` 残留（崩溃写一半）；只清 1h 以上的，避免误删并发写。"""
    if not base.is_dir():
        return
    now = time.time()
    for child in base.iterdir():
        if not child.is_dir() or not child.name.startswith(".tmp-"):
            continue
        try:
            if now - child.stat().st_mtime > TMP_TTL:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def _content_sha(payload: dict) -> str:
    """幂等判定用：忽略各处 generated_at（report/perf_diff/evidence/raw），但保留 raw 清单。"""

    def _strip(value):
        if isinstance(value, dict):
            value.pop("generated_at", None)
            for item in value.values():
                _strip(item)
        elif isinstance(value, list):
            for item in value:
                _strip(item)

    content = json.loads(json.dumps(payload, ensure_ascii=False))
    _strip(content)
    # gate_report.json 每次生成都会带新的 generated_at（sha 变），但它是从 run 证据派生的：
    # 幂等判定忽略它，否则同一状态每归档一次都会多一个 seq。
    evidence = content.get("evidence") if isinstance(content.get("evidence"), dict) else {}
    raw = evidence.get("raw") if isinstance(evidence.get("raw"), list) else []
    if raw:
        evidence["raw"] = [item for item in raw
                           if not (isinstance(item, dict)
                                   and str(item.get("kind") or "").endswith("-gate"))]
    return _sha_bytes(_canonical(content))


def _reuse(report_dir: Path, manifest: dict) -> dict | None:
    """同内容且深度校验通过 → 复用现有归档。"""
    report = _read_json(report_dir / "report.json")
    if report is None:
        return None
    if not _verify(manifest, report_dir, deep=True)["valid"]:
        return None
    merged = dict(report)
    merged.update({
        "report_sha256": manifest.get("report_sha256"),
        "content_sha256": manifest.get("content_sha256"),
        "seq": _seq_of(manifest),
        "report_dir": str(report_dir),
        "written_to": str(report_dir / "report.json"),
        "manifest": manifest,
        "report_valid": True,
        "stored": False,
    })
    return merged


def _manifest_raw(payload: dict) -> list[dict]:
    """取 payload.evidence.raw（类型安全）。"""
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    raw = evidence.get("raw")
    return raw if isinstance(raw, list) else []


def write(db: Session, loop: Loop) -> dict:
    """生成一次 append-only 报告（同内容幂等复用）；返回报告 + 归档元数据。

    返回字段 = 原 `optimization.report(write=False)` payload 平铺 + `report_sha256/
    content_sha256/seq/report_dir/written_to/manifest/report_valid/stored`。
    """
    task = optimization._require_task(db, loop)  # noqa: SLF001 —— 同包编排，复用口径
    project = db.get(Project, task.project_id)
    if project is None:
        raise optimization.OptimizationError("项目不存在，无法归档报告", status=409)

    payload = optimization.report(db, loop, write=False)
    # 归档前置：必须有「优化后」的 benchmark run + 可读 manifest。
    after_bench = (payload.get("runs") or {}).get("after", {}).get("benchmark") \
        if isinstance(payload.get("runs"), dict) else None
    after_run = db.get(Run, after_bench) if isinstance(after_bench, int) else None
    after_dir = _run_dir_of(after_run)
    if after_run is None or after_dir is None or not (after_dir / "manifest.json").is_file():
        raise optimization.OptimizationError(
            "报告无法归档：找不到优化后的 benchmark run 目录（先跑候选 benchmark）", status=409)

    # 尽量先落 gate_report.json（best-effort），再收集 run 元数据/计划 raw。
    runs, rows = _collect_runs(db, payload)
    _ensure_gate_reports(rows)
    runs, _rows = _collect_runs(db, payload)  # gate 落盘后重扫一遍产物清单
    payload["evidence"] = {
        "schema": "opx-report-evidence/1",
        "generated_at": _iso(_now()),
        "runs": runs,
        "oracle": _oracle_fingerprint(runs),
        "comparability": (payload.get("perf_diff") or {}).get("comparability"),
        "env": (payload.get("repro") or {}).get("env") if isinstance(payload.get("repro"), dict) else None,
        "reproduction": payload.get("repro") if isinstance(payload.get("repro"), dict) else None,
        "raw": _plan_raw(runs),
        "note": "raw 副本用于独立目录校验/复算；本工具不自动在目标机重跑真机命令",
    }
    content_sha = _content_sha(payload)

    with _LOCK:
        existing = _load_manifests(loop.id)
        if existing:
            report_dir, manifest = existing[0]
            if manifest.get("content_sha256") == content_sha:
                reused = _reuse(report_dir, manifest)
                if reused is not None:
                    return reused

        base = loop_root(loop.id)
        base.mkdir(parents=True, exist_ok=True)
        _clean_tmp(base)
        seq = max([_seq_of(m) for _d, m in existing] or [0]) + 1
        tmp_dir = base / f".tmp-{os.getpid()}-{seq}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        try:
            _copy_raw(_manifest_raw(payload), tmp_dir)
            report_bytes = _canonical(payload)
            report_sha = _sha_bytes(report_bytes)
            (tmp_dir / "report.json").write_bytes(report_bytes)
            final_dir = base / f"{seq:04d}-{report_sha[:12]}"
            if final_dir.exists():  # 理论上 seq 唯一；防御性再让一步
                seq += 1
                final_dir = base / f"{seq:04d}-{report_sha[:12]}"

            repro_lines = [
                "# 复现与校验（自动优化报告）",
                "",
                f"- loop: #{loop.id}",
                f"- report_sha256: `{report_sha}`",
                f"- 生成时间: {_iso(_now())}",
                "",
                "## 校验（不依赖控制台/DB；在本报告目录内运行）",
                "",
                "```bash",
                "python scripts/repro_report.py .",
                "```",
                "",
                "## 真机复现命令（人工执行；v1 不自动重跑）",
                "",
            ]
            repro = payload.get("repro") if isinstance(payload.get("repro"), dict) else {}
            for side in ("before", "after"):
                for action in ("test", "benchmark"):
                    cmd = ((repro.get(side) or {}).get(f"{action}_command")
                           if isinstance(repro.get(side), dict) else None)
                    if cmd:
                        repro_lines.append(f"- {side} {action}: `{cmd}`")
            repro_lines += ["", "raw/ 下是两侧 run 的关键产物副本；超上限/源缺失未复制的文件见 manifest.raw。",
                            "manifest/source 里的绝对路径是本机生成时的原路径，换机后以 raw/ 副本为准。", ""]
            (tmp_dir / "repro.md").write_text("\n".join(repro_lines), encoding="utf-8")

            files = []
            total_bytes = 0
            for name in ("report.json", "repro.md"):
                p = tmp_dir / name
                sha = _sha_file(p)
                size = p.stat().st_size
                files.append({"path": name, "sha256": sha, "bytes": size})
                total_bytes += size
            raw_entries = _manifest_raw(payload)
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "loop_id": loop.id,
                "task_id": task.id,
                "seq": seq,
                "report_sha256": report_sha,
                "content_sha256": content_sha,
                "generated_at": _iso(_now()),
                "files": files,
                "raw": raw_entries,
                "report": {
                    "schema": payload.get("schema"),
                    "verdict": (payload.get("objective_summary") or {}).get("verdict"),
                    "conclusion": (payload.get("objective_summary") or {}).get("conclusion"),
                },
                "size_bytes": total_bytes + sum(
                    int(item.get("bytes") or 0) for item in raw_entries if item.get("copied")),
                "raw_copied_count": sum(1 for item in raw_entries if item.get("copied")),
                "raw_total": len(raw_entries),
                "source": {"app": settings.app_name, "repo_root": settings.repo_root},
            }
            (tmp_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(tmp_dir, final_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    # 写盘后实测校验，而不是写死 report_valid=True（防 raw 复制/路径问题造成假有效）。
    verdict = _verify(manifest, final_dir, deep=True)
    if not verdict["valid"]:
        raise optimization.OptimizationError(
            "报告归档后校验未通过（manifest/raw 不一致）：" + "；".join(verdict["errors"][:3]),
            status=409)
    merged = dict(payload)
    merged.update({
        "report_sha256": report_sha,
        "content_sha256": content_sha,
        "seq": manifest["seq"],
        "report_dir": str(final_dir),
        "written_to": str(final_dir / "report.json"),
        "manifest": manifest,
        "report_valid": True,
        "stored": True,
    })
    logger.info("优化报告已归档（loop=%s seq=%s sha=%s）", loop.id, manifest["seq"],
                report_sha[:12])
    return merged
