#!/usr/bin/env python3
"""校验一份归档的自动优化报告（独立目录，不依赖 DB/控制台）。

用法：
    python scripts/repro_report.py reports/loops/12/0001-abcdef123456
    python scripts/repro_report.py reports/loops/12/0001-abcdef123456/report.json
    python scripts/repro_report.py <dir> --json
    python scripts/repro_report.py <dir> --show-commands

默认只校验、不执行任何真机命令：
1) report.json 的 sha256 == manifest.report_sha256；
2) manifest.files 里每个文件存在且 sha256 一致；
3) manifest.raw 里 `copied=true` 的副本存在且 sha256 一致。
`--show-commands` 额外打印报告里记录的两侧复现命令（供人工到目标机执行）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _safe_child(report_dir: Path, rel: str) -> Path | None:
    """限制 manifest 里的相对路径在报告目录内（防御被篡改的 manifest 读到目录外）。"""
    text = str(rel or "").strip()
    if not text or text.startswith("/") or ".." in Path(text).parts:
        return None
    target = (report_dir / text).resolve()
    root = report_dir.resolve()
    if target != root and root not in target.parents:
        return None
    return target


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def verify_dir(target: Path) -> dict:
    report_dir = target.parent if target.is_file() else target
    manifest_path = report_dir / "manifest.json"
    report_path = report_dir / "report.json"
    result: dict = {"report_dir": str(report_dir), "valid": False, "files": [],
                    "raw": [], "errors": [], "repro_commands": []}
    manifest = _load_json(manifest_path)
    if manifest is None:
        result["errors"].append(f"manifest.json 缺失或不可解析：{manifest_path}")
        return result
    want_sha = str(manifest.get("report_sha256") or "")
    actual = _sha_file(report_path) if report_path.is_file() else None
    ok = bool(want_sha) and actual == want_sha
    result["files"].append({"path": "report.json", "ok": ok, "expected": want_sha or None,
                            "actual": actual})
    if not ok:
        result["errors"].append("report.json 的 sha256 与 manifest.report_sha256 不一致")
    for item in manifest.get("files") or []:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path") or "")
        if not rel or rel == "report.json":
            continue
        p = _safe_child(report_dir, rel)
        got = _sha_file(p) if p is not None and p.is_file() else None
        good = got is not None and got == item.get("sha256")
        result["files"].append({"path": rel, "ok": good, "expected": item.get("sha256"),
                                "actual": got})
        if not good:
            result["errors"].append(f"文件校验失败：{rel}")
    for item in manifest.get("raw") or []:
        if not isinstance(item, dict) or not item.get("copied"):
            continue
        rel = str(item.get("copy") or "")
        p = _safe_child(report_dir, rel)
        got = _sha_file(p) if p is not None and p.is_file() else None
        good = got is not None and got == item.get("sha256")
        result["raw"].append({"kind": item.get("kind"), "path": rel, "ok": good,
                              "expected": item.get("sha256"), "actual": got})
        if not good:
            result["errors"].append(f"raw 副本校验失败：{rel}")
    report = _load_json(report_path) or {}
    repro = report.get("repro") if isinstance(report.get("repro"), dict) else {}
    for side in ("before", "after"):
        block = repro.get(side) if isinstance(repro.get(side), dict) else {}
        for action in ("test", "benchmark"):
            cmd = block.get(f"{action}_command")
            if cmd:
                result["repro_commands"].append({"side": side, "action": action, "command": cmd})
    result["report_sha256"] = want_sha or None
    result["valid"] = not result["errors"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验归档的自动优化报告（只读，不重跑真机）")
    parser.add_argument("path", help="报告目录或 report.json 路径")
    parser.add_argument("--json", action="store_true", help="输出结构化 JSON")
    parser.add_argument("--show-commands", action="store_true", help="打印复现命令")
    args = parser.parse_args(argv)

    target = Path(args.path).expanduser()
    if not target.exists():
        print(f"路径不存在：{target}", file=sys.stderr)
        return 2
    result = verify_dir(target)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"报告目录：{result['report_dir']}")
        print(f"report_sha256：{result.get('report_sha256')}")
        print(f"校验：{'PASS' if result['valid'] else 'FAIL'}")
        if result["errors"]:
            for err in result["errors"][:20]:
                print(f"  - {err}")
        else:
            print(f"  - {len(result['files'])} 个文件 + {len(result['raw'])} 个 raw 副本全部一致")
    if args.show_commands and result["repro_commands"]:
        print("\n复现命令（人工到目标机执行；本脚本不自动跑）：")
        for item in result["repro_commands"]:
            print(f"- [{item['side']}/{item['action']}] {item['command']}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
