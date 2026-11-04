#!/usr/bin/env python3
"""导入「算子案例库」bundle（独立 zip，仓库不预置案例）。

设计口径（2026-10-20 用户定）：
- 案例库是**项目的补充**，不是预置内容 —— 仓库里不放大批案例；
- 案例库由用户自己保存一份 zip；需要时用本脚本导入到**仓库外**的库目录；
- 导入后写一份 `index.json`，后续控制台（M2）按索引读取；现在只做导入与校验。

Bundle 结构（最小）：

    my-cases.zip
    ├── manifest.json                 # 见下
    └── cases/
        ├── relu/
        │   ├── cases.json            # 用例矩阵（必需）
        │   ├── benchmark.json        # 基准定义（可选）
        │   └── …                     # 实现/参考/脚本（可选）
        └── softmax/
            └── cases.json

manifest.json：

    {
      "schema": "opx-case-library/1",
      "name": "cann-cases",           # bundle 名（索引主键；重复导入会覆盖）
      "version": "2026.10",
      "cases": [
        {"op": "relu", "scenario": "激活", "path": "cases/relu"}
      ]
    }

用法：

    python3 scripts/import_cases.py --zip ~/Downloads/my-cases.zip
    python3 scripts/import_cases.py --zip … --dest /path/to/library --force
    python3 scripts/import_cases.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DEST = Path(os.environ.get("OPX_CASE_LIBRARY", "~/.local/share/op-agent/case-library")).expanduser()
SCHEMA = "opx-case-library/1"

#: bundle 安全上限：防 zip bomb / 打满磁盘（案例库是本地 CLI，但仍按不可信输入处理）。
MAX_FILES = 5000
MAX_FILE_BYTES = 100 * 1024 * 1024  # 单文件解压后
MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 全部解压后
MAX_COMPRESSION_RATIO = 200  # 解压/压缩 比超过它视为可疑
MAX_NAME_LEN = 120


class BundleError(ValueError):
    """bundle 不合法（结构 / manifest / 路径安全）。"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(name: str) -> Path:
    """拒绝绝对路径 / 路径穿越 / Windows 盘符。"""
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or (p.parts and p.parts[0].endswith(":")):
        raise BundleError(f"bundle 里有不安全的路径：{name!r}")
    return p


def _safe_bundle_name(raw) -> str:
    """manifest.name 必须是**单段**安全目录名（索引主键，也是落盘目录名）。"""
    name = str(raw or "").strip()
    if not name:
        raise BundleError("manifest.name 不能为空")
    if len(name) > MAX_NAME_LEN:
        raise BundleError(f"manifest.name 太长（>{MAX_NAME_LEN}）：{name!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise BundleError(f"manifest.name 含控制字符：{name!r}")
    if name in (".", "..") or "/" in name or "\\" in name:
        raise BundleError(f"manifest.name 必须是单段目录名：{name!r}")
    rel = _safe_rel(name)
    if len(rel.parts) != 1:
        raise BundleError(f"manifest.name 必须是单段目录名：{name!r}")
    return name


def _check_zip_limits(zf: zipfile.ZipFile) -> None:
    """条目数 / 单文件解压 / 总解压 / 压缩比上限（按 central directory 的声明大小）。"""
    infos = zf.infolist()
    if len(infos) > MAX_FILES:
        raise BundleError(f"bundle 条目数超过上限（{len(infos)} > {MAX_FILES}）")
    total = 0
    for info in infos:
        size = int(info.file_size or 0)
        comp = int(info.compress_size or 0)
        if size > MAX_FILE_BYTES:
            raise BundleError(
                f"bundle 单文件解压后超过上限：{info.filename}（{size} > {MAX_FILE_BYTES}）")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise BundleError(f"bundle 解压总量超过上限（>{MAX_TOTAL_BYTES}）")
        if size > 0 and (comp <= 0 or size / comp > MAX_COMPRESSION_RATIO):
            raise BundleError(
                f"bundle 压缩比异常（{info.filename}：{size}/{comp} > {MAX_COMPRESSION_RATIO}）")


def _atomic_write_json(path: Path, data: dict) -> None:
    """临时文件 + `os.replace` 原子写（进程中断不留半截 index.json）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        raw = zf.read("manifest.json")
    except KeyError as e:
        raise BundleError("bundle 根目录缺 manifest.json") from e
    try:
        man = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BundleError(f"manifest.json 不是合法 JSON：{e}") from e
    if not isinstance(man, dict):
        raise BundleError("manifest.json 必须是对象")
    if man.get("schema") != SCHEMA:
        raise BundleError(f"manifest.schema 必须是 {SCHEMA!r}（收到 {man.get('schema')!r}）")
    man["name"] = _safe_bundle_name(man.get("name"))
    return man


def validate(zf: zipfile.ZipFile, man: dict) -> list[dict]:
    """校验 bundle 并返回案例清单（op / scenario / path / target_chip）。"""
    _check_zip_limits(zf)
    declared = man.get("cases")
    names = set(zf.namelist())

    def is_dir(path: str) -> bool:
        p = path.rstrip("/")
        return any(n == p or n.startswith(p + "/") for n in names)

    def chip_of(rel: str) -> str | None:
        """cases.json 顶层 target_chip（可选）。"""
        try:
            data = json.loads(zf.read(f"{rel.rstrip('/')}/cases.json").decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError, OSError):
            return None
        return data.get("target_chip") if isinstance(data, dict) else None

    rows: list[dict] = []
    if declared is None:
        # 没写 cases 列表：按 cases/*/cases.json 推导
        for n in sorted(names):
            parts = Path(n).parts
            if len(parts) == 3 and parts[0] == "cases" and parts[2] == "cases.json":
                chip = chip_of(f"cases/{parts[1]}") or man.get("target_chip")
                rows.append({"op": parts[1], "scenario": None, "path": f"cases/{parts[1]}",
                             "target_chip": chip})
        if not rows:
            raise BundleError("bundle 里没有 cases/<op>/cases.json")
        return rows

    if not isinstance(declared, list) or not declared:
        raise BundleError("manifest.cases 必须是非空数组")
    for item in declared:
        if not isinstance(item, dict) or not item.get("op"):
            raise BundleError(f"manifest.cases 条目缺 op：{item!r}")
        rel = _safe_rel(str(item.get("path") or f"cases/{item['op']}"))
        if not str(rel).startswith("cases/"):
            raise BundleError(f"案例路径必须在 cases/ 下：{rel}")
        if not is_dir(rel.as_posix()):
            raise BundleError(f"案例目录不存在：{rel}")
        report = (rel / "cases.json").as_posix()
        if report not in names:
            raise BundleError(f"案例 {item['op']} 缺 cases.json（{report}）")
        try:
            data = json.loads(zf.read(report).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise BundleError(f"{report} 不是合法 JSON：{e}") from e
        if not isinstance(data, dict) or not data.get("cases"):
            raise BundleError(f"{report} 里没有 cases 数组")
        chip = item.get("target_chip") or (data.get("target_chip") if isinstance(data, dict) else None) \
            or man.get("target_chip")
        rows.append({"op": str(item["op"]), "scenario": item.get("scenario"),
                     "path": rel.as_posix(), "target_chip": chip})
    ops = [r["op"] for r in rows]
    if len(set(ops)) != len(ops):
        raise BundleError(f"manifest.cases 有重复 op：{ops}")
    return rows


def import_bundle(zip_path: Path, dest: Path = DEFAULT_DEST, *, force: bool = False) -> dict:
    """校验并导入一个 bundle，返回本次导入的摘要（同时更新 dest/index.json）。"""
    if not zip_path.is_file():
        raise BundleError(f"zip 不存在：{zip_path}")
    dest = dest.expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        _check_zip_limits(zf)
        for name in zf.namelist():
            _safe_rel(name)
        man = read_manifest(zf)
        rows = validate(zf, man)
        name = str(man["name"]).strip()
        # name 已经过单段校验；这里再校验落点父目录 == dest（防符号链接/未来改动绕过）
        dest_resolved = dest.resolve()
        target = (dest / name).resolve()
        if target.parent != dest_resolved:
            raise BundleError(f"manifest.name 越界（落点不在库目录内）：{name!r}")
        if target.exists() and not force:
            raise BundleError(f"目标已存在：{target}（加 --force 覆盖）")

        staging = Path(tempfile.mkdtemp(prefix="opx-cases-", dir=str(dest)))
        try:
            for name_in_zip in zf.namelist():
                rel = _safe_rel(name_in_zip)
                if rel.parts and rel.parts[0] not in ("cases", "manifest.json"):
                    continue
                out = staging / rel
                if name_in_zip.endswith("/"):
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name_in_zip) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            if target.exists():
                # 只允许删除库目录内的目标目录；文件/库外路径一律拒绝（旧实现可 rmtree 任意目录）
                if target.parent != dest_resolved or not target.is_dir():
                    raise BundleError(f"拒绝覆盖：目标不在库目录内或不是目录：{target}")
                shutil.rmtree(target)
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    digest = _sha256(zip_path)
    entry = {
        "name": name,
        "version": man.get("version"),
        "source_zip": str(zip_path),
        "sha256": digest,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
    }
    index_path = dest / "index.json"
    index = {"schema": "opx-case-library-index/1", "dest": str(dest), "bundles": []}
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("bundles"), list):
                index = loaded
        except (OSError, json.JSONDecodeError):
            pass
    index["schema"] = "opx-case-library-index/1"
    index["dest"] = str(dest)
    index["bundles"] = [b for b in index.get("bundles", []) if b.get("name") != name] + [entry]
    _atomic_write_json(index_path, index)
    return {"dest": str(dest), **entry, "index": str(index_path)}


def list_library(dest: Path = DEFAULT_DEST) -> dict:
    index_path = dest.expanduser() / "index.json"
    if not index_path.is_file():
        return {"dest": str(dest.expanduser()), "bundles": []}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"dest": str(dest.expanduser()), "bundles": [], "error": "index.json 读不了"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="导入算子案例库 bundle（独立 zip）")
    ap.add_argument("--zip", help="bundle zip 路径")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help=f"库目录（默认 {DEFAULT_DEST}）")
    ap.add_argument("--force", action="store_true", help="同名 bundle 已存在时覆盖")
    ap.add_argument("--list", action="store_true", help="列出已导入的 bundle")
    args = ap.parse_args(argv)

    if args.list:
        print(json.dumps(list_library(Path(args.dest)), ensure_ascii=False, indent=1))
        return 0
    if not args.zip:
        ap.error("需要 --zip（或用 --list 查看已导入）")
    try:
        result = import_bundle(Path(args.zip), Path(args.dest), force=args.force)
    except BundleError as e:
        print(f"[import_cases] {e}", file=sys.stderr)
        return 2
    print(f"[import_cases] 已导入 {result['name']}（{len(result['cases'])} 个案例）→ {result['dest']}")
    for c in result["cases"]:
        print(f"  - {c['op']}（{c.get('scenario') or '未标场景'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
