"""硬件画像（`ops/chips/*.yaml`）读侧（方案 §13-A11）。

画像由 agent/人维护在**任务工作区**里（`<workspace>/ops/chips/<arch>.yaml`），本模块只读不写：
把它读出来给「极限 / Roofline 对比」用 —— 达成带宽对比 HBM 峰值、达成算力对比向量峰值。
读不到就返回空 / None（不猜、不用内置默认值顶替）。
"""

from __future__ import annotations

from pathlib import Path

import yaml


def chips_dir(workspace: Path) -> Path:
    return workspace / "ops" / "chips"


def _load(path: Path) -> dict | None:
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def list_chips(workspace: Path) -> list[dict]:
    """列出画像的**摘要**（不返回整份，列表接口要轻）。"""
    d = chips_dir(workspace)
    if not d.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(d.glob("*.yaml")):
        doc = _load(p)
        if not doc:
            continue
        out.append({
            "name": p.stem,
            "schema": doc.get("schema"),
            "arch": doc.get("arch"),
            "npu_arch_flag": doc.get("npu_arch_flag"),
            "product_family": doc.get("product_family"),
            "models": doc.get("models") or [],
        })
    return out


def load_chip(workspace: Path, name: str) -> dict | None:
    """读一份完整画像（按文件名，不带 .yaml）。"""
    p = chips_dir(workspace) / f"{name}.yaml"
    return _load(p) if p.is_file() else None


def find_chip_for(workspace: Path, target: str | None) -> dict | None:
    """按目标芯片（如 `910B1`）选画像：先比 `models`，再比 `arch`/`npu_arch_flag` 子串，最后退第一个。"""
    d = chips_dir(workspace)
    if not d.is_dir():
        return None
    docs = [(_load(p), p) for p in sorted(d.glob("*.yaml"))]
    docs = [(doc, p) for doc, p in docs if doc]
    if not docs:
        return None
    if target:
        t = str(target).lower()
        for doc, _ in docs:
            for m in doc.get("models") or []:
                if t in str(m).lower() or str(m).lower() in t:
                    return doc
        for doc, _ in docs:
            if t in str(doc.get("arch") or "").lower() or t in str(doc.get("npu_arch_flag") or "").lower():
                return doc
    return docs[0][0]
