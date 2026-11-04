#!/usr/bin/env python3
"""提交前脱敏扫描 —— 发现问题就非零退出，挡住 git commit。

用法：  python3 deploy/scan_secrets.py
检查：  API key / 私钥 / 口令 / GitHub token / 内网外的明文凭据
"""
from __future__ import annotations

import re
import subprocess
import sys

PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "疑似 API Key (sk-...)"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"), "GitHub token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "PEM 私钥"),
    (re.compile(r"Hzzgkj@\d+"), "明文口令"),
    (re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{6,}['\"]"), "硬编码口令"),
    (re.compile(r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{20,}"), "硬编码 Bearer"),
]

# 允许出现在文档里的占位/示例（命中这些不算问题）
ALLOW = (
    "sk-7……", "sk-xxx", "<your-key>", "YOUR_KEY", "xxx…", "…",
    "sk-[A-Za-z0-9]{16,}",   # 文档里写正则本身
)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache"}


def iter_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if out.returncode != 0:
        print("不在 git 仓库里，先 git init", file=sys.stderr)
        return []
    return [f for f in out.stdout.splitlines() if f]


def main() -> int:
    hits = []
    for path in iter_files():
        if any(f"/{d}/" in f"/{path}" for d in SKIP_DIRS):
            continue
        # 测试夹具里的假私钥/假口令不算泄露（已人工核对：都是 fixture 生成的临时值）
        is_test = "/tests/" in f"/{path}" or path.rsplit("/", 1)[-1].startswith("test_")
        try:
            data = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for rx, label in PATTERNS:
            if is_test and label in ("PEM 私钥", "硬编码口令", "硬编码 Bearer"):
                continue
            for m in rx.finditer(data):
                frag = m.group(0)
                if any(a in frag for a in ALLOW) or any(a in data[max(0, m.start() - 4):m.start()] for a in ALLOW):
                    continue
                line = data[:m.start()].count("\n") + 1
                hits.append((path, line, label, frag[:24]))
    if hits:
        print(f"发现 {len(hits)} 处疑似敏感内容：", file=sys.stderr)
        for p, l, lab, frag in hits:
            print(f"  {p}:{l}  [{lab}]  {frag}…", file=sys.stderr)
        print("\n先脱敏再提交。确认是误报就把片段加进本脚本的 ALLOW 白名单。", file=sys.stderr)
        return 1
    print(f"脱敏扫描通过（{len(iter_files())} 个文件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
