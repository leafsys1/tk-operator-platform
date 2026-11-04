#!/usr/bin/env python3
"""最小用例冒烟（smoke 动作的核心）—— 开发完成后先证明「编得出来、跑得起来」。

为什么要有它（用户 2026-10-23 要求）：算子刚开发完就等全套 test/benchmark 才发现构建失败或
根本跑不通，返工成本高。smoke 只挑**一条最小 required match 用例**：
  - 优先 `expected=match`、`n_a == n_b`、required 的用例里 `n_a` 最小的那条（同 n 取 case_id 字典序）；
  - 用与正式 test 完全相同的输入生成、候选调用协议、golden、判据（复用 `tools/run_cases.py`
    的单用例执行逻辑，不另写一套判定）；
  - 产物 `/out/smoke.json`（机读）+ `/out/smoke.md`（人读），门禁按 `verdict=PASS` 判。

它不是正式正确性测试：用例覆盖只有 1 条；四动作门禁里的正确性仍由 `test` 动作（全部用例）负责。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_cases  # noqa: E402 —— 复用同一套输入生成/判据/候选调用

WS = Path(__file__).resolve().parents[1]


def _env_or_arg(name: str, arg):
    return arg if arg not in (None, "") else os.environ.get(name, "")


def pick_minimal_case(cases: list[dict], dtype: str) -> dict:
    """挑一条最小 required match 用例（n_a == n_b，整数 n，n 最小）。"""
    candidates = [
        c for c in cases
        if c.get("expected", "match") == "match"
        and c.get("required", True)
        and isinstance(c.get("n_a"), int)
        and c.get("n_a") == c.get("n_b")
        and (c.get("dtype") or dtype) == dtype
    ]
    if not candidates:
        # 退一步：任意 required match（dtype 可能不同；仍按 n_a==n_b）
        candidates = [
            c for c in cases
            if c.get("expected", "match") == "match"
            and c.get("required", True)
            and isinstance(c.get("n_a"), int)
            and c.get("n_a") == c.get("n_b")
        ]
    if not candidates:
        raise SystemExit("cases 里没有可用的 required match 用例（smoke 需要至少一条能比对的用例）")
    return min(candidates, key=lambda c: (int(c["n_a"]), str(c.get("case_id", ""))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="elementwise_add")
    ap.add_argument("--workspace", default=str(WS))
    ap.add_argument("--exe", default="")
    ap.add_argument("--cases", default="")
    ap.add_argument("--golden", default="")
    ap.add_argument("--out", default="/out/smoke.json")
    ap.add_argument("--md", default="")
    ap.add_argument("--workdir", default="/tmp/opx-smoke")
    ap.add_argument("--evidence-dir", default="")
    ap.add_argument("--case-timeout", type=int, default=300)
    ap.add_argument("--spec-version-id", default="")
    ap.add_argument("--criteria-ref", default="")
    ap.add_argument("--code-snapshot-sha256", default="")
    ap.add_argument("--suite-ref", default="")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    op = args.op
    cases_path = Path(args.cases) if args.cases else ws / "cases" / op / "cases.json"
    exe = Path(args.exe) if args.exe else ws / "ops" / op / "build" / op
    golden_path = Path(args.golden) if args.golden else ws / "ops" / op / "scripts" / "golden.py"
    out = Path(args.out)
    md = Path(args.md) if args.md else out.with_suffix(".md")
    workdir = Path(args.workdir)
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else out.parent / "evidence"

    spec_version_id = _env_or_arg("OPX_SPEC_VERSION_ID", args.spec_version_id)
    criteria_ref = _env_or_arg("OPX_CRITERIA_REF", args.criteria_ref)
    snapshot = _env_or_arg("OPX_CODE_SNAPSHOT_SHA256", args.code_snapshot_sha256)
    suite_ref = _env_or_arg("OPX_SUITE_REF", args.suite_ref)
    run_id = _env_or_arg("OPX_RUN_ID", "")

    missing = [k for k, v in (("spec_version_id", spec_version_id), ("criteria_ref", criteria_ref),
                              ("code_snapshot_sha256", snapshot), ("suite_ref", suite_ref)) if not v]
    if missing:
        print(f"[smoke] 版本对齐字段缺失：{missing}（应由 nm_test.sh 注入环境变量）", file=sys.stderr)
        return 2
    if not cases_path.is_file():
        print(f"[smoke] 用例矩阵不存在：{cases_path}", file=sys.stderr)
        return 2
    if not exe.is_file():
        print(f"[smoke] 候选实现不存在：{exe}（先跑 build）", file=sys.stderr)
        return 2
    if not golden_path.is_file():
        print(f"[smoke] 参考实现不存在：{golden_path}", file=sys.stderr)
        return 2

    suite = json.loads(cases_path.read_text(encoding="utf-8"))

    # ---- 显式按 op 分派：scan 分支（A1）。elementwise 旧路径保持原样。 ----
    if op == "jax_lax_scan":
        if suite.get("op") != op:
            print(f"[smoke] suite.op={suite.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
            return 2
        try:
            scripts = ws / "ops" / "jax_lax_scan" / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            import scan_engine  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] scan 执行器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return scan_engine.run_smoke(
            ws=ws, op=op, exe=exe, cases_path=cases_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, timeout=args.case_timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    # ---- 显式按 op 分派：jax_lax_associative_scan 分支。elementwise / scan 旧路径保持原样。 ----
    if op == "jax_lax_associative_scan":
        if suite.get("op") != op:
            print(f"[smoke] suite.op={suite.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
            return 2
        try:
            scripts = ws / "ops" / "jax_lax_associative_scan" / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            import assoc_scan_engine  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] assoc-scan 执行器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return assoc_scan_engine.run_smoke(
            ws=ws, op=op, exe=exe, cases_path=cases_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, timeout=args.case_timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    # ---- 显式按 op 分派：jax_lax_cond 分支。elementwise / scan / assoc-scan 旧路径保持原样。 ----
    if op == "jax_lax_cond":
        if suite.get("op") != op:
            print(f"[smoke] suite.op={suite.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
            return 2
        try:
            scripts = ws / "ops" / "jax_lax_cond" / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            import cond_engine  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] cond 执行器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return cond_engine.run_smoke(
            ws=ws, op=op, exe=exe, cases_path=cases_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, timeout=args.case_timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    dtype = suite.get("dtype", "float32")
    case = pick_minimal_case(suite.get("cases") or [], dtype)
    print(f"[smoke] 最小用例：{case['case_id']} n={case['n_a']} dtype={case.get('dtype', dtype)}")

    criteria = run_cases._load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    golden_mod = run_cases._load_module(golden_path, f"{op}_golden")
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    rec = run_cases._run_case(case, op=op, exe=exe, criteria=criteria, golden_mod=golden_mod,
                              workdir=workdir, evidence_dir=evidence_dir, dtype=dtype,
                              timeout=args.case_timeout)
    passed = rec.get("status") == "PASS"
    report = {
        "schema": "opx-smoke/1",
        "op": op,
        "action": "smoke",
        "run_id": run_id or None,
        # 版本对齐（与 correctness 报告同口径，门禁逐字比对）
        "spec_version_id": int(spec_version_id),
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": snapshot,
        "suite_ref": suite_ref,
        "candidate": {
            "exe": str(exe.relative_to(ws)) if str(exe).startswith(str(ws)) else str(exe),
            "sha256": run_cases._sha256_file(exe),
            "exec_path_marker_required": True,
        },
        "case_source": str(cases_path.relative_to(ws)) if str(cases_path).startswith(str(ws)) else str(cases_path),
        "case_source_sha256": run_cases._sha256_file(cases_path),
        "case": rec,
        "verdict": "PASS" if passed else "FAIL",
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 冒烟结果：{op}",
        "",
        f"- 用例：`{rec.get('case_id')}`（n={case.get('n_a')}，dtype={case.get('dtype', dtype)}）",
        f"- 结论：**{report['verdict']}**",
        f"- 耗时：{rec.get('duration_ms', '—')} ms",
        f"- 候选：`{report['candidate']['exe']}` sha256 `{report['candidate']['sha256'][:16]}…`",
        f"- 判据档：`{criteria_ref}`；用例集：`{suite_ref}`",
    ]
    if not passed:
        lines += ["", f"- 失败原因：{rec.get('detail') or rec.get('status')}"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[smoke] SMOKE_VERDICT {report['verdict']}  → {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
