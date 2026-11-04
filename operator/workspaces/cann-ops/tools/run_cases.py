#!/usr/bin/env python3
"""用例矩阵执行器 —— 四动作 `test` 的核心（在**容器里**跑）。

流程：读 `cases/<op>/cases.json` → 逐例生成输入 → 调候选实现（Ascend C 直调可执行文件）→
与独立标杆比对（判据档机判 `criteria/check.py`）→ 写 `correctness.json` + `correctness.md`。

对上门禁契约（docs/gate-report-v1.md）的几个要害：

1. **版本对齐字段由 nm_test.sh 注入环境变量**（`OPX_SPEC_VERSION_ID` / `OPX_CRITERIA_REF` /
   `OPX_CODE_SNAPSHOT_SHA256` / `OPX_SUITE_REF`）。容器里**不重算**源码哈希 —— 两边各算一次就多一个
   漂移面，而门禁只认真实执行那一版。缺字段就用 CLI 参数；都没有则明确报错，不写假值。
2. **逐例状态枚举** `PASS|FAIL|ERROR|INVALID|NOT_RUN`。必需用例 `NOT_RUN` 不算通过。
3. **「拒得对不对」也要判**：`expected=reject` 的用例，实现必须非 0 退出**且**打印约定的拒绝码
   （否则「因为别的原因失败」会被当成拒绝通过）。
4. **不许静默回退**：候选可执行文件每例必须打印 `EXEC` 标记行（算子名 + kernel 名 + 架构）；
   缺标记 → `INVALID`，而不是「过了」（§4.5：验证项含「实际执行的是否为候选算子路径」）。
5. **证据按需留**：通过用例只留 `meta.json`；失败/异常用例才把输入与输出 bin 一起留下（复现用），
   免得 1M 元素的用例把产物目录撑爆。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "tools"))
sys.path.insert(0, str(WS / "criteria"))

import gen_inputs  # noqa: E402

CASE_STATUSES = ("PASS", "FAIL", "ERROR", "INVALID", "NOT_RUN")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"模块加载失败：{path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_scan_engine(ws: Path):
    """按 op 显式分派：scan 用例的核心执行器（scan 字段模型与 elementwise 不同）。

    实现放在 `ops/jax_lax_scan/scripts/scan_engine.py`，本 runner 仍是唯一门禁入口；
    导入失败说明工程缺失，直接报错而不是回退到 elementwise 分支。
    """
    scripts = ws / "ops" / "jax_lax_scan" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import scan_engine  # noqa: PLC0415

    return scan_engine


def _load_assoc_scan_engine(ws: Path):
    """按 op 显式分派：jax_lax_associative_scan 的核心执行器。

    实现放在 `ops/jax_lax_associative_scan/scripts/assoc_scan_engine.py`；assoc-scan 用例
    的字段模型（op_kind/axis/reverse/shape|elems/mode）与 elementwise、scan 都不同。
    导入失败说明工程缺失，直接报错而不是回退到其他分支。
    """
    scripts = ws / "ops" / "jax_lax_associative_scan" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import assoc_scan_engine  # noqa: PLC0415

    return assoc_scan_engine


def _load_cond_engine(ws: Path):
    """按 op 显式分派：jax_lax_cond 的核心执行器。

    实现放在 `ops/jax_lax_cond/scripts/cond_engine.py`；cond 用例的字段模型
    （pred/branch_ids/mode/variant/pytree）与 elementwise、scan、assoc-scan 都不同。
    导入失败说明工程缺失，直接报错而不是回退到其他分支。
    """
    scripts = ws / "ops" / "jax_lax_cond" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import cond_engine  # noqa: PLC0415

    return cond_engine


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_or_arg(name: str, arg):
    return arg if arg not in (None, "") else os.environ.get(name, "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="elementwise_add")
    ap.add_argument("--workspace", default=str(WS))
    ap.add_argument("--cases", default="")
    ap.add_argument("--exe", default="")
    ap.add_argument("--golden", default="")
    ap.add_argument("--out", default="/out/correctness.json")
    ap.add_argument("--md", default="")
    ap.add_argument("--workdir", default="/tmp/opx-cases")
    ap.add_argument("--evidence-dir", default="")
    ap.add_argument("--case-timeout", type=int, default=300)
    ap.add_argument("--run-id", default="")
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
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else out.parent / "evidence"
    workdir = Path(args.workdir)

    spec_version_id = _env_or_arg("OPX_SPEC_VERSION_ID", args.spec_version_id)
    criteria_ref = _env_or_arg("OPX_CRITERIA_REF", args.criteria_ref)
    snapshot = _env_or_arg("OPX_CODE_SNAPSHOT_SHA256", args.code_snapshot_sha256)
    suite_ref = _env_or_arg("OPX_SUITE_REF", args.suite_ref)
    run_id = _env_or_arg("OPX_RUN_ID", args.run_id)

    missing = [k for k, v in (("spec_version_id", spec_version_id), ("criteria_ref", criteria_ref),
                              ("code_snapshot_sha256", snapshot), ("suite_ref", suite_ref)) if not v]
    if missing:
        print(f"[run_cases] 版本对齐字段缺失：{missing}（应由 nm_test.sh 注入环境变量）", file=sys.stderr)
        return 2
    if not cases_path.is_file():
        print(f"[run_cases] 用例矩阵不存在：{cases_path}", file=sys.stderr)
        return 2
    if not exe.is_file():
        print(f"[run_cases] 候选实现不存在：{exe}（先跑 build 动作）", file=sys.stderr)
        return 2

    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = suite.get("cases") or []

    # ---- 显式按 op 分派：scan 分支（A1）。elementwise 旧路径保持原样。 ----
    if op == "jax_lax_scan":
        if suite.get("op") != op:
            print(f"[run_cases] suite.op={suite.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
            return 2
        try:
            scan_engine = _load_scan_engine(ws)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_cases] scan 执行器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        md = Path(args.md) if args.md else out.with_suffix(".md")
        return scan_engine.run_suite(
            ws=ws, op=op, exe=exe, cases_path=cases_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, timeout=args.case_timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    # ---- 显式按 op 分派：jax_lax_associative_scan 分支。elementwise / scan 旧路径保持原样。 ----
    if op == "jax_lax_associative_scan":
        if suite.get("op") != op:
            print(f"[run_cases] suite.op={suite.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
            return 2
        try:
            assoc_engine = _load_assoc_scan_engine(ws)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_cases] assoc-scan 执行器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        md = Path(args.md) if args.md else out.with_suffix(".md")
        return assoc_engine.run_suite(
            ws=ws, op=op, exe=exe, cases_path=cases_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, timeout=args.case_timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    # ---- 显式按 op 分派：jax_lax_cond 分支。elementwise / scan / assoc-scan 旧路径保持原样。 ----
    if op == "jax_lax_cond":
        if suite.get("op") != op:
            print(f"[run_cases] suite.op={suite.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
            return 2
        try:
            cond_engine = _load_cond_engine(ws)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_cases] cond 执行器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        md = Path(args.md) if args.md else out.with_suffix(".md")
        return cond_engine.run_suite(
            ws=ws, op=op, exe=exe, cases_path=cases_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, timeout=args.case_timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    criteria_path = ws / "criteria" / "check.py"
    criteria = _load_module(criteria_path, "opx_criteria_check")
    golden_mod = _load_module(golden_path, f"{op}_golden")

    def _rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(ws).as_posix()
        except ValueError:
            return str(path)

    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        results.append(_run_case(case, op=op, exe=exe, criteria=criteria, golden_mod=golden_mod,
                                 workdir=workdir, evidence_dir=evidence_dir,
                                 dtype=suite.get("dtype", "float32"),
                                 timeout=args.case_timeout))

    summary = {s: sum(1 for r in results if r["status"] == s) for s in CASE_STATUSES}
    summary["TOTAL"] = len(results)

    report = {
        "schema": "opx-correctness/1",
        "op": op,
        "action": "test",
        "run_id": run_id or None,
        # ---- 版本对齐（门禁逐字比对这一版 run）----
        "spec_version_id": int(spec_version_id),
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": snapshot,
        "suite_ref": suite_ref,
        # ---- 候选实现指纹 ----
        "candidate": {
            "exe": str(exe.relative_to(ws)) if str(exe).startswith(str(ws)) else str(exe),
            "sha256": _sha256_file(exe),
            "exec_path_marker_required": True,
        },
        "case_source": str(cases_path.relative_to(ws)) if str(cases_path).startswith(str(ws)) else str(cases_path),
        "case_source_sha256": _sha256_file(cases_path),
        # ---- 证据输入指纹（参考实现/判据）：优化前后不一致 → 精度对比失效（perf_diff 硬拦） ----
        "oracle": {
            "golden_path": _rel(golden_path),
            "golden_sha256": _sha256_file(golden_path),
            "criteria_path": _rel(criteria_path),
            "criteria_sha256": _sha256_file(criteria_path),
        },
        "cases": results,
        "summary": summary,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 用例集原件随产物留一份（manifest 会按 out/cases/ 前缀标成 cases 类型）
    cases_copy = out.parent / "cases" / f"{op}-cases.json"
    cases_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cases_path, cases_copy)

    md = Path(args.md) if args.md else out.with_suffix(".md")
    md.write_text(_render_md(report), encoding="utf-8")

    print(f"[run_cases] {summary}  → {out}")
    failed = [r["case_id"] for r in results if r["status"] in ("FAIL", "ERROR", "INVALID")]
    if failed:
        print(f"[run_cases] 未通过：{failed}", file=sys.stderr)
    return 0 if (summary["FAIL"] == 0 and summary["ERROR"] == 0 and summary["INVALID"] == 0
                 and summary["NOT_RUN"] == 0) else 1


def _run_case(case, *, op, exe, criteria, golden_mod, workdir, evidence_dir, dtype, timeout) -> dict:
    case_id = case["case_id"]
    n_a, n_b = int(case["n_a"]), int(case["n_b"])
    case_dtype = case.get("dtype", dtype)
    seed = int(case.get("seed", 0))
    expected = case.get("expected", "match")
    cdir = workdir / case_id
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=True)

    rec = {
        "case_id": case_id,
        "group": case.get("group"),
        "required": bool(case.get("required", True)),
        "shape": [n_a],
        "shape_b": [n_b],
        "dtype": case_dtype,
        "seed": seed,
        "gen": f"a={gen_inputs.describe(case.get('gen_a'))};b={gen_inputs.describe(case.get('gen_b'))}",
        "expected": expected,
        "status": "NOT_RUN",
        "note": case.get("note"),
    }

    try:
        a = gen_inputs.generate(case.get("gen_a") or {}, n_a, seed, case_dtype)
        b = gen_inputs.generate(case.get("gen_b") or {}, n_b, seed + 1, case_dtype)
    except Exception as exc:  # 生成失败 = 用例本身写错了
        rec.update(status="ERROR", detail=f"输入生成失败：{type(exc).__name__}: {exc}")
        _dump_evidence(evidence_dir, case_id, rec, [])
        return rec

    in_a, in_b, out_bin = cdir / "a.bin", cdir / "b.bin", cdir / "out.bin"
    a.tofile(in_a)
    b.tofile(in_b)

    # argv 级拒绝用例（参数缺失 / 未知参数）：用例用 `argv` 声明对标准 argv 的改写，
    # 否则 runner 总是传完整合法 argv，这类用例永远不会被拒绝。
    named = {"--n-a": str(n_a), "--n-b": str(n_b), "--dtype": case_dtype,
             "--in-a": str(in_a), "--in-b": str(in_b), "--out": str(out_bin)}
    argv_spec = case.get("argv") or {}
    for drop in argv_spec.get("drop", []):
        named.pop(drop, None)
    cmd = [str(exe)]
    for k, v in named.items():
        cmd += [k, v]
    for extra in argv_spec.get("append", []):
        cmd += [str(x) for x in extra]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cdir))
    except subprocess.TimeoutExpired:
        rec.update(status="ERROR", detail=f"候选实现超时（{timeout}s）", cmd=" ".join(cmd))
        _dump_evidence(evidence_dir, case_id, rec, [])
        return rec
    rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    rec["cmd"] = " ".join(cmd)
    stdout, stderr, rc = proc.stdout or "", proc.stderr or "", proc.returncode

    if expected == "reject":
        code = case.get("expect_reject_code", "")
        hit = code in (stdout + stderr)
        if rc != 0 and hit:
            rec.update(status="PASS", detail=f"按预期拒绝：rc={rc} {code}")
        elif rc == 0:
            rec.update(status="FAIL", detail=f"该拒绝却成功返回（rc=0）；期望拒绝码 {code}")
        else:
            rec.update(status="FAIL", detail=f"拒绝了但不是预期的理由：rc={rc}；期望 {code}；stderr={stderr.strip()[:200]}")
        _dump_evidence(evidence_dir, case_id, rec, [in_a, in_b, out_bin])
        return rec

    if rc != 0:
        rec.update(status="ERROR", detail=f"候选实现非 0 退出 rc={rc}；stderr={stderr.strip()[:300]}",
                   stdout_tail=stdout.strip()[-300:])
        _dump_evidence(evidence_dir, case_id, rec, [in_a, in_b, out_bin])
        return rec

    marker = next((ln for ln in stdout.splitlines() if ln.startswith("EXEC ")), "")
    if not marker:
        rec.update(status="INVALID",
                   detail="候选实现没有打印 EXEC 标记行 → 无法证明实际执行的是候选算子路径（§4.5）",
                   stdout_tail=stdout.strip()[-300:])
        _dump_evidence(evidence_dir, case_id, rec, [in_a, in_b, out_bin])
        return rec
    rec["exec_marker"] = marker.strip()

    if not out_bin.is_file():
        rec.update(status="ERROR", detail="候选实现返回 0 但没有写出输出文件（静默失败）")
        _dump_evidence(evidence_dir, case_id, rec, [in_a, in_b])
        return rec

    got = np.fromfile(out_bin, dtype=case_dtype)
    if got.shape != (n_a,):
        rec.update(status="ERROR", detail=f"输出 shape 不对：{got.shape} ≠ ({n_a},)")
        _dump_evidence(evidence_dir, case_id, rec, [in_a, in_b, out_bin])
        return rec

    try:
        want = golden_mod.compute_golden(a, b)
    except Exception as exc:
        rec.update(status="ERROR", detail=f"标杆计算失败：{type(exc).__name__}: {exc}")
        _dump_evidence(evidence_dir, case_id, rec, [in_a, in_b, out_bin])
        return rec

    try:
        verdict = criteria.check(got, want)
    except Exception as exc:
        rec.update(status="ERROR", detail=f"判据机判失败：{type(exc).__name__}: {exc}")
        _dump_evidence(evidence_dir, case_id, rec, [in_a, in_b, out_bin])
        return rec

    rec["status"] = "PASS" if verdict["is_pass"] else "FAIL"
    rec["max_abs_err"] = verdict["max_abs_error"]
    rec["matched_ratio"] = verdict["matched_ratio"]
    rec["tol"] = {"rtol": verdict["rtol"], "atol": verdict["atol"],
                  "required_matched_ratio": verdict["required_matched_ratio"],
                  "max_abs_error_limit": verdict["max_abs_error_limit"],
                  "profile": verdict["profile"]}
    rec["nonfinite"] = verdict["nonfinite"]
    keep: list[Path] = []
    if not verdict["is_pass"]:
        rec["detail"] = "；".join(verdict.get("failure_reasons") or ["未通过"])
        keep = [in_a, in_b, out_bin]  # 失败用例留原始输入与输出：证据要能复现，不只是结论
        # 失败用例给几条具体的坏点，方便定位
        bad = np.flatnonzero(~np.isclose(got.astype(np.float64), want.astype(np.float64),
                                         rtol=verdict["rtol"], atol=verdict["atol"]))[:5]
        rec["mismatch_index_sample"] = [int(i) for i in bad]
        rec["sample"] = [{"i": int(i), "got": float(got[i]), "want": float(want[i])} for i in bad]
    _dump_evidence(evidence_dir, case_id, rec, keep)
    return rec


def _dump_evidence(evidence_dir: Path, case_id: str, rec: dict, keep: list[Path]) -> None:
    """通过用例只留 meta；失败/异常用例把输入输出一起留下（最小可复现证据）。"""
    dst = evidence_dir / case_id
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in keep:
        if f.is_file():
            shutil.copy2(f, dst / f.name)


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _render_md(report: dict) -> str:
    lines = [
        f"# 正确性报告 · {report['op']}",
        "",
        f"- 用例集：`{report['case_source']}`（sha256 `{report['case_source_sha256'][:16]}…`）",
        f"- 用例集 ref：`{report['suite_ref']}`",
        f"- 规格版本：v{report['spec_version_id']} ｜ 判据档：`{report['criteria_ref']}`",
        f"- 代码快照：`{report['code_snapshot_sha256'][:16]}…` ｜ 候选实现 sha256：`{report['candidate']['sha256'][:16]}…`",
        f"- 汇总：{report['summary']}",
        "",
        "| 用例 | 组 | 必需 | shape | dtype | 期望 | 状态 | matched_ratio | max_abs_err | 说明 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["cases"]:
        lines.append("| {case_id} | {group} | {req} | {shape} | {dtype} | {expected} | **{status}** | {ratio} | {err} | {detail} |".format(
            case_id=r["case_id"], group=r.get("group", ""), req="是" if r.get("required", True) else "否",
            shape=r.get("shape"), dtype=r.get("dtype"), expected=r.get("expected"),
            status=r["status"], ratio=_fmt(r.get("matched_ratio")), err=_fmt(r.get("max_abs_err")),
            detail=(r.get("detail") or r.get("note") or "").replace("|", "/")[:120]))
    lines += ["", "> 本文件是给人看的伴生品；机判只认同目录的 `correctness.json`（门禁契约 docs/gate-report-v1.md）。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
