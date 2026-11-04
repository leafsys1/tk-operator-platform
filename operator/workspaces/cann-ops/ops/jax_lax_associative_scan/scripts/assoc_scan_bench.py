# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_associative_scan 的基准分支（被 `tools/run_benchmark.py` 按 op 分派复用）。

口径（设计 §2.2.3，不改测量口径）：

* 候选：``build/jax_lax_associative_scan --case-json ... --inputs ... --mode jit --outdir ...``；
  ``--mode jit`` 内部一次 ``associative_scan_compiled`` 并 ``block_until_ready()``，
  app 内不再循环；
* 测量：``msopprof`` 设备侧 ``Task Duration(us)``，``warm_up`` / ``invocations`` /
  ``aggregate`` 一律读 ``benchmark.json``（不得用 runner 旧默认 2/5 覆盖）；
* 基线：**不设 CPU / torch 基线**，``baseline: null``、``comparison: same_device_only``；
  CPU 延迟只作参考，不是性能门槛。``performance_target=UNKNOWN``，只报告不设阈值。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fixtures  # noqa: E402
import assoc_scan_engine as engine  # noqa: E402

ASSOC_OP = engine.ASSOC_OP
BASELINE_REASON = "CPU latency is not a gate; no same-device peer baseline is defined"


def _sha256_file(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(ws: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ws.resolve()))
    except (ValueError, OSError):
        return str(path)


def _render_md(report: dict) -> str:
    cond = report["conditions"]
    lines = [
        f"# 基准报告 · {report['op']}（assoc-scan 分支）",
        "",
        f"- 基线：{cond['baseline_impl']}（{report['comparison']}；{report['baseline_reason']}）",
        f"- 主值测量范围：{cond['measure_scope']}（msopprof 设备侧 Task Duration）",
        f"- 采集次数：{cond['invocations']}，msopprof warm-up：{cond['warm_up']}，聚合：{cond['aggregate']}",
        f"- 设备：{cond.get('device')}（锁 `{cond.get('lock_mode')}`）",
        f"- 规格版本：v{report['spec_version_id']} ｜ 判据档：`{report['criteria_ref']}`",
        f"- 用例集 ref：`{report['suite_ref']}` ｜ 代码快照：`{report['code_snapshot_sha256'][:16]}…`",
        "",
        "| 用例 | length | width | op_kind | axis | reverse | 候选设备中位数(us) | 样本数 | 主用例 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['length']} | {case['width']} | {case['op_kind']} | "
            f"{case['axis']} | {case['reverse']} | "
            f"{case['candidate']['device_us']['median_us']:.3f} | "
            f"{case['candidate']['device_us']['count']} | {'是' if case.get('primary') else '否'} |")
    lines += [
        "",
        "> 不设 CPU / torch 基线：assoc-scan 的语义参考是 CPU JAX，不是同设备 peer；",
        "> 性能只记同卡同镜像的候选稳态中位数，performance_target=UNKNOWN（不设通过阈值）。",
        "> 本文件是给人看的伴生品；机判只认同目录的 `performance.json`。",
    ]
    return "\n".join(lines) + "\n"


def run_benchmark(*, ws: Path, op: str, exe: Path, bench: dict, bench_path: Path,
                  out: Path, md: Path, workdir: Path, evidence_dir: Path,
                  measure: dict, invocations: int, retries: int, timeout: int,
                  spec_version_id: str, criteria_ref: str, code_snapshot_sha256: str,
                  suite_ref: str, run_id: str) -> int:
    if bench.get("op") != op:
        print(f"[run_benchmark] benchmark.op={bench.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
        return 2

    # `.so` preflight（设计 §2.2.3）：benchmark.sh 不在授权列表，由本分支承担。
    ok, detail = engine.ensure_host_so(ws, op, log_dir=out.parent / "evidence")
    if not ok:
        print(f"[run_benchmark] MISSING_SO {detail}（补编失败，不静默跳过）", file=sys.stderr)
        return 2

    tools_dir = ws / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import run_benchmark as rb  # noqa: PLC0415 —— 复用同一套 msopprof 采集/聚合，不另写一把尺子

    invocations = invocations or int(measure.get("invocations", 10))
    retries = retries or int(measure.get("retries_per_invocation", 3))
    warm_up = int(measure.get("warm_up", 3))
    aic_metrics = measure.get("aic_metrics", "Default")
    primary_case_id = bench.get("primary_case")
    prof_parse = rb._profile_module().parse_op_basic_info

    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for case in bench.get("cases") or []:
        cid = case["case_id"]
        fixture = fixtures.make_fixture(case)
        geometry = fixture["geometry"][0] if fixture["geometry"] else {}
        length = int(geometry.get("length", case.get("length")))
        width = int(case.get("width", case.get("shape", [0, 0])[-1]))
        dtype = case.get("dtype", fixture["dtype"])
        seed = int(case.get("seed", fixture["seed"]))
        cdir = workdir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        case_json = cdir / "case.json"
        inputs = cdir / "inputs.npz"
        fixtures.write_case_json(case, fixture, case_json)
        fixtures.write_inputs_npz(fixture, inputs)

        cand_out = cdir / "out"
        if cand_out.exists():
            shutil.rmtree(cand_out)
        cand_app = (f"{exe} --case-json {case_json} --inputs {inputs} "
                    f"--mode jit --outdir {cand_out}")
        cand_us, cand_info, cand_fail = rb.collect_device_samples(
            prof_parse, cand_app, cdir / "opprof_candidate",
            invocations=invocations, retries=retries, aic_metrics=aic_metrics,
            warm_up=warm_up, timeout=timeout,
        )
        if not cand_us:
            print(f"[run_benchmark] {cid} 候选采集全部失败：{cand_fail[-3:]}", file=sys.stderr)
            (evidence_dir / f"{cid}.json").write_text(json.dumps(
                {"case_id": cid, "length": length, "width": width, "dtype": dtype, "seed": seed,
                 "candidate_samples_device_us": [], "candidate_op": cand_info,
                 "collection_failures": cand_fail, "status": "ERROR"},
                ensure_ascii=False, indent=2), encoding="utf-8")
            return 4
        if cand_fail:
            print(f"[run_benchmark] {cid} 采集有重试：候选失败 {len(cand_fail)} 次"
                  f"（已用成功样本 {len(cand_us)} 个）", file=sys.stderr)

        candidate = {
            "impl": "ascendc-direct-invoke",
            "op_name": cand_info.get("op_name"),
            "op_type": cand_info.get("op_type"),
            "block_dim": cand_info.get("block_dim"),
            "device_us": rb.aggregate(cand_us),
        }
        results.append({
            "case_id": cid,
            "length": length,
            "width": width,
            "n": int(fixture["numel"]),
            "op_kind": fixture["op_kind"],
            "axis": fixture["axis"],
            "reverse": fixture["reverse"],
            "dtype": dtype,
            "seed": seed,
            "gen": f"fixture={fixture['kind']};length={length};width={width};seed={seed}",
            "required": bool(case.get("required", True)),
            "primary": cid == primary_case_id,
            "note": case.get("note"),
            "baseline": None,
            "comparison": "same_device_only",
            "baseline_reason": BASELINE_REASON,
            "candidate": candidate,
            "derived": {
                "speedup_baseline_over_candidate": None,
                "note": "no same-device peer baseline; CPU latency is not a gate",
            },
        })
        (evidence_dir / f"{cid}.json").write_text(json.dumps(
            {"case_id": cid, "length": length, "width": width, "op_kind": fixture["op_kind"],
             "axis": fixture["axis"], "reverse": fixture["reverse"], "dtype": dtype, "seed": seed,
             "inputs_sha256": fixture["inputs_sha256"],
             "candidate_samples_device_us": cand_us, "candidate_op": cand_info,
             "collection_failures": cand_fail, "status": "ok"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[run_benchmark] {cid}: median={candidate['device_us']['median_us']:.3f}us "
              f"samples={candidate['device_us']['count']}")

    if not results:
        print("[run_benchmark] benchmark.json 没有用例", file=sys.stderr)
        return 2
    primary = next((case for case in results if case.get("primary")), results[0])
    conditions = {
        "baseline_impl": "none (same-device only)",
        "baseline_app": None,
        "measure_scope": measure.get("scope", "blocking_jit_execution"),
        "measure_tool": "msopprof (msprof op) OpBasicInfo.csv Task Duration(us)",
        "measure_note": measure.get("note"),
        "invocations": invocations,
        "retries_per_invocation": retries,
        "warm_up": warm_up,
        "aic_metrics": aic_metrics,
        "aggregate": measure.get("aggregate", "median"),
        "synchronize": measure.get("synchronize", "block_until_ready"),
        "comparison": "same_device_only",
        "device": os.environ.get("OPX_DEVICES") or None,
        "lock_mode": os.environ.get("OPX_LOCK_MODE") or None,
        "candidate_exe": _rel(ws, exe),
        "candidate_exe_sha256": _sha256_file(exe),
        "bench_source": _rel(ws, bench_path),
    }
    report = {
        "schema": "opx-performance/1",
        "op": op,
        "action": "benchmark",
        "run_id": run_id or None,
        "spec_version_id": int(spec_version_id),
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": code_snapshot_sha256,
        "suite_ref": suite_ref,
        "baseline": None,
        "comparison": "same_device_only",
        "baseline_reason": BASELINE_REASON,
        "conditions": conditions,
        "cases": results,
        "metrics": [
            {"id": "op_time_us", "kind": "duration", "status": "measured", "required": True,
             "source": "msopprof OpBasicInfo.csv Task Duration(us), median over invocations",
             "value": primary["candidate"]["device_us"]["median_us"],
             "case_id": primary["case_id"],
             "note": "候选 associative_scan 整树（blocking jit）的设备侧 kernel 时长中位数；"
                     "performance_target=UNKNOWN，不设通过阈值"},
            {"id": "baseline_op_time_us", "kind": "duration", "status": "unsupported",
             "required": False,
             "source": "no same-device peer baseline",
             "assumption": BASELINE_REASON},
            {"id": "speedup", "kind": "derived", "status": "unsupported", "required": False,
             "source": "no same-device peer baseline",
             "assumption": BASELINE_REASON},
        ],
        "summary": {
            "case_count": len(results),
            "primary_case": primary["case_id"],
            "primary_median_us": primary["candidate"]["device_us"]["median_us"],
            "comparison": "same_device_only",
        },
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_render_md(report), encoding="utf-8")
    print(f"[run_benchmark] 主用例 {primary['case_id']}：候选 "
          f"{primary['candidate']['device_us']['median_us']:.3f}us → {out}")
    return 0
