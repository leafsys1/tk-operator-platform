# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_cond 的基准分支（被 ``tools/run_benchmark.py`` 按 op 分派复用）。

口径（设计 §2.2.3，不改测量口径）：

* 候选：``build/jax_lax_cond --case-json ... --inputs ... --mode jit --outdir ...``；
  ``--mode jit`` 内部一次 ``cond_compiled`` 并 ``block_until_ready()``；
* 每条基准用例分别测 ``pred=True`` 与 ``pred=False`` 两条序列（同一 plan key）；
* 测量：``msopprof`` 设备侧 ``Task Duration(us)``，``warm_up=3`` / ``invocations=10`` /
  ``aggregate=median`` 一律读 ``benchmark.json``；
* ``performance_target=null``：本轮只报告，不设通过阈值，不得声称性能达标。
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
import cond_engine as engine  # noqa: E402

COND_OP = engine.COND_OP
BASELINE_REASON = "performance_target=null (baseline only); no pass threshold"


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
    conditions = report["conditions"]
    lines = [
        f"# 基准报告 · {report['op']}（cond 分支）",
        "",
        f"- 基线：{report['baseline']}（{report['comparison']}；{report['baseline_reason']}）",
        f"- 主值测量范围：{conditions['measure_scope']}（msopprof 设备侧 Task Duration）",
        f"- 采集次数：{conditions['invocations']}，msopprof warm-up：{conditions['warm_up']}，"
        f"聚合：{conditions['aggregate']}，同步：{conditions['synchronize']}",
        f"- 设备：{conditions.get('device')}（锁 `{conditions.get('lock_mode')}`）",
        f"- 规格版本：v{report['spec_version_id']} ｜ 判据档：`{report['criteria_ref']}`",
        f"- 用例集 ref：`{report['suite_ref']}` ｜ 代码快照：`{report['code_snapshot_sha256'][:16]}…`",
        "",
        "| 用例 | operand | pred | 候选设备中位数(us) | 样本数 | 主用例 |",
        "|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        for series in case["series"]:
            lines.append(
                f"| {case['case_id']} | {case['operand']} | {series['pred']} | "
                f"{series['device_us']['median_us']:.3f} | "
                f"{series['device_us']['count']} | {'是' if case.get('primary') else '否'} |")
    lines += [
        "",
        "> 本轮只记基线：`performance_target=null`，不设通过阈值，不得声称性能达标。",
        "> 同一用例的 pred=True/False 共享同一 plan key（pred 是运行期参数）；",
        "> 主值 = msopprof 设备侧 Task Duration(us)，warm_up=3 / invocations=10 / median。",
        "> 本文件是给人看的伴生品；机判只认同目录的 `performance.json`。",
    ]
    return "\n".join(lines) + "\n"


def run_benchmark(*, ws: Path, op: str, exe: Path, bench: dict, bench_path: Path,
                  out: Path, md: Path, workdir: Path, evidence_dir: Path,
                  measure: dict, invocations: int, retries: int, timeout: int,
                  spec_version_id: str, criteria_ref: str, code_snapshot_sha256: str,
                  suite_ref: str, run_id: str) -> int:
    if bench.get("op") != op:
        print(f"[run_benchmark] benchmark.op={bench.get('op')!r} 与 --op={op!r} 不一致",
              file=sys.stderr)
        return 2

    # `.so` preflight（设计 §2.2.3）：benchmark.sh 不在授权列表，由本分支承担。
    ok, detail = engine.ensure_host_so(ws, op, log_dir=out.parent / "evidence")
    if not ok:
        print(f"[run_benchmark] MISSING_SO {detail}（补编失败，不静默跳过）", file=sys.stderr)
        return 2

    tools_dir = ws / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import run_benchmark as rb  # noqa: PLC0415 —— 复用同一套 msopprof 采集/聚合

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
        cid = str(case["case_id"])
        series_results = []
        for pred_bool in (True, False):
            series_name = "true" if pred_bool else "false"
            series_case = dict(case)
            series_case["pred"] = pred_bool
            series_case["mode"] = ["jit"]
            fixture = fixtures.make_fixture(series_case, "", "jit")
            cdir = workdir / cid / series_name
            cdir.mkdir(parents=True, exist_ok=True)
            case_json = cdir / "case.json"
            inputs = cdir / "inputs.npz"
            fixtures.write_case_json(series_case, fixture, case_json)
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
            series = {
                "pred": series_name,
                "n": int(fixture["numel"]),
                "inputs_sha256": fixture["inputs_sha256"],
                "candidate_samples_device_us": cand_us,
                "candidate_op": cand_info,
                "collection_failures": cand_fail,
                "device_us": rb.aggregate(cand_us) if cand_us else None,
                "status": "ok" if cand_us else "ERROR",
            }
            series_results.append(series)
            (evidence_dir / f"{cid}-{series_name}.json").write_text(
                json.dumps({"case_id": cid, **series}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            if not cand_us:
                print(f"[run_benchmark] {cid}/{series_name} 候选采集全部失败："
                      f"{cand_fail[-3:]}", file=sys.stderr)
                return 4
            print(f"[run_benchmark] {cid}/{series_name}: "
                  f"median={series['device_us']['median_us']:.3f}us "
                  f"samples={series['device_us']['count']}")
        if not series_results:
            print(f"[run_benchmark] {cid} 没有可测量序列", file=sys.stderr)
            return 2
        results.append({
            "case_id": cid,
            "operand": case.get("operand"),
            "branches": list(case.get("branches") or []),
            "dtype": case.get("dtype", "float32"),
            "seed": case.get("seed"),
            "required": bool(case.get("required", True)),
            "primary": cid == primary_case_id,
            "note": case.get("note"),
            "series": series_results,
        })

    if not results:
        print("[run_benchmark] benchmark.json 没有用例", file=sys.stderr)
        return 2
    primary = next((case for case in results if case.get("primary")), results[0])
    primary_true = next((series for series in primary["series"] if series["pred"] == "true"),
                        primary["series"][0])
    conditions = {
        "baseline_impl": "none (no per-op peer baseline; CPU latency is not a gate)",
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
        "series_per_case": 2,
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
        "performance_target": None,
        "conditions": conditions,
        "cases": results,
        "metrics": [
            {"id": "op_time_us", "kind": "duration", "status": "measured",
             "required": True,
             "source": "msopprof OpBasicInfo.csv Task Duration(us), median over invocations",
             "value": primary_true["device_us"]["median_us"],
             "case_id": primary["case_id"], "pred": primary_true["pred"],
             "note": "cond_compiled(jit) 整次调用的设备侧 kernel 时长中位数；"
                     "performance_target=null，不设通过阈值"},
            {"id": "baseline_op_time_us", "kind": "duration", "status": "unsupported",
             "required": False, "source": "no same-device peer baseline",
             "assumption": BASELINE_REASON},
            {"id": "speedup", "kind": "derived", "status": "unsupported",
             "required": False, "source": "no same-device peer baseline",
             "assumption": BASELINE_REASON},
        ],
        "summary": {
            "case_count": len(results),
            "series_count": sum(len(case["series"]) for case in results),
            "primary_case": primary["case_id"],
            "primary_pred": primary_true["pred"],
            "primary_median_us": primary_true["device_us"]["median_us"],
            "comparison": "same_device_only",
            "performance_target": None,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_render_md(report), encoding="utf-8")
    print(f"[run_benchmark] 主用例 {primary['case_id']}/{primary_true['pred']}：候选 "
          f"{primary_true['device_us']['median_us']:.3f}us → {out}")
    return 0
