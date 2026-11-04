#!/usr/bin/env python3
"""性能采集执行器 —— 四动作 `profile` 的核心（在**容器里**跑）。

工具：`msopprof`（= `msprof op`，MindStudio Profiler For Operator，CANN 9.1 自带）。
D9 摸底结论（2026-10-16 真机实测，镜像 `lfx-op-dev-910b:torch210-cann910`）：

| 需求要求的指标类 | msopprof 来源 | 状态 |
|---|---|---|
| 算子执行耗时 | `OpBasicInfo.csv` → `Task Duration(us)` | measured |
| 数据搬运 | `Memory.csv`（GM→UB / UB→GM / 主存读写带宽与字节数）+ `PipeUtilization.csv`（mte2/mte3 占比） | measured |
| 存储使用 | `MemoryUB.csv`（UB 读/写带宽）+ `Memory.csv` 的 `GM_to_UB_datas(KB)` | measured |
| 计算单元负载 | `ArithmeticUtilization.csv`（vec ratio / fp32 ratio / fops）+ `PipeUtilization.csv` | measured |

即四类指标在**本镜像上都能采到**（不是 unsupported）。cube 相关列对 vector 算子为 NA，
按 `unsupported` 如实标注（不是「采不到」而是「该算子没有这类单元」）。

产物 `/out/metrics.json`（机读，门禁按类型 `metrics` 找）：四类指标各带 `status` 与 `source`；
原始 CSV 落到 `/out/evidence/profile/`（`dump/*.bin` 是中间数据，不回读 —— 占地且不可读）。

版本对齐字段由 `nm_test.sh` 注入环境变量（与 test/benchmark 同一口径）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "tools"))


def _gen_inputs():
    """延迟导入 `gen_inputs`：它依赖 numpy；本模块的解析/报告函数要能在只装控制台依赖的环境里被单测。"""
    import gen_inputs  # noqa: PLC0415

    return gen_inputs

#: msopprof 的采集能力组（`--aic-metrics`）。默认给全：四类指标恰好分落在这几组里。
DEFAULT_METRICS = "Default"
#: 回读进证据目录的文件（不含 `dump/*.bin`）
EVIDENCE_FILES = ("OpBasicInfo.csv", "ArithmeticUtilization.csv", "Memory.csv", "MemoryUB.csv",
                  "MemoryL0.csv", "PipeUtilization.csv", "L2Cache.csv", "ResourceConflictRatio.csv",
                  "dump/op_basic_info.txt")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_or_arg(name: str, arg):
    return arg if arg not in (None, "") else os.environ.get(name, "")


# ---------------------------------------------------------------- CSV 解析（可单测）


def read_csv_rows(path: Path) -> list[dict]:
    """读 msopprof 的 CSV（表头带单位，如 `aiv_time(us)`）。"""
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def col_mean(rows: list[dict], column: str) -> float | None:
    """对某一列取均值，跳过 NA/空/非数值（msopprof 会把不适用单元填 NA）。"""
    vals = []
    for r in rows:
        v = (r.get(column) or "").strip()
        if v in ("", "NA", "N/A"):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return sum(vals) / len(vals) if vals else None


def col_max(rows: list[dict], column: str) -> float | None:
    vals = []
    for r in rows:
        v = (r.get(column) or "").strip()
        if v in ("", "NA", "N/A"):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return max(vals) if vals else None


def parse_op_basic_info(path: Path) -> dict:
    """`OpBasicInfo.csv` 只有一行（算子级），给出 Task Duration / Block Dim / Device。"""
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"{path.name} 为空")
    r = rows[0]

    def f(key):
        v = (r.get(key) or "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    return {
        "op_name": (r.get("Op Name") or "").strip(),
        "op_type": (r.get("Op Type") or "").strip(),
        "task_duration_us": f("Task Duration(us)"),
        "block_dim": int(f("Block Dim")) if f("Block Dim") is not None else None,
        "device_id": int(f("Device Id")) if f("Device Id") is not None else None,
        "pid": int(f("Pid")) if f("Pid") is not None else None,
        "current_freq_mhz": f("Current Freq"),
        "rated_freq_mhz": f("Rated Freq"),
    }


def build_metrics(op_basic: dict, per_block: dict) -> list[dict]:
    """把「算子级 + 逐核聚合」的数据整理成**四类指标**（状态 + 来源齐备）。"""
    dur = op_basic.get("task_duration_us")
    metrics = [
        {"id": "op_time_us", "kind": "duration", "status": "measured" if dur is not None else "failed",
         "required": True, "source": "msopprof OpBasicInfo.csv Task Duration(us)", "value": dur,
         "note": "算子级（非逐核聚合），单次 kernel 的任务时长"},
        {"id": "gm_to_ub_bw_gbps", "kind": "data_movement",
         "status": "measured" if per_block.get("gm_to_ub_bw") is not None else "unsupported",
         "required": False, "source": "msopprof Memory.csv aiv_gm_to_ub_bw(GB/s) 逐核均值",
         "value": per_block.get("gm_to_ub_bw"), "unit": "GB/s"},
        {"id": "main_mem_read_bw_gbps", "kind": "data_movement",
         "status": "measured" if per_block.get("main_mem_read_bw") is not None else "unsupported",
         "required": False, "source": "msopprof Memory.csv aiv_main_mem_read_bw(GB/s) 逐核均值",
         "value": per_block.get("main_mem_read_bw"), "unit": "GB/s"},
        {"id": "main_mem_write_bw_gbps", "kind": "data_movement",
         "status": "measured" if per_block.get("main_mem_write_bw") is not None else "unsupported",
         "required": False, "source": "msopprof Memory.csv aiv_main_mem_write_bw(GB/s) 逐核均值",
         "value": per_block.get("main_mem_write_bw"), "unit": "GB/s"},
        {"id": "mte2_ratio", "kind": "data_movement",
         "status": "measured" if per_block.get("mte2_ratio") is not None else "unsupported",
         "required": False, "source": "msopprof PipeUtilization.csv aiv_mte2_ratio 逐核均值",
         "value": per_block.get("mte2_ratio"), "note": "GM→UB 搬运流水占比（越高越像搬运受限）"},
        {"id": "mte3_ratio", "kind": "data_movement",
         "status": "measured" if per_block.get("mte3_ratio") is not None else "unsupported",
         "required": False, "source": "msopprof PipeUtilization.csv aiv_mte3_ratio 逐核均值",
         "value": per_block.get("mte3_ratio"), "note": "UB→GM 搬运流水占比"},
        {"id": "ub_read_bw_gbps", "kind": "storage",
         "status": "measured" if per_block.get("ub_read_bw") is not None else "unsupported",
         "required": False, "source": "msopprof MemoryUB.csv aiv_ub_read_bw_vector(GB/s) 逐核均值",
         "value": per_block.get("ub_read_bw"), "unit": "GB/s", "note": "UB 存储访问强度"},
        {"id": "ub_write_bw_gbps", "kind": "storage",
         "status": "measured" if per_block.get("ub_write_bw") is not None else "unsupported",
         "required": False, "source": "msopprof MemoryUB.csv aiv_ub_write_bw_vector(GB/s) 逐核均值",
         "value": per_block.get("ub_write_bw"), "unit": "GB/s"},
        {"id": "gm_to_ub_datas_kb", "kind": "storage",
         "status": "measured" if per_block.get("gm_to_ub_datas_kb") is not None else "unsupported",
         "required": False, "source": "msopprof Memory.csv GM_to_UB_datas(KB) 逐核均值",
         "value": per_block.get("gm_to_ub_datas_kb"), "unit": "KB", "note": "实际搬运字节量"},
        {"id": "vec_ratio", "kind": "compute",
         "status": "measured" if per_block.get("vec_ratio") is not None else "unsupported",
         "required": False, "source": "msopprof PipeUtilization.csv aiv_vec_ratio 逐核均值",
         "value": per_block.get("vec_ratio"), "note": "Vector 计算单元负载（近似占用率）"},
        {"id": "vec_fp32_fops", "kind": "compute",
         "status": "measured" if per_block.get("vec_fp32_fops") is not None else "unsupported",
         "required": False, "source": "msopprof ArithmeticUtilization.csv aiv_vec_fops 逐核均值",
         "value": per_block.get("vec_fp32_fops")},
        {"id": "l2_total_hit_rate_pct", "kind": "storage",
         "status": "measured" if per_block.get("l2_hit_rate") is not None else "unsupported",
         "required": False, "source": "msopprof L2Cache.csv aiv_total_hit_rate(%) 逐核均值",
         "value": per_block.get("l2_hit_rate"), "unit": "%"},
        # vector 算子上没有 cube（矩阵）单元 —— 如实标 unsupported，不伪造数值
        {"id": "cube_ratio", "kind": "compute", "status": "unsupported", "required": False,
         "source": "msopprof ArithmeticUtilization.csv aic_cube_ratio", "value": None,
         "assumption": "本算子为 vector 类型，无 cube 单元数据（msopprof 填 NA）"},
    ]
    return metrics


def parse_per_block(prof_dir: Path) -> dict:
    """把逐核 CSV 聚合成一组数（均值；NA 跳过）。"""
    out: dict = {}

    def rows(name):
        p = prof_dir / name
        return read_csv_rows(p) if p.is_file() else []

    mem = rows("Memory.csv")
    ub = rows("MemoryUB.csv")
    pipe = rows("PipeUtilization.csv")
    arith = rows("ArithmeticUtilization.csv")
    l2 = rows("L2Cache.csv")

    out["gm_to_ub_bw"] = col_mean(mem, "aiv_gm_to_ub_bw(GB/s)")
    out["ub_to_gm_bw"] = col_mean(mem, "aiv_ub_to_gm_bw(GB/s)")
    out["main_mem_read_bw"] = col_mean(mem, "aiv_main_mem_read_bw(GB/s)")
    out["main_mem_write_bw"] = col_mean(mem, "aiv_main_mem_write_bw(GB/s)")
    out["gm_to_ub_datas_kb"] = col_mean(mem, "GM_to_UB_datas(KB)")
    out["ub_read_bw"] = col_mean(ub, "aiv_ub_read_bw_vector(GB/s)")
    out["ub_write_bw"] = col_mean(ub, "aiv_ub_write_bw_vector(GB/s)")
    out["mte2_ratio"] = col_mean(pipe, "aiv_mte2_ratio")
    out["mte3_ratio"] = col_mean(pipe, "aiv_mte3_ratio")
    out["vec_ratio"] = col_mean(pipe, "aiv_vec_ratio")
    out["vec_fp32_ratio"] = col_mean(arith, "aiv_vec_fp32_ratio")
    out["vec_fp32_fops"] = col_mean(arith, "aiv_vec_fops")
    out["l2_hit_rate"] = col_mean(l2, "aiv_total_hit_rate(%)")
    out["blocks_sampled"] = len(pipe) or len(mem) or len(ub)
    return out


def find_prof_dir(root: Path) -> Path:
    cands = sorted(p for p in root.glob("OPPROF_*") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"{root} 下没有 OPPROF_* 采集目录")
    return cands[-1]


# ---------------------------------------------------------------- 报告


def build_metrics_report(*, op: str, run_id: str, spec_version_id: int, criteria_ref: str,
                         code_snapshot_sha256: str, suite_ref: str, tool: dict, case: dict,
                         op_basic: dict, per_block: dict, metrics: list[dict]) -> dict:
    return {
        "schema": "opx-metrics/1",
        "op": op,
        "action": "profile",
        "run_id": run_id or None,
        # ---- 版本对齐（门禁逐字比对这一版 run）----
        "spec_version_id": int(spec_version_id),
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": code_snapshot_sha256,
        "suite_ref": suite_ref,
        "tool": tool,
        "case": case,
        "op_info": op_basic,
        "per_block_aggregate": per_block,
        "metrics": metrics,
    }


def _render_md(report: dict) -> str:
    t = report["tool"]
    lines = [
        f"# 性能采集报告 · {report['op']}",
        "",
        f"- 工具：{t['name']}（能力组 `{t['aic_metrics']}`，warm-up {t['warm_up']}，launch-count {t['launch_count']}）",
        f"- 算子：`{report['op_info'].get('op_name')}`（{report['op_info'].get('op_type')}）"
        f"｜Task Duration {report['op_info'].get('task_duration_us')} us ｜Block Dim {report['op_info'].get('block_dim')}",
        f"- 用例：{report['case'].get('case_id')}（N={report['case'].get('n')}，{report['case'].get('dtype')}）",
        f"- 规格版本：v{report['spec_version_id']} ｜ 判据档：`{report['criteria_ref']}` ｜ 用例集 ref：`{report['suite_ref']}`",
        "",
        "| 指标 | 类别 | 状态 | 值 | 单位 | 来源 |",
        "|---|---|---|---|---|---|",
    ]
    for m in report["metrics"]:
        val = m.get("value")
        val = f"{val:.6g}" if isinstance(val, float) else ("—" if val is None else val)
        lines.append(f"| {m['id']} | {m.get('kind','')} | {m['status']} | {val} | {m.get('unit','')} | {m.get('source','')} |")
    lines += ["", "> 本文件是给人看的伴生品；机判只认同目录的 `metrics.json`（门禁契约 docs/gate-report-v1.md）。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="elementwise_add")
    ap.add_argument("--workspace", default=str(WS))
    ap.add_argument("--exe", default="")
    ap.add_argument("--bench", default="", help="基准用例定义（复用其 primary case 的形状/dtype）")
    ap.add_argument("--out", default="/out/metrics.json")
    ap.add_argument("--md", default="")
    ap.add_argument("--workdir", default="/tmp/opx-profile")
    ap.add_argument("--evidence-dir", default="")
    ap.add_argument("--aic-metrics", default=DEFAULT_METRICS)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--launch-count", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--spec-version-id", default="")
    ap.add_argument("--criteria-ref", default="")
    ap.add_argument("--code-snapshot-sha256", default="")
    ap.add_argument("--suite-ref", default="")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    op = args.op
    bench_path = Path(args.bench) if args.bench else ws / "cases" / op / "benchmark.json"
    exe = Path(args.exe) if args.exe else ws / "ops" / op / "build" / op
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
        print(f"[run_profile] 版本对齐字段缺失：{missing}（应由 nm_test.sh 注入环境变量）", file=sys.stderr)
        return 2
    if not bench_path.is_file():
        print(f"[run_profile] 基准用例定义不存在：{bench_path}", file=sys.stderr)
        return 2
    if not exe.is_file():
        print(f"[run_profile] 候选实现不存在：{exe}（先跑 build 动作）", file=sys.stderr)
        return 2
    if not shutil.which("msopprof"):
        print("[run_profile] 找不到 msopprof（应为 CANN 自带：<CANN>/bin/msopprof）", file=sys.stderr)
        return 2

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    primary_id = bench.get("primary_case")
    case = next((c for c in bench.get("cases") or [] if c["case_id"] == primary_id),
                (bench.get("cases") or [None])[0])
    if not case:
        print("[run_profile] 基准用例定义里没有可采的用例", file=sys.stderr)
        return 2
    n = int(case["n"])
    dtype = case.get("dtype", bench.get("dtype", "float32"))
    seed = int(case.get("seed", 0))

    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = workdir / "opprof"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    in_a, in_b, out_bin = workdir / "a.bin", workdir / "b.bin", workdir / "c.bin"
    gen_inputs = _gen_inputs()
    gen_inputs.generate(case.get("gen_a", {}), n, seed, dtype).tofile(in_a)
    gen_inputs.generate(case.get("gen_b", {}), n, seed + 1, dtype).tofile(in_b)

    app = (f"{exe} --n-a {n} --n-b {n} --dtype {dtype} "
           f"--in-a {in_a} --in-b {in_b} --out {out_bin}")
    cmd = ["msopprof", f"--application={app}", f"--output={raw_dir}",
           f"--aic-metrics={args.aic_metrics}", f"--warm-up={args.warm_up}",
           f"--launch-count={args.launch_count}"]
    print(f"[run_profile] 采集：msopprof --application=\"{app}\" --output={raw_dir} "
          f"--aic-metrics={args.aic_metrics} --warm-up={args.warm_up} --launch-count={args.launch_count}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
    if proc.returncode != 0:
        print(f"[run_profile] msopprof 非 0 退出 rc={proc.returncode}\n{tail}", file=sys.stderr)
        return 3

    try:
        prof_dir = find_prof_dir(raw_dir)
        op_basic = parse_op_basic_info(prof_dir / "OpBasicInfo.csv")
        per_block = parse_per_block(prof_dir)
    except Exception as exc:
        print(f"[run_profile] 采集结果解析失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    if op_basic.get("task_duration_us") is None:
        print(f"[run_profile] OpBasicInfo 里没有 Task Duration → 耗时指标采不到\n{tail}", file=sys.stderr)
        return 4

    metrics = build_metrics(op_basic, per_block)

    # 证据：只回读 CSV + basic info（`dump/*.bin` 是中间数据，占地且不可读）
    ev = evidence_dir / "profile"
    ev.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in EVIDENCE_FILES:
        src = prof_dir / name
        if src.is_file():
            dst = ev / Path(name).name
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(evidence_dir)))

    tool = {"name": "msopprof", "aic_metrics": args.aic_metrics, "warm_up": args.warm_up,
            "launch_count": args.launch_count, "raw_dir": str(raw_dir),
            "raw_tail": tail.strip()[-1500:]}
    case_info = {"case_id": case["case_id"], "n": n, "dtype": dtype, "seed": seed,
                 "gen": f"a={_gen_inputs().describe(case.get('gen_a', {}))};b={_gen_inputs().describe(case.get('gen_b', {}))}"}
    report = build_metrics_report(op=op, run_id=run_id, spec_version_id=int(spec_version_id),
                                  criteria_ref=criteria_ref, code_snapshot_sha256=snapshot,
                                  suite_ref=suite_ref, tool=tool, case=case_info,
                                  op_basic=op_basic, per_block=per_block, metrics=metrics)
    report["evidence_files"] = copied
    report["conditions"] = {"device": os.environ.get("OPX_DEVICES") or None,
                            "lock_mode": os.environ.get("OPX_LOCK_MODE") or None,
                            "candidate_exe": str(exe.relative_to(ws)) if str(exe).startswith(str(ws)) else str(exe),
                            "candidate_exe_sha256": _sha256_file(exe)}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = Path(args.md) if args.md else out.with_suffix(".md")
    md.write_text(_render_md(report), encoding="utf-8")

    print(f"[run_profile] op_time={op_basic['task_duration_us']}us vec_ratio={per_block.get('vec_ratio')} "
          f"mte2_ratio={per_block.get('mte2_ratio')} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
