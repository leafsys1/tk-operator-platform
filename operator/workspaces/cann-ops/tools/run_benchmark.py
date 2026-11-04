#!/usr/bin/env python3
"""基准执行器 —— 四动作 `benchmark` 的核心（在**容器里**跑）。

对照口径（方案 v1.1 §4.6，D9 升级为自动优化的硬前置）：

- **基线 = 官方对应实现** `torch.add`（同镜像里的 torch_npu，device 0），不是「上一次的自己」；
- **同卡同条件**：同一用例/同一 dtype/同一输入生成方式与种子/同一测量范围/同一聚合口径；
  设备由 `nm_test.sh` 的远端 flock 独占（邻居会污染耗时，§4.6）。
- **测量范围一致（关键）**：两侧主值都取 **msopprof 的设备侧 `Task Duration(us)`**。
  为什么不用墙钟：候选是 C++ 直调、基线是 PyTorch，两者的 host 端派发/启动开销差一个量级
  （实测 torch.add 的 host 端到端约 80us，而设备侧只有约 8us）—— 拿 host 墙钟算「加速比」
  只是把框架开销差异当成算子差异。msopprof 量的是同一个东西（AICore 任务时长），两侧同源可比。
  实现上：两侧都交给 msopprof 采集（候选跑可执行文件那一次 kernel，基线跑
  `benchmark.json` 的 `baseline.app`，缺省 `tools/torch_add_baseline.py`），每次采集读一次
  `OpBasicInfo.csv`；重复 `invocations` 次取中位数。

产物 `/out/performance.json`（机读，门禁按类型 `performance` 找）：固定条件 + 两侧样本 + 聚合 +
派生量（加速比 / 达成带宽）。原始逐次样本与每侧 OPPROF 目录摘要进 `/out/evidence/`。

版本对齐字段由 `nm_test.sh` 注入环境变量（与 test 同一口径，见 `tools/run_cases.py`）：
容器里**不重算**源码哈希 —— 门禁逐字比对报告与 run 行，重算就多一个漂移面。
"""

from __future__ import annotations

import argparse
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_or_arg(name: str, arg):
    return arg if arg not in (None, "") else os.environ.get(name, "")


# ---------------------------------------------------------------- 聚合（可单测，不依赖 torch）


def aggregate(samples_us: list[float]) -> dict:
    """把逐次样本聚合成一组描述量。**用中位数做主值**：共享机上偶发被邻居打扰时，均值会被拖偏。"""
    xs = sorted(float(x) for x in samples_us)
    n = len(xs)
    if n == 0:
        raise ValueError("样本为空，无法聚合")
    mid = n // 2
    median = xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {
        "count": n,
        "min_us": xs[0],
        "median_us": median,
        "mean_us": mean,
        "p90_us": xs[min(n - 1, int(round(0.9 * (n - 1))))],
        "max_us": xs[-1],
        "std_us": var ** 0.5,
    }


def dtype_bytes(dtype: str | None) -> int:
    """dtype → 每元素字节数（带宽/AI 推导用；未知按 4 字节）。"""
    d = str(dtype or "").lower()
    if "64" in d:
        return 8
    if "16" in d or d in ("bf16", "fp16", "half"):
        return 2
    if d in ("int8", "uint8", "bool"):
        return 1
    return 4


def derive(baseline_median_us: float, candidate_median_us: float, n: int,
           bytes_per_elem: int = 4, reads_per_elem: int = 2, writes_per_elem: int = 1,
           flops_per_elem: float = 1.0, moved_bytes: int | None = None,
           flops: float | None = None, include_intensity: bool = True) -> dict:
    """派生量：加速比 + 达成带宽 + 算术强度（flop/byte）。

    以前这里把「2 读 1 写、1 flop/元素」写死，只对 `elementwise_add` 成立。
    现在由 `benchmark.json` 的 `memory` / `compute`（或用例级覆盖）给出；
    默认值保持向后兼容（老用例仍是 2 读 1 写、1 flop/元素）。
    矩阵/注意力类若给了显式 `traffic_bytes` / `flops`，优先用显式值。
    """
    if moved_bytes is None:
        read_bytes = reads_per_elem * n * bytes_per_elem
        write_bytes = writes_per_elem * n * bytes_per_elem
        moved = read_bytes + write_bytes
    else:
        moved = int(moved_bytes)
        read_bytes = None
        write_bytes = None
    if flops is None:
        flops = flops_per_elem * n

    def bw(us: float) -> float:
        return (moved / (us * 1e-6)) / 1e9 if us > 0 else float("nan")

    return {
        "moved_bytes": moved,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "reads_per_elem": reads_per_elem,
        "writes_per_elem": writes_per_elem,
        "flops_per_elem": flops_per_elem,
        "flops": flops,
        # 算术强度：判「带宽受限 / 算力受限」用；moved=0 或画像非显式时给 None
        # （非显式 = benchmark.json 没写 memory/compute，默认值只保证带宽口径向后兼容，
        #   不能拿来给 relu / softmax 之类下 AI 结论；前端会退回算子画像）
        "arithmetic_intensity": (float(flops) / moved) if (moved and include_intensity) else None,
        "speedup_baseline_over_candidate": (baseline_median_us / candidate_median_us)
        if candidate_median_us > 0 else None,
        "baseline_achieved_bw_gbps": bw(baseline_median_us),
        "candidate_achieved_bw_gbps": bw(candidate_median_us),
    }


def case_profile(bench: dict, case: dict) -> dict:
    """从 benchmark.json 的基准级默认 + 用例级覆盖，取访存/计算画像。"""
    mem = bench.get("memory") or {}
    comp = bench.get("compute") or {}
    return {
        "reads": int(case.get("reads_per_elem", mem.get("reads_per_elem", 2))),
        "writes": int(case.get("writes_per_elem", mem.get("writes_per_elem", 1))),
        "flops_per_elem": float(case.get("flops_per_elem", comp.get("flops_per_elem", 1.0))),
        # 显式给整例流量/浮点总数（矩阵类 AI 随形状变，按元素表达不了）
        "moved_bytes": case.get("traffic_bytes"),
        "flops": case.get("flops"),
    }


def profile_is_explicit(bench: dict, case: dict) -> bool:
    """benchmark.json 是否**显式**声明了访存/计算画像（决定报告里的 AI 敢不敢给）。"""
    keys = ("reads_per_elem", "writes_per_elem", "flops_per_elem", "traffic_bytes", "flops")
    return bool(bench.get("memory") or bench.get("compute") or any(k in case for k in keys))


# ---------------------------------------------------------------- msopprof（两侧同一把尺子）


def _profile_module():
    """复用 `run_profile.py` 的解析（同一个 OPPROF 目录结构，一份代码）。"""
    import run_profile  # noqa: PLC0415

    return run_profile


def run_msopprof(application: str, out_dir: Path, *, aic_metrics: str = "Default",
                 warm_up: int = 2, launch_count: int = 1, timeout: int = 600) -> Path:
    """对一次应用做 msopprof 采集，返回 OPPROF_* 目录（其中应含 `OpBasicInfo.csv`）。"""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["msopprof", f"--application={application}", f"--output={out_dir}",
           f"--aic-metrics={aic_metrics}", f"--warm-up={warm_up}", f"--launch-count={launch_count}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"msopprof 非 0 退出 rc={proc.returncode}；"
                           f"tail={(proc.stdout or '')[-400:]}{(proc.stderr or '')[-400:]}")
    d = _profile_module().find_prof_dir(out_dir)
    if not (d / "OpBasicInfo.csv").is_file():
        raise RuntimeError(f"msopprof 没产出 OpBasicInfo.csv：{d}（可能这次没匹配到 kernel）；"
                           f"tail={(proc.stdout or '')[-400:]}")
    return d


def collect_device_samples(prof_parse, application: str, base_dir: Path, *,
                           invocations: int, retries: int = 3, aic_metrics: str = "Default",
                           warm_up: int = 2, timeout: int = 600,
                           run_fn=run_msopprof) -> tuple[list[float], dict, list[str]]:
    """采一侧的设备侧 Task Duration 样本，**对偶发失败做重试**。

    实测（2026-10-16）msopprof 采 torch 基线偶发「起得来但没匹配到 kernel」（输出目录在、
    却缺 `OpBasicInfo.csv`）。这里每个样本最多试 `retries` 次，成功的留、失败的记下来；
    只要还有 ≥1 个成功样本就继续（返回样本 + 最后一条 op 信息 + 失败记录）。
    """
    samples: list[float] = []
    info: dict = {}
    failures: list[str] = []
    for i in range(invocations):
        for attempt in range(max(1, retries)):
            try:
                d = run_fn(application, base_dir / f"i{i}_a{attempt}",
                           aic_metrics=aic_metrics, warm_up=warm_up, timeout=timeout)
                cur = prof_parse(d / "OpBasicInfo.csv")
                if cur.get("task_duration_us") is None:
                    raise RuntimeError("OpBasicInfo 没有 Task Duration")
                info = cur
                samples.append(float(cur["task_duration_us"]))
                break
            except Exception as exc:  # noqa: BLE001 —— 采集失败不该让整轮 abort
                failures.append(f"i={i} a={attempt}: {type(exc).__name__}: {exc}")
    return samples, info, failures



# ---------------------------------------------------------------- 报告


def _baseline_source_label(baseline: dict, conditions: dict) -> str:
    """性能报告里「基线设备耗时」的来源标签，按实际基线身份生成。

    - 优先实际 msopprof 报的 op_name（如 torch.sub 的 ``Sub``、torch.add 的 ``Add``）；
    - 其次 conditions.baseline_impl（如 ``torch.sub (torch_npu, device 0, alpha=1)``）；
    - 字段缺失退中性 ``baseline``。

    之前这里写死 ``torch.add``，会把 torch.sub 基线的报告在 source 里标错成 torch.add。
    只影响这一处说明文字，不改变采样 / 判据 / 耗时数值 / 用例 / baseline.app。
    """
    op_name = (baseline.get("op_name") or "").strip()
    impl = (baseline.get("impl") or conditions.get("baseline_impl") or "").strip()
    if op_name:
        return f"{op_name} ({impl})" if impl and impl != "基线" else op_name
    if impl and impl != "基线":
        return impl
    return "baseline"


def build_report(*, op: str, run_id: str, spec_version_id: int, criteria_ref: str,
                 code_snapshot_sha256: str, suite_ref: str, conditions: dict,
                 cases: list[dict]) -> dict:
    """组装 performance.json。`cases` 里每项含 case_id/n/dtype/baseline/candidate/derived。"""
    primary = next((c for c in cases if c.get("primary")), cases[0])
    b = primary["baseline"]
    c = primary["candidate"]
    metrics = [
        {"id": "op_time_us", "kind": "duration", "status": "measured", "required": True,
         "source": "msopprof OpBasicInfo.csv Task Duration(us), median over invocations",
         "value": c["device_us"]["median_us"], "case_id": primary["case_id"],
         "baseline_value": b["device_us"]["median_us"],
         "note": "候选 Ascend C 直调算子的设备侧 kernel 时长（中位数）"},
        {"id": "baseline_op_time_us", "kind": "duration", "status": "measured", "required": True,
         "source": f"msopprof Task Duration(us) of {_baseline_source_label(b, conditions)} kernel, "
                   "median over invocations",
         "value": b["device_us"]["median_us"], "case_id": primary["case_id"],
         "op_name": b.get("op_name")},
        {"id": "speedup", "kind": "derived", "status": "measured", "required": False,
         "source": "baseline_median / candidate_median (device side)",
         "value": primary["derived"]["speedup_baseline_over_candidate"], "case_id": primary["case_id"],
         "note": "设备侧对比；不设加速比硬线（需求调整已定：完成闭环 + 对比报告即可）"},
        {"id": "candidate_achieved_bw_gbps", "kind": "data_movement", "status": "measured",
         "required": False, "source": "moved_bytes / candidate_median",
         "value": primary["derived"]["candidate_achieved_bw_gbps"], "case_id": primary["case_id"]},
        {"id": "baseline_achieved_bw_gbps", "kind": "data_movement", "status": "measured",
         "required": False, "source": "moved_bytes / baseline_median",
         "value": primary["derived"]["baseline_achieved_bw_gbps"], "case_id": primary["case_id"]},
    ]
    return {
        "schema": "opx-performance/1",
        "op": op,
        "action": "benchmark",
        "run_id": run_id or None,
        # ---- 版本对齐（门禁逐字比对这一版 run）----
        "spec_version_id": int(spec_version_id),
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": code_snapshot_sha256,
        "suite_ref": suite_ref,
        "conditions": conditions,
        "cases": cases,
        "metrics": metrics,
        "summary": {
            "case_count": len(cases),
            "primary_case": primary["case_id"],
            "speedup_device": primary["derived"]["speedup_baseline_over_candidate"],
        },
    }


def _render_md(report: dict) -> str:
    cond = report["conditions"]
    lines = [
        f"# 基准报告 · {report['op']}",
        "",
        f"- 基线：{cond['baseline_impl']}",
        f"- 主值测量范围：{cond['measure_scope']}（msopprof 设备侧 Task Duration）",
        f"- 采集次数：{cond['invocations']}，msopprof warm-up：{cond['warm_up']}，聚合：{cond['aggregate']}",
        f"- 设备：{cond.get('device')}（锁 `{cond.get('lock_mode')}`）",
        f"- 规格版本：v{report['spec_version_id']} ｜ 判据档：`{report['criteria_ref']}`",
        f"- 用例集 ref：`{report['suite_ref']}` ｜ 代码快照：`{report['code_snapshot_sha256'][:16]}…`",
        "",
        "| 用例 | N | dtype | 基线设备(us) | 候选设备(us) | 加速比 | 基线带宽(GB/s) | 候选带宽(GB/s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in report["cases"]:
        lines.append(
            f"| {c['case_id']} | {c['n']} | {c['dtype']} | {c['baseline']['device_us']['median_us']:.3f} | "
            f"{c['candidate']['device_us']['median_us']:.3f} | {c['derived']['speedup_baseline_over_candidate']:.3f} | "
            f"{c['derived']['baseline_achieved_bw_gbps']:.1f} | {c['derived']['candidate_achieved_bw_gbps']:.1f} |")
    lines += ["", "> 主值 = msopprof 设备侧 kernel 时长（候选与基线同一把尺子）；不设加速比硬线。",
              "> 本文件是给人看的伴生品；机判只认同目录的 `performance.json`（门禁契约 docs/gate-report-v1.md）。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="elementwise_add")
    ap.add_argument("--workspace", default=str(WS))
    ap.add_argument("--exe", default="")
    ap.add_argument("--bench", default="")
    ap.add_argument("--out", default="/out/performance.json")
    ap.add_argument("--md", default="")
    ap.add_argument("--workdir", default="/tmp/opx-bench")
    ap.add_argument("--evidence-dir", default="")
    ap.add_argument("--invocations", type=int, default=0, help="覆盖 benchmark.json 的采集次数")
    ap.add_argument("--retries", type=int, default=0, help="覆盖每次采集的重试次数")
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
        print(f"[run_benchmark] 版本对齐字段缺失：{missing}（应由 nm_test.sh 注入环境变量）", file=sys.stderr)
        return 2
    if not bench_path.is_file():
        print(f"[run_benchmark] 基准用例定义不存在：{bench_path}", file=sys.stderr)
        return 2
    if not exe.is_file():
        print(f"[run_benchmark] 候选实现不存在：{exe}（先跑 build 动作）", file=sys.stderr)
        return 2
    if not shutil.which("msopprof"):
        print("[run_benchmark] 找不到 msopprof（应为 CANN 自带：<CANN>/bin/msopprof）", file=sys.stderr)
        return 2

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    measure = bench.get("measure") or {}

    # ---- 显式按 op 分派：scan 分支（A1）。elementwise 旧路径保持原样。 ----
    if op == "jax_lax_scan":
        if bench.get("op") != op:
            print(f"[run_benchmark] benchmark.op={bench.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
            return 2
        try:
            scripts = ws / "ops" / "jax_lax_scan" / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            import scan_bench  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[run_benchmark] scan 基准执行器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        md = Path(args.md) if args.md else out.with_suffix(".md")
        return scan_bench.run_benchmark(
            ws=ws, op=op, exe=exe, bench=bench, bench_path=bench_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, measure=measure,
            invocations=args.invocations, retries=args.retries, timeout=args.timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    # ---- 显式按 op 分派：jax_lax_associative_scan 分支。elementwise / scan 旧路径保持原样。 ----
    if op == "jax_lax_associative_scan":
        if bench.get("op") != op:
            print(f"[run_benchmark] benchmark.op={bench.get('op')!r} 与 --op={op!r} 不一致", file=sys.stderr)
            return 2
        try:
            scripts = ws / "ops" / "jax_lax_associative_scan" / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            import assoc_scan_bench  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[run_benchmark] assoc-scan 基准执行器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        md = Path(args.md) if args.md else out.with_suffix(".md")
        return assoc_scan_bench.run_benchmark(
            ws=ws, op=op, exe=exe, bench=bench, bench_path=bench_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, measure=measure,
            invocations=args.invocations, retries=args.retries, timeout=args.timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    # ---- 显式按 op 分派：jax_lax_cond 分支。elementwise / scan / assoc-scan 旧路径保持原样。 ----
    if op == "jax_lax_cond":
        if bench.get("op") != op:
            print(f"[run_benchmark] benchmark.op={bench.get('op')!r} 与 --op={op!r} 不一致",
                  file=sys.stderr)
            return 2
        try:
            scripts = ws / "ops" / "jax_lax_cond" / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            import cond_bench  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[run_benchmark] cond 基准执行器加载失败：{type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 2
        md = Path(args.md) if args.md else out.with_suffix(".md")
        return cond_bench.run_benchmark(
            ws=ws, op=op, exe=exe, bench=bench, bench_path=bench_path, out=out, md=md,
            workdir=workdir, evidence_dir=evidence_dir, measure=measure,
            invocations=args.invocations, retries=args.retries, timeout=args.timeout,
            spec_version_id=spec_version_id, criteria_ref=criteria_ref,
            code_snapshot_sha256=snapshot, suite_ref=suite_ref, run_id=run_id,
        )

    # 基线小程序由 benchmark.json 指定（缺省仍是首条算子的 torch.add 基线）
    baseline_impl = bench.get("baseline", {}).get("impl", "基线")
    baseline_app_rel = (bench.get("baseline") or {}).get("app") or "tools/torch_add_baseline.py"
    baseline_app = ws / baseline_app_rel
    if not baseline_app.is_file():
        print(f"[run_benchmark] 基线小程序不存在：{baseline_app}", file=sys.stderr)
        return 2
    invocations = args.invocations or int(measure.get("invocations", 5))
    retries = args.retries or int(measure.get("retries_per_invocation", 3))
    warm_up = int(measure.get("warm_up", 2))
    aic_metrics = measure.get("aic_metrics", "Default")
    default_dtype = bench.get("dtype", "float32")
    primary_case_id = bench.get("primary_case")
    prof_parse = _profile_module().parse_op_basic_info

    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for case in bench.get("cases") or []:
        cid = case["case_id"]
        n = int(case["n"])
        dtype = case.get("dtype", default_dtype)
        seed = int(case.get("seed", 0))
        cdir = workdir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        in_a, in_b, out_bin = cdir / "a.bin", cdir / "b.bin", cdir / "out.bin"
        gi = _gen_inputs()
        gi.generate(case.get("gen_a", {}), n, seed, dtype).tofile(in_a)
        gi.generate(case.get("gen_b", {}), n, seed + 1, dtype).tofile(in_b)

        cand_app = (f"{exe} --n-a {n} --n-b {n} --dtype {dtype} "
                    f"--in-a {in_a} --in-b {in_b} --out {out_bin}")
        base_app = f"python3 {baseline_app} {n}"

        cand_us, cand_info, cand_fail = collect_device_samples(
            prof_parse, cand_app, workdir / f"opprof_cand_{cid}",
            invocations=invocations, retries=retries, aic_metrics=aic_metrics,
            warm_up=warm_up, timeout=args.timeout)
        if not cand_us:
            print(f"[run_benchmark] 候选采集全部失败（{cid}）：{cand_fail[-3:]}", file=sys.stderr)
            return 4
        base_us, base_info, base_fail = collect_device_samples(
            prof_parse, base_app, workdir / f"opprof_base_{cid}",
            invocations=invocations, retries=retries, aic_metrics=aic_metrics,
            warm_up=warm_up, timeout=args.timeout)
        if not base_us:
            print(f"[run_benchmark] 基线（{baseline_impl}）采集全部失败（{cid}）：{base_fail[-3:]}", file=sys.stderr)
            return 3
        if base_fail or cand_fail:
            print(f"[run_benchmark] {cid} 采集有重试：候选失败 {len(cand_fail)} 次 / 基线失败 {len(base_fail)} 次"
                  f"（已用成功样本 {len(cand_us)}/{len(base_us)} 个）", file=sys.stderr)

        baseline = {"impl": baseline_impl,
                    "op_name": base_info.get("op_name"), "op_type": base_info.get("op_type"),
                    "block_dim": base_info.get("block_dim"), "device_us": aggregate(base_us)}
        candidate = {"impl": "ascendc-direct-invoke",
                     "op_name": cand_info.get("op_name"), "op_type": cand_info.get("op_type"),
                     "block_dim": cand_info.get("block_dim"), "device_us": aggregate(cand_us)}
        profile = case_profile(bench, case)
        results.append({
            "case_id": cid,
            "n": n,
            "dtype": dtype,
            "seed": seed,
            "gen": f"a={gi.describe(case.get('gen_a', {}))};b={gi.describe(case.get('gen_b', {}))}",
            "required": bool(case.get("required", True)),
            "primary": cid == primary_case_id,
            "note": case.get("note"),
            "baseline": baseline,
            "candidate": candidate,
            "derived": derive(
                baseline["device_us"]["median_us"], candidate["device_us"]["median_us"], n,
                bytes_per_elem=dtype_bytes(dtype), reads_per_elem=profile["reads"],
                writes_per_elem=profile["writes"], flops_per_elem=profile["flops_per_elem"],
                moved_bytes=profile["moved_bytes"], flops=profile["flops"],
                include_intensity=profile_is_explicit(bench, case),
            ),
        })
        (evidence_dir / f"{cid}.json").write_text(json.dumps(
            {"case_id": cid, "n": n, "dtype": dtype, "seed": seed,
             "baseline_samples_device_us": base_us, "baseline_op": base_info,
             "candidate_samples_device_us": cand_us, "candidate_op": cand_info,
             "collection_failures": {"candidate": cand_fail, "baseline": base_fail}},
            ensure_ascii=False, indent=2), encoding="utf-8")

    primary_derived = next((c["derived"] for c in results if c.get("primary")), results[0]["derived"] if results else {})
    conditions = {
        "baseline_impl": baseline_impl,
        "baseline_app": baseline_app_rel,
        "measure_scope": measure.get("scope", "device_kernel_task_duration"),
        "measure_tool": "msopprof (msprof op) OpBasicInfo.csv Task Duration(us)",
        "measure_note": measure.get("note"),
        "invocations": invocations,
        "retries_per_invocation": retries,
        "warm_up": warm_up,
        "aic_metrics": aic_metrics,
        "aggregate": measure.get("aggregate", "median"),
        "dtype": default_dtype,
        "device": os.environ.get("OPX_DEVICES") or None,
        "lock_mode": os.environ.get("OPX_LOCK_MODE") or None,
        "candidate_exe": str(exe.relative_to(ws)) if str(exe).startswith(str(ws)) else str(exe),
        "candidate_exe_sha256": _sha256_file(exe),
        "bench_source": str(bench_path.relative_to(ws)) if str(bench_path).startswith(str(ws)) else str(bench_path),
        # 访存/计算画像（Roofline 用）：老报告没有这些字段，前端会退回算子画像
        "reads_per_elem": (bench.get("memory") or {}).get("reads_per_elem"),
        "writes_per_elem": (bench.get("memory") or {}).get("writes_per_elem"),
        "flops_per_elem": (bench.get("compute") or {}).get("flops_per_elem"),
        "arithmetic_intensity": primary_derived.get("arithmetic_intensity"),
    }
    report = build_report(op=op, run_id=run_id, spec_version_id=int(spec_version_id),
                          criteria_ref=criteria_ref, code_snapshot_sha256=snapshot, suite_ref=suite_ref,
                          conditions=conditions, cases=results)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = Path(args.md) if args.md else out.with_suffix(".md")
    md.write_text(_render_md(report), encoding="utf-8")

    primary = next((c for c in results if c["primary"]), results[0])
    print(f"[run_benchmark] 主用例 {primary['case_id']}：基线 {primary['baseline']['device_us']['median_us']:.3f}us "
          f"（{primary['baseline'].get('op_name')}）/ 候选 {primary['candidate']['device_us']['median_us']:.3f}us "
          f"（{primary['candidate'].get('op_name')}）"
          f"（加速比 {primary['derived']['speedup_baseline_over_candidate']:.3f}） → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
