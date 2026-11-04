#!/usr/bin/env python3
"""核数–耗时曲线测量（**调试用**：不登记 run、不进 D8 预算）。

用途：给自动优化第三轮提供依据 —— 小 shape（64k）到底是「核启动/固定开销」占主导，
还是「单核搬运」受限。做法：固定 N，用候选实现的 `--force-blocks K` 强制启 K 个核，
每个 K 采 `invocations` 次 msopprof，取 Task Duration 中位数 + `aiv_time` 逐核均值。

跑法（容器内，由 nm_test.sh 的调试命令触发）：
    bash build.sh && python3 tools/measure_curve.py --n 65536,1048576 --blocks 16,8,4,2,1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "tools"))


def _aiv_time(prof_dir: Path) -> float | None:
    import run_profile as rp  # noqa: PLC0415 —— 与 benchmark 复用同一份解析

    for name in ("ArithmeticUtilization.csv", "PipeUtilization.csv"):
        p = prof_dir / name
        if p.is_file():
            v = rp.col_mean(rp.read_csv_rows(p), "aiv_time(us)")
            if v is not None:
                return v
    return None


def main() -> int:
    import gen_inputs  # noqa: PLC0415
    import run_benchmark as rb  # noqa: PLC0415
    import run_profile as rp  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="elementwise_add")
    ap.add_argument("--n", default="65536,1048576", help="逗号分隔的元素数")
    ap.add_argument("--blocks", default="16,8,4,2,1", help="逗号分隔的强制核数（0 = 自动）")
    ap.add_argument("--invocations", type=int, default=5)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--exe", default="")
    ap.add_argument("--workdir", default="/tmp/opx-curve")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    exe = Path(args.exe) if args.exe else WS / "ops" / args.op / "build" / args.op
    if not exe.is_file():
        print(f"[measure] 候选不存在：{exe}（先 build）", file=sys.stderr)
        return 2
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n in (int(x) for x in args.n.split(",")):
        cdir = work / f"n{n}"
        cdir.mkdir(parents=True, exist_ok=True)
        a, b, out = cdir / "a.bin", cdir / "b.bin", cdir / "out.bin"
        gen_inputs.generate({"dist": "zeros"}, n, 1, "float32").tofile(a)
        gen_inputs.generate({"dist": "zeros"}, n, 2, "float32").tofile(b)

        for k in (int(x) for x in args.blocks.split(",")):
            app = (f"{exe} --n-a {n} --n-b {n} --dtype float32 "
                   f"--in-a {a} --in-b {b} --out {out}")
            if k > 0:
                app += f" --force-blocks {k}"
            samples: list[float] = []
            aiv: list[float] = []
            info: dict = {}
            failures = 0
            for i in range(args.invocations):
                for attempt in range(args.retries):
                    try:
                        d = rb.run_msopprof(app, work / f"prof_n{n}_k{k}_{i}_{attempt}",
                                            aic_metrics="Default", warm_up=2)
                        cur = rp.parse_op_basic_info(d / "OpBasicInfo.csv")
                        if cur.get("task_duration_us") is None:
                            raise RuntimeError("OpBasicInfo 没有 Task Duration")
                        info = cur
                        samples.append(float(cur["task_duration_us"]))
                        at = _aiv_time(d)
                        if at is not None:
                            aiv.append(at)
                        break
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        if attempt == args.retries - 1:
                            print(f"[measure] n={n} k={k} i={i} 失败：{exc}", file=sys.stderr)
            if not samples:
                rows.append({"n": n, "blocks": k, "error": "采集全部失败", "failures": failures})
                continue
            moved = 3 * n * 4
            med = statistics.median(samples)
            rows.append({
                "n": n,
                "blocks": k,
                "block_dim": info.get("block_dim"),
                "median_us": med,
                "min_us": min(samples),
                "max_us": max(samples),
                "aiv_time_us": (statistics.mean(aiv) if aiv else None),
                "achieved_bw_gbps": (moved / (med * 1e-6)) / 1e9 if med > 0 else None,
                "samples": samples,
                "failures": failures,
            })
            print(f"[measure] n={n:>8} blocks={k:>2} → {med:.3f}us"
                  f"（min {min(samples):.3f} / max {max(samples):.3f}）"
                  f" aiv={rows[-1]['aiv_time_us']}", file=sys.stderr)

    print(json.dumps({"schema": "opx-core-curve/1", "rows": rows}, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"schema": "opx-core-curve/1", "rows": rows},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
