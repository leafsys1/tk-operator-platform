#!/usr/bin/env python3
"""torch.add 基线小程序（benchmark 动作的**基线侧**，在同一张卡上被 msopprof 采集）。

为什么是「另一侧的小程序」而不是在 `run_benchmark.py` 里直接 import torch：
基准与候选必须用**同一把尺子**，而唯一能给出「设备侧 kernel 任务时长」这把尺子的是 msopprof
（`Task Duration(us)`）。msopprof 采集的是**一个进程里的一次 kernel**，所以基线也要以一个
可执行/可启动的形态被它采集 —— 这里用最小 torch 脚本承载官方 `torch.add`。

约定：`python3 tools/torch_add_baseline.py <N>`。只做 `torch.add(a, b, out=c)`，
让 msopprof 采到占绝对主导的那个 Add kernel（实测 Op Name 形如 `Add_<hash>_high_performance_*`）。
"""

from __future__ import annotations

import sys


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1048576

    import torch  # noqa: PLC0415
    import torch_npu  # noqa: F401,PLC0415

    torch.npu.set_device(0)
    a = torch.randn(n, dtype=torch.float32, device="npu")
    b = torch.randn(n, dtype=torch.float32, device="npu")
    c = torch.empty(n, dtype=torch.float32, device="npu")

    for _ in range(3):  # 让算子进入稳定状态；msopprof 自己还会 replay + warm-up
        torch.add(a, b, out=c)
    torch.npu.synchronize()

    torch.add(a, b, out=c)
    torch.npu.synchronize()
    print("TORCH_ADD_BASELINE_DONE", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
