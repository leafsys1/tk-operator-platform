#!/usr/bin/env python3
"""torch.sub 基线小程序（elementwise_sub 的 benchmark 基线侧，在同一张卡上被 msopprof 采集）。

为什么新增一个脚本而不是复用 `tools/torch_add_baseline.py`：规格把性能基线定为 **torch.sub**
（`performance_baseline: "torch.sub"`、`benchmark.json` 的 `baseline.app` 指向本文件）。
口径与 add 基线完全一致：只做一次 `torch.sub(a, b, out=c)`（默认 alpha=1），让 msopprof 采到
占绝对主导的那个 Sub kernel；两侧仍是同一把尺子（设备侧 Task Duration）。

约定：`python3 tools/torch_sub_baseline.py <N>`。
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
        torch.sub(a, b, out=c)
    torch.npu.synchronize()

    torch.sub(a, b, out=c)
    torch.npu.synchronize()
    print("TORCH_SUB_BASELINE_DONE", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
