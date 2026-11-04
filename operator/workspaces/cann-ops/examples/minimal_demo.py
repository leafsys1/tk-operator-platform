"""bid_elementwise_sub_20261101 · 最小独立示例。

直接调用真实算子插入路径（Python 接口 → ctypes → 候选 kernel），不依赖 pytest。
用法（需真机 torch + torch_npu + 已构建 libbid_elementwise_sub_20261101_host.so）：

    python examples/minimal_demo.py

验证语义：``[3, 1, 2] - [1, 1, 1] == [2, 0, 1]``。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # 注册 npu 设备

WS = Path(__file__).resolve().parents[1]
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))

from ops.bid_elementwise_sub_20261101.python.bid_elementwise_sub_20261101 import (  # noqa: E402
    ABI_VERSION,
    subtract,
)


def main() -> None:
    a = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32).npu()
    b = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).npu()
    y = subtract(a, b)

    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (3,)
    assert y.dtype == torch.float32
    assert y.device == a.device
    assert y.is_contiguous()
    assert y.data_ptr() not in (a.data_ptr(), b.data_ptr())

    expect = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float32)
    got = y.cpu()
    assert torch.equal(got, expect), f"got {got} != expect {expect}"

    print("minimal_demo OK: [3,1,2] - [1,1,1] =", got.numpy().tolist())
    print("ABI version:", ABI_VERSION)


if __name__ == "__main__":
    main()