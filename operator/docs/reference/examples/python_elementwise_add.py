"""elementwise_add · Python 调用接口（ctypes 封装 host C ABI）—— 交付物示例。

契约见控制台仓库 `docs/reference/deliverable-standards.md`：
- ABI：`int elementwise_add_run(const void*, const void*, void*, int64_t, int, int)`
- 入参/出参都是 torch_npu 张量（float32、同 shape、连续）；
- 拒绝码与可执行文件 CLI 一致（用异常类型/消息表达，供 pytest 断言）；
- 四动作门禁仍走可执行文件，本接口不替代 `test.sh`。

真机验证（有卡时）：
    import torch, torch_npu
    from ops.elementwise_add.python.elementwise_add import add
    a = torch.randn(65536, dtype=torch.float32).npu(); b = torch.randn_like(a)
    y = add(a, b)
    assert torch.allclose(y.cpu(), a.cpu() + b.cpu(), rtol=2**-10, atol=2**-16)
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_LIB_PATH = Path(__file__).resolve().parents[1] / "build" / "libelementwise_add_host.so"

DTYPE_CODE_FLOAT32 = 0
N_MIN = 1
N_MAX = 1048576

_RETURN_CODES = {
    0: "OK",
    1: "RUN_FAILED",
    2: "REJECT_BAD_ARGS",
    4: "REJECT_N_OUT_OF_RANGE",
    5: "REJECT_UNSUPPORTED_DTYPE",
}


def _load():
    if not _LIB_PATH.is_file():
        raise RuntimeError(f"缺少 Python 接口共享库：{_LIB_PATH}（先跑 build.sh）")
    lib = ctypes.CDLL(str(_LIB_PATH))
    run = lib.elementwise_add_run
    run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    run.restype = ctypes.c_int
    version = lib.elementwise_add_abi_version
    version.restype = ctypes.c_char_p
    return lib, run, version().decode()


_lib, _run, ABI_VERSION = _load()


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """y = a + b（float32 / 同 shape / 不广播 / N∈[1, 1048576]）。"""
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        raise TypeError("REJECT_BAD_ARGS: 入参必须是 torch.Tensor")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("REJECT_UNSUPPORTED_DTYPE: 只支持 float32")
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError("REJECT_SHAPE_MISMATCH: 规格 v1 不广播，两个输入 shape 必须相同")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("REJECT_BAD_ARGS: 入参必须是连续张量")
    n = int(a.numel())
    if n < N_MIN or n > N_MAX:
        raise ValueError(f"REJECT_N_OUT_OF_RANGE: n={n}，合法范围 [{N_MIN}, {N_MAX}]")

    out = torch.empty_like(a)
    device_id = a.device.index if a.device.index is not None else 0
    rc = _run(int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr()),
              n, DTYPE_CODE_FLOAT32, int(device_id))
    if rc != 0:
        raise RuntimeError(f"{_RETURN_CODES.get(rc, f'UNKNOWN({rc})')}: elementwise_add_run 返回 {rc}")
    return out
