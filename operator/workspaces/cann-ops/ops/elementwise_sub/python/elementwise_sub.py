"""elementwise_sub · Python 调用接口（ctypes 封装 host C ABI）。

签名对齐 `torch.sub(input, other, alpha=1, out=None)`：

    sub(a, b, alpha=1, out=None) -> Tensor

契约（控制台仓库 `docs/reference/deliverable-standards.md`）：
- ABI：``int elementwise_sub_run(const void*, const void*, void*, int64_t, int, int)``
       ``int elementwise_sub_alpha_run(const void*, const void*, void*, int64_t, float, int)``
- 入参/出参都是 torch_npu 张量（float32、同 shape、连续），拒绝码与可执行文件 CLI 一致
  （用异常类型/消息表达，供 pytest 断言）；
- 四动作门禁仍走可执行文件，本接口不替代 `test.sh`。

alpha≠1 的语义：规格 oracle = 「fp64 计算 y = a - alpha*b，再按 float32 舍入」。本接口做**一次**
候选 kernel 调用（``elementwise_sub_alpha_run``），kernel 内部是 fp32「先舍入 alpha*b 再相减」
（DaVinci vector 核不支持 double，见 op_kernel/elementwise_sub_tiling.h 的 OP_MODE_FMASUB 说明）；
Python 层不做任何数值计算（编码规则 A1：包装层不得代算）。

该口径与 oracle 的逐位关系：alpha 为 2 的幂（2.0 / 0.5 / -1.0）时 alpha*b 无舍入损失，
与 oracle 逐位一致；|alpha*b| > FLT_MAX 时 kernel 先上溢为 ±inf，结果不做逐位承诺
（规格 v2 已把 |alpha*b| ≤ FLT_MAX 写入支持范围）。

缓冲对齐：kernel 的尾核按 32B 对齐长度写回（最多多写 7 个元素到填充槽，元素数仍由调用方
张量形状约束）。torch_npu 由 caching allocator 按 512B 倍数分配，``empty_like`` 得到的缓冲天然
留有余量；但 **如果调用方把非对齐的裸指针 / 切片视图作为 out 传进来，尾部 7 个元素可能写到
张量范围之外** —— 这是 ABI 的调用前提，见 README「已知限制」。
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_LIB_PATH = Path(__file__).resolve().parents[1] / "build" / "libelementwise_sub_host.so"

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
    run = lib.elementwise_sub_run
    run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    run.restype = ctypes.c_int
    alpha_run = lib.elementwise_sub_alpha_run
    alpha_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_int64, ctypes.c_float, ctypes.c_int]
    alpha_run.restype = ctypes.c_int
    version = lib.elementwise_sub_abi_version
    version.restype = ctypes.c_char_p
    return lib, run, alpha_run, version().decode()


_lib, _run, _alpha_run, ABI_VERSION = _load()


def _device_index(t: torch.Tensor) -> int:
    return int(t.device.index) if t.device.index is not None else 0


def _check(a, b) -> int:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        raise TypeError("REJECT_BAD_ARGS: 入参必须是 torch.Tensor")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError(f"REJECT_UNSUPPORTED_DTYPE: 只支持 float32（得到 {a.dtype} / {b.dtype}）")
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"REJECT_SHAPE_MISMATCH: 规格 v1 不广播，shape 必须相同（{tuple(a.shape)} vs {tuple(b.shape)}）")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("REJECT_BAD_ARGS: 入参必须是连续张量")
    n = int(a.numel())
    if n < N_MIN or n > N_MAX:
        raise ValueError(f"REJECT_N_OUT_OF_RANGE: n={n}，合法范围 [{N_MIN}, {N_MAX}]")
    return n


def _raise(rc: int, fn: str):
    if rc != 0:
        raise RuntimeError(f"{_RETURN_CODES.get(rc, f'UNKNOWN({rc})')}: {fn} 返回 {rc}")


def sub(a: torch.Tensor, b: torch.Tensor, alpha=1, out: torch.Tensor | None = None) -> torch.Tensor:
    """y = a - alpha*b（float32 / 同 shape / 不广播 / N∈[1, 1048576] / 支持 out 复用）。

    - ``alpha=1``：单次 sub kernel；
    - ``alpha≠1``：单次 alpha kernel，fp32 先算 ``alpha*b`` 再与 ``a`` 相减（与 torch 求值顺序一致；
      alpha 为 2 的幂时与规格 oracle 逐位一致）；
    - ``out`` 可以就是 ``a``（原地 ``a.sub_(b, alpha)`` 语义）。
    """
    n = _check(a, b)
    device_id = _device_index(a)

    if out is None:
        out = torch.empty_like(a)
    else:
        if not torch.is_tensor(out):
            raise TypeError("REJECT_BAD_ARGS: out 必须是 torch.Tensor")
        if out.dtype != torch.float32:
            raise ValueError(f"REJECT_UNSUPPORTED_DTYPE: out 只支持 float32（得到 {out.dtype}）")
        if tuple(out.shape) != tuple(a.shape):
            raise ValueError(f"REJECT_SHAPE_MISMATCH: out shape 必须与输入一致（{tuple(out.shape)} vs {tuple(a.shape)}）")
        if not out.is_contiguous():
            raise ValueError("REJECT_BAD_ARGS: out 必须是连续张量")

    alpha_f = float(alpha)

    if alpha_f == 1.0:
        rc = _run(int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr()),
                  n, DTYPE_CODE_FLOAT32, device_id)
        _raise(rc, "elementwise_sub_run")
        return out

    # alpha≠1：单次候选 kernel（FMASUB：fp32 先乘后减；2 的幂时逐位等价 oracle）
    rc = _alpha_run(int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr()),
                    n, alpha_f, device_id)
    _raise(rc, "elementwise_sub_alpha_run")
    return out
