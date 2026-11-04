"""bid_elementwise_sub_20261101 · Python 调用接口（ctypes 封装 host C ABI）。

语义：``subtract(a, b) -> Tensor``，``y[i] = a[i] - b[i]``。

契约（控制台仓库 ``docs/reference/deliverable-standards.md``）：
- ABI：``int bid_elementwise_sub_20261101_run(const void*, const void*, void*, int64_t, int, int)``
  + ``const char* bid_elementwise_sub_20261101_abi_version()`` = ``"opx-python-abi/1"``；
- float32 / 1-D 连续 / 同 shape / 同 NPU 设备；输出为新张量（非原地、不与输入共享 storage）；
- 不广播、无 alpha、无 out=；拒绝码与可执行文件 CLI 一致（用异常类型/消息表达，供 pytest 断言）；
- 拒绝集合（唯一合法）：REJECT_BAD_ARGS / REJECT_SHAPE_MISMATCH / REJECT_N_OUT_OF_RANGE /
  REJECT_UNSUPPORTED_DTYPE / RUN_FAILED。包装层不做任何数值计算（编码规则 A1）。

缓冲对齐：kernel 只搬运 N 个有效元素（count * sizeof(float) 字节）；非 32B 对齐尾块由
DataCopyPad 在 UB 侧补齐，GM 侧不读/写逻辑 in/out 元素之外的任何字节。
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401  # NPU 环境下由 torch_npu 提供后端与流同步入口
    _HAS_TORCH_NPU = True
except ImportError:  # pragma: no cover - 无 NPU 后端的环境
    torch_npu = None
    _HAS_TORCH_NPU = False

_LIB_PATH = Path(__file__).resolve().parents[1] / "build" / "libbid_elementwise_sub_20261101_host.so"

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
        raise RuntimeError(f"缺少 Python 接口共享库：{_LIB_PATH}（请先构建 libbid_elementwise_sub_20261101_host.so）")
    lib = ctypes.CDLL(str(_LIB_PATH))
    run = lib.bid_elementwise_sub_20261101_run
    run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    run.restype = ctypes.c_int
    version = lib.bid_elementwise_sub_20261101_abi_version
    version.restype = ctypes.c_char_p
    return run, version().decode()


_run, ABI_VERSION = _load()


def _device_index(t: torch.Tensor) -> int:
    return int(t.device.index) if t.device.index is not None else 0


def _sync_producer(device) -> None:
    """生产者 → 候选流 的保守同步。

    候选算子在 host C ABI 内部新建**独立 ACL 流**执行，与 torch_npu 生产 a/b 的流无依赖关系；
    直接把 ``data_ptr`` 传过去可能读到尚未写回显存的输入。单卡限制下，这里在交给候选流之前对
    目标设备做一次整体同步（torch_npu 同步入口），保证 a/b 已由 torch_npu 完全产出。仍是同步
    阻塞语义，不承诺多卡/跨流异步。
    """
    if _HAS_TORCH_NPU:
        torch_npu.npu.synchronize(device)


def _check(a, b) -> int:
    """输入契约校验，返回元素数 n。抛异常时消息携带 REJECT_* 子串供 pytest 断言。"""
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        raise TypeError("REJECT_BAD_ARGS: 入参必须是 torch.Tensor")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError(f"REJECT_UNSUPPORTED_DTYPE: 只支持 float32（得到 {a.dtype} / {b.dtype}）")
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"REJECT_SHAPE_MISMATCH: 规格 v3 不广播，shape 必须相同"
                         f"（{tuple(a.shape)} vs {tuple(b.shape)}）")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"REJECT_BAD_ARGS: 只支持 1-D 输入（得到 {a.ndim}D / {b.ndim}D）")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("REJECT_BAD_ARGS: 入参必须是连续张量")
    if a.device.type != "npu" or b.device.type != "npu":
        # CPU 输入 / CPU-NPU 混用：规格 v3 要求精确 TypeError（设备类型错误），非 ValueError。
        raise TypeError(f"REJECT_BAD_ARGS: 入参必须都在 NPU（得到 {a.device.type} / {b.device.type}）")
    if a.device != b.device:
        # 跨 NPU（同类型、不同设备索引）为独立拒绝情形，保持 ValueError。
        raise ValueError(f"REJECT_BAD_ARGS: 入参必须在同一 NPU 设备（{a.device} vs {b.device}）")
    n = int(a.numel())
    if n < N_MIN or n > N_MAX:
        raise ValueError(f"REJECT_N_OUT_OF_RANGE: n={n}，合法范围 [{N_MIN}, {N_MAX}]")
    return n


def _raise(rc: int, fn: str):
    if rc != 0:
        raise RuntimeError(f"{_RETURN_CODES.get(rc, f'UNKNOWN({rc})')}: {fn} 返回 {rc}")


def subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``y = a - b``（float32 / 1-D 连续 / 同 shape / 同 NPU 设备 / N∈[1, 1048576]）。

    输出为新张量：非原地、不与输入共享 storage。不广播、无 alpha、无 out=。

    拒绝规则：非 tensor / CPU / CPU-NPU 混用 → TypeError(REJECT_BAD_ARGS)；非 float32 →
    REJECT_UNSUPPORTED_DTYPE；shape 不等 → REJECT_SHAPE_MISMATCH；rank≠1 / 非连续 / 跨 NPU →
    ValueError(REJECT_BAD_ARGS)；N∉[1,1048576] → REJECT_N_OUT_OF_RANGE。调用前对目标设备做
    torch_npu 保守同步（阻塞语义），保证候选独立流能看到生产者已写完的输入。
    """
    n = _check(a, b)
    device_id = _device_index(a)
    _sync_producer(a.device)  # 生产者→候选流 保守同步：确保 a/b 已产出到显存
    out = torch.empty_like(a)
    rc = _run(int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr()),
              n, DTYPE_CODE_FLOAT32, device_id)
    _raise(rc, "bid_elementwise_sub_20261101_run")
    return out