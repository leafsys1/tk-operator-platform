#!/usr/bin/env bash
# 环境自检（**在容器里跑**）：证明「能起 ≠ 能跑算子」里的后半句。
# 这条不是四动作之一，而是四动作第 ① 步「检查本动作的必要环境条件」的一次真实执行。
set -uo pipefail

echo "== 环境自检 =="
echo "--- 身份/系统 ---"
id -un; cat /etc/os-release 2>/dev/null | head -2
echo "--- npu-smi -t board ---"
npu-smi info -t board -i 0 2>&1 | head -14 || true
echo "--- python / torch / torch_npu ---"
python3 -V || true
python3 - <<'PY' || true
try:
    import torch, torch_npu
    print("torch", torch.__version__, "| torch_npu", torch_npu.__version__,
          "| npu devices", torch_npu.npu.device_count())
    print("device 0:", torch_npu.npu.get_device_name(0))
except Exception as e:
    print("torch/torch_npu 不可用:", type(e).__name__, e)
PY
echo "--- CANN toolkit ---"
ls /usr/local/Ascend/ascend-toolkit/ 2>&1 | head -4 || true
echo "--- 编译链 ---"
for t in gcc g++ cmake make; do printf "%s: " "$t"; command -v "$t" || echo "(缺)"; done
echo "--- 工作区 ---"
ls -A /ws | head -10
echo "== 自检结束 =="
