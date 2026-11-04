#!/usr/bin/env bash
# 环境自检（**在容器里跑**）：证明「能起 ≠ 能跑算子」里的后半句。
# 这条不是四动作之一，而是四动作第 ① 步「检查本动作的必要环境条件」的一次真实执行。
set -uo pipefail

echo "== 环境自检 =="
echo "--- 身份/系统 ---"
id -un; cat /etc/os-release 2>/dev/null | head -2
echo "--- 容器内设备节点 ---"
ls /dev | grep -E "^davinci|devmm_svm|hisi_hdc" | head -12 || echo "(没有 davinci 设备节点)"
echo "--- ascend_install.info ---"
head -6 /etc/ascend_install.info 2>&1 || true
echo "--- driver ---"
head -3 /usr/local/Ascend/driver/version.info 2>/dev/null || echo "(读不到 driver/version.info)"
ls /usr/local/Ascend/driver/lib64 2>&1 | head -4
echo "--- dcmi ---"
ls /usr/local/dcmi 2>&1 | head -4
echo "--- npu-smi -t board ---"
npu-smi info -t board -i 0 2>&1 | head -12 || true
echo "--- python / torch / torch_npu（带 90s 护栏：torch_npu 初始化可能挂死）---"
python3 -V || true
timeout --kill-after=5 90s python3 - <<'PY' || echo "(torch 检查超时或被拒 —— 记为 FAIL)"
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
echo "--- 结论（机器可读；按输出判定，不只看退出码 —— 实测 npu-smi 会「打印报错却返回 0」）---"
if timeout 30 npu-smi info -t board -i 0 2>/dev/null | grep -q "Product Name"; then
  echo "VERDICT npu: OK"
else
  echo "VERDICT npu: FAIL（npu-smi 没能返回板卡信息）"
fi
if timeout 120 python3 -c "import torch_npu, sys; sys.exit(0 if torch_npu.npu.device_count() > 0 else 1)" >/dev/null 2>&1; then
  echo "VERDICT torch_npu: OK"
else
  echo "VERDICT torch_npu: FAIL（导入成功但看不到设备也算 FAIL）"
fi
echo "== 自检结束 =="
