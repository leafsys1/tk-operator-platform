#!/bin/bash
# 判别实验：把可见设备设为不存在的 99，若仍 PASS 说明走 CPU 路径（非真机计算）。
set -u
IMG=lfx-op-dev-910b:torch210-cann910
WS=/data/lfx/op-agent-console-20261101/app/workspaces/cann-ops
NAME=tk-devprobe-$(date +%s)
BIN=/ws/ops/bid_elementwise_sub_20261101/build/bid_elementwise_sub_20261101
SECCOMP=/data/lfx/op-agent-console-20261101/config/docker-18.09-default.json
echo "=== 容器内可见设备数量探测 (ASCEND_RT_VISIBLE_DEVICES=99) ==="
docker run --rm --name "$NAME" \
  --privileged --security-opt seccomp="$SECCOMP" \
  -e OPENBLAS_NUM_THREADS=1 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e ASCEND_RT_VISIBLE_DEVICES=99 \
  -v "$WS":/ws -w /ws "$IMG" bash -lc '
set -e
echo "VISIBLE=$ASCEND_RT_VISIBLE_DEVICES"
python3 - <<PY
import torch, torch_npu
print("torch", torch.__version__)
print("npu_device_count", torch_npu.npu.device_count())
try:
    x = torch.ones(4, device="npu:0"); print("NPU_ALLOC_OK", float(x.sum().item()))
except Exception as e:
    print("NPU_ALLOC_FAIL", type(e).__name__, str(e)[:200])
PY
echo "--- 运行算子二进制 ---"
BIN=/ws/ops/bid_elementwise_sub_20261101/build/bid_elementwise_sub_20261101
mkdir -p /tmp/probe
python3 - <<PY
import numpy as np
np.array([1.0],dtype=np.float32).tofile("/tmp/probe/a.bin")
np.array([0.25],dtype=np.float32).tofile("/tmp/probe/b.bin")
PY
set +e
$BIN --n-a 1 --n-b 1 --dtype float32 --in-a /tmp/probe/a.bin --in-b /tmp/probe/b.bin --out /tmp/probe/out.bin
echo "BIN_RC=$?"
set -e
python3 -c "
import numpy as np
try:
    print(\"OUT\", np.fromfile(\"/tmp/probe/out.bin\",dtype=np.float32).tolist())
except Exception as e: print(\"OUT_ERR\",e)"
echo "--- 再试合法设备 4 ---"
'
echo "EXIT=$?"
docker rm -f "$NAME" >/dev/null 2>&1 || true
