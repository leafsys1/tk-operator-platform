#!/bin/bash
# 正例对照：同一容器/同一二进制，ASCEND_RT_VISIBLE_DEVICES=4 应成功并输出正确结果。
set -u
IMG=lfx-op-dev-910b:torch210-cann910
WS=/data/lfx/op-agent-console-20261101/app/workspaces/cann-ops
NAME=tk-devprobe-ok-$(date +%s)
SECCOMP=/data/lfx/op-agent-console-20261101/config/docker-18.09-default.json
exec 9>/data/lfx/op-agent/runs/.locks/dev4.lock
if ! flock -n 9; then echo "LOCK_BUSY_STOP"; exit 3; fi
echo "LOCK_ACQUIRED dev4"
cleanup(){ docker rm -f "$NAME" >/dev/null 2>&1 || true; exec 9>&-; echo CLEANED; }
trap cleanup EXIT
if npu-smi info -i 4 -t proc-mem 2>/dev/null | grep -q "Process id:"; then echo "CARD4_BUSY_STOP"; exit 4; fi
docker run --rm --name "$NAME" \
  --privileged --security-opt seccomp="$SECCOMP" \
  -e OPENBLAS_NUM_THREADS=1 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e ASCEND_RT_VISIBLE_DEVICES=4 \
  -v "$WS":/ws -w /ws "$IMG" bash -lc '
echo "VISIBLE=$ASCEND_RT_VISIBLE_DEVICES"
python3 - <<PY
import torch, torch_npu
print("npu_device_count", torch_npu.npu.device_count())
x = torch.ones(4, device="npu:0"); print("NPU_ALLOC_OK", float(x.sum().item()))
PY
BIN=/ws/ops/bid_elementwise_sub_20261101/build/bid_elementwise_sub_20261101
mkdir -p /tmp/probe
python3 - <<PY
import numpy as np
np.array([1.0,3.5,-2.0],dtype=np.float32).tofile("/tmp/probe/a.bin")
np.array([0.25,1.5,0.5],dtype=np.float32).tofile("/tmp/probe/b.bin")
PY
set +e
$BIN --n-a 3 --n-b 3 --dtype float32 --in-a /tmp/probe/a.bin --in-b /tmp/probe/b.bin --out /tmp/probe/out.bin
echo "BIN_RC=$?"
set -e
python3 -c "
import numpy as np
print(\"OUT\", np.fromfile(\"/tmp/probe/out.bin\",dtype=np.float32).tolist())
print(\"EXPECT\", [0.75,2.0,-2.5])"
'
echo "EXIT=$?"
