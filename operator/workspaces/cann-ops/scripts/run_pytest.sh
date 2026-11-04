#!/usr/bin/env bash
# 受控 pytest + 独立示例执行入口（TC-004）。
#
# 在任务镜像的容器内运行 tests/test_<op>.py 和 examples/<op>_demo.py：
#   - 自动带上 py 依赖（/opt/opx-pydeps），避免镜像内 pytest 缺 py 时收集失败；
#   - 缺 build 产物时先在容器内 build.sh；
#   - 固定任务镜像、设备、CANN 挂载和 seccomp 参数；
#   - 日志/JUnit 落到 /data/lfx/op-agent-console-20261101/pytest-runs/<timestamp>-<op>/。
#
# 用法：
#   scripts/run_pytest.sh --op bid_elementwise_sub_review_01 --device 4
set -o pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OP=""
DEVICE="${OPX_DEVICE:-2}"
IMAGE="${OPX_IMAGE:-lfx-op-dev-910b:torch210-cann910}"
while [ $# -gt 0 ]; do
  case "$1" in
    --op) OP="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done
[ -n "$OP" ] || { echo "必须指定 --op <算子名>" >&2; exit 2; }
[ -f "$ROOT/tests/test_${OP}.py" ] || { echo "缺测试脚本：tests/test_${OP}.py" >&2; exit 2; }
[ -f "$ROOT/examples/${OP}_demo.py" ] || { echo "缺独立示例：examples/${OP}_demo.py" >&2; exit 2; }

OUT="/data/lfx/op-agent-console-20261101/pytest-runs/$(date +%Y%m%d-%H%M%S)-${OP}"
PYDEPS="/data/lfx/op-agent-console-20261101/pydeps"
SECCOMP="/data/lfx/op-agent-console-20261101/config/docker-18.09-default.json"
mkdir -p "$OUT"
if [ ! -d "$PYDEPS/py" ]; then
  echo "[run_pytest] pydeps 缺 py，尝试安装 py==1.11.0"
  mkdir -p "$PYDEPS"
  python3 -m pip install --quiet --target "$PYDEPS" py==1.11.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple || {
      echo "安装 py 失败；请手动把 py 放到 $PYDEPS，或使用自带 py 的镜像" >&2
      exit 2
    }
fi

cat > "$OUT/run.sh" <<'EOS'
set -o pipefail
OP="${OP:?}"
for f in /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/*/share/info/ascendnpu-ir/bin/set_env.sh \
         /usr/local/Ascend/nnal/atb/set_env.sh; do
  [ -f "$f" ] && . "$f" >/dev/null 2>&1 || true
done
cd /ws || exit 2
if [ ! -x "/ws/ops/$OP/build/$OP" ] || [ ! -f "/ws/ops/$OP/build/lib${OP}_host.so" ]; then
  echo "[run_pytest] 缺构建产物，先在容器内 build.sh"
  OPX_OP="$OP" OPX_WS=/ws bash /ws/build.sh || exit 2
fi
python3 -m pytest "tests/test_${OP}.py" -v -s --junitxml=/out/junit.xml 2>&1 | tee /out/pytest.log
rc=${PIPESTATUS[0]}
echo "PYTEST_RC=$rc"
python3 "/ws/examples/${OP}_demo.py" 2>&1 | tee /out/demo.log
drc=${PIPESTATUS[0]}
echo "DEMO_RC=$drc"
[ "$rc" = "0" ] && [ "$drc" = "0" ]
EOS
chmod +x "$OUT/run.sh"

echo "[run_pytest] op=$OP device=$DEVICE image=$IMAGE"
echo "[run_pytest] out=$OUT"
docker run --rm --name "opx-pytest-$$" \
  --privileged --security-opt "seccomp=$SECCOMP" \
  -e OPENBLAS_NUM_THREADS=1 -e "ASCEND_RT_VISIBLE_DEVICES=$DEVICE" \
  -e PYTHONPATH=/opt/opx-pydeps -e "OP=$OP" \
  -v "$PYDEPS:/opt/opx-pydeps:ro" \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$ROOT:/ws" -v "$OUT:/out" -v "$OUT/run.sh:/run.sh:ro" \
  -w /ws --entrypoint bash "$IMAGE" -lc 'bash /run.sh'
rc=$?
echo "[run_pytest] container_rc=$rc"
echo "[run_pytest] pytest.log=$OUT/pytest.log junit=$OUT/junit.xml demo.log=$OUT/demo.log"
exit "$rc"
