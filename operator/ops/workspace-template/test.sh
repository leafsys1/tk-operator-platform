#!/usr/bin/env bash
# 四动作契约之一：test（正确性测试）。
# 契约：容器里跑用例集（常规/边界/异常），与**独立**参考实现对比，逐用例结果与
# 正确性报告写到 /out（correctness.json + correctness.md），日志 /out/run.log。
# TODO(P1.5-a)：接上参考实现与用例矩阵 —— 现在故意非 0 退出。
set -euo pipefail
echo "[test] NOT IMPLEMENTED —— 工作区根的 test.sh 还没实现"
echo "[test] 契约：用例矩阵(常规/边界/异常) + 独立参考实现 → /out/correctness.{json,md}"
exit 2
