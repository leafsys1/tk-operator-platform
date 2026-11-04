#!/usr/bin/env bash
# 四动作契约之一：build（编译）。
# 约定（方案 v1.1 §6）：统一执行入口在容器里调用工作区根的 <动作>.sh，
# 产物写到 /out，日志由入口脚本 tee 到 /out/run.log。
# TODO(P1.5-a)：由交付约定的构建脚本替代 —— 现在故意以非 0 退出，避免「看起来跑过了」。
set -euo pipefail
echo "[build] NOT IMPLEMENTED —— 工作区根的 build.sh 还没实现"
echo "[build] 契约：容器内编译本算子的 kernel/接口，日志进 /out/run.log"
exit 2
