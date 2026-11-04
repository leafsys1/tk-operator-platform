#!/usr/bin/env bash
# 四动作契约之一：profile（性能采集）。
# 契约：msopprof/msprof 采单算子；指标四类（耗时/搬运/存储/计算单元负载）每项标
# measured|estimated|unsupported|failed 并记来源；采不到就明确标，不按已采集验收。
# 指标支持范围待真机验证（D9）。TODO(P1.5-a)：现在故意非 0 退出。
set -euo pipefail
echo "[profile] NOT IMPLEMENTED —— 工作区根的 profile.sh 还没实现"
exit 2
