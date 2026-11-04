# cann-ops 工作区（演示用）

这是控制台里「CANN 算子开发」项目的 agent 工作区。dsh 会在这个目录里改代码；
真机验证走工作区里的 `scripts/nm.sh`（转发控制台仓库根的 `scripts/nm_test.sh`：rsync + 容器 + 回读产物 +
远端设备锁 + 台账回填）

```bash
scripts/nm.sh --dry-run --op <op> --action build   # 先看计划（不动容器）
scripts/nm.sh --op <op> --action build            # 上机编译
scripts/nm.sh --op <op> --action smoke            # 冒烟：编译 + 一条最小用例跑通（开发内自检）
scripts/nm.sh --op <op> --action test             # 上机正确性测试
scripts/nm.sh --op <op> --action benchmark|profile # 基准 / 采集（独占卡，慢）
```

当前内容：`ops/elementwise_add/`（Ascend C 直调实现）、`cases/`（用例矩阵 + 基准定义）、
`criteria/`（判据档）、`tools/`（用例/基准/采集执行器）、四个动作入口 `build.sh`/`test.sh`/
`benchmark.sh`/`profile.sh` + 冒烟入口 `smoke_check.sh`（根目录 `smoke.sh` 是环境自检，不是动作） —— 全部已入仓（`.dsh/` `.cannbot/` `runs/` `**/build/` 除外）。
