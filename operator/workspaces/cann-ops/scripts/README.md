# scripts/（工作区本地工具）

## `nm.sh` —— 真机验证入口（转发到控制台仓库根 `scripts/nm_test.sh`）

`nm_test.sh`、`ops/nm.conf`（SSH / 镜像 / 设备锁 / 台账回填）都在**控制台仓库根**，不在本工作区；
`nm.sh` 是给工作区里 agent 用的入口（`exec` 原脚本，规则完全一致）。

```bash
scripts/nm.sh --dry-run --op elementwise_mul --action build   # 先看计划，不动服务器
scripts/nm.sh --op elementwise_mul --action build             # 上机编译
scripts/nm.sh --op elementwise_mul --action test              # 上机正确性测试
scripts/nm.sh --op elementwise_mul --action benchmark         # 基准（独占卡，慢）
scripts/nm.sh --op elementwise_mul --action profile           # 性能采集（独占卡，慢）
scripts/nm.sh --task 11 --loop 6 --op elementwise_mul --action benchmark # 归属 Loop + 预算审计
```

- 远端设备由脚本自行加锁（flock），同一张卡不会并发跑；
- 每次执行会向控制台登记 run（`OPX_API=http://127.0.0.1:8077`），门禁/结果在「概览 → 四动作」与「测试与结果」可见；
- 不要与「概览 → 四动作」同时触发同一动作（会排队/等待锁）。
