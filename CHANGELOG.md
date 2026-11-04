# 变更记录

格式：`版本 — 日期 — 一句话`，下面列**影响运维的改动**（改文件/改配置/改行为），
纯文档修改不列。回退点看 git tag。

---

## v0.1.0 — 2026-11-04 — 首个可复现基线

融合系统首次入库：1 个外壳 + 2 个平台 + 算力机侧探针，附部署/运维/验证文档。

**入库时线上状态**（用于核对是否漂移）：

| 文件 | SHA-256 前 16 位 |
|---|---|
| `shell/gateway.py` | `f427b024b4a4a9c9` |
| `shell/portal.html` | `78fb21207384477d` |
| `shell/tanku-logo.png` | `cd75e9b829711b8c` |
| `model-platform/app.py` | `071229de59bf99fc` |
| `model-platform/static/index.html` | `92d190b4ff58feba` |

**组件版本**

- 算子平台后端 `op-agent-console 0.1.0`（FastAPI，143 个 .py）
- 算子平台前端 `op-agent-web 0.1.0`（React + Vite，dist 已入库）
- agent 运行时 `@deepseek-ai/dsh 0.2.0-rc.2`（Node ≥ 24.2）
- 算子平台 LLM `deepseek-v4.1-flash-1017` @ `https://tokens.tiankuan.com/v1`
- 模型平台 `_platform/app.py` 1432 行（纯 stdlib，**无 LLM**）
- 外壳 `gateway.py` 24.5KB（纯 stdlib）
- 依赖锁 `deploy/requirements.lock.txt`（34 项精确版本）

**已验证**（详见 `docs/03-verification.md`）

- 算子平台六动作全通（build/smoke/test 22-22/benchmark/profile/delivery），卡 4 真机
- 交付包 `bid_elementwise_sub_20261101_v2.0_20261103_r30.zip` sha256 `5b9d6ce5b7d4…`
- 模型平台真迁移到卡 4（加载/验证/性能/镜像构建全通），仅 OBS 上传受外部隧道故障阻塞
- 外壳 42 项单测 + 浏览器实测无报错
- 他人卡 0–3 全程未受影响

**已知未闭合**（详见 `docs/04-known-issues.md`）

1. 两平台之间无卡互斥（离线原型 10 tests OK，未接入）
2. 模型平台 OBS 上传走死代理隧道（外部故障）
3. 模型平台命令拼接未净化
4. 入口匿名可写（知情决定，上线前必改）
5. 整机冷启动、反代真实链路、数据切换 均未测

---

## 模板（下次改动用这个）

```
## v0.1.x — YYYY-MM-DD — 一句话

**改了什么**
- 文件 + 行为变化

**为什么改**
- 触发问题（附现象与根因）

**影响面**
- 哪些组件要重启 / 哪些数据要迁移 / 回退是否涉及 schema

**验证**
- 跑了什么，结果如何（数字）

**回退**
- bash deploy/rollback.sh <组件>
```
