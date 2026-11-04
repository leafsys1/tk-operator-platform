# 文档索引

按「接手的人第一次要什么」排序，建议顺序读 01 → 02 → 04。

| 文件 | 解决什么问题 | 什么时候看 |
|---|---|---|
| [`01-architecture.md`](01-architecture.md) | 谁在哪台机器上跑什么；外壳做了哪四件事；**两个平台的本质差别**（一个有 agent 一个没有） | 第一次接手、排查"为什么两个平台行为不一样" |
| [`02-deploy.md`](02-deploy.md) | 从零在一台等价机器上装起来；只改一个 `site.env`；大模型端点怎么配怎么验 | 新环境部署、换机器 |
| [`03-verification.md`](03-verification.md) | **已经验到什么程度**（真数字）+ **明确没验的清单** | 交付前对账、别把"没测"当"已支持" |
| [`04-known-issues.md`](04-known-issues.md) | 12 条已知问题：现象 → 根因 → 影响 → 现在怎么绕 → 正确修法 | 遇到怪现象先查这里，别重复踩 |
| [`05-operations.md`](05-operations.md) | 日常命令、故障定位顺序、升级/回退/备份/数据切换、**严禁的操作** | 值班、升级、出事 |

仓库根目录还有：

- `README.md` —— 架构总览 + 快速开始 + 版本基线指纹
- `CHANGELOG.md` —— 每次改动的记录模板与当前版本说明
- `LEDGER.md` —— 施工台账：每一次真机验证的原始读数、失败与修正过程（排查历史问题时最有价值）

## 代码在哪

| 想改什么 | 改哪个文件 |
|---|---|
| 统一视觉（颜色/字号/表格密度/导航） | `shell/gateway.py` 的 `apply_visual_theme()` |
| 路由、写闸、来源限制、缓存策略 | `shell/gateway.py` 的 `route()` / `do_GET()` / `make_server()` |
| 外壳页面（页签、Logo、双 iframe） | `shell/portal.html` |
| 算子平台的业务逻辑 | `operator/backend/app/`（路由在 `routers/`，业务在 `services/`） |
| 算子平台的 agent 调用方式 | `operator/backend/app/services/agent_runner.py` + `llm_config.py` |
| 算子平台的容器启动参数 | `operator/ops/nm.conf` |
| 算子平台前端 | `operator/web/src/`，改完必须 `npm run build` 并把 `dist/` 一起提交 |
| 模型迁移平台的流水线 | `model-platform/app.py` 的 `run_migration()` |
| 模型平台的远端探针 | `compute-side/_probe/*.sh` |

## 三条最容易搞错的事

1. **模型迁移平台没有 agent**。它不调用任何 LLM，`app.py` 里连 HTTP 客户端都没 import。
   页面上看到的"分析/适配"是 `emit()` 发的流水线事件。要给它加真 agent 得新写，不是改配置。
2. **视觉统一是网关注入 CSS，没改业务源码**。所以调样式只动 `shell/gateway.py`，
   改坏了 `rollback.sh gateway` 一条命令回去。别去改 React 源码调颜色。
3. **回退代码 ≠ 回退数据**。算子平台的数据库 schema 不随 tag 回退，
   跨 migration 的回退要手工 `alembic downgrade`。
