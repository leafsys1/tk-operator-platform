# 天宽科技 · 融合工作台（1 个壳 + 2 个平台）

两个产品共用一个 Web 入口的融合部署：一个统一外壳（shell）反向代理两套独立平台，
业务功能各自保留，视觉与导航在壳层统一。

```
                    ┌──────────────────────────────────────────────┐
   浏览器 ──────────▶│  shell / gateway      0.0.0.0:18080          │
   (内网)            │  ├─ GET /            → portal.html（双 iframe 壳）│
                    │  ├─ /brand/tanku.png → 公司 Logo              │
                    │  ├─ /operator/*      → 127.0.0.1:18078       │
                    │  └─ /model/*         → 127.0.0.1:8090        │
                    └───────────────┬──────────────┬───────────────┘
                                    │              │
                    ┌───────────────▼───┐   ┌──────▼────────────────┐
                    │ 算子开发平台        │   │ 模型迁移工作台          │
                    │ FastAPI 18078      │   │ http.server 8090       │
                    │ agent = dsh        │   │ 无 LLM，确定性流水线     │
                    └────────┬───────────┘   └──────┬────────────────┘
                             │  SSH 派发             │  SSH 派发
                             └──────────┬───────────┘
                                        ▼
                       算力服务器 昇腾 910B ×8
                       /data/lfx/op-agent      ← 算子容器任务 + 共享设备锁
                       /data/model_corpus      ← 模型权重 + 探针脚本
```

## 三个组件的本质差别（先看这个再看代码）

| | 算子开发平台 | 模型迁移工作台 | shell |
|---|---|---|---|
| 语言/框架 | Python FastAPI + React | 纯 Python `http.server` | 纯 Python `http.server` |
| 有 agent 吗 | **有**，`@deepseek-ai/dsh` 0.2.0-rc.2 | **没有**，一行 LLM 代码都没有 | 没有 |
| 大模型 | `deepseek-v4.1-flash-1017`（天宽网关） | 不调用任何 LLM | 不调用 |
| "智能"从哪来 | 真 LLM + 工具调用 + 技能 | 配方库 + 坑库关键词打分 + 远端真推理 | — |
| 端口 | 127.0.0.1:18078 | 127.0.0.1:8090 | 0.0.0.0:18080 |

> 模型迁移平台的“agent 感”（解析→匹配配方→适配→验证→交付）全部来自 `emit()` 发出的
> 流水线事件，不是模型生成。全仓 grep `openai|anthropic|chat/completions` 命中 0，
> 连 HTTP 客户端都没有 import —— 详见 `docs/01-architecture.md`。

## 目录结构

```
shell/                  统一外壳（网关 + 页面 + 测试 + systemd 单元）
  gateway.py            ★ 全部路由/主题注入/写闸/缓存控制都在这里
  portal.html           双 iframe 壳（64px 页签轨 + 118×29 Logo）
  tanku-logo.png        公司 Logo（180×44 RGBA 源图，显示 118×29）
  tests/                42 项单元测试
  deploy/               tk-gateway-preview.service

operator/               算子开发平台（同事产品，此处为可复现快照）
  backend/              FastAPI 后端（143 个 .py）
  web/                  React 前端源码（30 个 ts/tsx）+ dist 构建产物
  workspaces/cann-ops/  算子工作区模板（脚本 + 用例 + 判据 + 5 个算子实现）
  ops/                  nm.conf（容器启动参数）、dsh patch、workspace 模板
  scripts/              nm_test.sh（远端执行总入口）、cannbot 等
  docs/                 平台自身设计文档（gate-report-v1、impl-contract-v2 等）
  config/               docker-18.09-default.json（seccomp profile）
  deploy/               systemd 单元 + 环境变量模板

model-platform/         模型迁移工作台（同事产品，此处为可复现快照）
  app.py                单文件后端，1432 行，仅 stdlib
  static/               单页前端
  start.sh              启停脚本
  deploy/               systemd 单元 + 看门狗

compute-side/_probe/    算力机侧探针脚本（probe.sh / mk_image.sh / npu_shim.py …）

tools/                  端到端验证脚本（Playwright UI 驱动 + NPU 真伪判别）
docs/                   部署、回滚、复现、已知问题
LEDGER.md               施工台账（含每一次真机验证的原始结论）
```

## 快速开始（在一台全新的等价服务器上）

前提：Ubuntu 22.04+、Python 3.12、能 SSH 到算力机、算力机上已装 Ascend 驱动与 CANN。

```bash
# 0. 克隆 + 改唯一一个配置文件
git clone git@github.com:leafsys1/tk-operator-platform.git
cd tk-operator-platform
cp deploy/site.env.example deploy/site.env    # 改里面的 5 处路径/地址

# 1. 算子平台后端（先起，网关依赖它）
bash deploy/setup-operator.sh

# 2. 模型迁移平台
bash deploy/setup-model-platform.sh

# 3. 统一外壳
bash deploy/setup-shell.sh

# 4. 验收（不 PASS 不要交付）
bash deploy/verify.sh
```

每个脚本都会自检前置条件、跑对应单测、装 systemd unit、回读健康检查；
`verify.sh` 给一张 `PASS/FAIL/SKIP` 汇总表，明细落 `/tmp/tk-verify-*.log`。
详见 `docs/02-deploy.md`。

| 脚本 | 干什么 | 失败时会怎样 |
|---|---|---|
| `deploy/setup-operator.sh` | venv + 34 项锁版依赖 + 后端 571 tests + alembic + unit | 测试不过直接退出，**不启动服务** |
| `deploy/setup-model-platform.sh` | 同步 app.py/static + 推探针到算力机 + unit | 探针推送失败会告警（任务会在远端失败） |
| `deploy/setup-shell.sh` | 装网关/页面/Logo + 改上游端口 + **42 项单测** + 端到端冒烟 | 单测不过直接退出 |
| `deploy/verify.sh` | 进程/端口/回环/壳页/缓存头/两侧 API/端点探测 | FAIL 非零退出 |
| `deploy/rollback.sh` | 回退任意组件 | 回退后自动 curl 验证 |
| `deploy/scan_secrets.py` | 提交前脱敏扫描 | 有敏感内容非零退出 |
| `deploy/selftest-render.sh` | 用假 site.env 渲染三个 unit，检查路径/端口/用户是否都被正确替换 | 有任何一项没替换就失败 |
| `deploy/selftest-run.sh` | 全程 28078/28090/28080 + /tmp，起两个假上游实际验路由 | 路由错就失败 |

### 1.3 部署脚本本身怎么验的（不碰生产）

改端口是最容易「静默失败」的一步（网关上游端口写死在代码里）。所以有两条离线自测：

```bash
bash deploy/selftest-render.sh   # 渲染三个 unit + 改端口，逐项断言 16 项
bash deploy/selftest-run.sh      # 起假上游真跑一遍，验 /model/ 与 /operator/ 路由
```

两条都在 `/tmp` 与 28078/28090/28080 上跑，**不安装 unit、不重启任何服务、不碰生产端口**。
实测结果：render 16 项全过；run 5 项全过（`/model/api/x` → model 上游且前缀被剥掉、
`/operator/api/y` → operator 上游、`/` 返回壳页、未知路径 404）。

> 踩过的坑：最初用 `sed` 改 `("127.0.0.1", 8090)`，而源码里其实是**单引号**
> `('127.0.0.1', 8090)`，sed 一条都没替换成功却只打了 warning。现在改成
> python 正则 + `subn` 计数 + 回读 grep 双重确认，替换不到就 `die`。

## 版本与回退

本仓库**就是回退基线**。每个可上线状态打一个 `v*` tag：

```bash
git tag -l
git log --oneline v0.1.0..HEAD        # 相对某个基线改了什么
git diff v0.1.0 -- shell/gateway.py   # 只看网关改动
```

当前基线 `v0.1.1` 的文件指纹（与生产机实测一致，可用于核对是否漂移）：

| 文件 | SHA-256 前 16 位 |
|---|---|
| `shell/gateway.py` | `824ae2f888a0bf71` |
| `shell/portal.html` | `78fb21207384477d` |
| `shell/tanku-logo.png` | `cd75e9b829711b8c` |
| `model-platform/app.py` | `071229de59bf99fc` |
| `model-platform/static/index.html` | `bfce3ced31ca78dc` |
| `operator/web/dist/index.html` | `4c96ef8a3d56b484` |
| `operator/web/dist/assets/index-Cy1qBUO3.js` | `23c44e4eb522e614` |

回退、灰度、双跑、数据切换的做法见 `docs/05-operations.md`。

## 克隆下来就能验（实测）

在一台干净目录里 clone 之后直接跑（不需要先部署）：

```bash
cd shell && python3 -m pytest tests -q        # 42 passed, 3 subtests (4.4s)
cd ../operator/backend && python -m pytest -q # 571 collected, rc=0
python3 -c "import ast;ast.parse(open('../model-platform/app.py').read())"   # OK, 1432 行
```

需要在算力机上跑测试时，用 `control-venv312` 里的解释器（它带齐了 pytest 与后端依赖）。

## 重要：这不是生产级安全配置

基线里 `WORKBENCH_AUTH_REQUIRED=0`、`PREVIEW_GATE=0`、两侧都**可匿名写**，只靠来源网段
`172.16.111.0/24` 兜底。上公网前必须做鉴权（反代 + 域名 + 证书），见 `docs/04-known-issues.md`
的「上线前必做」清单。仓库里**不含**任何密钥、私钥、密码。
