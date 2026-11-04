# CANNBot

[![npm](https://img.shields.io/npm/v/%40cannbot-plugin%2Fcannbot?style=flat-square&label=npm)](https://www.npmjs.com/package/@cannbot-plugin/cannbot)

## 🚀 概述

CANNBot 是 [CANN](https://hiascend.com/software/cann) 社区的 Infra 智能体层，用 Agent 完成 AscendC/PyPTO/TileLang/Triton 等各类语言的算子开发、模型迁移与推理优化，并延伸至图模式、Runtime 等更多 CANN 开发场景。

本仓（cannbot）是其应用仓与用户入口，提供工作流编排与场景插件；仓群还包括 [cannbot-skills](https://gitcode.com/cann/cannbot-skills)、[cannbot-knowledge](https://gitcode.com/cann/cannbot-knowledge)、[cannbot-dsl](https://gitcode.com/cann/cannbot-dsl)、[cann-bench](https://gitcode.com/cann/cann-bench)、[cannbot-sentry](https://gitcode.com/cann/cannbot-sentry) 等仓库，结构如下。

![CANNBot 仓群结构](docs/figures/cannbot-repo-map.png)

## ⚡ 快速开始

### npm 安装

```bash
npx @cannbot-plugin/cannbot@latest install ${plugin} --tool opencode
```

### 源码安装

```bash
git clone --recurse-submodules --shallow-submodules https://gitcode.com/cann/cannbot.git
cd cannbot/plugins-official/${plugin}
bash init.sh
```

`${plugin}`：插件名称，见[官方插件](#-官方插件)，例如 `ops-direct-invoke`。

## 📦 官方插件

| 插件 | 能力 |
|------|------|
| [`ops-direct-invoke`](plugins-official/ops-direct-invoke/) | Ascend C 直调算子开发：以 `<<<>>>` 直接调用核函数的方式接入 |
| [`ops-registry-invoke`](plugins-official/ops-registry-invoke/) | Ascend C 注册算子开发：经 ACLNN / GEIR 接入框架，以标准算子形式调用 |
| [`model-infer-optimize`](plugins-official/model-infer-optimize/) | NPU 模型迁移、精度对齐与推理性能优化 |
| [`hccl-op-generator`](plugins-official/hccl-op-generator/) | HCCL AICPU 标准通信算子开发，四 Agent 状态机工作流 |
| [`triton-op-generator`](plugins-official/triton-op-generator/) | Triton-Ascend 算子代码生成与优化（Phase 0-7 直接执行模式），知识卡片由 cannbot-knowledge 知识库供给 |

## 🧩 社区插件

社区插件属实验孵化性质，清单与完整介绍见 [社区插件介绍](docs/community-plugins.md)。

## 📖 文档

- [插件安装与工作流](script/docs/cannbot-workflows.md)
- [社区插件介绍](docs/community-plugins.md)
- [仓库架构与维护](docs/repository-guide.md)
- [插件目录说明](plugins-official/README.md)
- [npm 构建与发布](script/README.md)

## 💬 相关信息

- [问题反馈](https://gitcode.com/cann/cannbot/issues)
- [许可证](LICENSE)
