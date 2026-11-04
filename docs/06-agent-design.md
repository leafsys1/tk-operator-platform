# 模型迁移工作台的 Agent 介入层

> 对应代码：`model-platform/agent_bridge.py`（协议与循环）、`app.py` 的
> `decide_route / AgentSandbox / run_agent`（判定与工具实现）、
> `compute-side/agent_exec.sh`（容器内执行）。
> 本文所有数字均来自 2026-11-04 在算力机 61.241.77.3（910B）上的真机实测。

## 1. 要解决的问题

改之前，模型平台是一条**确定性流水线**：`probe_model.py` 按固定顺序试几条加载路线
（modelscope → transformers → diffusers → speechbrain → 自带框架），全失败就报错结束。
失败的模型只能靠人读日志、改入口、再跑一遍。

现在的规则只有两条：

| 情况 | 走法 | LLM 调用 |
|---|---|---|
| 这套流程**曾经跑通过**，环境没变 | 回放复现包，和以前一模一样地再跑一次 | **0 次** |
| **从来没跑通过**，或者回放失败 | Agent 介入：读证据 → 改入口 → 容器里验算 → 复跑，直到跑通并出适配包 | 有界（≤16 轮） |

## 2. 路线判定（`decide_route()`）

判定发生在**跑探针之前**，结果写进 `jobs.route`，前端任务列表和详情弹窗都显示它。

```
命中 `_probe/<M>/recipe.json` 或 `probe.json` 且 loaded=true ?
├─ 否  → route=agent（无复现包，首次迁移）
└─ 是  → 配方里有 fingerprint.token ?
         ├─ 无（历史配方）→ route=replay，但标注「先回放一次验证」
         └─ 有 → 与 env_fingerprint().token 比对
                  ├─ 一致   → route=replay（LLM 0 次）
                  └─ 不一致 → route=agent（环境已变，配方不可信）
```

**为什么历史配方先回放而不是直接判 agent**：语料库里 111 条历史配方都没有指纹，
一上来全判 agent 会白烧大量 LLM 调用。先回放一次，成功就补齐指纹（自愈），
失败才升级——两种情况下都没有隐瞒：日志里写的是「先按原样回放一次验证」。

**环境指纹** = `探针镜像 ID` + `驱动 version.info` + `npu-smi 板卡固件`，取 sha256 前 16 位。
它只回答一个问题：**这套配方是不是在同一套推理栈上验证的**。

## 3. Agent 能做什么（工具白名单）

| 工具 | 作用 | 边界 |
|---|---|---|
| `list_dir` | 列模型目录文件（含大小） | 只能在模型目录内 |
| `read_text` | 读文本文件（config / 入口 / README） | ≤60KB，只能在模型目录内 |
| `run_py` | **在昇腾容器里跑 python**，模型挂 `/workspace/model`(ro) | ≤40KB 代码，≤600s，拒绝破坏性命令 |
| `write_entry` | 写模型的专用入口 `entry.py` | 必须通过 `compile()`；≤200KB；写前自动备份 |
| `patch_json` | 合并改 config.json 等 | 写前自动备份 |
| `run_probe` | 用当前入口重跑完整探针 | 与平台同一张卡、同一镜像 |
| `finish` | 交结论（ok / summary / root_cause / fix） | ok=true 仍需平台复验 |

`run_py` 是关键：固定流水线只能试预先写死的路线，agent 可以在容器里直接问
「这个类的 forward 签名是什么」「喂这个输入会怎么样」，再把验证过的调用形式写进 `entry.py`。

## 4. 判定口径（不许自证成功）

agent 说「跑通了」**不算**。`run_agent()` 之后平台**自己再跑一次** `_run_probe()`：

```python
ok2, err2, tail2 = _run_probe(...)      # 平台复验
if ok2:  # 只有容器真的打印 VERDICT=LOAD-OK 才算
    agent_res["verified_by_platform"] = True
else:
    agent_res["ok"] = False             # agent 自述成功也判失败
```

`VERDICT=LOAD-OK` 是容器里模型真的加载完、并且跑完一次前向才打印的那一行，
不是平台写的事件。这条是整套设计里唯一的成功判据。

## 5. 跑通之后沉淀什么

1. **复现包**：`_probe/<M>/recipe.json` 增加 `fingerprint`（镜像+驱动+固件 token）、
   `entry_sha16`、`verified_at`、`verified_by_job`、`agent`（轮数/根因/修法/LLM 用量）；
2. **入口落位**：agent 写的 `entry.py` 留在语料目录（下次探针直接走专用入口），
   同时同步进交付包；
3. **适配库**：`_manifest/adaptations.json` 追加 `现象 → 根因 → 修法`，
   下次同模型或同家族可以直接参考；
4. **审计**：`_probe/<M>/agent.jsonl`（逐轮动作）+ 交付包里的 `AGENT_REPORT.md`。

于是同一个模型第二次跑，`decide_route()` 会看到「配方 + 指纹一致 + 入口哈希一致」→ `replay`，
不再调用 agent。

## 6. 真机实测（2026-11-04）

### 6.1 已跑通的模型：零介入回放

| 模型 | 判定 | 结果 |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | `route=replay`（历史配方无指纹 → 回放验证） | ✅ `agent_turns=0 / agent_used=false`，交付包 10 个文件，归档指纹 `7186ebe1ba5f8b4d` |

事件流原文（节选）：

```
[match_recipe/ok] 复现包落位：已把交付包里的专用入口 entry.py 装回语料目录（sha d60a2a419d1cc8d7）
[match_recipe/ok] 路线：复现包回放 —— 命中历史配方（未记录环境指纹）：先按原样回放一次验证，通过后自动补指纹
[adapt/ok]        复现包回放成功（全程未调用 Agent）
[quality/ok]      产物校验：probe.json loaded=true（回同步 5 个文件）
[deliver/ok]      复现包已归档：环境指纹 7186ebe1ba5f8b4d，入口 entry.py sha d60a2a419d1cc8d7
```

### 6.2 从没跑通的模型：Agent 介入并出包

**`microsoft/Florence-2-base`（多模态）** —— 4 轮解决：

| 轮 | 动作 | 结论 |
|---|---|---|
| 1 | `run_py` | 确认 `forward` 真实签名：`input_ids` + `pixel_values`（+ `decoder_input_ids`） |
| 2 | `write_entry` | 按真签名重写入口（4905 B，sha `9db6e9d2470af08e`），fp16 对齐 |
| 3 | `run_probe` | `VERDICT=LOAD-OK`，logits `(1,4,51289)` 全有限值 |
| 4 | `finish` | 交根因/修法 |

平台复验通过 → 交付包 `microsoft-florence-2-base.tar.gz`（9 个文件）→ 配方归档。
**这一条同时补上了 TC-024 里「多模态类没有可复现成功实例」的空缺。**

**`AI-ModelScope/yolo_x_layout`（多模态 · 版面检测）** —— 16 轮：

- 原状态：`probe.json loaded=false`，通用探针只走到 `KeyError: Unknown task document-understanding`；
  历史 `recipe.json` 却记着「torchscript:jit 可用」——典型的「配方命中但重跑跑不通」；
- Agent 判定 `yolox_l.pt` 是 TorchScript 归档，自己写入口 `torch.jit.load` + YOLOX 解码，
  在 NPU 上 3.4s 跑通，解出真实版面框（`Picture 0.122 [0,0,768,775.5]`）；
- 中途还纠正了自己一次：第一版解码把坐标裁成 `[1,1,1,1]`，它没有拿「VERDICT 过了」当成功，
  而是回头验证原始输出通道再改解码。

### 6.3 本轮暴露并修掉的 agent 自身缺陷

| 缺陷 | 现象 | 修法 |
|---|---|---|
| 容器内脚本路径错位 | `run_py` 报 `can't open file /workspace/out/_agent/turn_01.py` | `agent_exec.sh` 里脚本挂载点与写入点对齐 |
| `max_tokens` 太小 | 代码类动作被截断成非法 JSON，连续两轮解析失败 | 默认 1600 → 4000，`MODEL_AGENT_MAX_TOKENS` 可调 |
| 回灌历史用了脱敏参数 | 模型把 `<1150 字符 sha256=…>` 当成真代码原样重发 | 喂回模型的 assistant 消息用**原始参数**，只有落盘 transcript 才脱敏 |
| 轮数上限偏小 | Florence-2 第一次跑到第 10 轮正要复验就被截断 | 默认 10 → 16 |
| 平台 `start.sh stop` 假停 | pid 文件过期 → 旧代码继续服务，改了半天没生效 | `stop` 按端口反查真实监听者 |

前三条都补了回归用例（`model-platform/tests/`，32 条，离线可跑）。

## 7. 配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_AGENT_ENABLED` | `1` | 置 0 则完全不走 agent（失败即失败） |
| `MODEL_AGENT_BASE_URL` | `https://tokens.tiankuan.com/v1` | OpenAI 兼容网关 |
| `MODEL_AGENT_MODEL` | `deepseek-v4.1-flash-1017` | 与算子平台同一个模型 |
| `MODEL_AGENT_KEY_FILE` | `<_platform>/agent.key` | 600 权限；缺省回落到算子平台的 secrets |
| `MODEL_AGENT_MAX_TURNS` | `16` | 单任务最多动作轮数 |
| `MODEL_AGENT_MAX_WALL` | `1800` | 单任务 agent 总时长上限（秒） |
| `MODEL_AGENT_MAX_TOKENS` | `4000` | 单轮输出上限；太小会截断代码 |

`GET /api/agent` 可直接看当前生效配置与 key 是否存在。

## 8. 已知边界（不要当成能自动解决一切）

1. **权重级不兼容救不了**：算子/精度问题（例如 `embedding_symint` 在某些 dtype 下的
   NPU 实现缺陷）不是改入口能解决的，agent 会在 16 轮内放弃并如实报原因；
2. **需要真数据的模型**：要标注文件、要外部权重的模型，agent 只能做出「能加载能前向」的
   最小验证，不等于业务精度可用；
3. **性能测试对入口模型会 SKIP**：`perf.sh` 目前只对通用探针路线有效
   （Florence-2 / yolo 的实测都是 `PERF_VERDICT=SKIP`），这是待办，不是通过；
4. **agent 改动只落在模型目录**：不碰镜像、不碰驱动、不装系统包（缺依赖仍走
   `pip install --target` 的老路径）；
5. **成本**：一次 4 轮介入约 4 次 LLM 调用；16 轮满打满算约 16 次。常态（回放）为 0。
