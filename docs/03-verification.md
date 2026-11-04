# 已验证到什么程度（证据，不是承诺）

本文件只记**真跑出来的**结果。凡是"推断"/"未测"都单独标注，不要把这里当 SLA。

## 1. 当前已通过（2026-11-03 ~ 11-04 实测）

### 1.1 算子开发平台 —— 网页前端全流程，卡 4 真跑

全部动作在浏览器上点击完成（`tools/ui_execute_step.py` 可复现），不是 API 直调。

| 动作 | run | 结果 | 关键证据 |
|---|---|---|---|
| 新建 + 需求澄清 | job 27/28/29 | ok | 两轮真 LLM 追问后规格齐备，定版 v2.0 |
| 用例组确认 | job 30/32 | ok | `cases_confirmed_at` 落库，`cases@83713d1950aa6f5c` |
| ① 编译 build | #29 | `gate=pass` | `exit 0`、device=4、`OPX_LOCK_OK devices=4 mode=exclusive` |
| ② 冒烟 smoke | #25 | `SMOKE_VERDICT PASS` | `max_abs_err=0.0`、`matched_ratio=1.0` |
| ③ 正确性 test | #26 | `TEST_VERDICT PASS` | `{PASS:22, FAIL:0, ERROR:0, INVALID:0, TOTAL:22}`，误差全 0 |
| ④ 基准 benchmark | #27 | `gate=pass` | 3 用例 msopprof 设备侧实测（见下表） |
| ⑤ 性能采集 profile | #28 | `gate=pass` | 13 项指标 measured |
| ⑥ 交付包 delivery | #30 | HTTP 201 | `missing=[]`，sha256 `5b9d6ce5b7d4…`，58,902 B |

**基准真值**（msopprof 设备侧 Task Duration，5 次中位数；同一把尺子量基线与候选）：

| 用例 | N | 基线 torch.sub | 候选 kernel | 加速比 | 基线带宽 | 候选带宽 |
|---|---|---|---|---|---|---|
| bench_4k | 4,096 | 2.060 us | 2.720 us | 0.757× | 23.9 GB/s | 18.1 GB/s |
| bench_64k（主） | 65,536 | 3.680 us | 5.740 us | **0.641×** | 213.7 GB/s | 137.0 GB/s |
| bench_1m | 1,048,576 | 8.500 us | 8.380 us | 1.014× | 1480.3 GB/s | 1501.5 GB/s |

**结论**：候选算子在 1M 规模才追平，小规模因 `block_dim=8` 与启动开销落后。
这是真实测量，判据档 `ops-precision-standard@1` **不设加速比硬线**——不要为了让数字好看去调判据。

**采集真值**（run #28）：`op_time_us=5.1401`、`gm_to_ub_bw=3.777 GB/s`、`mte2_ratio=0.555`、
`mte3_ratio=0.308`、`vec_ratio=0.0254`、`l2_hit=2.67%`、`cube_ratio=unsupported`（vector 算子无 cube）。

**交付包内容**（解开数过，19 个文件）：算子源码 7 个（`.asc`/`.h`）+ `CMakeLists.txt` + `cases.json` +
`benchmark.json` + `golden.py` + `design.md`(23KB) + `evidence/correctness.json` + `evidence/performance.json` +
`manifest.json` + 测试脚本。sha256 本地复核一致。

### 1.2 「真在 NPU 上跑」的判据（负例/正例对照）

不靠 `npu-smi` 抓进程（时间太短抓不到），靠对照：

| | 设置 | 结果 |
|---|---|---|
| 负例 | `ASCEND_RT_VISIBLE_DEVICES=99` | `npu_device_count=0`、`aclrtSetDevice failed ret=107001`、`BIN_RC=1`、无输出 |
| 正例 | `=4`（持 dev4 锁） | `npu_device_count=1`、`NPU_ALLOC_OK 4.0`、`OK n=3 blocks=1 cores=48`、输出与期望一致 |

→ 证明不是 CPU 兜底。脚本：`tools/npu_device_discriminator.sh`、`tools/npu_device_positive.sh`。

### 1.3 模型迁移平台 —— 从融合入口页面提交，真跑

`job-20261103-140712-c57e87`（Qwen2.5-0.5B-Instruct），全程在页面上操作：

```
22:07:12  任务创建 → resolve ok
22:07:13  命中已验证配方 transformers.AutoModel（历史 18.6s）→ fast route
22:07:15  语料目录 …/Qwen__Qwen2.5-0.5B-Instruct，0.9 GB
22:07:15  已分配 NPU 卡 4（共 8 张，独占使用）
22:07:45  NPU 兼容层就绪
22:07:54  模型加载成功
22:07:58  端到端加载验证通过 → probe.json 已生成
22:08:34  性能测试完成：OK（P50/P95/P99 实测）
         权重镜像构建成功 → model-corpus-weights:qwen__qwen2.5-0.5b-instruct（tar 795 MB）
```

**卡住的唯一一步**：`obsutil cp` 上传 OBS 失败，`obssdk.log` 实测
`use of closed network connection` / `EOF` / `context deadline exceeded`。

根因**不在本系统**：算力机 root 的 `HTTPS_PROXY=http://127.0.0.1:17891` 那条 sshd 隧道已死
（`curl -x http://127.0.0.1:17891 http://example.com` 5 秒 0 字节超时，直连
`https://61.241.77.58/` 也拿不到响应）。对照证据：同日 07:20 同模型同路径
`权重镜像：OK 759 MB` + `交付包已生成（10 个文件）` 成功过。

→ 属于**外部基础设施故障**，不是迁移引入。修法与影响见 `docs/04-known-issues.md` 第 2 条。

### 1.4 环境隔离

| 项 | 实测 |
|---|---|
| 卡 4–7 | 任务跑完全部释放，`dev4.lock` 可独占 |
| 遗留容器 | 无 `opx-*` |
| 卡 0–3（他人的 VLLM） | PID 947833/947834/970214/970215、显存 56865/56869/55425/55425 MB，**与开跑前逐项一致** |
| 源机 `op-agent-console-20261101.service` | 一直在跑，**未被改动** |

### 1.5 外壳

42 项单测（`shell/tests/`）覆盖：主题注入位置与权重、缓存头、壳页在只读/可写两种状态下都返回
`portal.html`、表格列宽与 nowrap、两平台导航字重一致、模型工作空间块样式、0.0.0.0 绑定必须有显式 CIDR。

浏览器实测：`page_errors=[]`、无 `http>=400`、两个 iframe 同时保活、页签切换正常。

## 2. 未验证（明确标注，不要当成已支持）

| 项 | 状态 | 影响 |
|---|---|---|
| **整机断电重启后自动恢复** | 未测（只测了服务级 `systemctl restart`） | 开机自启由 systemd `enable` 保证，但没实测过冷启动顺序（外壳依赖算子后端先起） |
| **Lucky/反代真实链路** | 未测（只在目标机上用合成 Host 头验过） | 用户侧域名访问、证书、SSE 透传都未验 |
| **模型迁移平台的交付包整体成功** | 未闭合 | 被 OBS 上传隧道挡住，`deliver` 阶段走到"生成交付包"前会降级为 warn |
| **跨平台卡互斥** | 未部署（离线原型 10 tests OK） | 两平台可能同时抢同一张卡，见 known-issues 第 1 条 |
| **模型平台的输入安全** | 未修（候选 26 tests 里 5F+1E，不可部署） | `app.py` 存在命令拼接，只在内网可信输入下可用 |
| **数据切换** | 未做 | 源机仍是权威且在产生新数据 |
| **多并发真机任务** | 未测（一次只跑一个） | `MAX_CARDS=8` 的排队逻辑只在代码层，未压测 |

## 3. 复跑方法

```bash
# 外壳单测
cd shell && python3 -m pytest tests -q          # 42 passed

# 算子后端回归
cd operator/backend && ../../控制venv/bin/python -m pytest -q   # 571 passed（需装好依赖）

# 页面自动化（需要 playwright + chromium）
python3 tools/unified_workbench_check.py        # 壳页结构/两 iframe/无报错
python3 tools/typography_parity_check.py        # 两侧字号/字重逐元素比对
python3 tools/density_check.py                  # 表格列宽/行高/nowrap
python3 tools/ui_execute_step.py 基准测试 500    # 在页面上点某个动作并等它跑完

# NPU 真伪判别
bash tools/npu_device_discriminator.sh          # 负例必须失败
bash tools/npu_device_positive.sh               # 正例必须成功
```

> `tools/*.py` 里的 `BASE = 'http://172.16.111.102:18080'` 与
> `--host-resolver-rules=MAP 172.16.111.102 127.0.0.1` 是配合 SSH 隧道用的，
> 换环境要改这两处（没有做成配置，因为只在验证时用）。
