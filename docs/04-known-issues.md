# 已知问题与风险（按优先级）

每一条都写清：现象 → 根因（已测到哪一层）→ 影响 → 现在怎么绕 → 正确修法。
**没有标"已证实"的都是候选，不要当结论用。**

---

## 1. 【高】两个平台之间没有卡互斥，可能同时抢同一张卡

**现象**：算子平台锁住卡 4，模型平台在同一时刻扫到卡 4 空闲（`npu-smi` 还没看到容器进程），
两边同时往卡 4 上跑。真发生会看到显存争抢、`aclrtSetDevice` 失败或推理精度异常。

**根因（已证实到代码层）**：

| 平台 | 机制 | 生效范围 |
|---|---|---|
| 算子 | `flock` on `$COMPUTE_OPAGENT_ROOT/runs/.locks/dev<N>.lock` | 跨进程、跨平台（文件系统级） |
| 模型 | `_CARD_BUSY` 字典 + 8 秒 TTL 的 `npu-smi` 缓存 | **仅本进程** |

两者不在同一个命名空间，谈不上互斥。这不是"配置不对"，是设计缺口。

**影响**：中。目前单人使用、一次只跑一个任务时不会触发。多人同时用就会。

**现在怎么绕**：约定同一时间只让一个人在网页上跑真机任务；或在跑之前手工
`flock $COMPUTE_OPAGENT_ROOT/runs/.locks/dev4.lock -c 'npu-smi info'` 确认锁空闲。

**正确修法**（离线原型已验证）：
把模型平台的 `_CARD_BUSY` 换成与算子平台**同一个 inode** 的 `flock`，
并在整个任务生命周期（含 probe + perf 两段）持有同一把锁，不能中途放了再拿。

离线原型 10 个测试已通过（并发互斥、崩溃后残留隔离、inode 漂移 fail-closed、
多设备按序加锁逆序释放、部分失败释放已持有锁…），但**尚未接入两个平台**。
接入前还要解决：锁文件在两台机器？不——两个平台都在同一台通算机上，所以本地 `flock` 就够。

---

## 2. 【高·外部+平台自身记账】模型平台的权重镜像上传：真的成功了，但平台记成"未完成"

**现象**：`deliver` 阶段「构建模型权重镜像并上传 OBS…」卡住，事件不推进；一小时后平台写
`warn 权重镜像未完成（算力服务器命令超时（3600s））`，然后继续生成交付包。

**实测真相（2026-11-04 复核）**——三份证据互相矛盾，以实物为准：

| 证据 | 内容 |
|---|---|
| 平台事件流 | `15:08:34 deliver warn 权重镜像未完成（算力服务器命令超时（3600s））` |
| 算力机 `weights_image.json` | `ok: true, size_mb: 759, obs: obs://model-corpus/images/weights/qwen__qwen2.5-0.5b-instruct.tar.gz, ts: 2026-11-03T15:27:14+0800` |
| **OBS 实物**（`obsutil ls -s`） | `Total size of prefix ... is: 758.46MB, File number: 1` ← **对象确实存在** |

**根因（两层，都已实测）**：

1. **上传慢/卡的直接原因**：算力机 root 的 `HTTPS_PROXY=http://127.0.0.1:17891` 是别人
   （codex 进程）起的 sshd 动态转发隧道。上传期间隧道断过，
   `obssdk.log` 反复 `use of closed network connection` / `EOF` / `context deadline exceeded`，
   obsutil 一直重试（multipart 分片上传，5 片）。隧道恢复后上传在 `15:27:14` 完成。
2. **平台记错的机制原因**：平台用 `npu(..., 3600)` 跑远端命令，**超时只杀掉本地 SSH 子进程，
   不会杀远端 `mk_image.sh`**。远端脚本继续跑完并写了 `weights_image.json`，
   而平台的事件流永远停在"超时 warn"，交付包在 `15:12` 就已生成（早于上传真正完成 15 分钟）。

**影响**：不是功能缺失，是**状态口径不可信**。看到 `warn 权重镜像未完成` 不代表没上传成功，
必须去算力机读 `_probe/<模型>/weights_image.json` 并用 `obsutil ls` 核对象是否存在。
反过来更危险：交付包可能在哪天真的没上传成功时也照样生成（因为只看事件流）。

**现在怎么绕**：判据固定成三步——① 平台事件流 ② 算力机 `weights_image.json` 的 `ok` ③
`obsutil ls -s <obs 路径>` 拿到的字节数。三者一致才算交付完成。

**正确修法**（未做）：

- `mk_image.sh` 里对 `obsutil cp` 显式 `env -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy`
  （或走内网 endpoint 直连），不要继承别人的临时隧道；
- 平台侧把「上传」做成可查询的独立状态（读 `weights_image.json` 而不是靠退出码），
  超时后继续轮询该文件，别把 warn 写死；
- 给 `npu()` 的远端命令加真正的远端超时与清理（`timeout` 包一层 + 结束后 `pkill -f mk_image`），
  否则超时后远端还在跑，既不受控又会被记账成失败。

**注意**：那条隧道被别人的进程共用。**不要 kill 或重启别人的进程**，报告给对应的人。

---

## 2.5 【高·真实】「命中已验证配方」不等于现在能跑：配方没有环境指纹

**现象**：语料库里标为「已迁移」的模型，重新提交迁移却在「验证推理」阶段失败，
事件流先给「命中已验证配方：加载路线 X，依赖 Y，历史耗时 Zs」，紧接着 `模型加载失败`。

**实测（2026-11-04，一次批量提交 11 个模型）**：

| 类别 | 模型 | 结果 | 错误分桶 |
|---|---|---|---|
| nlp | `facebook/esm2_t6_8M_UR50D` | ✅ 成功 | — |
| cv | `OpenGVLab/internimage_t_1k_224` | ✅ 成功 | — |
| audio | `speechbrain/soundchoice-g2p` | ✅ 成功 | — |
| audio | `pyannote/segmentation` | ❌ | 未分类（`HFValidationError: Repo id must be…'/workspace/model'`） |
| multi-modal | `AI-ModelScope/yolo_x_layout` | ❌ | 未分类（`KeyError: Unknown task document-understanding`） |
| multi-modal | `microsoft/Florence-2-base` | ❌ | **C · 算子**（所有前向签名均失败） |
| multi-modal | `AI-ModelScope/PhotoMaker` | ❌ | **E · 模型结构与自定义代码**（`no model_index.json`） |
| scientific | `ai4s/viscosity_of_gas` | ❌ | 未分类（`未实现该框架: unknown`） |
| scientific | `ai4s/MDGEN` | ❌ | 未分类（同上） |
| scientific | `ai4s/HelixFold3` | ❌ | 未分类（同上） |

**根因（已实测到代码层）**：`load_recipes()` 只判断 `probe.json` / `recipe.json` 里
`loaded == true`，就把它当「已验证配方」走 fast route；配方里**不含镜像 digest、依赖版本、
框架版本**，也没有「这条配方是什么时候、在什么环境下验证的」。推理栈或镜像一变，
历史配方就静默失效，而页面仍在日志里宣称「命中已验证配方」。

**影响**：高。会让人误以为「这个模型迁过了、肯定能过」，实际跑起来失败；批量迁移时失败率被低估。
不是并发导致——上表失败项换成单跑同样失败（本次也观察到：并发期间失败的三个任务，
分别拿到不同卡号，失败原因全部是模型侧报错，与卡竞争无关）。

**现在怎么绕**：把「命中配方」只当**加速提示**，不当成功承诺；判据一律以
`validate_infer` 阶段的 `模型加载成功` / `端到端加载验证通过` 为准。

**正确修法**（未做）：
- 配方随记录存环境指纹（镜像 digest、CANN/驱动版本、关键依赖版本、验证时间）；
- fast route 前比对当前环境指纹，不一致则降级为 slow route 并在日志里说明「配方环境已变化，转逐步迁移」；
- 失败诊断里区分「配方过期」与「模型本身不支持」，别都归到「未分类」。

---

## 3. 【高·安全】模型平台有命令拼接，输入未净化

**现象**：`app.py` 里 `download_model()` / `run_migration()` 把用户提交的模型名
拼进 shell 命令（SSH 派发到算力机执行）。构造特殊的模型名可以注入命令。

**已测**：离线候选 `model-security-dev` 26 个测试里 **5 失败 + 1 报错**，不可部署。
另有两个独立审查结论（都是只读审查，未部署）：

- **下载互斥候选不宜部署**：① `app.py:722–736` 用 glob 取模型目录**首个子项**而不是下载返回的
  精确源路径，可能把不完整模型判为可用（离线 mock 已复现）；② 锁只覆盖下载、不覆盖
  "已有目录被发现使用"，别的任务能读到半成品；③ 异常时 `finally` 只释放占位，
  不保证远端下载进程真停了。
- **旧配方身份**：96 条已加载配方中 **51 条没有显式 `model` 字段**。不能用目录名或
  registry 唯一匹配去"猜"身份并放行——那等于把猜测当授权。必须把
  "旧配方展示" 与 "执行授权" 拆开：可展示，不可据此执行。

**影响**：只在内网可信输入下可用。当前入口是**匿名可写**，等于把注入面暴露给整个
`172.16.111.0/24`。

**现在怎么绕**：收紧 `ALLOWED_CIDRS` 到具体机器；或临时把 `WORKBENCH_MODEL_READONLY=1`。

**正确修法**：模型名先按白名单正则校验（`^[\w.-]+/[\w.-]+$`），再 `shlex.quote`；
远端路径用 `realpath` 校验必须落在 `COMPUTE_MODEL_ROOT` 内。
（这条和网关层的 403/400 校验是两件事，网关挡不住已授权的内部人。）

---

## 4. 【中·安全】入口匿名可写可执行

**现象**：`WORKBENCH_AUTH_REQUIRED=0`、`PREVIEW_GATE=0`、两侧 `*_READONLY=0`。
内网任何人可以：提交真机算子任务（起 `--privileged` 容器）、提交模型迁移任务。

**为什么现在这样**：用户明确要求先跑通流程、鉴权自己用反代做。这是**知情决定**，不是疏忽。

**上线前必做**（按顺序）：

1. 反代 + 域名 + 证书，反代侧做认证。
2. 打开网关 Basic auth 作为兜底：
   `WORKBENCH_AUTH_REQUIRED=1`，凭据文件 `$SHELL_ROOT/access.txt`（网关自读）。
3. 收紧 `ALLOWED_CIDRS`，把 `ALLOWED_HOSTS` 从 `*` 改成实际域名。
4. 改完跑 `bash deploy/verify.sh`，确认匿名 GET 变 401 而不是页面打不开。

---

## 5. 【中】`--privileged` 容器是必须的，不是配错了

**现象**：去掉 `--privileged` 后容器内 `npu-smi` 报
`dcmi module initialize failed ret=-8005`，容器 exit 187。

**根因**：宿主驱动 26.0.rc1 在非特权设备直通容器里拒绝 ACL init。
（另有一处历史注释写 `-8020 "device is used"`，是同一类行为的另一个错误码。）

**影响**：任务容器等于宿主机 root 权限。配合第 4 条（匿名可写）风险叠加。

**不要做**：不要为了"降低风险"去掉 `--privileged`，那会让所有任务失败。要降风险就收紧入口。

---

## 6. 【中】设备映射曾经错位（已修，但要知道怎么查）

**历史现象**：锁了卡 4，容器里却挂载 `/dev/davinci2`。任务能跑但用的是错的卡。

**根因**：`ops/nm.conf` 的 `OPX_DOCKER_ARGS` 里写死了 `--device=/dev/davinci2`，
而卡号实际由 `ASCEND_RT_VISIBLE_DEVICES=<锁到的卡>` 决定，两者打架。

**已修**：从 `nm.conf` 删掉 `--device=/dev/davinciN`，只保留 3 个管理设备
（`davinci_manager` / `devmm_svm` / `hisi_hdc`）。

**怎么自查**：跑完任务看 run 目录的 `docker_cmd.txt`：

```bash
grep -o 'ASCEND_RT_VISIBLE_DEVICES=[0-9]*' $OPERATOR_ROOT/app/runs/<run-id>/docker_cmd.txt
grep -c 'device=/dev/davinci[0-9]' $OPERATOR_ROOT/app/runs/<run-id>/docker_cmd.txt   # 必须是 0
```

---

## 7. 【低】算子页有 GET 请求会写文件

`GET /operator/api/proto` 会写文件（生成 proto 定义），被 `OPERATOR_READONLY=1` 时单独拦掉。
`GET /api/capacity` 会 SSH 到算力机跑 `npu-smi`（有副作用但是只读查询），
`GET /operator/api/jobs` 会碰数据库 WAL。

这三个是**已知副作用**，做只读发布时不要以为 GET 就一定安全。

---

## 8. 【低】任务 #22 是一次失败的历史记录，别当 bug 修

`r4-20261103-124331` 的 `smoke` 是 `failed`。原因是当初给的规格（支持广播、空张量返回空）
与算子二进制实现（`n_min=1`、不支持广播、空张量返回 `REJECT_*`）冲突。

**处理方式是改规格而不是改算子**——以现有实现为准，产出 v2.0 规格并重跑。这条 run 保留作为证据。
若以后有人跑同类算子又失败，先核对"规格口径 vs 二进制口径"，不要先怀疑环境。

---

## 9. 【低】`gate` 接口的 `checks=0` 不代表没检查

`/api/runs/<id>/gate` 返回 `{"checks":0,"fail":0,"warn":0}` 是**门禁项统计为空**，
不等于"全部检查通过"。判据要看 `verdict` 字段和 run 目录里的机读报告
（`correctness.json` / `performance.json` / `metrics.json`）。

---

## 10. 【低】交付包门禁会因"版本证据过期"拒收

**现象**：`POST /operator/api/tasks/<id>/deliveries` 返回
`409 E_VERSION_EVIDENCE_STALE`，提示「编译（没有与当前规格/算子代码/用例集一致且门禁通过的证据）」。

**根因（正常行为）**：编译证据是旧规格版本跑的 run。改了规格/用例后必须重跑 build，
否则交付的代码与验证过的代码不是同一份。

**做法**：按提示重跑 build 出新的 run，再提交付。**不要绕过门禁**。

---

## 11. 【低】`access.txt` 与凭据文件

`WORKBENCH_CREDENTIAL_FILE` 指向 `$SHELL_ROOT/access.txt`（42 字节，`0600`）。
`AUTH_REQUIRED=0` 时它不被读，但文件留着。里面是明文 `user:pass`。

**本仓库不含该文件**。换机器时由部署者自己创建：

```bash
umask 077; printf 'admin:%s' "$(openssl rand -base64 18)" > "$SHELL_ROOT/access.txt"
```

---

## 12. 【提示】模型平台的"agent"是假的，别按 agent 方式排查

它没有 LLM、没有工具调用、没有会话。看到"分析中/适配中"是 `emit()` 打的事件。
要看真实进展只能看 `events` 表和算力机上的进程：

```bash
sqlite3 $MODEL_ROOT/_platform/platform.db "select seq,phase,status,message from events where job_id='<id>' order by seq;"
ssh ... 'docker ps | grep opx; ps -ef | grep -E "probe|mk_image"'
```

要它具备真正的 agent 能力（自动读错误日志给根因、自动改写配方），
**当前代码里没有任何基础，得新写**，不是改配置能出来的。
