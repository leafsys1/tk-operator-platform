# JAX 优先算子 / 原语清单（2026-10-26）

## 1. 选择口径

本批次优先覆盖截图中提出的 **JAX 特有能力**，不是把通用神经网络算子改名为 JAX 算子。
JAX 官方接口作为语义真相源；开发前先在现有 910B1 环境验证 JAX runtime / 后端能力，再决定交付形态：

- **框架原语**：验证 `jit`、控制流、自动微分和 pytree 语义，不能伪装成单个 Ascend C kernel；
- **数值算子**：可落成独立 Ascend C 实现，并用 JAX CPU 结果作 oracle；
- **Pallas 内核原语**：先验证当前 JAX/Pallas 后端是否支持 Ascend；不支持时只交付兼容性证据，不宣称 NPU 跑通。

## 2. 第一批 10 项

| # | JAX 接口 | 类型 | 首轮最小验收 | 预期交付形态 | 状态 |
|---:|---|---|---|---|---|
| 1 | `jax.lax.scan` | 可微静态循环 | carry/stack、reverse、`jit`、正反向梯度 | JAX 兼容用例与执行报告 | #13 前置已通过；用例组待人工确认 |
| 2 | `jax.lax.associative_scan` | 并行前缀扫描 | sum/max、reverse、非 2 次幂长度 | JAX 兼容用例与执行报告 | CPU 已穿刺；910B1 后端阻塞 |
| 3 | `jax.lax.cond` | 编译条件分支 | 真/假分支、pytree、`jit`、梯度 | JAX 兼容用例与执行报告 | CPU 已穿刺；910B1 后端阻塞 |
| 4 | `jax.lax.while_loop` | 动态循环 | 动态次数、固定 carry shape、前向梯度边界 | JAX 兼容用例与执行报告 | CPU 已穿刺；910B1 后端阻塞 |
| 5 | `jax.lax.fori_loop` | 计数循环 | 静态/动态边界、unroll、梯度差异 | JAX 兼容用例与执行报告 | CPU 已穿刺；910B1 后端阻塞 |
| 6 | `jax.lax.custom_linear_solve` | 隐式微分线性求解 | primal 残差、transpose solve、VJP | JAX 兼容用例与执行报告 | CPU 已穿刺；910B1 后端阻塞 |
| 7 | `jax.lax.axis_index` | SPMD 分片索引 | 单/多设备轴、`pmap`/mesh 可用性 | 分布式兼容性报告 | 单 CPU 已穿刺；910B1/SPMD 阻塞 |
| 8 | `jax.nn.dot_product_attention` | 数值算子 | MHA、causal、mask、GQA/MQA、梯度 | 优先做 Ascend 算子 + JAX oracle | 可转 Ascend C 正式开发 |
| 9 | Pallas `load` | 内核访存原语 | mask/bounds、GMEM→片上、编译后端 | Pallas 后端兼容性报告 / 最小 kernel | CPU interpret 已穿刺；Ascend 后端阻塞 |
| 10 | Pallas `store` | 内核访存原语 | mask/bounds、片上→GMEM、写回一致性 | Pallas 后端兼容性报告 / 最小 kernel | CPU interpret 已穿刺；Ascend 后端阻塞 |

候补：`jax.lax.custom_root`、`jax.lax.switch`、Pallas atomics。是否替换第一批条目，以穿刺证据为准。

## 3. 与仓库既有 10 案例要求的关系

`docs/PLAN-web-console-v1.1.md` 的 M2 要求是“矩阵 2 / 归一化 3 / 激活 3 / 注意力 2，10/10 真机全链路”。
本清单是用户新指定的 **JAX 优先批次**，先独立推进，不把框架控制流原语冒充 M2 的 Ascend 单算子案例。
其中 `dot_product_attention` 后续可以计入注意力场景；其余条目是否纳入 M2，需在交付形态和真机口径明确后再决定。

## 4. 统一硬要求

1. 语义与当前 JAX 官方文档对齐，记录 JAX/JAXLIB 版本和设备后端；
2. 所有测试固定 seed，保留输入、命令、退出码、结构化结果和版本哈希；
3. 控制流原语分别验证 eager / `jit` / `grad` 支持边界，不把不支持的反向梯度写成通过；
4. 数值算子使用独立高精度 oracle，覆盖常规、边界、异常和非有限值；
5. 真机只认 910B1 容器里的真实执行证据；CPU 跑通只能标为语义基线；
6. Pallas 属实验接口，先穿刺后开发，后端不支持时保留阻塞证据，不做假实现；
7. 任何独立 Ascend 算子仍需遵守 `docs/reference/deliverable-standards.md` 的五件交付物与 build→smoke→test→benchmark 门禁。

## 5. 开发顺序

1. **环境/路线穿刺**：10 项导入、CPU 语义、`jit/grad`、远端镜像与 Pallas/Ascend 后端能力；
2. **框架原语组**：`scan / associative_scan / cond / while_loop / fori_loop`；
3. **微分与分布式组**：`custom_linear_solve / axis_index`；
4. **数值算子组**：`dot_product_attention`（按正式 Ascend 算子流程）；
5. **Pallas 组**：`load / store`，仅在后端穿刺通过后进入正式开发。

## 6. 2026-10-26 技术穿刺结论

标准 `ops-direct-invoke` / `ascendc/feasibility` 工作流已完成三次执行与独立验收。
工作流最终按门禁判定为 `fail`，原因不是本机探针失败，而是目标环境不具备
JAX→Ascend 运行栈，不能形成 910B1 编译、运行和精度证据：

- 本机隔离环境：JAX/JAXLIB `0.12.0.1`，CPU backend；8 项 JAX API 的最小
  eager/jit/grad 或边界断言通过，Pallas Ref 索引只在 CPU `interpret=True` 下验证；
- 目标镜像：`lfx-op-dev-910b:torch210-cann910`，固定 digest 下未安装 JAX、
  JAXLIB 或 JAX Ascend 插件；虽能枚举 8 张 910B1，但 JAX 代表用例实际执行为 `0/10`；
- `scan / associative_scan / cond / while_loop / fori_loop / custom_linear_solve /
  axis_index` 是框架、编译器、自动微分或 SPMD 原语，不按独立 Ascend C kernel 交付；
- Pallas `load/store` 有明确的版本差异：目标 JAX 0.8.1 仍保留已废弃的顶层
  `pl.load/pl.store`，JAX 0.12 已移除并改用 Ref 索引/后端专用入口；两版 CPU
  `interpret=True` 皆已穿刺通过，但本轮仍没有 Ascend Pallas backend/lowering 证据；
- `jax.nn.dot_product_attention` 是本批唯一可继续定义为独立 Ascend C 数值算子的候选，
  可使用 JAX CPU 结果作 oracle，但不能声称已经打通 JAX→Ascend 调用链。

因此下一阶段只推进 `dot_product_attention` 的正式 Ascend C 直调任务；其余 9 项保留
兼容性/阻塞证据，等提供包含兼容 JAX/JAXLIB 和明确支持 910B1 插件的镜像后再重启。
正式代码开发前仍需完成用例组确认并通过 `POST /api/tasks/<task_id>/cases/confirm` 留痕。

## 7. 无卡阶段准备状态（2026-10-26）

### 7.1 公共环境

- 官方 `cann/xla-npu` 已固定到 commit
  `2a55287e4e4f0c8c9c4e80b2b0a53377a758eae0`；运行组合固定为 Python 3.11、
  JAX/JAXLIB 0.8.1、CANN 9.2.0-beta.2、AFIR backend、Ascend910B1；
- 可复现构建配方、版本探针和验收说明已准备在
  `.cannbot/jax_xla_npu/workflow1/`（运行时交付件，按仓库规则不提交 git）；
- 目标版 JAX/JAXLIB 0.8.1 和本机新版 0.12.0.1 都已跑通同一套 CPU 语义探针；
  探针已做版本感知，不再把 Pallas 顶层 API 的废弃/移除误判为 Ascend 能力；
- 远端正在构建镜像 `opx-xla-npu:jax081-cann920-910b`，该步骤只占 CPU/内存/磁盘，
  不占 NPU；镜像完成后先做无设备的文件/版本检查，有卡后才做 backend/device/jit 真机验收；
- Web 任务 #13 是公共环境前置；#12 → #11 → #10 → #9 → #8 → #7 → #6 → #3 →
  #5 → #4 已按依赖顺序建立。除 #13 外保持草稿，不在没有真机证据时伪造阶段进展。

### 7.4 公共环境真机验收（2026-10-26）

- 镜像 `opx-xla-npu:jax081-cann920-910b` 构建完成，最终本地 image id
  `sha256:b0a1aaec9133b8a7614940c74e889aa8abe1af05999f6e630d28eacf4ea6ec41`，revision label
  为 `2a55287e4e4f0c8c9c4e80b2b0a53377a758eae0`；
- 无卡验收通过：Python `3.11.17`、JAX/JAXLIB `0.8.1`、PJRT 插件文件存在，CPU 语义探针通过；
- 第一次设备探针真实失败于缺少 `ASCEND_MLIR_PYTHON_PATH`，未误报通过；补齐官方
  `build/xla_npu_env` 三项运行变量并固化进镜像配置后重跑；
- 设备 0 独占探针通过：`default_backend()=="npu"`、`jax.devices()==[npu:0]`，AFIR 完成三段
  fusion 编译，`jax.jit(x*2+1)` 返回 `[3,5,7,9]`；前后 `npu-smi` 与完整日志保存在远端
  `/data/lfx/op-agent/runs/task13-xla-npu-probe/`；
- 公共前置 #13 判定通过。下一项仅准备 #12 `lax.scan` 用例组，等待用户人工确认后才正式执行。

### 7.2 每项开工输入与验收草案

| 顺序 / 任务 | 无卡阶段已明确的输入 | 有卡后的最小验收 | 卡数 |
|---|---|---|---:|
| 01 / #12 `lax.scan` | carry/stack、reverse、静态 length、CPU oracle、固定 seed | NPU eager/jit；前向结果；可支持的 JVP/VJP 边界 | 1 |
| 02 / #11 `associative_scan` | sum/max、reverse、非 2 次幂长度、CPU oracle | NPU jit；正反向扫描；边界长度 | 1 |
| 03 / #10 `lax.cond` | 真/假分支、同构 pytree、梯度语义 | 两分支 NPU jit；pytree；支持的梯度 | 1 |
| 04 / #9 `while_loop` | 动态次数、固定 carry shape、反向模式不支持边界 | NPU jit 前向；0/1/多轮；明确反向梯度边界 | 1 |
| 05 / #8 `fori_loop` | 静态/动态 bounds、unroll、grad 差异 | NPU jit；静态与动态边界；梯度边界 | 1 |
| 06 / #7 `custom_linear_solve` | primal residual、transpose solve、VJP oracle | NPU jit；残差；隐式微分/VJP | 1 |
| 07 / #6 `axis_index` | axis name、`pmap`/mesh、单设备降级口径 | 2/4 设备轴索引、设备唯一性与聚合一致性 | 2（建议 4） |
| 08 / #3 `dot_product_attention` | MHA/GQA/MQA、causal/mask、dtype/shape、JAX CPU oracle | Ascend C build/smoke/test/benchmark；JAX CPU 精度对比 | 1 |
| 09 / #5 Pallas `load` | Ref 索引、mask/bounds、非法访问拒绝口径 | 仅在 xla-npu 明确提供 Pallas lowering 后验证 | 1 |
| 10 / #4 Pallas `store` | Ref 写回、mask/bounds、未写区域保持 | 仅在 xla-npu 明确提供 Pallas lowering 后验证 | 1 |

### 7.3 开卡后的固定顺序

1. 先完成 #13：镜像 digest、插件文件、`default_backend()=="npu"`、设备枚举、最小
   `jax.jit` 数值用例；失败则停止后续 9 个兼容任务并保留环境证据；
2. 每个任务先在页面展示完整用例组，由用户一次性确认，再调用 cases confirm 留痕；
3. 严格串行，一次只正式化一个任务；build → smoke → test → benchmark，必要时 profile；
4. 单卡任务固定同一 device；`axis_index` 单独申请连续 2/4 卡；任何任务不得挪用别的任务
   的 run 作为通过证据；
5. Pallas 若无 Ascend lowering，交付“明确阻塞 + 可复现探针”，不做 CPU interpret 冒充 NPU。

## 8. 可直接进入“整组确认”的用例草案

以下只是任务输入草案；真正写入 `cases/<op>/cases.json` 前仍须在任务对话中整组展示并由用户确认。
所有数值输入固定 seed `20261026`；CPU oracle 与 NPU 实测分别记录 backend、device、版本和编译模式。

### 8.1 `lax.scan`

| case_id | 覆盖点 | 输入要点 | 预期 |
|---|---|---|---|
| `scan_sum_len8_f32` | 基本 carry + stacked ys | `xs=[8,16]`，加法 carry | eager/jit 与 CPU 一致 |
| `scan_reverse_len7` | `reverse=True`、非 2 次幂 | `xs=[7]` | 输出顺序与 CPU 一致 |
| `scan_pytree_carry` | tuple/dict carry | 两叶子不同 shape | pytree 结构、shape/dtype 一致 |
| `scan_jvp_vjp` | 正反向梯度 | 标量 loss | 支持时梯度与 CPU 一致；不支持要明确分类 |
| `scan_bad_carry_shape` | carry shape 不变量 | body 改变 carry shape | 编译期拒绝，错误分类稳定 |

### 8.2 `lax.associative_scan`

| case_id | 覆盖点 | 输入要点 | 预期 |
|---|---|---|---|
| `assoc_sum_len8` | 加法前缀 | f32 长度 8 | 与 `numpy.cumsum` 一致 |
| `assoc_max_len7` | max、非 2 次幂 | f32 长度 7 | 与 CPU oracle 一致 |
| `assoc_reverse_sum` | reverse | f32 长度 9 | 反向前缀顺序正确 |
| `assoc_pytree` | pytree operator | tuple 两叶子 | 结构与逐叶 oracle 一致 |
| `assoc_grad` | 梯度 | sum 输出聚合 | 梯度与 CPU 一致或明确不支持 |

### 8.3 `lax.cond`

| case_id | 覆盖点 | 输入要点 | 预期 |
|---|---|---|---|
| `cond_true_jit` | true 分支 | 标量 pred=true | 只执行/返回 true 语义 |
| `cond_false_jit` | false 分支 | 标量 pred=false | 只执行/返回 false 语义 |
| `cond_pytree` | 分支 pytree 输出 | tuple/dict 输出 | 两分支结构和 dtype 一致 |
| `cond_grad_both` | 两分支梯度 | 分别取 true/false | 梯度与 CPU 一致 |
| `cond_bad_branch_shape` | 分支签名不一致 | 输出 shape 不同 | trace/编译期拒绝 |

### 8.4 `lax.while_loop`

| case_id | 覆盖点 | 输入要点 | 预期 |
|---|---|---|---|
| `while_zero_iter` | 0 次 | 初值已满足退出 | carry 原样返回 |
| `while_one_iter` | 1 次 | 单次递增 | 与 Python oracle 一致 |
| `while_dynamic_count` | 运行时循环次数 | 0/3/17 | NPU jit 结果一致 |
| `while_pytree_carry` | pytree carry | counter + tensor | shape/dtype/值一致 |
| `while_reverse_grad_boundary` | 已知反向模式边界 | 标量 loss | 不支持时必须是预期拒绝，不记 PASS |
| `while_bad_carry_shape` | carry shape 不变量 | body 改 shape | 编译期拒绝 |

### 8.5 `lax.fori_loop`

| case_id | 覆盖点 | 输入要点 | 预期 |
|---|---|---|---|
| `fori_static_0_8` | 静态上下界 | `[0,8)` | 与 Python loop 一致 |
| `fori_empty` | 空区间 | `[4,4)` | 初值原样返回 |
| `fori_dynamic_bounds` | 动态上下界 | device scalar bounds | 前向结果一致 |
| `fori_unroll_2` | unroll | static bounds, `unroll=2` | 与未 unroll 一致 |
| `fori_grad_static` | 静态边界梯度 | 标量 loss | 与 CPU 一致 |
| `fori_grad_dynamic_boundary` | 动态边界梯度边界 | 动态 bounds | 支持矩阵如实记录 |

### 8.6 `lax.custom_linear_solve`

| case_id | 覆盖点 | 输入要点 | 预期 |
|---|---|---|---|
| `cls_spd_4x4` | 基本 primal | 4×4 SPD, f32 | `||Ax-b||` 达标 |
| `cls_batch_2x4x4` | batch | batch=2 | 与 NumPy/JAX CPU 一致 |
| `cls_transpose_solve` | transpose solve | 非对称可逆矩阵 | 转置系统残差达标 |
| `cls_vjp_b` | 对 RHS 的 VJP | 标量 loss | 与 CPU autodiff 一致 |
| `cls_bad_solver_result` | solver 形状错误 | 错误 shape | trace/编译期拒绝 |

### 8.7 `lax.axis_index`

| case_id | 覆盖点 | 设备要求 | 预期 |
|---|---|---:|---|
| `axis_index_single` | 单设备降级 | 1 | 索引为 0 |
| `axis_index_pmap_2` | `pmap` 轴索引 | 2 | `{0,1}` 各一次 |
| `axis_index_pmap_4` | 4 卡连续轴 | 4 | `{0,1,2,3}` 各一次 |
| `axis_index_named_axis_sum` | 索引参与 collective | 2/4 | 聚合与解析公式一致 |
| `axis_index_unbound` | 未绑定 axis name | 1 | 明确拒绝，不误返回常量 |

### 8.8 `jax.nn.dot_product_attention`

| case_id | 覆盖点 | shape / dtype | 预期 |
|---|---|---|---|
| `dpa_mha_b1_h8_q128_k128_d64_f16` | 标准 MHA | `[1,8,128,64]` f16 | JAX CPU oracle 容差内 |
| `dpa_causal_q128` | causal mask | Q=K=128 | 上三角屏蔽正确 |
| `dpa_boolean_mask` | 外部 bool mask | 局部屏蔽 | 被屏蔽位置不贡献 |
| `dpa_gqa_h8_kv2` | GQA | Hq=8,Hkv=2 | 分组映射正确 |
| `dpa_mqa_h8_kv1` | MQA | Hq=8,Hkv=1 | 广播语义正确 |
| `dpa_q1_decode` | decode 小 Q | Q=1,K=1024,D=128 | 精度通过并作为性能候选 |
| `dpa_non_aligned_d80` | 非常见 head dim | D=80 | 支持则正确；不支持须明确拒绝 |
| `dpa_grad_qkv_small` | Q/K/V 梯度 | 小 shape f32 | 与 JAX CPU autodiff 一致 |
| `dpa_nan_inf_policy` | 非有限值 | Q/K/V 注入 NaN/Inf | 与确认后的特殊行为口径一致 |

### 8.9 Pallas `load`

| case_id | 覆盖点 | 输入要点 | 预期 |
|---|---|---|---|
| `pl_load_full_block` | 完整 block | 对齐二维 tile | Ref 与后端结果一致 |
| `pl_load_tail_mask` | 尾块 mask | 尺寸非 tile 整倍数 | 越界不读，mask 值正确 |
| `pl_load_dynamic_index` | 动态索引 | program_id 派生 | 索引映射正确 |
| `pl_load_oob_reject` | 无 mask 越界 | 故意越界 | 编译或运行期明确拒绝 |

### 8.10 Pallas `store`

| case_id | 覆盖点 | 输入要点 | 预期 |
|---|---|---|---|
| `pl_store_full_block` | 完整写回 | 对齐二维 tile | 全量写回正确 |
| `pl_store_tail_mask` | 尾块 mask | 非整倍数 | 未写区域保持 sentinel |
| `pl_store_dynamic_index` | 动态写位置 | program_id 派生 | 无重叠时映射正确 |
| `pl_store_oob_reject` | 无 mask 越界 | 故意越界 | 编译或运行期明确拒绝 |

## 9. 统一机读结果要求

每项任务至少产出：`environment.json`（镜像 digest / Python / JAX / JAXLIB / plugin commit /
backend / devices）、`correctness.json`（逐 case PASS/FAIL/SKIP + 误差/异常分类）、`performance.json`
（只对有意义的数值算子记录 warmup/重复次数/中位数/离散度）和原始日志。`SKIP` 不能算通过；
因后端不支持而停止时必须写清最小复现命令与首个稳定错误，不得用 CPU PASS 覆盖 NPU FAIL。
