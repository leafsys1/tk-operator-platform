# 为什么 GPU 迁移到 NPU 要改这么多

> 核心原因一句话：**CUDA 不只是一个 API，它是一整套用了十几年的生态。NPU 不是"另一个 CUDA"，是另一套栈。**
>
> 改动之所以"多"，是因为**问题分散在五个完全不同的层次**，而它们只能一层层试出来。

---

## 五个层次，从软到硬

### 第 1 层：设备 API（最简单，一层适配搞定）

```python
model.cuda()               →  model.npu()
torch.cuda.is_available()  →  torch.npu.is_available()
```

这一层昇腾已经给了 `transfer_to_npu`，**自动把 `torch.cuda.*` 全重定向到 `torch.npu.*`**。

实测效果：

```text
[npu_shim] torch.cuda.* 重定向到 npu    patched(transfer_to_npu)
```

**光这一层就能让大约一半的标准模型直接跑起来。**

但只要模型里有**硬断言**就漏了：

```python
assert device.type == "cuda"           # 这个不会被重定向
if not torch.cuda.is_available(): ...  # 自定义代码里的一样漏
```

所以还要额外补一层 shim 去改这些断言。

---

### 第 2 层：依赖包（中等麻烦，但很隐蔽）

这是最反直觉的一层。**代码一行没改，就是装不上。**

原因：整个 Python AI 生态的 wheel 是按 `manylinux + x86_64 + CUDA` 构建的。

| 障碍 | 典型表现 |
|---|---|
| 没有 aarch64 wheel | `No matching distribution found` |
| 有 wheel 但 Python 版本不匹配 | 只有 cp310，你跑 cp312 |
| C++ 扩展要编译 | 编译时找 `cuda_runtime.h`，找不到 |
| 依赖链里有 CUDA-only 包 | `torch-scatter`、`flash-attn`、`torchmcubes`、`spconv` |

而且**报错常常是骗人的**：

> 官方提示 `pip install modelscope[audio]`
> 真因：缺 `funasr`（提示完全不沾边）

> `pip show SwissArmyTransformer` 说装了
> 真因：wheel 是空壳，只有 `dist-info` 没有代码目录

> `megatron_util` 装不上
> 真因：装出来的目录叫 `megatron-util`（连字符），Python 要下划线

---

### 第 3 层：自定义算子（真正的硬骨头）

这是 GPU 和 NPU 之间**最本质的差异**。

- GPU 上模型用的算子，是 cuBLAS / cuDNN / 各种手写 CUDA kernel
- NPU 上对应的是 CANN 的 aclnn 算子库 + AscendC 自定义算子

**两者不是一一对应的。** NPU 的算子集更小，很多"GPU 上很常见"的 op 没有实现：

```text
aclnnXxx not implemented
Unsupported op: aten::xxx
```

更麻烦的是**报错位置常常不是根因**：

> 调用栈指向的那个算子，往往只是"第一个做同步/内存搬移的调用点"，
> 真正崩的地方更早埋下。

所以要花大量力气做「最小复现脚本」，把崩点精确定位到具体 op。

**能绕过的部分**：只在导出/可视化阶段用的 C++ 扩展 → 写纯 Python 占位 stub。

| 包 | 用途 | 能否 stub |
|---|---|---|
| `torchmcubes` | 把体素提取成 mesh | ✅ 可（导出阶段才用） |
| `rembg` | 抠图预处理 | ✅ 可 |
| `open3d` | 3D 可视化 | ✅ 可 |
| `FlexiCubes` | mesh 提取 | ✅ 可 |
| `spconv` | 稀疏卷积 | ❌ 不可（参与前向） |

---

### 第 4 层：框架层（最难，因为要改的是"继承关系"）

有些模型不是"用了某个库"，而是**本身就是某个框架的子类**。

实测 `OFA`：

```python
class OFAModel(TransformerModel):   # 直接继承 fairseq 的类
    ...
```

模型代码里到处是 `FairseqEncoder`、`fsdp_wrap`、`quant_noise`。

**这不是"补几个 stub"能解决的** —— 你要么把整个 fairseq 搬过来，要么重写模型。

同类问题：

- `mplug` / `hitea` 也是 fairseq 系
- `uni-fold-multimer` 依赖 `unicore`
- GLM-130B 系依赖 SwissArmyTransformer + Megatron 格式
- Paddle 系模型要单独的 Python 解释器（`py3.10` + `paddle-custom-npu`）

---

### 第 5 层：生态惯性（结构性问题）

这一层不是"bug"，是**整个行业的事实标准**：

```text
论文代码默认 CUDA
   ↓
README 说 "pip install xxx"，xxx 只出 CUDA wheel
   ↓
模型代码里写死 .cuda()
   ↓
自定义 kernel 用 CUDA 写
   ↓
CI 只测 CUDA
```

**每一环都在加深对 CUDA 的依赖**，所以迁移时你要逆向走完整条链。

---

## 为什么"改动看起来那么零散"

因为**它们是不同层次的问题，且只能一个个撞出来**。

实测一个多模态模型的迁移过程：

```text
1. 装不上 packageA          ← 第 2 层
2. packageA 装上了，import 报错  ← 第 2 层（目录名问题）
3. 能 import 了，config 不认     ← 第 4 层（格式版本）
4. config 改了，权重对不上       ← 第 5 层（键名映射）
5. 权重对上了，前向崩在算子      ← 第 3 层
6. 算子绕过了，OOM              ← D 桶（资源）
```

**六个问题分属四个层次，不可能一次性预判。**

---

## 但也有"不用改那么多"的情况

反过来说清楚：**标准 HF 架构的模型，改动几乎为零。**

```text
google/electra-small-discriminator   零改动，2.6 秒跑通
AI-ModelScope/bert-base-cased        零改动
OpenGVLab/pvt_v2_b2                  零改动
```

因为 `transformers` / `diffusers` 这些上游库**已经把 device 差异抽象掉了**。

**改动量的分布极度不均：**

| 模型类型 | 改动量 |
|---|---|
| 标准 transformers 架构 | **0 行** |
| 标准 + 一两个额外依赖 | 几行（装包） |
| 自定义架构但代码可得 | 几十行（device 断言、算子） |
| 框架级依赖 / 代码不可得 | **几百行**（要么重写，要么重建架构） |

---

## 一句话总结

> **GPU → NPU 不是"换个后端"，是"换一整套地基"。**
>
> 上层（模型代码）看起来只差一个 `device` 参数，
> 但下面压着**算子库、编译产物、依赖生态、框架约定**四层东西，每一层都可能塌。

---

## 一个可验证的类比

这有点像把 C 程序从 x86 移植到 ARM：

| x86 → ARM | GPU → NPU |
|---|---|
| 纯 C 代码，改编译器就能编 | 标准 transformers 模型，零改动 |
| 用了 SSE intrinsics，要换 NEON | 自定义算子要换实现 |
| 依赖只有 x86 版的库，找替代或自己写 | fairseq 这类框架级依赖 |
| 依赖闭源 `.so`，完全没戏 | 代码不可得的模型（checkpoint ≠ 公开代码） |

---

## 对照本项目的数据

| 层次 | 对应坑库桶 | 本项目遇到的情况 |
|---|---|---|
| 第 1 层 设备 API | B 桶 | `transfer_to_npu` 覆盖大部分；硬断言需 shim 补 |
| 第 2 层 依赖包 | A 桶 | 条目最多的一类；含 3 次"同名不同项目"误判 |
| 第 3 层 自定义算子 | C 桶 | 5 个 C++ 扩展中 4 个可 stub，1 个（spconv）不可 |
| 第 4 层 框架层 | E 桶 | fairseq 3 个、unicore 1 个、SAT 1 个、Paddle 2 个 |
| 第 5 层 生态惯性 | — | 表现为 E 桶的"键名/配置格式/权重布局"问题 |

**最终结果**：117 个模型，108 个迁移成功（92%）。
其中标准架构模型占大多数，是"零改动"那一档。

---

## 给后来者的三条判据

遇到"某个依赖拿不到"时，**下结论前至少验证三件事**：

1. 它是否**真的**不在包里（去翻 checkpoint 的键，可能权重本来就有）
2. 它是否**真的**不在 PyPI（换个名字查、查官方源、数文件数）
3. 它是否**真的**必需（看它被调用的位置，可能只是导出阶段）

> 本项目因为跳过这三步，至少误判了 6 次：
> 虚报 6 个模型已迁移 / Paddle 误判不可用 / 权重扫描漏扩展名 /
> `megatron_util` 误判不在 PyPI / SAT 空壳 wheel / GLU 判反。
>
> **六次纠错里没有一次是靠灵感发现的，全部来自同一个动作——再验一次。**
