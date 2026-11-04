# 交付物规范（任务生命周期 v2 · 2026-10-23）

> **给 agent 看的执行规范**：任务对话进入「开发阶段」后，按本文件产出交付物；任务对话 prompt 与
> `workspaces/cann-ops/AGENTS.md` 都引用本文件。设计依据：`docs/plan-task-lifecycle-v2.md` §6、
> `docs/impl-contract-v2.md` §5/§7。路径一律相对**任务工作区（cwd）**。

## 0. 硬规则（先读，违反即返工）

1. **用例组未确认，不动代码、不跑四动作。** 确认状态以控制台 `tasks.cases_confirmed_at` +
   `tasks.cases_confirmed_sha256` 为准；对话 prompt 会告知当前是「未确认 / 已确认且一致 / 已变更需重新确认」。
   用例组必须先在对话里**整组呈现、用户确认**，再写 `cases/<op>/cases.json` + `cases/<op>/benchmark.json`
   并调用 `POST /api/tasks/<task_id>/cases/confirm`。确认前不得修改算子代码，不得触发
   build/test/benchmark/profile（四动作 plan 也会以「用例组未确认」拦下）。
2. **四动作链路不变。** build/test/benchmark/profile 仍走工作区根脚本 + **可执行文件**
   `ops/<op>/build/<op>`（agent 用 `scripts/nm.sh --op <op> --action ...` 调）；benchmark 测量口径
   （设备侧 Task Duration、同卡同条件、median、warm-up/重复次数）不变。新增的 Python 接口与 pytest
   **只是交付件**，不替代四动作门禁。
3. **五件交付物缺一不可**（详见 §1）：算子代码 / Python 调用接口 / pytest 测试脚本 / 算子说明文档 /
   规划文档。另外 `cases/<op>/{cases,benchmark}.json` 与参考实现 `ops/<op>/scripts/golden.py` 是
   开发前置/依赖，不属于「开发阶段新产出」，但交付包会带上。
4. **不伪造、不静默。** 每步在对话里简短汇报；pytest 结果如实报告；跑不了就说明阻塞，不写假结论。
5. **开发顺序：代码写完 → build → smoke → 才谈正式 test/benchmark。**
   `scripts/nm.sh --op <op> --action build` 之后必须跑 `--action smoke`（编译 + 一条最小 required match
   用例跑通）。**smoke 不 PASS 不许进入正式 test/benchmark，也不许说「开发完成」**；smoke 会登记
   `action=smoke` 的 run，控制台阶段「开发」要求 build + smoke 门禁都 pass。smoke 失败先修代码再重跑；
   设备被占/跑不了就如实说阻塞，禁止绕过。

## 1. 交付物路径表（相对任务工作区 cwd）

| 产物 | 路径 | 必需 | 说明 |
|---|---|---|---|
| 算子代码 | `ops/<op>/`（`op_kernel/`、`op_host/`、`CMakeLists.txt`） | ✅ | 沿用现有 AscendC 直调工程；`build/` 产物不进交付包 |
| Python 调用接口 | `ops/<op>/python/<op>.py`（+ 构建产物 `lib<op>_host.so`） | ✅ | ctypes 封装 host C ABI，见 §2 |
| pytest 测试脚本 | `tests/test_<op>.py` | ✅ | 读 `cases/<op>/cases.json`，见 §3 |
| 精度用例（已确认） | `cases/<op>/cases.json` | ✅ | 开发**前**已在对话里整组确认；开发阶段只读 |
| 性能用例（已确认） | `cases/<op>/benchmark.json` | ✅ | 同上；benchmark 脚本读它 |
| 参考实现 | `ops/<op>/scripts/golden.py` | 沿用 | 独立 oracle（如 torch CPU / numpy fp64），pytest 与 test.sh 共用 |
| 算子说明文档 | `ops/<op>/README.md` | ✅ | 结构见 §4 |
| 规划文档（agent 补全） | `docs/<op>/design.md` | ✅ | 按模板五段，见 §5 |
| 设计文档原文/解析稿 | `task-docs/task-<id>/` | 若有上传 | 由控制台上传接口落盘，交付时原文随包 |

- 命名：`<op>` 必须与任务规格里的算子名一致（= `OPX_OP` = `cases/<op>/` 目录名）。
- 不要另起目录（如 `src/`、`impl/`）放交付物；不要动共享资产（`scripts/`、`tools/`、`criteria/`、
  根入口 `build.sh/test.sh/benchmark.sh/profile.sh`）——那是门禁权威，不在交付范围内。

## 2. Python 调用接口规范

目标：在算子开发镜像的普通 Python 进程里（`import torch, torch_npu` 之后）用**候选算子实现**跑
精度用例；接口语义对齐 torch（同 shape/dtype、不广播、非法输入拒绝）。

### 2.1 总体约束

- 实现路径：Python ctypes 直接加载 host 共享库 `lib<op>_host.so`，调 C ABI；**不**在 Python 里重算、
  **不**用 `subprocess` 调 CLI 冒充 Python 接口（若真机 spike 证明 ctypes 不可行，退回子进程 wrapper
  是唯一例外，且必须在 README/对话里写明原因与限制）。
- **四动作不受影响**：`ops/<op>/build/<op>` 可执行文件保留、路径不变；build.sh 仍只 `make <op>`，
  所以共享库要挂到可执行文件目标上（见 §2.3）。
- 开发前先做最小 spike（真机）：Python 里 `torch_npu` 建 tensor → `data_ptr()` → 调 `.so` → 读回输出
  与 CPU 参考对比一致。spike 不通过（ACL 上下文冲突等）先与用户/控制台确认再退回备选方案。
- **已验证（2026-10-23，cann 910b 算子开发镜像）**：按本节做法（独立 `*_pyapi.asc` 导出 C ABI +
  CMake `SHARED` target + `add_dependencies(<op> <op>_host)`，**不改共享 build.sh**）在容器内真编通过；
  `ctypes.CDLL` 能加载 `lib<op>_host.so` 并取到 `elementwise_add_run` 与 ABI 版本
  （run `r3-20261023-114033`，`.so` sha256 `8a31e87c…`）。参考实现见
  `docs/reference/examples/elementwise_add_pyapi.asc` 与 `examples/python_elementwise_add.py`。
  **运行时**（torch_npu 已初始化 ACL 上下文后实际调 kernel）尚未验证 —— 有卡时必须按 §2.1 做最小 spike，
  不得只看「编译通过」就认为 Python 接口可用。

### 2.2 C ABI（host `.asc` 导出）

把 host 侧「校验 + tiling + kernel 调用 + 结果写回」从 `main()` 抽成一个导出函数；`main()` 改成
解析 argv 后调用它，保证**同一套逻辑**同时服务 CLI 与 Python（拒绝码必须一致，不许两套校验）。

```c
// ops/<op>/op_host/<op>.asc（示意；输出/属性参数按算子语义调整）
extern "C" __attribute__((visibility("default")))
int <op>_run(const void* a, const void* b, void* out, int64_t n, int dtype_code, int device_id);
```

约定：

- 参数：`a`/`b` 为 device 侧输入指针（来自 torch_npu tensor 的 `data_ptr()`）；`out` 为 device 侧输出
  指针（Python 侧 `torch.empty_like` 分配）；`n` 元素数；`dtype_code` 见下表；`device_id` 为 ACL device。
  多输入/多输出/带属性的算子按语义扩展参数（属性用标量或小 struct），但保持「int 返回拒绝码」。
- 返回：`0` = 成功（返回前必须完成 device→host 同步，保证 Python 侧可立即读 `out`）；非 0 = 拒绝/失败，
  **数值与现有 CLI 退出码映射一致**（例：2 BAD_ARGS、3 SHAPE_MISMATCH、4 N_OUT_OF_RANGE、
  5 UNSUPPORTED_DTYPE；具体以本算子 README/用例 `expect_reject_code` 为准）。
- 拒绝时若有 detail，可另导出 `int <op>_last_error(char* buf, int buflen)` 供 Python 取错误串（可选）。
- **ACL 上下文**：Python 进程里 `torch_npu` 已初始化 ACL；host 入口**不要**调 `aclInit/aclFinalize`，
  只按需 `aclrtSetDevice(device_id)`（幂等）；内部自行创建/销毁 stream 或复用默认 stream，调用
  kernel 后 `aclrtSynchronizeStream`；设备内存由 Python 侧持有，host 只读写指针。
- 共享库与可执行文件共用同一份 kernel/ tiling 源文件；不要把 kernel 逻辑复制两份。

`dtype_code`（Python 与 host **必须**用同一张表；新增 dtype 时两边同时改）：

| dtype_code | dtype |
|---|---|
| 0 | float32 |
| 1 | float16 |
| 2 | bfloat16 |
| 3 | int32 |
| 4 | int64 |

### 2.3 CMake：同时产出可执行文件与共享库

保留原有可执行文件 target（四动作入口）；新增共享库 target，并让 build.sh 的 `make <op>` 顺带编出 `.so`：

```cmake
# 原有：四动作入口，路径 ops/<op>/build/<op> 不变
add_executable(<op> op_host/<op>.asc)
# 新增：Python ctypes 用的共享库
add_library(<op>_host SHARED op_host/<op>.asc)   # 建议拆出无 main 的 <op>_api.asc，两个 target 共用
set_target_properties(<op>_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(<op> <op>_host)                  # build.sh 只 make <op>，靠这行确保 .so 也编出来
```

- `.so` 默认落在 `ops/<op>/build/lib<op>_host.so`；若 CMake 输出到别处，Python 侧负责按候选路径查找。
- 编译失败时 build 动作就是失败（门禁会看 build 日志），不要为了 Python 接口牺牲原 target。

### 2.4 Python 包装 `ops/<op>/python/<op>.py`

职责（按顺序）：

1. 定位并 `ctypes.CDLL` 加载 `lib<op>_host.so`：先找 `ops/<op>/python/`，再找 `ops/<op>/build/`；
   用 `Path(__file__).resolve()` 相对定位，**不写死绝对路径**。
2. 显式声明 `argtypes` / `restype`（`c_void_p, c_void_p, c_void_p, c_int64, c_int, c_int` → `c_int`）。
3. 输入校验（与规格一致）：dtype 在支持表、shape 一致（不广播）、`contiguous()`、device 正确、
   n 在范围内；校验失败抛 `ValueError`/`TypeError`，并尽量带拒绝码文案。
4. `torch_npu.npu.set_device(device)`（或沿用当前 device）→ `torch.empty_like(a)` 分配输出。
5. 调 `_lib.<op>_run(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, dtype_code, device_id)`；返回非 0 时
   按拒绝码抛 `ValueError`（非法输入）或 `RuntimeError`（设备/运行失败）。
6. 返回 `out`（新 tensor，shape/dtype 与期望一致）；同时校验输出非空、shape 正确。

示例骨架（按算子语义改参数）：

```python
"""<op> 的 Python 调用接口（ctypes 封装 host C ABI）。"""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch
import torch_npu

_DTYPE_CODE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2, torch.int32: 3, torch.int64: 4}


def _lib_path() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / "lib<op>_host.so", here.parent / "build" / "lib<op>_host.so"):
        if cand.is_file():
            return cand
    raise RuntimeError("找不到 lib<op>_host.so：先跑 build（scripts/nm.sh --action build）")


_lib = ctypes.CDLL(str(_lib_path()))
_lib.<op>_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_int64, ctypes.c_int, ctypes.c_int]
_lib.<op>_run.restype = ctypes.c_int


def <op>(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype not in _DTYPE_CODE:
        raise TypeError(f"不支持的 dtype：{a.dtype}")
    if a.shape != b.shape:
        raise ValueError(f"shape 不一致：{tuple(a.shape)} vs {tuple(b.shape)}")
    if a.device != b.device:
        raise ValueError("输入必须在同一 device")
    a, b = a.contiguous(), b.contiguous()
    device_id = a.device.index if a.device.index is not None else 0
    torch_npu.npu.set_device(device_id)
    out = torch.empty_like(a)
    rc = _lib.<op>_run(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(),
                       _DTYPE_CODE[a.dtype], device_id)
    if rc != 0:
        raise ValueError(f"<op>_run 拒绝/失败 rc={rc}")  # 按拒绝码映射成具体异常
    return out
```

- 交付前在镜像里自测一次：`python3 -c "import ...; 小 shape 与 golden 对比"`，结果写进对话（不替代四动作）。

## 3. pytest 规范：`tests/test_<op>.py`

定位：**交付件**（可在算子开发镜像里 `pytest -q tests/` 直接跑，不依赖控制台）；**门禁权威仍是共享
runner**（`test.sh` + `tools/run_cases.py`），两者读同一份用例文件、同口径判定。

> **实测提醒（2026-10-23）**：cann 910b 算子开发镜像**未预装 pytest**（`python3 -c "import pytest"` 失败）。
> pytest 脚本仍是必需交付件；要在镜像里跑需先 `python3 -m pip install pytest`（镜像内 pip/网络可用时），
> 或在开发机/CI 跑。**四动作门禁不受影响**：`test` 动作走 `test.sh`+`run_cases.py`，不依赖 pytest。

要求：

1. 读 `cases/<op>/cases.json`（`json.load`），逐 case 生成一个 pytest 参数（`pytest.mark.parametrize`）。
2. 候选：用 §2 的 **Python 接口**跑（不用 CLI 子进程）。
3. 参考：`ops/<op>/scripts/golden.py` 的 `compute_golden(...)`（torch CPU / numpy 独立 oracle）。
4. 判据：与 `criteria/check.py` 同口径 —— 用 `importlib.util.spec_from_file_location` 加载工作区
   `criteria/check.py`，调 `check(actual_np, golden_np)`，`is_pass` 为 True 才 PASS（含非有限位规则）。
5. 拒绝用例（`expected == "reject"`）：必须**按预期抛出/拒绝且拒绝码匹配** `expect_reject_code`
   （与 `tools/run_cases.py` 同口径：不能只判「非 0/抛异常」，要认出期望的码）；实现拒绝时没给码算 FAIL。
6. seed 固定：按 case 里的 `seed` + `gen_a/gen_b` 生成输入（dtype 一致）；不要自己另造输入。
7. 路径用 `Path(__file__).resolve().parents[1]`（tests/ 的上一级 = 工作区根）相对定位，不写死绝对路径。
8. 跑完在对话里报告：`pytest -q tests/test_<op>.py` 的通过/失败数；失败要贴关键原因，不许只说「过了」。

示例骨架：

```python
import importlib.util, json
from pathlib import Path

import numpy as np
import pytest
import torch

WS = Path(__file__).resolve().parents[1]
OP = "<op>"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


candidate = _load(WS / "ops" / OP / "python" / f"{OP}.py", f"{OP}_python")
golden_mod = _load(WS / "ops" / OP / "scripts" / "golden.py", f"{OP}_golden")
criteria = _load(WS / "criteria" / "check.py", "opx_criteria")
CASES = json.loads((WS / "cases" / OP / "cases.json").read_text(encoding="utf-8"))["cases"]


def _make(case):
    ...  # 按 gen_a/gen_b + seed 生成 torch_npu tensor（或 numpy 后转 torch_npu）


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_case(case):
    if case.get("expected") == "reject":
        with pytest.raises(Exception) as ei:
            candidate(...)
        assert case.get("expect_reject_code", "") in str(ei.value)
        return
    a, b = _make(case)
    got = candidate(a, b).cpu().numpy()
    want = golden_mod.compute_golden(a.cpu().numpy(), b.cpu().numpy())
    assert criteria.check(got, want)["is_pass"], criteria.check(got, want)["failure_reasons"]
```

## 4. 算子说明文档 `ops/<op>/README.md`

必须包含：

1. **算子语义**：数学定义、与 torch/框架同名算子的对齐关系、支持 dtype/shape/约束。
2. **调用接口**：CLI（`ops/<op>/build/<op>` 参数、`EXEC` 标记、退出码/拒绝码表）与 Python
   （`from ... import <op>`、入参要求、返回、异常/拒绝码），给最小调用示例。
3. **构建**：容器里 `bash build.sh`（或 `scripts/nm.sh --action build`）与 CMake 双 target 说明。
4. **精度口径**：oracle、判据档 `ops-precision-standard@1`、容差/非有限位规则。
5. **性能口径**：基线实现、测量工具/口径（设备侧 Task Duration）、代表用例、实测值（若有）。
6. **已知限制**：不支持组合、已知偏差、Python 接口 spike 结论/限制（如有）。

## 5. 规划文档 `docs/<op>/design.md`
按 `docs/templates/算子设计文档模板.md` 的**五段主线**写（可裁剪，但不要缺段）：

1. 需求背景（需求来源/业务场景/数学语义/历史实现与现状）；
2. 需求分析（外部依赖/内部适配/模块设计/算子原型：输入输出属性与推导规则）；
3. 需求详细设计（使能方式/总体设计：Host tiling 分核、shape/dtype 校验、workspace；Kernel 搬运/
   流水/精度/写回；支持硬件；约束限制）；
4. 特性交叉分析（混合精度/图模式/动态 shape/其他，逐项写「不涉及」并说明）；
5. 可维可测分析（精度标准/性能标准/兼容性）。

规则：用户上传过设计文档 → 保留原文语义，补全缺失章节并标注来源（用户原文 / agent 补全）；
不确定的事实写 **UNKNOWN** 或「待确认」，**不要编造**。交付包会同时带用户原文（若有）。

## 6. 用例文件（`cases/<op>/`）——开发阶段只读

- `cases.json`（精度组，schema `opx-cases/1`）：`schema/op/spec_version/dtype/criteria_ref/constraints/
  case_groups/cases[]`，每条 case 含 `case_id/group/required/shape 或 n_a,n_b/dtype/gen_a,gen_b/seed/
  expected(match|reject)/expect_reject_code/note`。
- `benchmark.json`（性能组，schema `opx-benchmark/1`）：`schema/op/dtype/baseline/memory/compute/
  measure/primary_case/cases[]`，每条 case 含 `case_id/required/n/dtype/gen_a,gen_b/seed/note`。
- 两份文件由**任务对话里的用例确认流程**产出并 `POST /cases/confirm` 留痕；开发阶段视为只读。
  用户要求改用例 → 走对话 → 改文件 → 重新 confirm；`suite_ref`（`cases@<hash16>`）变化后旧
  test/benchmark 证据失效，交付前必须用新用例集重跑。
- pytest 必须读这份 `cases.json`，不要内置私有用例集；`test.sh`/`run_cases.py` 也读它。

## 7. 交付前自检清单

- [ ] 用例组已确认且与当前 `cases/` 哈希一致（对话/控制台可见 in_sync）。
- [ ] 五件交付物齐全（§1 表），路径/命名正确，无绝对路径、无 `build/` 产物混入源码目录。
- [ ] **开发完成前已跑过 `scripts/nm.sh --op <op> --action build` + `--action smoke`，smoke 门禁 pass**
      （烟测只挑 1 条最小 required match 用例；smoke 未 PASS 不许进入正式 test/benchmark）。
- [ ] `bash build.sh` 通过，且 `ops/<op>/build/lib<op>_host.so` 已产出。
- [ ] Python 接口 spike/自测通过；`pytest -q tests/test_<op>.py` 结果已如实报告。
- [ ] 动作链（build → smoke → test → benchmark，profile 可选）已按当前版本跑过且门禁 pass；未跑/重跑需求要在
      对话里说清楚（交付门禁会拦证据失效；smoke 是开发段门禁，不在交付必需动作里）。
- [ ] README / `docs/<op>/design.md` 无编造内容，UNKNOWN 已标注。
- [ ] 改过用例/规格 → 已重新确认并重跑，或已取得用户对话确认的豁免（留痕）。
