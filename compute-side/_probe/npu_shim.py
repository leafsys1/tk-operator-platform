#!/usr/bin/env python3
"""npu_shim —— 让 ModelScope / 第三方推理栈在昇腾 NPU 上可用的通用补丁层。

每一条补丁都对应一个实测踩到的坑，是"经验沉淀"的载体。
新增补丁时必须写清：现象 / 根因 / 影响面 / 修法。

用法（在 import modelscope 之后、建 pipeline 之前调用）:
    import npu_shim; npu_shim.apply_all()
"""
import importlib
import os
import sys

PATCHES = []


def _patch(name, why):
    def deco(fn):
        PATCHES.append((name, why, fn))
        return fn
    return deco


# ---------------------------------------------------------------- P1
@_patch("modelscope.verify_device 接受 npu",
        "现象: AssertionError device should be either cpu, cuda, gpu...\n"
        "根因: modelscope/utils/device.py 的 verify_device 白名单写死 cpu/cuda/gpu\n"
        "影响面: 所有走 modelscope.pipelines 的原生模型")
def p1_modelscope_device():
    try:
        from modelscope.utils import device as msdev
    except Exception:  # noqa: BLE001
        return "skip: modelscope 未安装"
    if getattr(msdev, "_npu_shim_patched", False):
        return "already"
    orig = msdev.verify_device

    def verify_device(device_name):
        if isinstance(device_name, str):
            n = device_name.strip().lower()
            head = n.split(":")[0]
            if head in ("npu", "ascend"):
                parts = n.split(":")
                return "npu", (int(parts[1]) if len(parts) > 1 else 0)
        return orig(device_name)

    verify_device._npu_shim_orig = orig
    msdev.verify_device = verify_device
    msdev._npu_shim_patched = True

    # 关键：很多 pipeline 模块写的是 `from modelscope.utils.device import verify_device`，
    # 那是按值绑定，事后改模块属性对它们**无效**。必须把已加载模块里
    # 指向原函数的引用一并替换掉（实测 FunASRPipeline 就是这么绕过补丁的）。
    import sys
    fixed = 0
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod, "__dict__", None):
            continue
        for k, v in list(mod.__dict__.items()):
            if v is orig:
                setattr(mod, k, verify_device)
                fixed += 1
    return "patched (含 %d 处按值引用)" % fixed


# ---------------------------------------------------------------- P2
@_patch("torch.cuda.* 重定向到 npu",
        "现象: 模型代码里的 .cuda()/torch.cuda.* 在 NPU 上无效\n"
        "根因: 大量第三方库按 CUDA 写死\n"
        "影响面: 通用；torch_npu 自带 transfer_to_npu 可整体重定向")
def p2_transfer_to_npu():
    try:
        import torch_npu  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return "skip: 无 torch_npu (%s)" % type(e).__name__
    if os.environ.get("NPU_SHIM_NO_TRANSFER") == "1":
        return "skip: 显式禁用"
    try:
        import torch_npu.contrib.transfer_to_npu  # noqa: F401
        return "patched(transfer_to_npu)"
    except Exception as e:  # noqa: BLE001
        return "skip: transfer_to_npu 不可用 (%s)" % type(e).__name__


# ---------------------------------------------------------------- P3
@_patch("屏蔽 npu-smi 缺失告警",
        "现象: 'can not use command: npu-smi info' 刷屏\n"
        "根因: 容器内没装 npu-smi，torch_npu 探测设备时告警\n"
        "影响面: 仅日志噪音，不影响功能")
def p3_quiet_npusmi():
    os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    return "ok"


# ---------------------------------------------------------------- P4
@_patch("transformers 远程代码免交互确认",
        "现象: 探针日志出现 'Do you wish to run the custom code? [y/N]' 并被自动答 N，加载失败\n"
        "根因: transformers/dynamic_module_utils.py:715 在 trust_remote_code is None 时交互提问；\n"
        "      魔搭内部的 transformers 回退路径没传这个参数\n"
        "影响面: 所有需要自定义代码的模型（本批约 1/3）")
def p4_trust_remote_code():
    try:
        from transformers import dynamic_module_utils as dmu
    except Exception:  # noqa: BLE001
        return "skip: 无 transformers"
    if getattr(dmu, "_npu_shim_patched", False):
        return "already"
    orig = dmu.resolve_trust_remote_code

    def resolve(trust_remote_code=None, *a, **kw):
        # 批处理场景没有 stdin 可交互，默认信任（模型来自受控的魔搭语料库）
        if trust_remote_code is None:
            trust_remote_code = True
        return orig(trust_remote_code, *a, **kw)

    dmu.resolve_trust_remote_code = resolve
    dmu._npu_shim_patched = True
    return "patched(默认信任)"


# ---------------------------------------------------------------- P5
@_patch("补齐旧自定义模型的 transformers 新版属性",
        "现象: AttributeError: 'XxxModel' object has no attribute 'all_tied_weights_keys'\n"
        "根因: 自定义模型代码按旧版 transformers 写，新版 PreTrainedModel 要求该属性\n"
        "影响面: 所有携带自定义代码的模型（BiRefNet / RMBG-1.4 等，本批约 1/3）")
def p5_tied_weights_keys():
    try:
        import transformers
        from transformers import PreTrainedModel
    except Exception:  # noqa: BLE001
        return "skip: 无 transformers"
    if getattr(PreTrainedModel, "_npu_shim_p5", False):
        return "already"
    # 只补缺失的默认值，不覆盖已有实现。
    # 注意 all_tied_weights_keys 必须是 **dict**：transformers 对其调用
    # .update()/.items()/.pop()，给 list 会报 'list' object has no attribute 'keys'（实测踩过）。
    #
    # 这一类「旧自定义模型 vs 新版 transformers」的属性期望差，实测会连续撞很多次
    # （all_tied_weights_keys -> forced_bos_token_id -> _supports_sdpa -> ...），
    # 所以这里一次性把新版会读取的开关属性都补上默认值。
    MODEL_DEFAULTS = {
        "all_tied_weights_keys": {},
        "_tied_weights_keys": {},
        "_supports_sdpa": False,
        "_supports_flash_attn_2": False,
        "_supports_flash_attn_3": False,
        "_supports_flex_attn": False,
        "_supports_attention_backend": False,
        "_supports_cache_class": False,
        "_supports_static_cache": False,
        "_supports_quantized_cache": False,
        "_supports_param_buffer_assignment": False,
        "_supports_offloading": False,
        "_keep_in_fp32_modules": [],
        "_keep_in_fp32_modules_strict": [],
        "_no_split_modules": [],
        "_skip_keys_device_placement": [],
    }
    added = []
    for k, v in MODEL_DEFAULTS.items():
        if not hasattr(PreTrainedModel, k):
            setattr(PreTrainedModel, k, v)
            added.append(k)

    # 光补类属性不够：远端自定义模型可能不继承本版 PreTrainedModel，
    # 于是 torch.nn.Module.__getattr__ 直接抛 AttributeError（实测 Florence-2）。
    # 这里做一层**有界兜底**：只对下表里的名字生效，且仅在正常查找失败后才返回默认值。
    import torch.nn as _nn
    if not getattr(_nn.Module, "_npu_shim_p5b", False):
        _orig = _nn.Module.__getattr__

        def _getattr(self, name):
            if name in MODEL_DEFAULTS:
                return MODEL_DEFAULTS[name]
            return _orig(self, name)

        _nn.Module.__getattr__ = _getattr
        _nn.Module._npu_shim_p5b = True

    PreTrainedModel._npu_shim_p5 = True
    return "patched(类上补 %d 个; 另加 Module.__getattr__ 兜底 %d 个)" % (
        len(added), len(MODEL_DEFAULTS))


# ---------------------------------------------------------------- P6
@_patch("补齐 config 缺失的生成参数默认值",
        "现象: AttributeError: 'XxxConfig' object has no attribute 'forced_bos_token_id'\n"
        "      （同类还有 conv_pos_batch_norm / all_tied_weights_keys 等）\n"
        "根因: 自定义模型代码按旧版 transformers 写，新版生成/前向路径会读取这些属性\n"
        "影响面: 携带自定义代码的模型，实测本批已撞 3 次（RMBG / MERT / Florence-2）")
def p6_config_defaults():
    try:
        from transformers import PretrainedConfig
    except Exception:  # noqa: BLE001
        return "skip: 无 transformers"
    if getattr(PretrainedConfig, "_npu_shim_p6", False):
        return "already"
    # 这些是生成/解码路径会读、但旧自定义 Config 可能没设的属性。
    # 只补类级默认值，实例上已有的会覆盖它，因此不会改变正常模型的行为。
    DEFAULTS = {
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "decoder_start_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "pad_token_id": None,
        "sep_token_id": None,
        "vocab_size": 0,
        "is_encoder_decoder": False,
        "tie_word_embeddings": False,
    }
    added = []
    for k, v in DEFAULTS.items():
        if not hasattr(PretrainedConfig, k):
            setattr(PretrainedConfig, k, v)
            added.append(k)
    PretrainedConfig._npu_shim_p6 = True
    return "patched(补 %d 个默认值: %s)" % (len(added), ",".join(added[:5]) or "无缺失")


# ---------------------------------------------------------------- P7
@_patch("补齐 torchaudio 被移除的旧 API",
        "现象: AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'\n"
        "根因: 第三方音频库（如 pyannote.audio）按旧版 torchaudio 写，新版移除了该符号\n"
        "影响面: 所有自带音频框架的模型（pyannote / speechbrain 等）")
def p7_torchaudio_compat():
    try:
        import torchaudio
    except Exception:  # noqa: BLE001
        return "skip: 无 torchaudio"
    if getattr(torchaudio, "_npu_shim_p7", False):
        return "already"
    added = []
    if not hasattr(torchaudio, "AudioMetaData"):
        from typing import NamedTuple

        class AudioMetaData(NamedTuple):
            sample_rate: int = 0
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""

        torchaudio.AudioMetaData = AudioMetaData
        added.append("AudioMetaData")
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
        added.append("list_audio_backends")
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
        added.append("get_audio_backend")
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
        added.append("set_audio_backend")
    torchaudio._npu_shim_p7 = True
    return "patched(补 %s)" % (", ".join(added) or "无缺失")


def apply_all(verbose=True):
    report = []
    for name, why, fn in PATCHES:
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            r = "ERROR: %s: %s" % (type(e).__name__, e)
        report.append({"patch": name, "result": r, "why": why.splitlines()[0]})
        if verbose:
            print("[npu_shim] %-34s %s" % (name, r))
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(apply_all(), ensure_ascii=False, indent=2))
