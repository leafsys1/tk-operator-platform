"""transformers 5.x 兼容 shim —— 让按 4.x 写的第三方模型代码能跑。

实测救活：Qwen/Qwen3-TTS-12Hz-0.6B-Base（qwen_tts 包）。

四类不兼容（都是 4.x -> 5.x 的破坏性变更）：
 1) check_model_inputs 签名由 () 变为 (func)，旧代码写 @check_model_inputs()
 2) PretrainedConfig 不再自动设置 pad_token_id / *token_id 等属性
 3) ROPE_INIT_FUNCTIONS 去掉了 'default' 键
 4) 5.x 的 linear rope 需要 rope_parameters['factor']，而 4.x 的 default 不需要

用法：在 import 目标模型库**之前** `import tf5_compat`。
"""
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 1)
try:
    from transformers.utils import generic as _G

    if hasattr(_G, "check_model_inputs"):
        _G.check_model_inputs = lambda func=None, *a, **kw: (
            func if func is not None else (lambda f: f)
        )
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------- 2)
# 5.x 对若干 config 键做了**重命名**。老模型卡里还是旧名，构造 config 时会被
# PretrainedConfig 当未知 kwargs 吞掉（于是新名取默认值），最终在 forward 里报
# AttributeError。这里统一做一次旧名 -> 新名映射。
_RENAMES = {
    "dim_embed": "embed_dim",   # Florence2 视觉塔（davit）
}

try:
    from transformers import PretrainedConfig as _PC

    _COMMON = (
        "pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id",
        "sep_token_id", "cls_token_id", "mask_token_id", "unk_token_id",
        "image_token_id", "video_token_id",
    )
    if not getattr(_PC, "_compat_patched", False):
        _orig_init = _PC.__init__

        def _init(self, **kw):
            for _old, _new in _RENAMES.items():
                if _old in kw and _new not in kw:
                    kw[_new] = kw.pop(_old)
            _orig_init(self, **kw)
            for a in _COMMON:
                try:
                    if not hasattr(self, a):
                        object.__setattr__(self, a, None)
                except Exception:  # noqa: BLE001
                    pass

        _PC.__init__ = _init
        _PC._compat_patched = True
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------- 3) + 4)
try:
    import torch as _t
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _R

    def _compute_default_rope_parameters(config=None, device=None, seq_len=None,
                                         layer_type=None):
        """4.x 的 default RoPE：无 scaling，直接按 rope_theta 生成 inv_freq。"""
        base = getattr(config, "rope_theta", 10000.0)
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        head_dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (_t.arange(0, head_dim, 2, dtype=_t.int64).float().to(device) / head_dim)
        )
        return inv_freq, 1.0

    if "default" not in _R:
        _R["default"] = _compute_default_rope_parameters
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------- 5)
def patch_rotary_embedding_classes():
    """给「远端代码自定义的 *RotaryEmbedding* 类」补上 compute_default_rope_parameters。

    为什么需要：transformers 5.x 的 `modeling_utils._init_weights` 里有无条件分支
        elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
            rope_fn = (ROPE_INIT_FUNCTIONS[module.rope_type]
                       if module.rope_type != "default"
                       else module.compute_default_rope_parameters)   # ← 假定一定有这个方法
    远端模型代码（按 4.x 写）自定义的 RotaryEmbedding 没有它，于是
    `from_pretrained` 走到 _init_weights 时抛 AttributeError。

    用法（关键在**顺序**）：
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        get_class_from_dynamic_module("modeling_xxx.XxxForConditionalGeneration", model_dir)
        import tf5_compat; tf5_compat.patch_rotary_embedding_classes()
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    """
    import sys as _sys

    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _RR

        fn = _RR.get("default")
        if fn is None:
            return 0
    except Exception:  # noqa: BLE001
        return 0

    n = 0
    for mod in list(_sys.modules.values()):
        if mod is None:
            continue
        for attr in dir(mod):
            if "RotaryEmbedding" not in attr:
                continue
            cls = getattr(mod, attr, None)
            if not isinstance(cls, type):
                continue
            if hasattr(cls, "compute_default_rope_parameters"):
                continue
            try:
                cls.compute_default_rope_parameters = staticmethod(fn)
                n += 1
            except Exception:  # noqa: BLE001
                pass
    return n



# ---------------------------------------------------------------- 6)
# 5.x 删掉了一些老代码常用的工具函数 / 常量。这里按需补回（默认给安全值）。
try:
    import transformers.utils.import_utils as _IU

    _MISSING_UTILS = {
        # 老代码用来判断是否走 torch.fx 分支；返回 False 表示不走（最安全）
        "is_torch_fx_available": lambda: False,
        "is_torch_fx_available_for_dtype": lambda *a, **k: False,
        "is_accelerate_available": lambda: True,
        "is_safetensors_available": lambda: True,
        "is_tokenizers_available": lambda: True,
        "is_torch_available": lambda: True,
    }
    for _n, _v in _MISSING_UTILS.items():
        if not hasattr(_IU, _n):
            setattr(_IU, _n, _v)
except Exception:  # noqa: BLE001
    pass




# ---------------------------------------------------------------- 7)
# 5.x 移除了 bert 模块里的 BasicTokenizer / WordpieceTokenizer，
# 老代码（如 modelscope 的 ofa preprocessor）仍按老路径 import。
def _install_tokenizer_stubs():
    import types

    def _mk(name):
        def _init(self, *a, **k):
            raise NotImplementedError(
                "tf5_compat stub: %s 在 transformers 5.x 已移除，仅作 import 占位；"
                "分词请改用 AutoTokenizer。" % name)

        def _tok(self, *a, **k):
            raise NotImplementedError("tf5_compat stub: %s" % name)

        return type(name, (), {"__init__": _init, "tokenize": _tok})

    targets = [
        "transformers.models.bert.tokenization_bert",
        "transformers.tokenization_utils",
        "transformers.tokenization_utils_base",
    ]
    import importlib
    for t in targets:
        try:
            m = importlib.import_module(t)
        except Exception:
            continue
        for n in ("BasicTokenizer", "WordpieceTokenizer", "WordPieceTokenizer"):
            if not hasattr(m, n):
                setattr(m, n, _mk(n))


try:
    _install_tokenizer_stubs()
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------- 8)
# pandas 2.2+ 移除了大写频率别名（"6H" 要写 "6h"），而老代码大量在用大写。
# 这里包装 pd.date_range / pd.Timedelta / pd.to_timedelta，自动把 H/M/S/T 等
# 大写频率降成小写，避免 ValueError: Invalid frequency: 6H。
try:
    import pandas as _pd

    if not getattr(_pd, "_compat_freq_patched", False):
        _dr_orig = _pd.date_range

        def _fix_freq(v):
            if isinstance(v, str):
                # 只替换频率里的日期单位字母（注意别动 %H 这种格式串）
                if "%" in v:
                    return v
                return (v.replace("H", "h").replace("T", "min")
                         .replace("S", "s").replace("M", "ME")
                         if v[-1:] in ("H", "T", "S", "M") else v)
            return v

        def _date_range(*a, **k):
            if "freq" in k:
                k["freq"] = _fix_freq(k["freq"])
            return _dr_orig(*a, **k)

        _pd.date_range = _date_range

        # pd.Timedelta("12H") 同样会炸。必须用**子类**而不是普通函数：
        # 它出现在联合类型注解里（xarray: str | datetime.timedelta | pd.Timedelta | ...），
        # 普通函数会让 `|` 报 TypeError；子类仍是 class，`|` 正常。
        _td_base = _pd.Timedelta

        class _TimedeltaCompat(_td_base):
            def __new__(cls, *a, **k):
                if a and isinstance(a[0], str):
                    a = (_fix_freq(a[0]),) + a[1:]
                return _td_base(*a, **k)

        _TimedeltaCompat.__name__ = "Timedelta"
        _TimedeltaCompat.__qualname__ = "Timedelta"
        _pd.Timedelta = _TimedeltaCompat
        _pd._compat_freq_patched = True
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------- 9)
# 5.x 从 transformers.tokenization_utils / utils 里删掉了一批常量，
# 老代码（含 modelscope）仍按老路径 import。
try:
    import importlib as _il

    _CONSTS = {
        "TOKENIZER_CONFIG_FILE": "tokenizer_config.json",
        "CONFIG_NAME": "config.json",
        "WEIGHTS_NAME": "pytorch_model.bin",
        "TF2_WEIGHTS_NAME": "tf_model.h5",
        "TF_WEIGHTS_NAME": "model.ckpt",
        "FLAX_WEIGHTS_NAME": "flax_model.msgpack",
        "SAFE_WEIGHTS_NAME": "model.safetensors",
        "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
        "WEIGHTS_INDEX_NAME": "pytorch_model.bin.index.json",
        "SPM_FILE": "spiece.model",
        "VOCAB_FILES_NAMES": {},
    }
    for _mod in ("transformers.tokenization_utils",
                 "transformers.tokenization_utils_base",
                 "transformers.utils",
                 "transformers.utils.hub"):
        try:
            _m = _il.import_module(_mod)
        except Exception:  # noqa: BLE001
            continue
        for _k, _v in _CONSTS.items():
            if not hasattr(_m, _k):
                # ⚠️ 必须直接写 __dict__：transformers 5.x 的 tokenization_utils 是
                # _LazyModule，setattr 会被它的 __setattr__ 吞掉（且模块显示
                # "(unknown location)"），导致 `from ... import X` 仍报 ImportError。
                try:
                    _m.__dict__[_k] = _v
                except Exception:  # noqa: BLE001
                    try:
                        setattr(_m, _k, _v)
                    except Exception:  # noqa: BLE001
                        pass
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------- 10)
# 5.x 从 transformers.utils 删掉的一批函数（老代码/modelScope 仍在用）。
try:
    import importlib as _il2

    def _mk_removed(name):
        def _f(*a, **k):
            raise NotImplementedError(
                "tf5_compat stub: %s 在 transformers 5.x 已移除，仅作 import 占位。" % name)
        _f.__name__ = name
        return _f

    _FUNCS = [
        "get_file_from_repo", "has_file", "is_remote_url", "cached_path",
        "filename_to_url", "get_from_cache", "get_list_of_files",
        "is_torch_fx_available", "is_torch_tpu_available",
        "is_torch_cuda_available", "is_torch_bf16_gpu_available",
        "is_ipex_available", "is_bitsandbytes_available",
        "is_flash_attn_2_available", "is_flash_attn_greater_or_equal_2_10",
        "is_ninja_available", "is_vision_available", "is_pytesseract_available",
    ]
    for _mod in ("transformers.utils", "transformers.utils.hub",
                 "transformers.utils.import_utils"):
        try:
            _m2 = _il2.import_module(_mod)
        except Exception:  # noqa: BLE001
            continue
        for _n in _FUNCS:
            if not hasattr(_m2, _n):
                try:
                    _m2.__dict__[_n] = _mk_removed(_n)
                except Exception:  # noqa: BLE001
                    try:
                        setattr(_m2, _n, _mk_removed(_n))
                    except Exception:  # noqa: BLE001
                        pass
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------- 11)
# 上面第 9/10 节的 setattr 注入对 transformers 5.x 的 _LazyModule 无效
# （它的 __setattr__ 会拦截；模块显示 "(unknown location)"）。
# 这里改用**模块代理**：把 sys.modules 里的条目换成代理对象，
# `from X import Y` 时会走代理的 __getattr__，从而拿到我们补的符号。
def _inject_module_proxy(modname, extra):
    import sys as _s
    import types as _t

    class _Proxy(_t.ModuleType):
        def __getattr__(self, name):
            if name in extra:
                return extra[name]
            real = self.__dict__.get("_real")
            if real is not None:
                return getattr(real, name)
            raise AttributeError(name)

        def __setattr__(self, name, value):
            if name.startswith("_") or name in extra:
                object.__setattr__(self, name, value)
            else:
                real = self.__dict__.get("_real")
                if real is not None:
                    setattr(real, name, value)
                object.__setattr__(self, name, value)

    try:
        real = __import__(modname, fromlist=["__x__"])
    except Exception:  # noqa: BLE001
        return False
    cur = _s.modules.get(modname)
    if isinstance(cur, _Proxy):
        cur.__dict__["_real"] = real
        return True
    prox = _Proxy(modname)
    object.__setattr__(prox, "_real", cur if cur is not None else real)
    prox.__dict__.update({k: v for k, v in getattr(real, "__dict__", {}).items()
                          if not k.startswith("__")})
    for k, v in extra.items():
        prox.__dict__[k] = v
    _s.modules[modname] = prox
    return True


try:
    _CONSTS2 = {
        "TOKENIZER_CONFIG_FILE": "tokenizer_config.json",
        "CONFIG_NAME": "config.json",
        "WEIGHTS_NAME": "pytorch_model.bin",
        "SAFE_WEIGHTS_NAME": "model.safetensors",
        "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
        "WEIGHTS_INDEX_NAME": "pytorch_model.bin.index.json",
        "SPM_FILE": "spiece.model",
        "TF2_WEIGHTS_NAME": "tf_model.h5",
        "FLAX_WEIGHTS_NAME": "flax_model.msgpack",
    }
    for _mn in ("transformers.tokenization_utils",
                "transformers.tokenization_utils_base"):
        _inject_module_proxy(_mn, _CONSTS2)
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------- 12)
# 我们给 flash_attn 提供了基于 SDPA 的等价实现（_stubs/flash_attn），
# 但 transformers 5.x 的 PACKAGE_DISTRIBUTION_MAPPING 里已删掉 flash_attn 键，
# 于是 is_flash_attn_2_available() 会抛 KeyError。
# 补上这个键，让"有 flash_attn"这件事对 transformers 自洽。
try:
    import transformers.utils.import_utils as _iu

    _iu.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash-attn"])
except Exception:  # noqa: BLE001
    pass
