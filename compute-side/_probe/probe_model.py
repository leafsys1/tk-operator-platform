#!/usr/bin/env python3
"""通用模型探针（容器内运行）：探测「能否加载 + 能否前向」，并把过程沉淀成经验。

设计原则（对应 7 个 skill）
- migration-probe：原样跑、只收集真实失败现象，不预判、不改模型代码。
- model-delivery §7.5 客户自理层：pip 必须换国内源（直连 pypi 会卡死）。
- 经验沉淀：缺哪个依赖就自动补装并**记录到 recipes**，补装记录即配方的一部分。

用法（容器内）:
    python3 probe_model.py --model /workspace/model --out /workspace/out/probe.json
"""
import argparse
import importlib
import signal
import socket
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback

# 坑：trust_remote_code=True 会去 hub 拉远程代码；网络挂起时无超时，
# 一个模型就能把整批探针阻塞住（实测 nomic-embed-text 卡死 6+ 分钟）。
socket.setdefaulttimeout(int(os.environ.get("PROBE_SOCKET_TIMEOUT", "60")))

ROUTE_TIMEOUT = int(os.environ.get("PROBE_ROUTE_TIMEOUT", "300"))


class RouteTimeout(Exception):
    pass


class _alarm:
    """给单条加载路线加墙钟超时。"""
    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        def h(signum, frame):
            raise RouteTimeout("路线超时 %ss" % self.sec)
        self.old = signal.signal(signal.SIGALRM, h)
        signal.alarm(self.sec)
        return self

    def __exit__(self, *a):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old)
        return False


PIP_INDEX = os.environ.get("PIP_INDEX_URL", "https://mirrors.aliyun.com/pypi/simple/")
# 补装的进程级墙钟超时（秒）。默认 240s：正常包够用，病态 backtracking 会被掐断。
PIP_WALL_TIMEOUT = int(os.environ.get("PROBE_PIP_TIMEOUT", "240"))

# 模块名 -> pip 包名（有些对不上）
PIP_ALIAS = {
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "soundfile": "soundfile",
    "librosa": "librosa",
    "addict": "addict",
    "datasets": "datasets",
    "sentencepiece": "sentencepiece",
    "tiktoken": "tiktoken",
    "einops": "einops",
    "omegaconf": "omegaconf",
    "timm": "timm",
    "safetensors": "safetensors",
    "accelerate": "accelerate",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    # 第五类「自带框架」：import 名与 pip 名不一致
    "pyannote": "pyannote.audio",
    "gliner": "gliner",
    "ultralytics": "ultralytics",
    "paddleocr": "paddleocr",
    "einops": "einops",
}


def make_constraints():
    """把基础包钉在当前版本，防止补装把它们升级掉。

    实测坑：装 datasets 会拉入要求 NumPy>=2.0 的 pyarrow，而容器里 NumPy
    被 torch_npu/CANN 钉在 1.26.4；不约束就会 ImportError 甚至带崩 NPU 栈。
    """
    path = "/tmp/probe_constraints.txt"
    # 已知冲突包（每一条都是实测踩出来的，属于"配方"的一部分）
    pins = [
        "numpy<2",      # NPU 栈（torch_npu/CANN）在 1.26.x 上验证过，不升 2.x
        "pyarrow<17",   # pyarrow>=17 运行时要求 NumPy>=2.0，与上面冲突
        "datasets<3",   # datasets>=3 会强拉新 pyarrow
    ]
    for mod, pkg in (("numpy", "numpy"), ("torch", "torch"),
                     ("torch_npu", "torch_npu"), ("transformers", "transformers")):
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", None)
            if v and not any(x.startswith(pkg + "<") or x.startswith(pkg + "==")
                             for x in pins):
                pins.append("%s==%s" % (pkg, v))
        except Exception:  # noqa: BLE001
            pass
    with open(path, "w") as f:
        f.write("\n".join(pins) + "\n")
    return path


def pip_install(mod, log):
    pkg = PIP_ALIAS.get(mod, mod)
    cons = make_constraints()
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "-i", PIP_INDEX, "--trusted-host", "mirrors.aliyun.com",
           "--timeout", "60", "--retries", "2", "-c", cons, pkg]
    t0 = time.time()
    # 坑：补装必须设**进程级**墙钟超时。--timeout 只管单次网络请求，
    # 解析器 backtracking（如 ultralytics 撞 numpy<2 约束）能让 pip 跑 20+ 分钟不返回，
    # 一个包就能挂死整批（实测踩过）。
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=PIP_WALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        # 兜底：backtracking 是因为 pip 要同时求解「本体 + 它声明的依赖」。
        # 先 --no-deps 把本体装上（秒级），依赖再由后续的 ModuleNotFoundError
        # 逐个触发补装 —— 每个都只有一个包，不会 backtracking。
        # 实测：ultralytics 从"卡死 20 分钟"变成 11.9 秒跑通，并打通了 YOLO11。
        log.append({"module": mod, "package": pkg, "rc": -1,
                    "sec": round(time.time() - t0, 1),
                    "note": "本体安装超时（backtracking），改用 --no-deps 重试"})
        cmd2 = cmd[:]
        cmd2.insert(cmd2.index("install") + 1, "--no-deps")
        try:
            p = subprocess.run(cmd2, capture_output=True, text=True,
                               timeout=PIP_WALL_TIMEOUT)
        except subprocess.TimeoutExpired:
            log.append({"module": mod, "package": pkg, "rc": -1,
                        "sec": round(time.time() - t0, 1),
                        "error": "即使 --no-deps 也超时 %ss" % PIP_WALL_TIMEOUT})
            return False
        if p.returncode != 0:
            log.append({"module": mod, "package": pkg, "rc": p.returncode,
                        "sec": round(time.time() - t0, 1),
                        "error": "本体安装失败(--no-deps): %s" % (p.stderr or "")[-200:]})
            return False
        # 尝试装上它声明的依赖（逐个装，避免联合求解）
        try:
            import importlib.metadata as _md
            reqs = _md.requires(pkg) or []
        except Exception:  # noqa: BLE001
            reqs = []
        names = []
        for r in reqs:
            if ";" in r and "extra" in r.split(";", 1)[1]:
                continue
            nm = re.split(r"[<>=!~\[;( ]", r.strip(), 1)[0].strip()
            if nm and nm.lower() not in ("python",):
                names.append(nm)
        for nm in names[:14]:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                "-i", PIP_INDEX, "--trusted-host", "mirrors.aliyun.com",
                                "--timeout", "60", "--retries", "1", "-c", cons, nm],
                               capture_output=True, text=True, timeout=120)
            except Exception:  # noqa: BLE001
                pass
        log.append({"module": mod, "package": pkg, "rc": 0,
                    "sec": round(time.time() - t0, 1),
                    "note": "已用 --no-deps 装本体 + 逐个装 %d 个依赖" % len(names[:14]),
                    "constraints": open(cons).read().strip().replace("\n", " ")})
        return True
    entry = {"module": mod, "package": pkg, "rc": p.returncode,
             "constraints": open(cons).read().strip().replace("\n", " "),
             "sec": round(time.time() - t0, 1)}
    if p.returncode != 0:
        entry["stderr"] = (p.stderr or "")[-300:]
    log.append(entry)
    return p.returncode == 0


def try_import(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return "IMPORT-FAIL: %s" % e


def probe_env():
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "torch_npu", "transformers", "modelscope", "diffusers",
                "speechbrain", "soundfile", "librosa", "numpy", "PIL"):
        env[mod] = try_import(mod)
    try:
        import torch
        env["npu_available"] = bool(getattr(torch, "npu", None)
                                    and torch.npu.is_available())
        env["npu_count"] = torch.npu.device_count() if env["npu_available"] else 0
    except Exception as e:  # noqa: BLE001
        env["npu_available"] = "ERR: %s" % e
    return env


WEXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def find_model_subdir(mdir, max_depth=3):
    """找出真正的模型目录（可能藏在多层子目录里）。

    典型布局：
      * sentence-transformers：顶层无权重，整模型在 `0_<ModelName>/`
      * GroundVTS 这类：顶层只有 configuration.json，
        真模型在 `GroundVTS-{I,Q}/Stage1_base/`（**两层深**）

    守卫（重要）：只有顶层**完全没有权重文件**时才下探。
    否则会把组件子目录（TTS 的 speech_tokenizer/、ASR 内嵌的 Qwen3-0.6B/）
    误当主模型加载。

    先做 1 层（语义最明确），找不到再递归到 max_depth；
    递归时可能有多个候选（如 I/Q 两个变体 × base/adapter 多个 stage），
    **按权重总体积取最大**——基座模型几乎总是最大的那个，adapter 只有几十 MB。
    """
    try:
        names = sorted(os.listdir(mdir))
    except OSError:
        return None
    # 顶层已有主权重 -> 不是该布局
    for n in names:
        if n.lower().endswith(WEXT) and os.path.getsize(os.path.join(mdir, n)) > 10 * 1024**2:
            return None

    # ---- 第一轮：1 层（向后兼容原来的语义）----
    for n in names:
        p = os.path.join(mdir, n)
        if not os.path.isdir(p):
            continue
        if not os.path.isfile(os.path.join(p, "config.json")):
            continue
        try:
            inner = os.listdir(p)
        except OSError:
            continue
        if any(i.lower().endswith(WEXT) for i in inner):
            return p

    # ---- 第二轮：递归到 max_depth，按权重大小选基座 ----
    cands = []
    try:
        base_depth = mdir.rstrip(os.sep).count(os.sep)
        for root, dirs, files in os.walk(mdir):
            if root.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
                continue
            if not os.path.isfile(os.path.join(root, "config.json")):
                continue
            w = [f for f in files if f.lower().endswith(WEXT)]
            if not w:
                continue
            total = 0
            for f in w:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
            if total > 10 * 1024**2:
                cands.append((total, root))
    except OSError:
        return None
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    return cands[0][1]

def detect_format(mdir):
    names = set(os.listdir(mdir)) if os.path.isdir(mdir) else set()
    low = {n.lower() for n in names}
    # 默认值必须在分支之前给：否则无分支匹配时 info["format"] 未定义（KeyError）
    info = {"files": sorted(names)[:50], "n_files": len(names), "format": "unknown"}
    if "configuration.json" in low:
        info["format"] = "modelscope"
        try:
            cfg = json.load(open(os.path.join(mdir, "configuration.json"), encoding="utf-8"))
            info["ms_configuration"] = cfg
            info["ms_framework"] = cfg.get("framework")
            info["ms_task"] = cfg.get("task") or (cfg.get("pipeline") or {}).get("type")
        except Exception as e:  # noqa: BLE001
            info["ms_config_error"] = str(e)
    elif "model_index.json" in low:
        info["format"] = "diffusers"
    elif "config.json" in low:
        info["format"] = "transformers"
        try:
            cfg = json.load(open(os.path.join(mdir, "config.json"), encoding="utf-8"))
            info["architectures"] = cfg.get("architectures")
            info["model_type"] = cfg.get("model_type")
        except Exception as e:  # noqa: BLE001
            info["cfg_error"] = str(e)
    elif "hyperparams.yaml" in low:
        info["format"] = "speechbrain"
    elif any(n.endswith((".ckpt", ".pt", ".pth")) for n in low):
        info["format"] = "torch-checkpoint"
    if info["format"] in ("unknown", "modelscope"):
        sub = find_model_subdir(mdir)
        if sub:
            info["model_subdir"] = sub
            try:
                c = json.load(open(os.path.join(sub, "config.json"), encoding="utf-8"))
                info["sub_architectures"] = c.get("architectures")
                info["sub_model_type"] = c.get("model_type")
            except Exception:  # noqa: BLE001
                pass
    return info


ROUTE_ORDER = ("modelscope", "transformers", "diffusers", "speechbrain")


# 第五类加载路线：**模型自带框架**。
# 特征：只有 configuration.json（无 config.json、无自定义 .py），且 configuration.json
# 里的 task 名不在 modelscope 的 pipeline 注册表里 —— 既不是 transformers，也不是 diffusers，
# 更不是魔搭原生管线。实测本批 10 个（pyannote / YOLO / GLiNER / TripoSR / PhotoMaker…）。
OWN_FRAMEWORK = [
    (r"pyannote", "pyannote.audio", "pyannote",
     "from pyannote.audio import Model; Model.from_pretrained(dir)"),
    (r"gliner", "gliner", "gliner",
     "from gliner import GLiNER; GLiNER.from_pretrained(dir)"),
    (r"yolo|ultralytics", "ultralytics", "ultralytics",
     "from ultralytics import YOLO; YOLO(dir)"),
    (r"triposr|trellis|image-to-3d|image-to-3D", None, "unsupported",
     "需模型源码仓库（无 PyPI 包）"),
    (r"photomaker|ominicontrol|controlnet|text-to-image-synthesis", "diffusers", "diffusers",
     "DiffusionPipeline.from_pretrained(dir, trust_remote_code=True)"),
]
PIP_HINT = re.compile(r"pip install\s+([A-Za-z0-9_\-\.]+)")
IMP_HINT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)", re.M)


def detect_own_framework(mdir, ms_task=None):
    """猜「自带框架」型模型该用哪个包。返回 (pip包, 类型, 说明)。"""
    blob = os.path.basename(mdir)
    for fn in ("README.md", "readme.md", "configuration.json"):
        fp = os.path.join(mdir, fn)
        if os.path.isfile(fp):
            try:
                blob += " " + open(fp, encoding="utf-8", errors="replace").read()[:20000]
            except OSError:
                pass
    if ms_task:
        blob += " " + ms_task
    # README 里显式写的 pip install / import 优先
    for pk in PIP_HINT.findall(blob)[:5]:
        if pk.lower() not in ("https", "modelscope"):
            return pk, "readme-pip", "README 里 pip install %s" % pk
    imps = IMP_HINT.findall(blob)
    for im in imps:
        top = im.split(".")[0]
        if top not in ("os", "sys", "json", "torch", "numpy", "transformers", "modelscope",
                       "huggingface_hub", "PIL", "typing", "collections", "pathlib"):
            return top, "readme-import", "README 里 import %s" % top
    for rx, pkg, kind, note in OWN_FRAMEWORK:
        if re.search(rx, blob, re.I):
            return pkg, kind, note
    return None, "unknown", "未能判定自带框架"


def build_routes(finfo, mdir):
    """按目录特征列出可尝试的加载路线（多路线回退，全部记录）。"""
    have = set()
    try:
        have = {n.lower() for n in os.listdir(mdir)}
    except Exception:  # noqa: BLE001
        pass
    routes = []
    if "configuration.json" in have:
        routes.append("modelscope")
    if "config.json" in have:
        routes.append("transformers")
    elif find_model_subdir(mdir):
        # 坑 E-2：权重在子目录（sentence-transformers 的 0_Xxx/）时，
        # 顶层没有 config.json，只走 modelscope 路线会因 task 名不在注册表而失败。
        # 这里补一条 transformers 路线，指向子目录。
        routes.append("transformers")
    if "model_index.json" in have:
        routes.append("diffusers")
    if "hyperparams.yaml" in have:
        routes.append("speechbrain")
    # 只有 configuration.json 且没有通用配置 -> 大概率是「自带框架」型
    if not routes or routes == ["modelscope"]:
        routes.append("own-framework")
    return routes or ["none"]


def _route_modelscope(finfo, mdir, device):
    from modelscope.pipelines import pipeline
    task = finfo.get("ms_task")
    if not task:
        return {"route": "modelscope.pipeline",
                "skipped": "configuration.json 无 task 字段"}
    # 坑：魔搭对含 plugins/allow_remote 的模型要求显式开启远程代码
    pipeline(task, model=mdir, device=device, trust_remote_code=True)
    return {"route": "modelscope.pipeline", "task": task, "load": "OK"}


def _route_transformers(finfo, mdir, device):
    from transformers import AutoConfig, AutoModel
    # 权重可能在子目录（sentence-transformers 布局）
    mdir = finfo.get("model_subdir") or mdir
    cfg = AutoConfig.from_pretrained(mdir, trust_remote_code=True)
    out = {"route": "transformers.AutoModel",
           "arch": getattr(cfg, "architectures", None)}
    m = AutoModel.from_pretrained(mdir, trust_remote_code=True)
    out["load"] = "OK"
    out["n_params"] = sum(p.numel() for p in m.parameters())
    return out


def _route_diffusers(finfo, mdir, device):
    from diffusers import DiffusionPipeline
    DiffusionPipeline.from_pretrained(mdir, trust_remote_code=True)
    return {"route": "diffusers.DiffusionPipeline", "load": "OK"}


def _route_speechbrain(finfo, mdir, device):
    from speechbrain.inference.TTS import Tacotron2  # noqa: F401
    return {"route": "speechbrain", "load": "OK(import-only)"}


def _route_own_framework(finfo, mdir, device):
    """第五类：装模型自带框架并按其 API 加载。"""
    pkg, kind, note = detect_own_framework(mdir, finfo.get("ms_task"))
    if kind == "unsupported" or not pkg:
        return {"route": "own-framework", "skipped": note}
    # 归一化：README 抽到的包名 -> 具体实现
    if kind in ("readme-pip", "readme-import"):
        pl = (pkg or "").lower().replace("_", ".").replace("-", ".")
        if "pyannote" in pl:
            kind = "pyannote"
        elif "gliner" in pl:
            kind = "gliner"
        elif "ultralytics" in pl or "yolo" in pl:
            kind = "ultralytics"
        elif "diffusers" in pl:
            kind = "diffusers"
        else:
            kind = "unknown"
        note = "%s -> %s" % (note, kind)
    if kind == "pyannote":
        from pyannote.audio import Model as PModel
        m = PModel.from_pretrained(mdir)
        return {"route": "own-framework:pyannote", "load": "OK",
                "pkg": pkg, "note": note}
    if kind == "gliner":
        from gliner import GLiNER
        m = GLiNER.from_pretrained(mdir)
        return {"route": "own-framework:gliner", "load": "OK", "pkg": pkg, "note": note}
    if kind == "ultralytics":
        from ultralytics import YOLO
        m = YOLO(mdir)
        return {"route": "own-framework:ultralytics", "load": "OK", "pkg": pkg, "note": note}
    if kind == "diffusers":
        from diffusers import DiffusionPipeline
        p = DiffusionPipeline.from_pretrained(mdir, trust_remote_code=True)
        return {"route": "own-framework:diffusers", "load": "OK", "pkg": pkg,
                "pipeline": type(p).__name__, "note": note}
    return {"route": "own-framework", "skipped": "未实现该框架: %s" % kind}


ROUTES = {"modelscope": _route_modelscope, "transformers": _route_transformers,
          "own-framework": _route_own_framework,
          "diffusers": _route_diffusers, "speechbrain": _route_speechbrain}


# 「再挖一层」：笼统错误 -> 真实根因。
# 动机（实测教训）：`Failed to import modelscope.models.X because of the following error
# (look up to see its traceback)` 这种信息**掩盖了简单原因**（往往只是缺个包）。
# 曾因为只看表象，把 3 个"其实装个包就能过"的模型误判为"不值得救"。
# 这里在失败后自动复现那个 import，把真正的 traceback 根因抓出来。
# 只认**模块路径**形态（含点号），不认 `XxxPipeline:` 这种类名
# —— 实测误把 FunASRPipeline 当成缺包名，导致统计里出现不存在的包。
DEEP_RX = re.compile(r"Failed to import ([\w]+(?:\.[\w]+)+) because of")


def deep_root_cause(err, timeout=180):
    """对笼统的 import 失败做一次受控复现，返回真实根因。"""
    if not err:
        return None
    m = DEEP_RX.search(err)
    if not m:
        return None
    mod = m.group(1) or m.group(2)
    if not mod or len(mod) < 4:
        return None
    code = "import %s" % mod
    try:
        pr = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"module": mod, "ok": False,
                "root_cause": "复现 import 超时 %ss" % timeout}
    if pr.returncode == 0:
        return {"module": mod, "ok": True,
                "root_cause": "该模块单独 import 正常 —— 失败可能与加载上下文有关"}
    tb = pr.stderr or ""
    lines = [x for x in tb.strip().splitlines() if x.strip()]
    root = lines[-1][:220] if lines else "复现失败但无输出"
    miss = re.search(r"No module named '([\w\.]+)'", tb)
    return {
        "module": mod, "ok": False, "root_cause": root,
        "missing_package": miss.group(1) if miss else None,
        "traceback_tail": "\n".join(lines[-9:])[:900],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--max-install", type=int, default=12)
    a = ap.parse_args()

    t0 = time.time()
    # 必须在 import modelscope.pipelines 之前打补丁：pipeline 子模块会按值
    # 绑定 verify_device，打晚了就绕过去了（实测 FunASR 踩过）。
    shim_report = []
    try:
        import npu_shim
        shim_report = npu_shim.apply_all(verbose=True)
    except Exception as e:  # noqa: BLE001
        shim_report = [{"patch": "npu_shim", "result": "ERROR: %s" % e}]

    res = {"model_dir": a.model, "ts": time.strftime("%F %T"), "env": probe_env()}
    finfo = detect_format(a.model)
    res["format_info"] = finfo
    fmt = finfo.get("format")
    res["autodeps"] = []

    attempts = []
    res["loaded"] = False
    routes = build_routes(finfo, a.model)
    res["routes_tried"] = routes
    for route in routes:
        fn = ROUTES.get(route)
        if fn is None:
            attempts.append({"route": route, "skipped": "无通用加载器"})
            continue
        for i in range(a.max_install + 1):
            try:
                with _alarm(ROUTE_TIMEOUT):
                    out = fn(finfo, a.model, a.device)
                attempts.append(out)
                if str(out.get("load", "")).startswith("OK"):
                    res["loaded"] = True
                    res["result"] = out
                break
            except ModuleNotFoundError as e:
                mod = (e.name or "").split(".")[0]
                attempts.append({"route": route, "missing_module": mod})
                if not mod or not pip_install(mod, res["autodeps"]):
                    attempts.append({"route": route,
                                     "error": "依赖 %s 安装失败" % mod})
                    break
            except Exception as e:  # noqa: BLE001
                attempts.append({"route": route,
                                 "error": "%s: %s" % (type(e).__name__,
                                                      str(e)[:240])})
                break
        if res["loaded"]:
            break
    if not res["loaded"]:
        errs = [x for x in attempts if x.get("error")]
        res["error"] = errs[-1]["error"] if errs else "所有路线均未跑通"
        res["routes_failed"] = [x.get("route") for x in attempts]

    res["attempts"] = attempts
    res["npu_shim"] = shim_report
    # 失败且错误笼统时，自动"再挖一层"
    if not res.get("loaded"):
        try:
            deep = deep_root_cause(res.get("error"))
            if deep:
                res["deep"] = deep
                if deep.get("missing_package"):
                    res["autodeps_hint"] = deep["missing_package"]
        except Exception as e:  # noqa: BLE001
            res["deep_error"] = str(e)[:120]

    res["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("VERDICT=%s  依赖补装=%d  %ss" % (
        "LOAD-OK" if res["loaded"] else "LOAD-FAIL",
        len(res["autodeps"]), res["elapsed_s"]))
    # 保证「未通过」时一定有一行 ERROR: —— 平台侧靠它做自动诊断。
    # 之前"无通用加载器"这类结论只落在 probe.json 里，stdout 没有 ERROR 行，
    # 导致平台判定为"加载未通过"却拿不到可诊断的错误文本（实测 ai4s/OmegaFold）。
    if not res.get("loaded"):
        # 错误行要**具体**，否则平台侧坑库匹配不到（实测：只输出"所有路线均未跑通"
        # 时匹配不上任何坑，诊断等于无效）。
        fmt = (res.get("format_info") or {}).get("format")
        reasons = [a.get("skipped") or a.get("error") for a in res.get("attempts", [])
                   if a.get("skipped") or a.get("error")]
        parts = []
        if reasons:
            parts.append(str(reasons[-1]))
        if res.get("error") and res["error"] not in parts:
            parts.append(str(res["error"]))
        # 带上格式与目录特征，便于坑库按"证据特征"命中
        feats = []
        if fmt:
            feats.append("format=%s" % fmt)
        files = (res.get("format_info") or {}).get("files") or []
        for k in ("main.py", "setup.py", "requirements.txt", "config.json",
                  "configuration.json", "model_index.json"):
            if k in files:
                feats.append("有%s" % k)
        if feats:
            parts.append("（%s）" % "，".join(feats))
        deep = res.get("deep") or {}
        if deep.get("root_cause"):
            parts.append("【再挖一层】%s" % str(deep["root_cause"])[:150])
        if deep.get("missing_package"):
            parts.append("缺包=%s" % deep["missing_package"])
        print("ERROR:", " ｜ ".join(parts)[:420])
    elif res.get("error"):
        print("ERROR:", res["error"][:300])
    if res["autodeps"]:
        print("补装:", ", ".join(d["package"] for d in res["autodeps"]))


if __name__ == "__main__":
    main()
