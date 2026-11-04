#!/usr/bin/env python3
"""一键模型迁移平台 · 后端（零第三方依赖，纯标准库）。

设计对应最终目标：
  客户在网页输入模型 -> 后端判定走「经验快速路径」还是「逐步迁移路径」
  -> 实时把阶段进度推给网页。

复用 /data/model migration/platform/harness 的阶段模型：
  resolve -> match_recipe -> prepare_env -> adapt -> validate_infer -> quality -> deliver

存储：SQLite（jobs / events），与 harness 的 store.py 同构。
推送：SSE（Server-Sent Events）——events 是 append-only，天然匹配；
      断线重连带 ?after=<seq> 即可续上。

用法: python3 app.py [--port 8090] [--root /data/model_corpus]
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.environ.get("CORPUS_ROOT", "/data/model_corpus")
PROBE = os.path.join(ROOT, "_probe", "probe.sh")

# ---------------------------------------------------------------- 算力服务器
# 拓扑：虚拟机放代码 + 平台；算力服务器（61.241.77.3）放权重 + NPU。
# 因此"迁移"这件事必须通过 SSH 到算力服务器上执行，
# 平台本身只负责编排 + 把过程流式推给前端。
NPU_SH = os.environ.get("NPU_SH")   # 为空时按当前 ROOT 延迟推导
REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "/data/model_corpus")


def _npu_sh():
    """按当前 ROOT 推导 npu.sh 路径（ROOT 会被 --root 参数覆盖，不能提前算死）。"""
    if NPU_SH:
        return NPU_SH
    for cand in (os.path.join(os.path.dirname(ROOT), "npu.sh"),
                 os.path.join(ROOT, "npu.sh"),
                 os.path.expanduser("~/workspace/npu.sh")):
        if os.path.isfile(cand):
            return cand
    return os.path.expanduser("~/workspace/npu.sh")


def npu(cmd, timeout=120):
    """在算力服务器上执行命令，返回 (returncode, 合并输出)。"""
    sh = _npu_sh()
    if not os.path.isfile(sh):
        return 127, "算力服务器连接脚本不存在：%s" % sh
    try:
        p = subprocess.run(["bash", sh, cmd], capture_output=True,
                           text=True, errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "算力服务器命令超时（%ss）" % timeout
    except Exception as e:  # noqa: BLE001
        return 1, "%s: %s" % (type(e).__name__, str(e)[:200])


_REMOTE_CACHE = {"ts": 0.0, "dirs": set()}


def _remote_dirs(force=False):
    """缓存算力服务器上的 '<分类>/<org__name>' 目录集合（60 秒）。"""
    now = time.time()
    if not force and _REMOTE_CACHE["dirs"] and now - _REMOTE_CACHE["ts"] < 60:
        return _REMOTE_CACHE["dirs"]
    rc, out = npu("find %s -mindepth 2 -maxdepth 2 -type d -printf '%%P\\n' 2>/dev/null"
                  % REMOTE_ROOT, 90)
    dirs = set()
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if "/" in line and "__" in line and not line.startswith("_"):
                dirs.add(line)
    _REMOTE_CACHE["ts"] = now
    _REMOTE_CACHE["dirs"] = dirs
    return dirs
DB = os.path.join(ROOT, "_platform", "platform.db")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

PHASES = ["resolve", "match_recipe", "prepare_env", "adapt",
          "validate_infer", "quality", "deliver"]
PHASE_CN = {
    "resolve": "解析模型", "match_recipe": "匹配配方", "prepare_env": "准备环境",
    "adapt": "执行适配", "validate_infer": "验证推理", "quality": "冒烟测试",
    "deliver": "归档交付",
}
# 五个官方分类的中文名（前端与 /api/experience 共用同一套叫法）
CAT_CN = {
    "nlp": "自然语言处理", "cv": "计算机视觉", "multi-modal": "多模态",
    "audio": "语音", "scientific-computing": "科学计算",
}
# /static/* 的 Content-Type 映射（缺了会让浏览器按二进制处理，CSS/JS 不生效）
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".md": "text/markdown; charset=utf-8",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, model TEXT NOT NULL, route TEXT,
  status TEXT NOT NULL, phase TEXT, exit_code INTEGER, error TEXT,
  result_json TEXT, created_at REAL, started_at REAL, finished_at REAL);
CREATE TABLE IF NOT EXISTS events (
  job_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL,
  phase TEXT, status TEXT, message TEXT, PRIMARY KEY (job_id, seq));
"""

_lock = threading.Lock()


def db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    return c


def kill_orphan_downloads():
    """清理算力服务器上遗留的下载进程。

    平台重启后 SSH 连接断了，但远端的 `python3 /tmp/_dl.py` 不会跟着死，
    多次重启会堆积成多个并发下载抢带宽（实测堆过 7 个、缓存 47GB）。
    """
    try:
        rc, out = npu("pkill -9 -f '/tmp/_dl.py' 2>/dev/null; "
                      "sleep 1; ps aux | grep -c '[_]dl.py'", 60)
        n = (out or "").strip().splitlines()[-1] if out else "?"
        return n
    except Exception:  # noqa: BLE001
        return "?"


def reap_orphans(c):
    """平台重启后，上一次进程里的任务线程都没了，但库里还留着 running/pending。

    不清理的话：页面显示"进行中"但永远不动，而且这些任务占着卡调度名额。
    这里把它们标成 interrupted，并如实写明原因。
    """
    try:
        rows = c.execute(
            "SELECT job_id FROM jobs WHERE status IN ('running','pending','queued')").fetchall()
        if not rows:
            return 0
        now = time.time()
        for r in rows:
            c.execute(
                "UPDATE jobs SET status='failed', phase=COALESCE(phase,'adapt'), "
                "finished_at=?, error=? WHERE job_id=?",
                (now, "平台重启，任务线程已中断（重新提交即可继续）", r["job_id"]))
            emit(c, r["job_id"], "adapt", "warn",
                 "平台重启，本任务已中断；重新提交同一模型会走已缓存的内容，很快完成")
        c.commit()
        return len(rows)
    except Exception:  # noqa: BLE001
        return 0


def emit(c, job_id, phase, status, message):
    row = c.execute("SELECT COALESCE(MAX(seq),0)+1 n FROM events WHERE job_id=?",
                    (job_id,)).fetchone()
    c.execute("INSERT INTO events (job_id,seq,ts,phase,status,message) VALUES (?,?,?,?,?,?)",
              (job_id, row["n"], time.time(), phase, status, message))
    # 同步 jobs.phase：否则并发时队列里显示的阶段会一直停在旧值（实测停在 resolve）
    if phase:
        try:
            c.execute("UPDATE jobs SET phase=? WHERE job_id=?", (phase, job_id))
            c.commit()
        except Exception:  # noqa: BLE001
            pass
    return row["n"]


# ---------------------------------------------------------------- 配方库
def load_recipes():
    """已探通（LOAD-OK）的模型 = 已验证配方，走快速路径。"""
    reg = {}
    rp = os.path.join(ROOT, "_manifest", "registry.json")
    if os.path.isfile(rp):
        for m in json.load(open(rp, encoding="utf-8")).get("models", []):
            reg[m["_key"]] = m
    ok = {}
    for d in os.listdir(os.path.join(ROOT, "_probe")) if os.path.isdir(os.path.join(ROOT, "_probe")) else []:
        # 优先读权威配方记录 recipe.json（救援产物，重跑探针不会覆盖它）；
        # 退化到 probe.json（通用探针结果，可被重跑覆盖 —— 曾因此弄丢过配方）
        rj = os.path.join(ROOT, "_probe", d, "recipe.json")
        pj = os.path.join(ROOT, "_probe", d, "probe.json")
        src = rj if os.path.isfile(rj) else pj
        if not os.path.isfile(src):
            continue
        try:
            p = json.load(open(src, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if p.get("loaded"):
            mid = p.get("model") or d.replace("__", "/")
            ok[mid] = {"model": mid,
                       "route": p.get("route") or (p.get("result") or {}).get("route"),
                       "elapsed_s": p.get("elapsed_s"),
                       "autodeps": [(x.get("package") if isinstance(x, dict) else str(x))
                       for x in (p.get("autodeps") or [])],
                       "category": (reg.get(mid) or {}).get("category"),
                       "gb": (reg.get(mid) or {}).get("gb"),
                       "authoritative": bool(p.get("authoritative"))}
    return ok


def load_corpus_status():
    """100 模型迁移状态总览（离线汇总，看板数据源）。"""
    p = os.path.join(ROOT, "_manifest", "corpus_status.json")
    if not os.path.isfile(p):
        return {"total": 0, "items": []}
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"total": 0, "items": []}


_PIT = {"entries": [], "loaded": 0}


DELIV = os.path.join(ROOT, "_delivery")


def load_deliveries():
    """交付包清单：每个已迁移模型一个自包含目录（model-delivery skill 规范）。"""
    if not os.path.isdir(DELIV):
        return {"count": 0, "items": []}
    items = []
    for name in sorted(os.listdir(DELIV)):
        d = os.path.join(DELIV, name)
        if not os.path.isdir(d) or name in ("images", "archives"):
            continue
        it = {"package": name, "path": d}
        for f in ("DEPLOY.md", "TEST_REPORT.md", "ADAPTATION.md",
                  "run.sh", "entry.py", "validate_delivery.py",
                  "weights_manifest.json", "PERF_REPORT.md", "perf.json",
                  "weights_image.json", "ACCURACY_REPORT.md"):
            it[f] = os.path.isfile(os.path.join(d, f))
        it["has_adaptation"] = it.get("ADAPTATION.md", False)
        # 目录总体积（前端展示 + 估算 tar 大小）
        try:
            tot = 0
            for _rt, _ds, _fs in os.walk(d):
                for _f in _fs:
                    try:
                        tot += os.path.getsize(os.path.join(_rt, _f))
                    except OSError:
                        pass
            it["size"] = tot
        except Exception:  # noqa: BLE001
            it["size"] = 0
        # 从 TEST_REPORT 抓判定
        try:
            t = open(os.path.join(d, "TEST_REPORT.md"), encoding="utf-8").read()
            it["verdict"] = "PASS" if "✅ PASS" in t else ("未通过" if "❌" in t else "未记录")
            m = re.search(r"\|\s*耗时\s*\|\s*([^|]+)\|", t)
            it["elapsed"] = m.group(1).strip() if m else "—"
            m = re.search(r"\|\s*架构\s*\|\s*([^|]+)\|", t)
            it["arch"] = m.group(1).strip() if m else "—"
        except Exception:  # noqa: BLE001
            it["verdict"] = "未记录"
        items.append(it)
    return {"count": len(items), "items": items,
            "note": "每个模型一个自包含交付包（无总索引），遵循 model-delivery skill"}


def load_pitfalls():
    """加载坑库索引（build_pitfall_index.py 产出），供报错匹配。"""
    global _PIT
    fp = os.path.join(ROOT, "_manifest", "pitfalls.json")
    if not os.path.isfile(fp):
        return _PIT
    try:
        st = os.path.getmtime(fp)
    except OSError:
        st = 0
    if _PIT.get("mtime") != st:
        d = json.load(open(fp, encoding="utf-8"))
        _PIT = {"entries": d.get("entries", []), "loaded": d.get("count", 0), "mtime": st}
    return _PIT


def diagnose(err, topn=3):
    """按证据特征匹配坑库。对应 pitfall-index skill 的「先想特征，再找最像的经验」。"""
    ent = load_pitfalls().get("entries") or []
    if not err or not ent:
        return []
    low = err.lower()
    out = []
    for e in ent:
        score, hits = 0.0, []
        for sig in e.get("signatures") or []:
            sl = sig.lower()
            if len(sl) < 4:
                continue
            if sl in low:
                score += 1.0 + min(len(sl) / 20.0, 3.0)
                hits.append(sig)
        for kw, b in (("no module named", "A"), ("modulenotfound", "A"),
                      ("device should be", "B"), ("memoryformat", "B"),
                      ("not supported on the npu", "C"), ("aicore", "C"),
                      ("unknown task", "E"), ("unrecognized", "E")):
            if kw in low and e["bucket"] == b:
                score += 0.4
        if score > 0:
            out.append({"score": round(score, 2), "hits": hits[:5],
                        "bucket": e["bucket"], "bucket_cn": e["bucket_cn"],
                        "symptom": e["symptom"], "root_cause": e.get("root_cause", ""),
                        "direction": e.get("direction", ""), "case": e.get("case", ""),
                        "id": e.get("id", "")})
    out.sort(key=lambda x: -x["score"])
    return out[:topn]


def load_adaptations():
    """已救援模型的适配记录（现象→根因→修法），供网页展示人工介入点。"""
    p = os.path.join(ROOT, "_manifest", "adaptations.json")
    if not os.path.isfile(p):
        return {}
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {x["model"]: x for x in d.get("items", [])}


def find_corpus_dir(model):
    """把 'org/name' 映射到语料目录。

    先查虚拟机本地，再查算力服务器（权重实际存放处）。
    算力服务器命中时返回远程绝对路径，后续 probe 也在远程执行。
    """
    safe = model.replace("/", "__")
    for cat in ("nlp", "multi-modal", "audio", "cv", "scientific-computing"):
        d = os.path.join(ROOT, cat, safe)
        if os.path.isdir(d):
            return cat, d
        if "%s/%s" % (cat, safe) in _remote_dirs():
            return cat, "%s/%s/%s" % (REMOTE_ROOT, cat, safe)
    return None, None


# 魔搭 task -> 语料分类（自动下载后据此归位）
TASK2CAT = {
    "text-generation": "nlp", "text-classification": "nlp",
    "fill-mask": "nlp", "sentence-similarity": "nlp",
    "feature-extraction": "nlp", "text-ranking": "nlp",
    "token-classification": "nlp", "text2text-generation": "nlp",
    "image-classification": "cv", "object-detection": "cv",
    "image-segmentation": "cv", "depth-estimation": "cv",
    "image-to-image": "cv", "video-classification": "cv",
    "visual-question-answering": "multi-modal", "image-text-to-text": "multi-modal",
    "text-to-image": "multi-modal", "image-captioning": "multi-modal",
    "multi-modal-embedding": "multi-modal", "visual-entailment": "multi-modal",
    "text-to-speech": "audio", "automatic-speech-recognition": "audio",
    "audio-classification": "audio", "voice-activity-detection": "audio",
    "protein-structure": "scientific-computing",
}


def guess_category(model):
    """未入库时先按名字关键字猜，猜不到默认 nlp。"""
    m = model.lower()
    for kw, cat in (("speech", "audio"), ("asr", "audio"), ("tts", "audio"),
                    ("whisper", "audio"), ("voice", "audio"), ("audio", "audio"),
                    ("clip", "multi-modal"), ("blip", "multi-modal"),
                    ("vl", "multi-modal"), ("vlm", "multi-modal"),
                    ("vision-language", "multi-modal"), ("caption", "multi-modal"),
                    ("protein", "scientific-computing"), ("fold", "scientific-computing"),
                    ("molecul", "scientific-computing"), ("chem", "scientific-computing"),
                    ("vit", "cv"), ("resnet", "cv"), ("yolo", "cv"),
                    ("detr", "cv"), ("seg", "cv"), ("sam", "cv"),
                    ("swin", "cv"), ("convnext", "cv"), ("diffusion", "cv")):
        if kw in m:
            return cat
    return "nlp"


# ---------------------------------------------------------------- 卡调度
# 并发跑多个迁移时，每个任务必须落在不同的空闲卡上。
# 只用 npu-smi 判空闲还不够：两个任务可能同时抢到同一张，
# 所以再叠一层进程内的预留集合。
_CARD_LOCK = threading.Lock()
_CARD_BUSY = set()


def _free_cards():
    """返回算力服务器上当前无进程的卡号列表（0..7）。"""
    cmd = ('for i in 0 1 2 3 4 5 6 7; do '
           's=$(npu-smi info -t proc-mem -i $i 2>/dev/null); '
           'if echo "$s" | grep -q "No process in device"; then echo "$i"; fi; done')
    rc, out = npu(cmd, 90)
    cards = []
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit() and 0 <= int(line) <= 7:
                cards.append(int(line))
    return sorted(cards)


# 可用卡上限：算力服务器 8 张，但被别人占用的卡不算。
# 超过这个数就排队 —— 这就是"显存/算力已满"的判据。
MAX_CARDS = int(os.environ.get("MAX_CARDS", "8"))


def _usable_cards():
    """本平台可用的卡：无外进程、且不超过 MAX_CARDS。"""
    return _free_cards()[:MAX_CARDS]


def release_card(card):
    """释放卡位。必须在任务结束时调用（含异常路径），否则卡会被永久占用。"""
    if card is None:
        return
    with _CARD_LOCK:
        _CARD_BUSY.discard(card)
    # 让容量缓存尽快失效，排队任务能立刻看到空出来的卡
    _CAP_CACHE["ts"] = 0.0


_CAP_CACHE = {"ts": 0.0, "usable": None}


def _usable_cached(ttl=8):
    """_free_cards() 每次要 SSH 跑 8 张卡的 npu-smi（5-10 秒），
    前端每 4 秒轮询一次受不了，所以缓存 usable 列表。"""
    now = time.time()
    if _CAP_CACHE["usable"] is not None and now - _CAP_CACHE["ts"] < ttl:
        return _CAP_CACHE["usable"]
    u = _free_cards()[:MAX_CARDS]
    _CAP_CACHE["ts"] = now
    _CAP_CACHE["usable"] = u
    return u


def capacity():
    """当前算力容量快照，供前端展示与排队判定。"""
    with _CARD_LOCK:
        busy = set(_CARD_BUSY)
    usable = _usable_cached()
    # 用并集去重：一张卡刚被平台占用时，会同时出现在「缓存里的 usable」
    # 和「实时的 busy」中；直接 len(usable)+len(busy) 会把它算两次，
    # 导致 in_use + free != total（实测 total 在 5/6 之间跳变）。
    total = min(MAX_CARDS, len(set(usable) | busy))
    return {
        "max_cards": MAX_CARDS,
        "total": total,
        "in_use": len(busy),
        "free": len([c for c in usable if c not in busy]),
        "busy_cards": sorted(busy),
        "free_cards": [c for c in usable if c not in busy],
    }


def wait_for_card(job_id, c, model=""):
    """排队等卡：没卡就把任务置为 queued 并阻塞等待，拿到卡返回卡号。

    这是"并行 + 排队"的核心：
      - 有空卡 -> 立刻占用，状态回到 running
      - 无空卡 -> 标记排队，每隔几秒重试；前面任务释放后自动开始
    """
    notified = False
    while True:
        with _CARD_LOCK:
            # 关键：限流看的是"已预留卡数"，不是物理空闲卡数。
            # 否则刚预留、docker 还没起来的卡仍显示为空闲，
            # 新任务会再抢一张 -> 突破 MAX_CARDS（实测跑出 3 个并发）。
            busy = len(_CARD_BUSY)
            if busy < MAX_CARDS:
                usable = _usable_cached()
                free = [x for x in usable if x not in _CARD_BUSY]
                if free:
                    card = free[0]
                    _CARD_BUSY.add(card)
                    c.execute("UPDATE jobs SET status='running' WHERE job_id=?", (job_id,))
                    return card
            cap = MAX_CARDS
        if not notified:
            ahead = c.execute(
                "SELECT COUNT(*) n FROM jobs q WHERE q.status IN ('queued','running') "
                "AND q.created_at < (SELECT created_at FROM jobs WHERE job_id=?)",
                (job_id,)).fetchone()["n"]
            c.execute("UPDATE jobs SET status='queued', phase='adapt' WHERE job_id=?", (job_id,))
            emit(c, job_id, "adapt", "queued",
                 "算力已满（%d/%d 张卡在用）→ 进入排队，前面还有 %d 个任务"
                 % (busy, cap, max(0, ahead - busy)))
            notified = True
        time.sleep(5)


# ---------------------------------------------------------------- 体积上限
# 910B1 单卡 HBM = 64 GB（实测 HBM Capacity 65536 MB），8 卡合计 512 GB。
# 权重超过这个量级，任何并行策略都装不下 -> 直接报错，不要浪费几十分钟下载。
CARD_HBM_GB = int(os.environ.get("CARD_HBM_GB", "64"))
MAX_MODEL_GB = int(os.environ.get("MAX_MODEL_GB", str(CARD_HBM_GB * 8)))


# 同一模型只允许一个下载：多个任务同时点同一个模型时，
# 后到的等前一个下完直接用缓存，避免 N 份并发下载抢带宽。
_DL_LOCK = threading.Lock()
_DL_INFLIGHT = {}


def _acquire_download(model):
    """返回 True 表示由我来下；False 表示别人已在下（等下完用缓存）。"""
    with _DL_LOCK:
        if model in _DL_INFLIGHT:
            return False
        _DL_INFLIGHT[model] = time.time()
        return True


def _release_download(model):
    with _DL_LOCK:
        _DL_INFLIGHT.pop(model, None)


def _wait_download(model, timeout=1800):
    """等别的任务把同一个模型下完。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        with _DL_LOCK:
            if model not in _DL_INFLIGHT:
                return True
        time.sleep(5)
    return False


def remote_model_size_gb(model):
    """查询模型在魔搭上的总大小（GB）。失败返回 None（不阻塞流程）。"""
    py = (
        "import sys\n"
        "from modelscope.hub.api import HubApi\n"
        "api=HubApi()\n"
        "fs=api.get_model_files(sys.argv[1], recursive=True)\n"
        "t=0\n"
        "for f in fs:\n"
        "    n=f.get('Name') if isinstance(f,dict) else getattr(f,'Name',None)\n"
        "    sz=f.get('Size') if isinstance(f,dict) else getattr(f,'Size',None)\n"
        "    if sz is None: continue\n"
        "    t+=int(sz or 0)\n"
        "print('SIZE_GB', round(t/1024.0/1024/1024, 2))\n"
    )
    # ModelScope 的 get_model_files 实测要 47 秒（大仓库更久）。
    # 这里只给它 25 秒：查不到就放行，靠下载后的真实目录大小做权威判定。
    rc, out = npu("cat > /tmp/_sz.py <<'PYEOF'\n%s\nPYEOF\npython3 /tmp/_sz.py %s"
                  % (py, model), 25)
    for line in (out or "").splitlines():
        if line.startswith("SIZE_GB"):
            try:
                return float(line.split()[1])
            except Exception:  # noqa: BLE001
                return None
    return None


def dir_size_gb(path):
    """算力服务器上某目录的总大小（GB）。"""
    rc, out = npu("du -sb %s 2>/dev/null | cut -f1" % path, 120)
    try:
        return round(int(out.strip()) / 1024.0 / 1024 / 1024, 2)
    except Exception:  # noqa: BLE001
        return None


def enforce_size_limit(job_id, c, gb, label=""):
    """超限就置失败并给出明确原因；返回 True 表示放行。"""
    if gb is None:
        return True
    if gb <= MAX_MODEL_GB:
        return True
    emit(c, job_id, "prepare_env", "failed",
         "模型体积 %.1f GB，超过 %d 张卡的可用显存合计 %d GB —— 无法在本机迁移"
         % (gb, MAX_CARDS, MAX_MODEL_GB))
    c.execute("UPDATE jobs SET status='failed', phase='prepare_env', finished_at=?, "
              "error=? WHERE job_id=?",
              (time.time(),
               "模型过大：%.1f GB > %d GB（%d 卡 × %d GB）%s"
               % (gb, MAX_MODEL_GB, MAX_CARDS, CARD_HBM_GB, label), job_id))
    return False


def _register_in_corpus(model, cat, pkg):
    """把网页新迁成功的模型登记进 corpus_status.json（本机与算力服务器各一份）。

    没有这一步，make_delivery.py 会跳过它 -> 交付包为空目录。
    """
    entry = {"model": model, "category": cat, "status": "migrated",
             "package": pkg, "source": "web", "ts": time.time()}
    # 远程：用 argv 传参，避免把 JSON 内联进 shell 的转义地狱
    rpy = (
        "import json,os,sys\n"
        "model,cat,pkg=sys.argv[1],sys.argv[2],sys.argv[3]\n"
        "p='/data/model_corpus/_manifest/corpus_status.json'\n"
        "d=json.load(open(p,encoding='utf-8')) if os.path.isfile(p) else {'items':[]}\n"
        "items=[x for x in d.get('items',[]) if x.get('model')!=model]\n"
        "items.append({'model':model,'category':cat,'status':'migrated',"
        "'package':pkg,'source':'web'})\n"
        "sc={}\n"
        "for x in items: sc[x.get('status','')]=sc.get(x.get('status',''),0)+1\n"
        "d['items']=items; d['status_count']=sc; d['total']=len(items)\n"
        "json.dump(d,open(p,'w',encoding='utf-8'),ensure_ascii=False,indent=2)\n"
        "print('REGISTERED',len(items))\n"
    )
    try:
        npu("cat > /tmp/_reg.py <<'PYEOF'\n%s\nPYEOF\npython3 /tmp/_reg.py %s %s %s"
            % (rpy, model, cat, pkg), 120)
    except Exception:  # noqa: BLE001
        pass
    # 本机
    fp = os.path.join(ROOT, "_manifest", "corpus_status.json")
    try:
        if os.path.isfile(fp):
            d = json.load(open(fp, encoding="utf-8"))
            items = [x for x in d.get("items", []) if x.get("model") != model]
            items.append(entry)
            d["items"] = items
            sc = {}
            for x in items:
                k = x.get("status", "")
                sc[k] = sc.get(k, 0) + 1
            d["status_count"] = sc
            d["total"] = len(items)
            json.dump(d, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        pass

def download_model(model, job_id, c):
    """把模型从魔搭拉到算力服务器，并归位到对应分类目录。"""
    safe = model.replace("/", "__")
    cat = guess_category(model)
    dst = "%s/%s/%s" % (REMOTE_ROOT, cat, safe)
    emit(c, job_id, "prepare_env", "running",
         "语料库无此模型 → 从魔搭下载（归类：%s）" % CAT_CN.get(cat, cat))

    # 同一模型并发提交时只让一个真下，其余等缓存
    if not _acquire_download(model):
        emit(c, job_id, "prepare_env", "running",
             "已有其他任务在下载同一模型，等待其完成（避免重复下载抢带宽）…")
        _wait_download(model)
        rc0, out0 = npu("ls -d /tmp/ms_dl/models/%s/* 2>/dev/null | head -1"
                        % model.replace("/", "/"), 60)
        if out0.strip():
            src0 = out0.strip().splitlines()[0]
            rc1, _ = npu("mkdir -p %s && cp -rn %s/. %s/ 2>/dev/null" % (dst, src0, dst), 900)
            _release_download(model)
            _remote_dirs(force=True)
            emit(c, job_id, "prepare_env", "ok",
                 "复用已下载的缓存（未重复下载）：%s/（%.1f GB）"
                 % (dst.replace(REMOTE_ROOT, "语料库"), dir_size_gb(dst) or 0))
            return cat, dst
        _release_download(model)
    # 体积查询（ModelScope API 实测 47 秒）与下载**并行**：
    # 查到了就判定；查不到不影响下载，最后再用真实目录大小兜底。
    _size_box = {}

    def _probe_size():
        try:
            _size_box["gb"] = remote_model_size_gb(model)
        except Exception:  # noqa: BLE001
            _size_box["gb"] = None

    _st = threading.Thread(target=_probe_size, daemon=True)
    _st.start()
    py = (
        "import sys\n"
        "from modelscope.hub.snapshot_download import snapshot_download\n"
        "d=snapshot_download(sys.argv[1], cache_dir='/tmp/ms_dl')\n"
        "print('DOWNLOADED', d)\n"
    )
    # 下载用 Popen 跑，边下边看体积查询的结果：超限立刻杀进程，不白等几十分钟
    _cmd = "cat > /tmp/_dl.py <<'PYEOF'\n%s\nPYEOF\npython3 /tmp/_dl.py %s" % (py, model)
    _p = subprocess.Popen(["bash", _npu_sh(), _cmd],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    _killed = False
    _last_hb = time.time()
    _t0 = time.time()
    while True:
        try:
            _out, _ = _p.communicate(timeout=5)
            break
        except subprocess.TimeoutExpired:
            gb = _size_box.get("gb")
            if gb is not None and gb > MAX_MODEL_GB:
                _p.kill()
                _out, _ = _p.communicate()
                _killed = True
                break
            # 心跳：下载可能要几分钟，不发心跳前端会以为卡死了
            if time.time() - _last_hb > 30:
                _last_hb = time.time()
                rc2, cur = npu("du -sm /tmp/ms_dl 2>/dev/null | cut -f1", 30)
                got = ""
                try:
                    got = "（已缓存 %.1f GB）" % (int(cur.strip()) / 1024.0)
                except Exception:  # noqa: BLE001
                    pass
                emit(c, job_id, "prepare_env", "running",
                     "下载中…%s 已用 %ds%s"
                     % (model.split("/")[-1], int(time.time() - _t0), got))
    out = _out or ""
    rc = _p.returncode
    if _killed:
        _release_download(model)
        enforce_size_limit(job_id, c, _size_box.get("gb"), "，下载中发现超限已中止")
        return None, None
    if rc != 0 or "DOWNLOADED" not in out:
        emit(c, job_id, "prepare_env", "failed",
             "下载失败：%s" % (out.strip().splitlines()[-1][:180] if out.strip() else "未知"))
        return None, None
    src = out.split("DOWNLOADED", 1)[1].strip().splitlines()[0].strip()
    rc, out2 = npu("mkdir -p %s && cp -rn %s/. %s/ 2>/dev/null; ls %s | wc -l"
                   % (dst, src, dst, dst), 900)
    _release_download(model)
    real_gb = dir_size_gb(dst)
    emit(c, job_id, "prepare_env", "running",
         "已下载并归档到 %s/（%s 个文件）%s"
         % (dst.replace(REMOTE_ROOT, "语料库"), out2.strip()[:8],
            ("，实测 %.1f GB" % real_gb) if real_gb is not None else ""))
    # 权威判定：用下载后的真实体积。超限直接失败，不进入探路（免得白跑一轮）
    if not enforce_size_limit(job_id, c, real_gb, "，下载后实测"):
        return None, None
    emit(c, job_id, "prepare_env", "ok",
         "体积校验通过：%.1f GB ≤ %d GB" % (real_gb or 0, MAX_MODEL_GB))
    _remote_dirs(force=True)
    return cat, dst



# ---------------------------------------------------------------- 错误分桶
# 对应 pitfall-index skill 的 A/B/C/D 证据特征分类
BUCKETS = [
    ("A · 环境与依赖",
     r"ModuleNotFoundError|No module named|requires the .* library|"
     r"依赖 .* 安装失败|cannot import name",
     "按需补装依赖；补装时必须带版本约束（numpy<2 / pyarrow<17），否则会带崩 NPU 栈"),
    ("B · 设备 API",
     r"device should be either|CUDA|\.cuda\(|npu_available|Device|device_type",
     "启用 NPU 兼容层（verify_device 补丁 + transfer_to_npu 重定向）"),
    ("E · 模型结构与自定义代码",
     r"does not recognize this architecture|Unrecognized model|"
     r"Should have a .* key|no file named|is not a valid JSON|"
     r"Unrecognized configuration class",
     "该模型是自定义架构，通用加载器无法覆盖；需按 README 写模型专用入口（逐步迁移）"),
    ("C · 算子",
     r"aclnn|AscendC|kernel|not implemented|Unsupported op|operator",
     "走 operator-bridge：判定能否复用 / 需 aclnn 或 AscendC 桥接"),
    ("D · 性能与资源",
     r"out of memory|OOM|HBM|timeout|RouteTimeout",
     "检查卡占用与显存；必要时降 batch 或换空闲卡"),
]


def classify_error(err):
    """把报错归到 A/B/C/D 桶，返回 (桶名, 建议动作)。"""
    if not err:
        return None, None
    for name, rx, advice in BUCKETS:
        if re.search(rx, err, re.I):
            return name, advice
    return "未分类", "需人工判读原始 traceback"


def profile_model(mdir):
    """画像：读配置，说明这是什么模型（对应 model-profile skill）。

    路径可能在算力服务器上（权重存那边），此时改用 SSH 读取。
    """
    import glob
    if mdir.startswith(REMOTE_ROOT) and not os.path.isdir(mdir):
        py = (
            "import glob,json,os,sys\n"
            "d=sys.argv[1]\n"
            "i={'files':len(os.listdir(d)),'configs':[],'weights':0}\n"
            "for fn in ('configuration.json','config.json','model_index.json','hyperparams.yaml'):\n"
            "    if os.path.isfile(os.path.join(d,fn)): i['configs'].append(fn)\n"
            "cf=os.path.join(d,'config.json')\n"
            "if os.path.isfile(cf):\n"
            "    try:\n"
            "        c=json.load(open(cf,encoding='utf-8'))\n"
            "        i['arch']=c.get('architectures') or c.get('model_type')\n"
            "    except Exception: pass\n"
            "w=[]\n"
            "for e in ('.safetensors','.bin','.pt','.pth','.ckpt'):\n"
            "    w+=glob.glob(os.path.join(d,'**','*'+e),recursive=True)\n"
            "i['weights']=len(w)\n"
            "print('PROFILE',json.dumps(i,ensure_ascii=False))\n"
        )
        rc, out = npu("cat > /tmp/_pf.py <<'PYEOF'\n%s\nPYEOF\npython3 /tmp/_pf.py %s"
                      % (py, mdir), 120)
        for line in out.splitlines():
            if line.startswith("PROFILE"):
                try:
                    info = json.loads(line[len("PROFILE"):].strip())
                    info.setdefault("configs", [])
                    info["dir"] = mdir
                    return info
                except Exception:  # noqa: BLE001
                    break
        return {"dir": mdir, "files": 0, "configs": [], "weights": 0, "arch": None}

    info = {"dir": mdir, "files": len(os.listdir(mdir))}
    for fn in ("configuration.json", "config.json", "model_index.json",
               "hyperparams.yaml"):
        fp = os.path.join(mdir, fn)
        if os.path.isfile(fp):
            info.setdefault("configs", []).append(fn)
    try:
        cf = os.path.join(mdir, "config.json")
        if os.path.isfile(cf):
            c = json.load(open(cf, encoding="utf-8"))
            info["arch"] = c.get("architectures") or c.get("model_type")
    except Exception:  # noqa: BLE001
        pass
    w = []
    for ext in (".safetensors", ".bin", ".pt", ".pth", ".ckpt"):
        w += glob.glob(os.path.join(mdir, "**", "*" + ext), recursive=True)
    info["weights"] = len(w)
    return info


# ---------------------------------------------------------------- 迁移执行
STAGE_RULES = [
    (re.compile(r"\[npu_shim\]"), ("adapt", "ok", "NPU 兼容层就绪")),
    (re.compile(r"补装: |\"autodeps\": \[[^\]]"), ("prepare_env", "running", "补装缺失依赖")),
    (re.compile(r"VERDICT=LOAD-OK"), ("validate_infer", "ok", "模型加载成功")),
    (re.compile(r"VERDICT=LOAD-FAIL"), ("validate_infer", "failed", "模型加载失败")),
]


def run_migration(job_id, model):
    c = db()
    _card = None
    try:
        c.execute("UPDATE jobs SET status='running', phase='resolve', started_at=? WHERE job_id=?",
                  (time.time(), job_id))
        emit(c, job_id, "resolve", "ok", "解析模型：%s" % model)

        cat, mdir = find_corpus_dir(model)
        if not mdir:
            # 语料库里没有 -> 自动从魔搭拉取（客户输入即迁移的关键一环）
            cat, mdir = download_model(model, job_id, c)
            if not mdir:
                # 体积超限时 enforce_size_limit 已经写好具体原因，
                # 这里不能再用笼统的"下载失败"把它盖掉。
                row = c.execute("SELECT status, error FROM jobs WHERE job_id=?",
                                (job_id,)).fetchone()
                if not (row and row["status"] == "failed" and (row["error"] or "").strip()):
                    c.execute("UPDATE jobs SET status='failed', phase='prepare_env', "
                              "finished_at=?, error=? WHERE job_id=?",
                              (time.time(), "模型下载失败或不存在", job_id))
                return

        recipes = load_recipes()
        hit = model in recipes
        c.execute("UPDATE jobs SET route=? WHERE job_id=?", ("fast" if hit else "slow", job_id))
        if hit:
            r = recipes[model]
            emit(c, job_id, "match_recipe", "ok",
                 "命中已验证配方：加载路线 %s，依赖 %s，历史耗时 %ss"
                 % (r.get("route"), ",".join(r.get("autodeps") or []) or "无", r.get("elapsed_s")))
            # 该模型当初是靠这些修复才跑通的 —— 展示给客户，说明"经验"具体是什么
            ad = load_adaptations().get(model)
            if ad:
                emit(c, job_id, "adapt", "ok",
                     "该配方含 %d 步已沉淀修复（%d 步已自动化），本次直接复用，无需重做"
                     % (ad["n_steps"], ad.get("n_auto", 0)))
                for i, st in enumerate(ad["steps"], 1):
                    emit(c, job_id, "adapt", "ok",
                         "  历史修复 %d/%d [%s桶] %s → %s" % (
                             i, ad["n_steps"], st["bucket"],
                             st["symptom"][:42], st["fix"][:70]))
        else:
            emit(c, job_id, "match_recipe", "running",
                 "无已验证配方 → 走逐步迁移（画像→探路→分类→定位→适配→验证→交付）")
            pf = profile_model(mdir)
            emit(c, job_id, "match_recipe", "ok",
                 "画像：%d 个文件 / %s；架构=%s；权重文件 %d 个"
                 % (pf["files"], "/".join(pf.get("configs") or ["无标准配置"]),
                    pf.get("arch") or "未知", pf["weights"]))
            # 有已知适配记录 -> 把「已沉淀的修复步骤」逐步展示出来
            ad = load_adaptations().get(model)
            if ad:
                emit(c, job_id, "adapt", "running",
                     "命中已沉淀的适配方案（%d 步，其中 %d 步已自动化）"
                     % (ad["n_steps"], ad.get("n_auto", 0)))
                for i, st in enumerate(ad["steps"], 1):
                    emit(c, job_id, "adapt", "ok",
                         "修复 %d/%d [%s桶]%s → %s" % (
                             i, ad["n_steps"], st["bucket"],
                             st["symptom"][:46], st["fix"][:78]))

        # 已在库的模型：量大时也给出提示（只提示不拦，因为可能已分片优化过）
        _gb = dir_size_gb(mdir) if mdir.startswith(REMOTE_ROOT) else None
        if _gb is not None and _gb > MAX_MODEL_GB:
            emit(c, job_id, "prepare_env", "warn",
                 "该模型目录 %.1f GB，超过 %d 卡合计 %d GB，加载大概率 OOM"
                 % (_gb, MAX_CARDS, MAX_MODEL_GB))
        emit(c, job_id, "prepare_env", "ok",
             "语料目录 %s（%s 类）%s"
             % (mdir, CAT_CN.get(cat, cat),
                ("，%.1f GB" % _gb) if _gb is not None else ""))
        emit(c, job_id, "adapt", "running", "在算力服务器上启动 NPU 适配与加载…")

        # probe 必须在算力服务器上跑：docker + NPU 都在那边，权重也在那边。
        # 排队拿卡：有空卡立刻跑，没空卡进入 queued，等前面任务释放后自动开始
        card = _card = wait_for_card(job_id, c, model)
        emit(c, job_id, "prepare_env", "ok", "已分配 NPU 卡 %d（共 8 张，独占使用）" % card)
        remote_cmd = ("cd %s/_probe && PROBE_ROUTE_TIMEOUT=%s bash probe.sh %s %s %d"
                      % (REMOTE_ROOT,
                         os.environ.get("PROBE_ROUTE_TIMEOUT", "240"),
                         cat, os.path.basename(mdir), card))
        p = subprocess.Popen(["bash", _npu_sh(), remote_cmd],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace",
                             start_new_session=True)
        verdict, err = None, None
        tail = []
        emitted = set()          # 去重：同一条规则只推一次，避免刷屏
        for line in p.stdout:
            line = line.rstrip("\n")
            if line.strip():
                tail.append(line)
                if len(tail) > 30:
                    tail.pop(0)
            for rx, (ph, st, msg) in STAGE_RULES:
                if rx.search(line) and msg not in emitted:
                    emitted.add(msg)
                    emit(c, job_id, ph, st, msg)
            if "VERDICT=" in line:
                verdict = "LOAD-OK" in line
            if line.startswith("ERROR:"):
                err = line[6:].strip()[:300]
        p.wait(timeout=30)
        rc = p.returncode
        # 注意：这里**不能**释放卡。
        # deliver 阶段还要用同一张卡跑性能测试；释放后可能被别的任务抢走，
        # 导致 perf 跑到一半设备被占。卡统一在 finally 里释放。

        if verdict:
            emit(c, job_id, "validate_infer", "ok", "端到端加载验证通过")
            emit(c, job_id, "quality", "ok", "产物校验：probe.json 已生成")

            # ---- 真正生成交付包 ----
            # make_delivery.py 需要读模型目录（在算力服务器上），所以在那侧生成，
            # 再把这一个包同步回虚拟机供网页下载。
            # ---- ① 性能测试与调优（复用刚跑完探针的那张卡）----
            emit(c, job_id, "deliver", "running",
                 "性能测试与调优：测 P50/P95/P99 延迟、吞吐、显存…")
            rc_p, out_p = npu(
                "bash %s/_probe/perf.sh %s %s %d 3 20"
                % (REMOTE_ROOT, cat, os.path.basename(mdir), card), 1800)
            _pl = [x for x in (out_p or "").splitlines() if x.startswith("PERF_VERDICT=")]
            if _pl:
                emit(c, job_id, "deliver", "ok", "性能测试完成：%s" % _pl[-1].split("=", 1)[1])
            else:
                emit(c, job_id, "deliver", "warn",
                     "性能测试未产出结果（%s）" % (out_p or "")[-90:].replace("\n", " "))

            # ---- ② 权重镜像（三道闸门在 mk_image.sh 内）----
            emit(c, job_id, "deliver", "running", "构建模型权重镜像并上传 OBS…")
            rc_i, out_i = npu(
                "bash %s/_probe/mk_image.sh %s %s 0"
                % (REMOTE_ROOT, cat, os.path.basename(mdir)), 3600)
            _img_line = [x for x in (out_i or "").splitlines()
                         if x.startswith(("OK ", "SKIP_SIZE", "SKIP_DISK", "FAIL"))]
            if _img_line:
                emit(c, job_id, "deliver", "ok", "权重镜像：%s" % _img_line[-1][:110])
            else:
                emit(c, job_id, "deliver", "warn",
                     "权重镜像未完成（%s）" % (out_i or "")[-90:].replace("\n", " "))

            emit(c, job_id, "deliver", "running", "生成交付包（DEPLOY/TEST_REPORT/run.sh…）…")
            pkg = model.replace("/", "-").lower()
            # make_delivery.py 只处理 corpus_status.json 里的模型，
            # 网页上新迁的模型必须先登记进去，否则会被跳过（交付包为空）。
            _register_in_corpus(model, cat, pkg)
            rc2, out2 = npu(
                "cd %s/_tools && CORPUS_ROOT=%s timeout 300 python3 make_delivery.py"
                % (REMOTE_ROOT, REMOTE_ROOT), 360)
            if rc2 != 0:
                emit(c, job_id, "deliver", "warn",
                     "交付包生成脚本返回 %s：%s" % (rc2, out2.strip().splitlines()[-1][:140] if out2.strip() else ""))
            # 同步该包到虚拟机
            local_pkg = os.path.join(ROOT, "_delivery", pkg)
            os.makedirs(local_pkg, exist_ok=True)
            rc3, out3 = npu("ls %s/_delivery/%s 2>/dev/null | wc -l" % (REMOTE_ROOT, pkg), 60)
            have_remote = rc3 == 0 and out3.strip().isdigit() and int(out3.strip()) > 0
            if have_remote:
                subprocess.run(
                    ["rsync", "-az", "-e",
                     "ssh -i %s -p 31236 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
                     "-o UserKnownHostsFile=%s" % (
                         os.path.expanduser("~/.ssh/KeyPair-fae2027.pem"),
                         os.path.expanduser("~/.ssh/known_hosts")),
                     "root@61.241.77.3:%s/_delivery/%s/" % (REMOTE_ROOT, pkg),
                     local_pkg + "/"],
                    capture_output=True, text=True, timeout=180)
                nfiles = len(os.listdir(local_pkg)) if os.path.isdir(local_pkg) else 0
                emit(c, job_id, "deliver", "ok",
                     "交付包已生成：%s.tar.gz（%d 个文件），可在页面「交付产物」下载"
                     % (pkg, nfiles))
            else:
                emit(c, job_id, "deliver", "warn", "交付包未生成（%s）" % pkg)
            emit(c, job_id, "deliver", "ok", "配方已归档，可用于下次快速迁移")
            c.execute("UPDATE jobs SET status='succeeded', phase='deliver', exit_code=?, finished_at=?, result_json=? WHERE job_id=?",
                      (rc, time.time(), json.dumps({"verdict": "LOAD-OK", "package": pkg},
                                                   ensure_ascii=False), job_id))
        else:
            emit(c, job_id, "quality", "failed", err or "加载未通过")
            # 从 probe.json 读「再挖一层」结论。
            # 比解析 stdout 的 ERROR 行更稳：深挖是在子进程里做的，
            # 结果落在 probe.json 的 deep 字段（root_cause / missing_package）。
            try:
                pj = os.path.join(ROOT, "_probe",
                                  os.path.basename(mdir) or model.replace("/", "__"),
                                  "probe.json")
                if os.path.isfile(pj):
                    _d = (json.load(open(pj, encoding="utf-8")) or {}).get("deep") or {}
                    if _d.get("root_cause"):
                        emit(c, job_id, "quality", "warn",
                             "再挖一层：%s" % str(_d["root_cause"])[:170])
                        if _d.get("missing_package"):
                            emit(c, job_id, "adapt", "warn",
                                 "真实缺包 = %s（笼统报错常掩盖简单原因，勿据此判定不值得救）"
                                 % _d["missing_package"])
                        if not err:
                            err = str(_d["root_cause"])
            except Exception as e:  # noqa: BLE001
                # 不再静默：这类"读不到深挖结论"的 bug 之前被 pass 藏了整整一轮
                emit(c, job_id, "quality", "warn",
                     "深挖结论读取失败：%s: %s" % (type(e).__name__, str(e)[:90]))

            if err:
                bucket, advice = classify_error(err)
                emit(c, job_id, "quality", "warn",
                     "错误分桶：%s ｜ 证据：%s" % (bucket, err[:120]))
                emit(c, job_id, "adapt", "warn", "建议动作：%s" % (advice or "待人工判读"))
                # 自动诊断：按证据特征检索坑库，直接把「命中了哪条经验」推给客户，
                # 不用客户再手动粘报错（对应 pitfall-index skill 的检索动作）。
                try:
                    hits = diagnose(err, 2)
                    if hits:
                        for i, h in enumerate(hits, 1):
                            emit(c, job_id, "quality", "warn",
                                 "坑库命中 %d/%d [%s桶 %s·匹配度%s] %s → %s" % (
                                     i, len(hits), h["bucket"], h["bucket_cn"], h["score"],
                                     h["symptom"][:70], (h["direction"] or "")[:110]))
                        emit(c, job_id, "adapt", "warn",
                             "该模型可按上述已沉淀经验处置（逐步迁移路径）")
                    else:
                        emit(c, job_id, "adapt", "warn",
                             "坑库未命中 —— 属新坑；修好后请回写坑库（这是经验增长的来源）")
                except Exception as e:  # noqa: BLE001
                    emit(c, job_id, "adapt", "warn", "坑库检索异常：%s" % str(e)[:80])
            c.execute("UPDATE jobs SET status='failed', phase='quality', exit_code=?, finished_at=?, error=? WHERE job_id=?",
                      (rc, time.time(), err, job_id))
    except Exception as e:  # noqa: BLE001
        try:
            emit(c, job_id, "adapt", "failed", "执行异常：%s" % e)
            c.execute("UPDATE jobs SET status='failed', phase='adapt', error=?, finished_at=? WHERE job_id=?",
                      (str(e)[:300], time.time(), job_id))
        except Exception:  # noqa: BLE001
            pass
    finally:
        # 任何路径退出都要把卡还回去，否则并发跑几次后 8 张卡会被"泄漏"占满
        try:
            release_card(_card)
        except Exception:  # noqa: BLE001
            pass
        c.close()


# ---------------------------------------------------------------- HTTP
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _text(self, body, ctype="text/plain; charset=utf-8", code=200):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path, ctype):
        try:
            b = open(path, "rb").read()
        except OSError:
            return self._json({"error": "not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
        # ---- 静态资源 ----
        # index.html 里外链了 /static/preview.css，但此前没有这个路由，
        # 导致 CSS 404 -> 整页无样式、viewBox 的 SVG 撑满屏幕。
        if u.path.startswith("/static/"):
            rel = u.path[len("/static/"):]
            # 防目录穿越：解析后必须仍在 STATIC 之内
            base = os.path.realpath(STATIC)
            fp = os.path.realpath(os.path.join(base, rel))
            if not (fp == base or fp.startswith(base + os.sep)) or not os.path.isfile(fp):
                return self._json({"error": "not found"}, 404)
            ctype = MIME.get(os.path.splitext(fp)[1].lower(),
                             "application/octet-stream")
            return self._file(fp, ctype)
        if u.path == "/api/phases":
            return self._json({"phases": [{"key": p, "cn": PHASE_CN[p]} for p in PHASES]})
        if u.path == "/api/deliveries":
            return self._json(load_deliveries())
        # ---- 交付包内单文件下载（性能报告、镜像清单等）----
        if u.path == "/api/artifact":
            pkg = (q.get("package") or [""])[0].strip()
            name = (q.get("name") or [""])[0].strip()
            if not pkg or "/" in pkg or ".." in pkg or not name or "/" in name or ".." in name:
                return self._json({"error": "参数非法"}, 400)
            fp = os.path.join(ROOT, "_delivery", pkg, name)
            if not os.path.isfile(fp):
                return self._json({"error": "文件不存在：%s" % name}, 404)
            ct = MIME.get(os.path.splitext(name)[1].lower(), "application/octet-stream")
            if name.endswith(".md"):
                ct = "text/markdown; charset=utf-8"
            return self._file(fp, ct)
        # ---- 交付包下载：按需打成 tar.gz，不落盘（每个包只有几十 KB） ----
        if u.path == "/api/download":
            pkg = (q.get("package") or [""])[0].strip()
            if not pkg or "/" in pkg or ".." in pkg:
                return self._json({"error": "缺少或非法的 package 参数"}, 400)
            src = os.path.join(ROOT, "_delivery", pkg)
            if not os.path.isdir(src):
                return self._json({"error": "交付包不存在：%s" % pkg}, 404)
            import io
            import tarfile
            buf = io.BytesIO()
            # 解压后得到 <pkg>/ 目录，与 zip 版结构一致
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.add(src, arcname=pkg)
            data = buf.getvalue()
            fn = "%s.tar.gz" % pkg
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Disposition",
                             'attachment; filename="%s"' % fn)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Delivery-Package", pkg)
            self.end_headers()
            self.wfile.write(data)
            return
        # 交付包清单文件（供前端展示包内构成）
        if u.path == "/api/delivery-files":
            pkg = (q.get("package") or [""])[0].strip()
            if not pkg or "/" in pkg or ".." in pkg:
                return self._json({"error": "缺少或非法的 package 参数"}, 400)
            src = os.path.join(ROOT, "_delivery", pkg)
            if not os.path.isdir(src):
                return self._json({"error": "交付包不存在：%s" % pkg}, 404)
            files = []
            for rt, _dirs, fns in os.walk(src):
                for fn in sorted(fns):
                    fp = os.path.join(rt, fn)
                    files.append({
                        "name": os.path.relpath(fp, src),
                        "size": os.path.getsize(fp),
                    })
            return self._json({"package": pkg, "count": len(files),
                               "files": files,
                               "tar": "%s.tar.gz" % pkg})
        if u.path == "/api/report":
            fp = os.path.join(ROOT, "FINAL-REPORT.md")
            if not os.path.isfile(fp):
                return self._json({"error": "尚未生成总报告"}, 404)
            with open(fp, encoding="utf-8") as f:
                return self._text(f.read(), "text/markdown; charset=utf-8")
        if u.path == "/api/pitfalls":
            d = load_pitfalls()
            return self._json({"count": d.get("loaded", 0),
                               "entries": d.get("entries", [])})
        if u.path == "/api/validation":
            fp = os.path.join(ROOT, "_manifest", "testset_validation.json")
            if not os.path.isfile(fp):
                return self._json({"error": "尚未跑过验收"}, 404)
            return self._json(json.load(open(fp, encoding="utf-8")))
        if u.path == "/api/corpus":
            cs = load_corpus_status()
            only = (q.get("status") or [""])[0]
            cat = (q.get("category") or [""])[0]
            items = cs.get("items", [])
            if only:
                items = [x for x in items if x["status"] == only]
            if cat:
                items = [x for x in items if x["category"] == cat]
            head = {k: v for k, v in cs.items() if k != "items"}
            return self._json(dict(head, filtered=len(items), items=items))
        if u.path == "/api/experience":
            # 各分类的迁移经验：成功率 + 成功用到的加载路线分布。
            # 前端 loadExperience() 用它渲染顶部那排分类徽章。
            cs = load_corpus_status()
            rec = load_recipes()
            agg = {}
            for x in cs.get("items", []):
                c = x.get("category") or ""
                a = agg.setdefault(c, {"n_ok": 0, "n_total": 0, "routes": {}})
                a["n_total"] += 1
                if x.get("status") != "migrated":
                    continue
                a["n_ok"] += 1
                r = str((rec.get(x.get("model")) or {}).get("route") or "")
                r = r.split(":")[0].strip()
                if r:
                    a["routes"][r] = a["routes"].get(r, 0) + 1
            cats = []
            for c, a in agg.items():
                n = a["n_total"]
                cats.append({
                    "category": c,
                    "name": CAT_CN.get(c, c),
                    "n_ok": a["n_ok"],
                    "n_total": n,
                    "rate": int(round(a["n_ok"] * 100.0 / n)) if n else 0,
                    "routes": sorted(a["routes"].items(), key=lambda kv: -kv[1]),
                })
            cats.sort(key=lambda z: (-z["rate"], z["name"]))
            return self._json({"categories": cats, "total": len(cats)})
        if u.path == "/api/rescues":
            ad = load_adaptations()
            return self._json({"count": len(ad), "items": list(ad.values())})
        if u.path == "/api/recipes":
            r = load_recipes()
            return self._json({"count": len(r), "recipes": list(r.values())})
        if u.path == "/api/capacity":
            # 算力容量 + 排队情况，前端据此显示"N/6 卡在用 · M 个排队"
            cap = capacity()
            c = db()
            try:
                q = c.execute("SELECT COUNT(*) n FROM jobs WHERE status='queued'").fetchone()["n"]
                r = c.execute("SELECT COUNT(*) n FROM jobs WHERE status='running'").fetchone()["n"]
            finally:
                c.close()
            cap["queued"] = q
            cap["running"] = r
            return self._json(cap)
        if u.path == "/api/jobs":
            c = db()
            rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50").fetchall()
            c.close()
            return self._json([dict(x) for x in rows])
        m = re.match(r"^/api/jobs/([\w-]+)$", u.path)
        if m:
            c = db()
            j = c.execute("SELECT * FROM jobs WHERE job_id=?", (m.group(1),)).fetchone()
            ev = c.execute("SELECT * FROM events WHERE job_id=? ORDER BY seq", (m.group(1),)).fetchall()
            c.close()
            if not j:
                return self._json({"error": "no such job"}, 404)
            return self._json({"job": dict(j), "events": [dict(e) for e in ev]})
        m = re.match(r"^/api/jobs/([\w-]+)/stream$", u.path)
        if m:
            return self._stream(m.group(1), int((q.get("after") or ["0"])[0]))
        return self._json({"error": "not found"}, 404)

    def _stream(self, job_id, after):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        c = db()
        last = after
        t0 = time.time()
        try:
            while time.time() - t0 < 1800:
                rows = c.execute("SELECT * FROM events WHERE job_id=? AND seq>? ORDER BY seq",
                                 (job_id, last)).fetchall()
                for r in rows:
                    last = r["seq"]
                    self.wfile.write(("data: %s\n\n" % json.dumps(
                        {"seq": r["seq"], "phase": r["phase"], "status": r["status"],
                         "message": r["message"], "ts": r["ts"],
                         "phase_cn": PHASE_CN.get(r["phase"], r["phase"])},
                        ensure_ascii=False)).encode())
                j = c.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if j and j["status"] in ("succeeded", "failed", "cancelled"):
                    self.wfile.write(("data: %s\n\n" % json.dumps(
                        {"done": True, "status": j["status"]})).encode())
                    self.wfile.flush()
                    break
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            c.close()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/diagnose":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:  # noqa: BLE001
                return self._json({"error": "bad json"}, 400)
            err = body.get("error") or ""
            if not err.strip():
                return self._json({"error": "缺少 error 字段"}, 400)
            hits = diagnose(err, int(body.get("topn") or 3))
            return self._json({"error_excerpt": err[:400],
                               "matched": len(hits), "results": hits})
        if u.path != "/api/jobs":
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:  # noqa: BLE001
            return self._json({"error": "bad json"}, 400)
        # 支持两种入参：
        #   {"model": "org/name"}           单模型
        #   {"models": ["a/b", "c/d", ...]} 批量（前端多行输入走这个）
        raw = body.get("models")
        if raw is None:
            raw = [body.get("model") or ""]
        if not isinstance(raw, list):
            return self._json({"error": "models 必须是数组"}, 400)

        models, seen = [], set()
        for x in raw:
            m = str(x or "").strip()
            if not m:
                continue
            # 允许直接贴魔搭链接
            mm = re.search(r"modelscope\.cn/models/([^/?#]+/[^/?#]+)", m)
            if mm:
                m = mm.group(1)
            if m in seen:
                continue
            seen.add(m)
            models.append(m)
        if not models:
            return self._json({"error": "缺少 model 或 models 字段"}, 400)
        if len(models) > 50:
            return self._json({"error": "一次最多提交 50 个模型"}, 400)

        c = db()
        created = []
        for i, model in enumerate(models):
            job_id = "job-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            with _lock:
                c.execute("INSERT INTO jobs (job_id,model,status,created_at) VALUES (?,?,?,?)",
                          (job_id, model, "queued", time.time() + i * 1e-6))
            emit(c, job_id, "resolve", "running", "任务已创建，模型 %s" % model)
            created.append({"job_id": job_id, "model": model})
        c.close()
        for it in created:
            threading.Thread(target=run_migration,
                             args=(it["job_id"], it["model"]), daemon=True).start()
        first = created[0]
        return self._json({"job_id": first["job_id"], "model": first["model"],
                           "count": len(created), "jobs": created})


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()

    # 启动清理：把上次进程遗留的 running/pending 标为中断
    try:
        _c0 = db()
        _k = kill_orphan_downloads()
        if _k not in ("0", "?", ""):
            print("[startup] 清理了遗留下载进程，剩余:", _k)
        _n = reap_orphans(_c0)
        _c0.close()
        if _n:
            print("[startup] 已清理 %d 个中断任务" % _n)
    except Exception as _e:  # noqa: BLE001
        print("[startup] 清理中断任务失败:", _e)
    ROOT = a.root
    db().close()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print("迁移平台后端已启动: http://0.0.0.0:%d  语料根目录=%s" % (a.port, ROOT), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
