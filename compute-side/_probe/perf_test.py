#!/usr/bin/env python3
"""性能测试：在 NPU 上测推理延迟、吞吐、显存，并给出调优建议。

测三组数据：
  1. 首次推理（cold）    —— 含算子编译/图下沉，通常远高于稳态
  2. 稳态延迟（hot）      —— warmup 后的 P50 / P95 / P99
  3. 显存占用            —— HBM 峰值分配

判定口径：不设"过/不过"，只给事实数据 + 调优建议
（性能没有绝对标准，要跟基线比才有意义；本环境无 GPU，所以只留实测值）。

用法：
  python3 perf_test.py --model <权重目录> --out <输出目录> [--warmup 3] [--iters 20]
"""
import argparse
import json
import os
import statistics
import sys
import time


def shim():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, "/workspace/probe"):
        if os.path.isfile(os.path.join(p, "npu_shim.py")):
            sys.path.insert(0, p)
            import npu_shim
            npu_shim.apply_all(verbose=False)
            return


def detect_kind(model_dir):
    cf = os.path.join(model_dir, "config.json")
    cfg = {}
    if os.path.isfile(cf):
        try:
            cfg = json.load(open(cf, encoding="utf-8"))
        except Exception:
            pass
    mt = str(cfg.get("model_type") or "").lower()
    arch = " ".join(cfg.get("architectures") or []).lower()
    if "vision" in mt or "vit" in mt or "clip" in mt or "swin" in mt or "deit" in mt:
        return "vision"
    if "whisper" in mt or "wav2vec" in mt or "audio" in mt or "speech" in mt:
        return "audio"
    if any(k in mt for k in ("bert", "gpt", "qwen", "llama", "t5", "roberta")):
        return "text"
    if "ForImageClassification" in arch or "ForObjectDetection" in arch:
        return "vision"
    if "ForCausalLM" in arch or "ForMaskedLM" in arch or "ForSequenceClassification" in arch:
        return "text"
    return "text"


def pick_inputs(kind, model_dir, seed=20240521):
    import torch
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    cfg = {}
    cf = os.path.join(model_dir, "config.json")
    if os.path.isfile(cf):
        try:
            cfg = json.load(open(cf, encoding="utf-8"))
        except Exception:
            cfg = {}
    if kind == "vision":
        size = cfg.get("image_size")
        h = w = size if isinstance(size, int) else (
            size[0] if isinstance(size, list) else 224)
        if h > 448:
            h = w = 224
        return {"pixel_values": torch.randn(1, 3, h, w, generator=g)}
    if kind == "audio":
        return {"input_values": torch.randn(1, 16000, generator=g)}
    vocab = cfg.get("vocab_size") or (cfg.get("text_config") or {}).get("vocab_size") or 30000
    ids = torch.randint(0, min(vocab, 30000), (1, 16), generator=g)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--kind", default="auto")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    shim()
    try:
        import tf5_compat  # noqa: F401
    except Exception:
        pass
    import torch

    res = {"model_dir": a.model, "device": "npu:0",
           "warmup": a.warmup, "iters": a.iters, "baseline": None,
           "note": "本环境无 GPU 基线，只记录 NPU 实测值"}
    t_all = time.time()

    kind = a.kind if a.kind != "auto" else detect_kind(a.model)
    res["kind"] = kind
    inputs = pick_inputs(kind, a.model)
    res["input_spec"] = {k: list(v.shape) for k, v in inputs.items()}

    try:
        from transformers import AutoModel
        t_load0 = time.time()
        m = AutoModel.from_pretrained(a.model, trust_remote_code=True,
                                      attn_implementation="eager")
        m.eval()
        res["load_s"] = round(time.time() - t_load0, 2)
        n_params = sum(p.numel() for p in m.parameters())
        res["n_params"] = n_params
        t_to0 = time.time()
        m = m.to("npu:0")
        ins = {k: v.to("npu:0") for k, v in inputs.items()}
        res["to_npu_s"] = round(time.time() - t_to0, 2)
    except Exception as e:  # noqa: BLE001
        res.update({"verdict": "SKIP", "reason": "加载失败：%s: %s" % (type(e).__name__, str(e)[:180])})
        json.dump(res, open(os.path.join(a.out, "perf.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("PERF_VERDICT=SKIP"); print("  reason:", res["reason"]); return

    def sync():
        try:
            torch.npu.synchronize()
        except Exception:
            pass

    # ---- 首次推理（cold）----
    cold = None
    try:
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            m(**ins)
        sync(); cold = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        res["forward_err"] = "%s: %s" % (type(e).__name__, str(e)[:180])

    # ---- warmup ----
    try:
        for _ in range(a.warmup):
            with torch.no_grad():
                m(**ins)
        sync()
    except Exception as e:  # noqa: BLE001
        res["warmup_err"] = str(e)[:160]

    # ---- 稳态 ----
    lat = []
    try:
        for _ in range(a.iters):
            sync(); t0 = time.perf_counter()
            with torch.no_grad():
                m(**ins)
            sync(); lat.append(time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        res["latency_err"] = str(e)[:160]

    # ---- 显存 ----
    try:
        res["hbm_peak_mb"] = round(torch.npu.max_memory_allocated() / 1024**2, 1)
        res["hbm_reserved_mb"] = round(torch.npu.max_memory_reserved() / 1024**2, 1)
    except Exception:
        pass

    if cold is not None:
        res["cold_s"] = round(cold, 4)
    if lat:
        ms = [x * 1000 for x in lat]
        res["latency_ms"] = {
            "mean": round(statistics.mean(ms), 2),
            "p50": round(pct(ms, 50), 2),
            "p95": round(pct(ms, 95), 2),
            "p99": round(pct(ms, 99), 2),
            "min": round(min(ms), 2),
            "max": round(max(ms), 2),
        }
        res["throughput_qps"] = round(1000.0 / statistics.mean(ms), 2)
    res["verdict"] = "OK" if lat else "SKIP"
    res["elapsed_s"] = round(time.time() - t_all, 1)

    json.dump(res, open(os.path.join(a.out, "perf.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    write_report(res, os.path.join(a.out, "PERF_REPORT.md"))
    print("PERF_VERDICT=%s" % res["verdict"])
    if res.get("latency_ms"):
        L = res["latency_ms"]
        print("  P50=%.2fms P95=%.2fms P99=%.2fms  QPS=%.2f  HBM峰值=%sMB  cold=%ss"
              % (L["p50"], L["p95"], L["p99"], res.get("throughput_qps", 0),
                 res.get("hbm_peak_mb", "?"), res.get("cold_s", "?")))


def write_report(res, path):
    """生成人可读的性能测试与调优报告。"""
    L = []
    L.append("# 性能测试与调优报告")
    L.append("")
    L.append("> 本报告记录 **NPU 单卡实测性能**，并给出可执行的调优建议。")
    L.append("> 本环境无 GPU，**不提供跨硬件对比**；所有数值为 Ascend 910B1 单卡实测。")
    L.append("")
    L.append("## 1. 结论")
    L.append("")
    L.append("| 项 | 结果 |")
    L.append("|---|---|")
    L.append("| 状态 | **%s** |" % res.get("verdict"))
    L.append("| 设备 | `npu:0`（Ascend 910B1，64 GB HBM） |")
    L.append("| 输入类型 | `%s` |" % res.get("kind"))
    n = res.get("n_params")
    L.append("| 参数量 | %s |" % (format(n, ",") if n else "—"))
    L.append("| 权重加载耗时 | %ss |" % res.get("load_s", "—"))
    L.append("| 搬到 NPU 耗时 | %ss |" % res.get("to_npu_s", "—"))
    L.append("")

    if res.get("latency_ms"):
        m = res["latency_ms"]
        L.append("## 2. 推理延迟（稳态，%d 次平均）" % res.get("iters", 0))
        L.append("")
        L.append("| 指标 | 数值 |")
        L.append("|---|---:|")
        L.append("| **P50 延迟** | **%s ms** |" % m["p50"])
        L.append("| **P95 延迟** | **%s ms** |" % m["p95"])
        L.append("| P99 延迟 | %s ms |" % m["p99"])
        L.append("| 平均延迟 | %s ms |" % m["mean"])
        L.append("| 最小 / 最大 | %s / %s ms |" % (m["min"], m["max"]))
        L.append("| **吞吐** | **%s 次/秒** |" % res.get("throughput_qps"))
        L.append("")
        cold = res.get("cold_s")
        if cold:
            ratio = round(cold * 1000 / m["p50"], 1) if m["p50"] else None
            L.append("## 3. 首次推理 vs 稳态")
            L.append("")
            L.append("| 指标 | 数值 |")
            L.append("|---|---:|")
            L.append("| 首次推理（cold） | %s ms |" % round(cold * 1000, 1))
            L.append("| 稳态 P50 | %s ms |" % m["p50"])
            if ratio:
                L.append("| 首/稳 倍数 | **%s×** |" % ratio)
            L.append("")
            L.append("> 首次推理包含算子编译与图下沉，所以显著高于稳态。")
            L.append("> **对外承诺性能时，必须用 P50/P95，不要用首次耗时**——那会把性能说低几倍。")
            L.append("")

    hb = res.get("hbm_peak_mb")
    if hb:
        L.append("## 4. 显存占用")
        L.append("")
        L.append("| 指标 | 数值 |")
        L.append("|---|---:|")
        L.append("| HBM 峰值分配 | **%s MB**（%.1f GB） |" % (hb, hb / 1024))
        rr = res.get("hbm_reserved_mb")
        if rr:
            L.append("| HBM 保留 | %s MB（%.1f GB） |" % (rr, rr / 1024))
        L.append("| 单卡 HBM 总量 | 64 GB |")
        if hb:
            L.append("| 占用率 | %.1f%% |" % (hb / 1024 / 64 * 100))
        L.append("")

    L.append("## 5. 调优建议")
    L.append("")
    tips = []
    L.append("### 5.1 通用")
    L.append("")
    tips.append("**首次推理慢是正常的**（算子编译），不要按首次耗时做容量规划。"
                "生产环境先跑一次 warmup 再对外服务。")
    tips.append("**P99 明显高于 P50** 说明有长尾抖动，通常是内存分配或算子 fallback 导致；"
                "可尝试固定 batch、预分配显存（`torch.npu.set_per_process_memory_fraction`）。")
    tips.append("**batch 提升吞吐但增加延迟**：吞吐敏感场景（离线批处理）加大 batch；"
                "延迟敏感场景（在线服务）用 batch=1 + 多实例。")
    for t in tips:
        L.append("- " + t)
    L.append("")

    L.append("### 5.2 本模型相关")
    L.append("")
    if res.get("n_params"):
        gb = res["n_params"] * 4 / 1024**3
        L.append("- 参数量 %s，fp32 权重约 **%.1f GB**；若显存紧张可考虑 fp16/bf16（能省一半）。"
                 % (format(res["n_params"], ","), gb))
    if res.get("hbm_peak_mb"):
        pct_use = res["hbm_peak_mb"] / 1024 / 64 * 100
        if pct_use > 80:
            L.append("- 显存占用 %.0f%%，接近上限：建议降 batch 或改用 fp16。" % pct_use)
        else:
            L.append("- 显存占用 %.0f%%，还有余量：可以尝试加大 batch 提升吞吐。" % pct_use)
    L.append("- 若需进一步优化，优先看：① 是否有算子 fallback 到 CPU（会慢几十倍）"
             " ② 是否可用 `torch.compile`（NPU 上需谨慎验证正确性）")
    L.append("")

    L.append("## 6. 未覆盖项")
    L.append("")
    L.append("- **无 GPU / CPU 基线对比**：本环境无 GPU，只有 NPU 单卡实测值。")
    L.append("- **未做端到端吞吐测试**（含预处理/后处理）：只测了模型前向本身。")
    L.append("- **未做多卡 / 多实例并发测试**：单卡单进程数据。")
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
