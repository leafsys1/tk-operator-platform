#!/usr/bin/env python3
"""精度验证：同一份权重 + 同一份输入，分别在 CPU 和 NPU 上前向，逐元素对比。

为什么用 CPU 做基线：
  - 本环境没有 GPU，CPU 是唯一可得的"另一套实现"参考
  - 它验证的是"NPU 算子实现与参考实现是否一致"，这正是迁移要回答的问题
  - 不宣称"与 GPU 逐位一致" —— 那需要 GPU 才能比

判定口径（写在报告里，不模糊）：
  PASS  : 所有输出 max|Δ| <= atol 且 相对误差 <= rtol，且余弦相似度 >= 0.9999
  WARN  : 能用，但存在超出阈值的偏差（记录具体数值）
  SKIP  : 模型过大/无可用输入构造器/CPU 基线跑不起来

用法：
  python3 acc_check.py --model <权重目录> --out <输出目录> [--kind text|vision|audio|auto]
"""
import argparse
import json
import os
import sys
import time

# 判定阈值（改这里就要同步改报告里的说明）
#
# 为什么不用逐元素相对误差：输出里有大量接近 0 的分量，
# 除以它们会让 max_rel_err 爆到几百（实测 648），完全没有判别力。
# 改用：
#   rel_l2    = ||cpu - npu|| / ||cpu||      —— 整体能量层面的相对偏差
#   cosine    = 两向量的夹角余弦              —— 方向一致性
# 这两个指标对"接近 0 的分量"不敏感，是数值等价性更稳的度量。
REL_L2_PASS = 1e-2      # 1%：fp32 跨后端实现的正常范围
REL_L2_WARN = 1e-1      # 10%：能用但需关注
COS_PASS = 0.9999
COS_WARN = 0.99


def shim():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, "/workspace/probe"):
        if os.path.isfile(os.path.join(p, "npu_shim.py")):
            sys.path.insert(0, p)
            import npu_shim
            npu_shim.apply_all(verbose=False)
            return


def pick_inputs(kind, model_dir, seed=20240521):
    """构造固定输入。用固定 seed，保证 CPU/NPU 两次输入完全相同。"""
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

    if kind == "text":
        vocab = cfg.get("vocab_size") or (cfg.get("text_config") or {}).get("vocab_size") or 30000
        ids = torch.randint(0, min(vocab, 30000), (1, 16), generator=g)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}
    if kind == "vision":
        img_size = cfg.get("image_size")
        if isinstance(img_size, list):
            h, w = img_size[0], img_size[1]
        elif isinstance(img_size, int):
            h = w = img_size
        else:
            h = w = 224
        # 太大就砍到 224，避免 CPU 太慢
        if h > 448 or w > 448:
            h = w = 224
        px = torch.randn(1, 3, h, w, generator=g)
        return {"pixel_values": px}
    if kind == "audio":
        n = 16000
        wav = torch.randn(1, n, generator=g)
        return {"input_values": wav}
    return None


def detect_kind(model_dir):
    """按 config 猜输入类型。猜不到返回 None（跳过精度检查）。"""
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
    if "bert" in mt or "gpt" in mt or "qwen" in mt or "llama" in mt or "t5" in mt or "roberta" in mt:
        return "text"
    if "ForImageClassification" in arch or "ForObjectDetection" in arch:
        return "vision"
    if "ForCausalLM" in arch or "ForMaskedLM" in arch or "ForSequenceClassification" in arch:
        return "text"
    return None


def run_forward(model_dir, kind, inputs, device):
    """在指定 device 上前向一次，返回展平后的 float32 张量。"""
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                  attn_implementation="eager")
    m.eval()
    m = m.to(device)
    ins = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = m(**ins)
    # 取主输出
    t = None
    if hasattr(out, "last_hidden_state"):
        t = out.last_hidden_state
    elif hasattr(out, "logits"):
        t = out.logits
    elif hasattr(out, "pooler_output"):
        t = out.pooler_output
    elif isinstance(out, (tuple, list)) and out:
        t = out[0]
    elif torch.is_tensor(out):
        t = out
    if t is None:
        raise RuntimeError("无法从输出里取到张量")
    v = t.detach().float().cpu().flatten()
    del m
    try:
        torch.npu.empty_cache()
    except Exception:
        pass
    return v


def compare(a, b):
    """a = CPU 基线, b = NPU 结果。返回差异指标。"""
    import torch
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    diff = (a - b).abs()
    na, nb = a.norm(), b.norm()
    cos = float((a @ b) / (na * nb)) if float(na) > 0 and float(nb) > 0 else None
    rel_l2 = float((a - b).norm() / na) if float(na) > 0 else None
    cpu_abs_mean = float(a.abs().mean())
    # 只在"信号足够大"的分量上算相对误差，避免被接近 0 的分量带爆
    strong = a.abs() > max(1e-3, 0.01 * cpu_abs_mean)
    rel_p99 = None
    if int(strong.sum()) > 0:
        r = (diff[strong] / a[strong].abs())
        rel_p99 = float(torch.quantile(r, 0.99)) if r.numel() > 1 else float(r.max())
    return {
        "n_compared": int(n),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff_norm": float(diff.max() / cpu_abs_mean) if cpu_abs_mean > 0 else None,
        "rel_l2": rel_l2,
        "cosine_sim": cos,
        "rel_err_p99_strong": rel_p99,
        "cpu_abs_mean": cpu_abs_mean,
        "npu_abs_mean": float(b.abs().mean()),
    }


def verdict(metrics):
    rl = metrics.get("rel_l2")
    cos = metrics.get("cosine_sim")
    if rl is None and cos is None:
        return "SKIP"
    # 两个指标都满足才 PASS；任一满足 WARN 条件则 WARN
    if (rl is not None and rl <= REL_L2_PASS) and (cos is None or cos >= COS_PASS):
        return "PASS"
    if (rl is not None and rl <= REL_L2_WARN) and (cos is None or cos >= COS_WARN):
        return "WARN"
    return "FAIL"


def write_report(res, path):
    """生成人可读的精度验证报告。"""
    v = res.get("verdict", "SKIP")
    m = res.get("metrics") or {}
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭"}.get(v, "·")
    L = []
    L.append("# 精度验证报告")
    L.append("")
    L.append("> 本报告验证的是：**同一份权重、同一份输入，在 NPU 与 CPU 两套实现上的数值一致性**。")
    L.append("> 它回答「迁移后算子实现是否等价」，不回答「与 GPU 是否逐位相同」——那需要 GPU 基准才能比。")
    L.append("")
    L.append("## 1. 判定")
    L.append("")
    L.append("| 项 | 结果 |")
    L.append("|---|---|")
    L.append("| 结论 | **%s %s** |" % (icon, v))
    L.append("| 基线设备 | `cpu` |")
    L.append("| 目标设备 | `npu:0` |")
    L.append("| 输入类型 | `%s` |" % res.get("kind"))
    L.append("| 实测耗时 | %ss |" % res.get("elapsed_s"))
    if res.get("reason"):
        L.append("| 跳过/失败原因 | %s |" % res["reason"])
    L.append("")

    if m:
        L.append("## 2. 数值对比")
        L.append("")
        L.append("| 指标 | 数值 | 说明 |")
        L.append("|---|---:|---|")
        L.append("| 对比元素数 | %s | 输出张量展平后的元素数 |" % format(m.get("n_compared"), ","))
        rl = m.get("rel_l2")
        L.append("| **相对 L2 误差** | **%.3e** | `‖NPU−CPU‖ / ‖CPU‖`，整体能量层面的相对偏差 |"
                 % (rl if rl is not None else float("nan")))
        cos = m.get("cosine_sim")
        L.append("| **余弦相似度** | **%.9f** | 两向量的夹角余弦，1.0 = 方向完全一致 |"
                 % (cos if cos is not None else float("nan")))
        L.append("| 最大绝对误差 | %.3e | 逐元素最大差值 |" % m.get("max_abs_diff", 0))
        L.append("| 平均绝对误差 | %.3e | 逐元素平均差值 |" % m.get("mean_abs_diff", 0))
        madn = m.get("max_abs_diff_norm")
        L.append("| 归一化最大误差 | %s | `max|Δ| / mean(|CPU|)`，扣掉信号幅度后的相对量 |"
                 % ("%.3e" % madn if madn is not None else "—"))
        L.append("| CPU 输出均值|值| | %.4f | 参考信号幅度 |" % m.get("cpu_abs_mean", 0))
        L.append("| NPU 输出均值|值| | %.4f | 目标信号幅度 |" % m.get("npu_abs_mean", 0))
        p99 = m.get("rel_err_p99_strong")
        if p99 is not None:
            L.append("| 强信号分量相对误差 P99 | %.3e | 只在 |值|>1%%均值 的分量上统计 |" % p99)
        L.append("")

    L.append("## 3. 判定标准")
    L.append("")
    c = res.get("criteria") or {}
    L.append("| 结论 | 条件 |")
    L.append("|---|---|")
    L.append("| ✅ PASS | 相对 L2 误差 ≤ %.0e **且** 余弦相似度 ≥ %s |"
             % (c.get("rel_l2_pass", 1e-2), c.get("cos_pass", 0.9999)))
    L.append("| ⚠️ WARN | 相对 L2 误差 ≤ %.0e **且** 余弦相似度 ≥ %s |"
             % (c.get("rel_l2_warn", 1e-1), c.get("cos_warn", 0.99)))
    L.append("| ❌ FAIL | 不满足以上任一 |")
    L.append("| ⏭ SKIP | 无法判定输入类型 / CPU 基线跑不起来 |")
    L.append("")
    L.append("### 为什么不用逐元素相对误差")
    L.append("")
    L.append("输出张量里存在大量接近 0 的分量，除以它们会让逐元素相对误差放大到几百"
             "（实测 `max_rel_err` 达 648），完全失去判别力。")
    L.append("改用「相对 L2 误差 + 余弦相似度」，两者都对接近 0 的分量不敏感。")
    L.append("")

    L.append("## 4. 输入规格")
    L.append("")
    spec = res.get("input_spec") or {}
    if spec:
        L.append("| 输入名 | 形状 |")
        L.append("|---|---|")
        for k, v in spec.items():
            L.append("| `%s` | `%s` |" % (k, v))
        L.append("")
        L.append("> 输入由固定随机种子（20240521）生成，保证 CPU / NPU 两次前向用的是**同一份数据**。")
    else:
        L.append("未构造输入（已跳过）。")
    L.append("")

    L.append("## 5. 未覆盖项")
    L.append("")
    L.append("- **未与 GPU 基线对比**：本环境无 GPU，CPU 是唯一可得的参考实现。")
    L.append("- **未在任务数据集上算指标**（Accuracy / F1 / BLEU / mAP）：")
    L.append("  本轮验证的是「同一输入下两套实现的数值一致性」，不是「任务精度」。")
    L.append("- **随机输入**：用固定种子生成，非真实样本；它足以暴露算子实现差异，")
    L.append("  但不能代表真实分布下的精度表现。")
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--kind", default="auto")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    shim()
    # tf5_compat 必须先 import：它补 transformers 5.x 的若干兼容点
    # （含 flash_attn 映射表，缺了会 KeyError: 'flash_attn'）
    try:
        import tf5_compat  # noqa: F401
    except Exception:
        pass
    import torch

    res = {"model_dir": a.model, "baseline": "cpu", "target": "npu:0",
           "criteria": {"rel_l2_pass": REL_L2_PASS, "rel_l2_warn": REL_L2_WARN,
                        "cos_pass": COS_PASS, "cos_warn": COS_WARN}}

    kind = a.kind if a.kind != "auto" else detect_kind(a.model)
    res["kind"] = kind
    if kind is None:
        res.update({"verdict": "SKIP", "reason": "无法判定输入类型（config 里没有可识别的 model_type/architectures）"})
    else:
        inputs = pick_inputs(kind, a.model)
        if inputs is None:
            res.update({"verdict": "SKIP", "reason": "无法为该类型构造固定输入"})
        else:
            res["input_spec"] = {k: list(v.shape) for k, v in inputs.items()}
            try:
                v_cpu = run_forward(a.model, kind, inputs, "cpu")
                res["cpu_ok"] = True
            except Exception as e:  # noqa: BLE001
                res.update({"verdict": "SKIP", "cpu_ok": False,
                            "reason": "CPU 基线跑不起来：%s: %s" % (type(e).__name__, str(e)[:200])})
                v_cpu = None
            if v_cpu is not None:
                try:
                    v_npu = run_forward(a.model, kind, inputs, "npu:0")
                    res["npu_ok"] = True
                except Exception as e:  # noqa: BLE001
                    res.update({"verdict": "FAIL", "npu_ok": False,
                                "reason": "NPU 前向失败：%s: %s" % (type(e).__name__, str(e)[:200])})
                    v_npu = None
                if v_npu is not None:
                    m = compare(v_cpu, v_npu)
                    res["metrics"] = m
                    res["verdict"] = verdict(m)

    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, "accuracy.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    write_report(res, os.path.join(a.out, "ACCURACY_REPORT.md"))
    print("ACC_VERDICT=%s" % res["verdict"])
    if res.get("metrics"):
        m = res["metrics"]
        print("  rel_l2=%.3e  cosine=%.9f  max_abs_diff=%.3e  n=%d"
              % (m["rel_l2"] if m["rel_l2"] is not None else -1,
                 m["cosine_sim"] if m["cosine_sim"] is not None else -1,
                 m["max_abs_diff"], m["n_compared"]))
    else:
        print("  reason:", res.get("reason", ""))


if __name__ == "__main__":
    main()
