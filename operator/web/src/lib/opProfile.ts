/**
 * 算子算术强度画像（Roofline / 极限对比用）。
 *
 * **不要**再把 AI 写死在页面里：不同算子的读写元素数、每元素浮点运算数不同，
 * `elementwise_add` 的 1 flop / 12 B 不能套到 relu / softmax / matmul 上。
 *
 * 优先级：
 *  1. 机读报告里显式给出的 `arithmetic_intensity`（新 benchmark 报告会写）；
 *  2. 本文件里的常见算子画像（近似值，页面会标明是「估算」）；
 *  3. 查不到 → 返回 null，页面不做 Roofline 判定（不猜）。
 */

export type OpProfile = {
  /** 每个元素读几次（GM→UB 的元素数） */
  reads: number;
  /** 每个元素写几次 */
  writes: number;
  /** 每个元素的浮点运算数（近似；用于判「带宽/算力受限」即可） */
  flops: number;
  /** 估算依据 / 注意事项 */
  note?: string;
};

function normalize(op: string | null | undefined): string {
  return (op ?? "")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
}

const PROFILES: Record<string, OpProfile> = {
  // 逐元素二元
  elementwise_add: { reads: 2, writes: 1, flops: 1, note: "2 读 1 写；add 每元素 1 flop" },
  add: { reads: 2, writes: 1, flops: 1, note: "2 读 1 写；add 每元素 1 flop" },
  addrelu: { reads: 2, writes: 1, flops: 2, note: "add + relu，每元素约 2 op" },
  add_relu: { reads: 2, writes: 1, flops: 2, note: "add + relu，每元素约 2 op" },
  sub: { reads: 2, writes: 1, flops: 1 },
  mul: { reads: 2, writes: 1, flops: 1 },
  div: { reads: 2, writes: 1, flops: 1 },
  maximum: { reads: 2, writes: 1, flops: 1 },
  minimum: { reads: 2, writes: 1, flops: 1 },

  // 逐元素一元（近似）
  relu: { reads: 1, writes: 1, flops: 1, note: "1 读 1 写；比较/选择按 1 op 估" },
  abs: { reads: 1, writes: 1, flops: 1 },
  neg: { reads: 1, writes: 1, flops: 1 },
  exp: { reads: 1, writes: 1, flops: 1, note: "超越函数按 1 op 估（实际指令数更多）" },
  log: { reads: 1, writes: 1, flops: 1 },
  sqrt: { reads: 1, writes: 1, flops: 1 },
  rsqrt: { reads: 1, writes: 1, flops: 1 },
  sigmoid: { reads: 1, writes: 1, flops: 4, note: "exp + 加/除，按约 4 op 估" },
  tanh: { reads: 1, writes: 1, flops: 4 },
  leaky_relu: { reads: 1, writes: 1, flops: 2 },
  gelu: { reads: 1, writes: 1, flops: 8, note: "gelu 按约 8 op 估" },
  silu: { reads: 1, writes: 1, flops: 4, note: "silu = x·sigmoid(x)，按约 4 op 估" },
  swish: { reads: 1, writes: 1, flops: 4 },

  // 归约 / 归一化（按整轴归约的近似）
  softmax: { reads: 1, writes: 1, flops: 5, note: "max/sub/exp/sum/div，按约 5 op 估" },
  log_softmax: { reads: 1, writes: 1, flops: 6 },
  layer_norm: { reads: 1, writes: 1, flops: 8, note: "均值/方差/归一化/缩放，按约 8 op 估" },
  rms_norm: { reads: 1, writes: 1, flops: 6 },
  reduce_sum: { reads: 1, writes: 1, flops: 1 },
  reduce_mean: { reads: 1, writes: 1, flops: 2 },
  reduce_max: { reads: 1, writes: 1, flops: 1 },
};

// 常见命名变体（camelCase / 连写）归一到上面的键
const ALIASES: Record<string, string> = {
  elementwiseadd: "elementwise_add",
  add_relu: "addrelu",
  leakyrelu: "leaky_relu",
  layernorm: "layer_norm",
  rmsnorm: "rms_norm",
  logsoftmax: "log_softmax",
  reducesum: "reduce_sum",
  reducemean: "reduce_mean",
  reducemax: "reduce_max",
};

export function opProfile(op: string | null | undefined): OpProfile | null {
  const key = normalize(op);
  return PROFILES[key] ?? PROFILES[ALIASES[key]] ?? null;
}

export function dtypeBytes(dtype: string | null | undefined): number {
  const d = (dtype ?? "").toLowerCase();
  if (d.includes("64")) return 8;
  if (d.includes("16") || d === "bf16" || d === "fp16" || d === "half") return 2;
  if (d === "int8" || d === "uint8" || d === "bool") return 1;
  return 4; // fp32 / int32 / 默认
}

export type ArithmeticIntensity = {
  value: number; // flop / byte
  /** report = 报告显式给出；profile = 按算子画像估算 */
  source: "report" | "profile";
  note?: string;
};

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null;
}

/**
 * 从机读报告 / 算子名推导算术强度。返回 null = 没有可靠画像（页面不应下 Roofline 结论）。
 *
 * @param report  `performance.json` 原样
 * @param opName  报告里的 op（优先）或任务 op_name
 * @param primary 主 benchmark 用例（取 dtype；新报告也带 derived.arithmetic_intensity）
 */
export function arithmeticIntensity(
  report: Record<string, unknown>,
  opName: string | null | undefined,
  primary?: { dtype?: string | null; derived?: { arithmetic_intensity?: number | null } | null } | null,
): ArithmeticIntensity | null {
  const cond = (report.conditions as Record<string, unknown> | undefined) ?? {};
  const explicit =
    asNumber(primary?.derived?.arithmetic_intensity) ??
    asNumber(cond.arithmetic_intensity);
  if (explicit != null) {
    return { value: explicit, source: "report", note: "报告显式给出的 AI" };
  }

  const profile = opProfile(opName);
  if (!profile) return null;
  const elemBytes = dtypeBytes(primary?.dtype);
  const moved = (profile.reads + profile.writes) * elemBytes;
  if (moved <= 0) return null;
  return {
    value: profile.flops / moved,
    source: "profile",
    note: profile.note,
  };
}

/** 报告里的基线名称（不再写死 torch.add）。 */
export function baselineLabel(
  report: Record<string, unknown>,
  primary?: { baseline?: { impl?: string } | null } | null,
): string {
  const cond = (report.conditions as Record<string, unknown> | undefined) ?? {};
  const fromCond = typeof cond.baseline_impl === "string" ? cond.baseline_impl.trim() : "";
  const fromCase = (primary?.baseline?.impl ?? "").trim();
  return fromCond || fromCase || "基线";
}

