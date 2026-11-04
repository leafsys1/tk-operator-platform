"""把「先流式说人话 + 末尾附 JSON」的输出切成**可上屏的 prose** 与**吞掉的 JSON 尾**。

契约（`chat.SYSTEM_PROMPT` 同款）：模型先写给人看的自然语言回复（逐 token 上屏），
写完另起一行输出分隔符 `<<<SPEC>>>`，其后是承载结构化产物的 JSON。前端只该看到 prose。

为什么要在后端切：dsh 的增量是**裸文本**，JSON 尾如果直接转给前端，用户会看见一串
`{"spec": ...` 从气泡里流出来。这里用一个小状态机解决，三种模式：

- `prose`   ：正常。发"安全前缀"，把**可能是分隔符前缀**的尾巴留在缓冲区
              （分隔符被网络/插件切成两段是常态，不留尾巴就会把 `<<<SPE` 当正文喷出去）。
- `tail`    ：已见到分隔符，之后所有增量只累积、不上屏。
- `suppress`：**起手判别**——输出以 `{` / `[` / ``` ``` ``` 开头说明模型退了旧格式
              （整段 JSON），此时全都不上屏，等跑完由 `chat.parse_output` 取出 `reply` 补发。

本模块**不依赖 dsh、不依赖 FastAPI**，纯函数 + 一个缓冲类，方便单测。
"""

from __future__ import annotations

#: 分隔符（独占一行；prose 与 JSON 尾的边界）
SPEC_SEP = "<<<SPEC>>>"

#: 起手就判定为「整段都是 JSON」的前缀
_JSON_HEADS = ("{", "[", "`")


class ProseRelay:
    """增量喂入，返回**这次该上屏**的 prose 片段（可能为空串）。"""

    def __init__(self, sep: str = SPEC_SEP) -> None:
        self.sep = sep
        self.mode = "probe"  # probe | prose | tail | suppress
        self.buf = ""
        self.raw = ""  # 模型原始输出（含 JSON 尾）
        self.prose = ""  # 已上屏的部分
        self.tail = ""  # 分隔符之后的部分（JSON）

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self.raw += text
        if self.mode == "suppress":
            return ""
        if self.mode == "tail":
            self.tail += text
            return ""
        self.buf += text
        if self.mode == "probe":
            head = self.buf.lstrip()
            if not head:
                return ""  # 还在空白，等
            if head.startswith(_JSON_HEADS):
                self.mode = "suppress"
                return ""
            self.mode = "prose"
        return self._drain()

    def _drain(self) -> str:
        i = self.buf.find(self.sep)
        if i >= 0:
            out = self.buf[:i].rstrip("\n")
            self.tail += self.buf[i + len(self.sep):]
            self.buf = ""
            self.mode = "tail"
        else:
            hold = self._holdback()
            if hold:
                out, self.buf = self.buf[:-hold], self.buf[-hold:]
                if self.buf.startswith("<"):  # 扣住的是分隔符前缀 → 末尾换行是它的前导
                    out = out.rstrip("\n")
            else:
                out, self.buf = self.buf, ""
        self.prose += out
        return out

    def _holdback(self) -> int:
        """缓冲区末尾「可能是分隔符前缀」的长度（0 = 可以全发）。"""
        for k in range(min(len(self.sep) - 1, len(self.buf)), 0, -1):
            if self.buf.endswith(self.sep[:k]):
                return k
        # 换行也留一留：分隔符是独占一行的，末尾这个 \n 很可能就是它的前导
        # —— 先扣住，等下一段到了再决定是发出去（多段正文）还是丢掉（分隔符前导）。
        return 1 if self.buf.endswith("\n") else 0

    def flush(self) -> str:
        """跑完了：把缓冲区里剩下的（没等到分隔符的）当 prose 发出去。"""
        if self.mode in ("prose", "probe") and self.buf:
            if self.mode == "probe" and self.buf.lstrip().startswith(_JSON_HEADS):
                self.mode = "suppress"
                self.buf = ""
                return ""
            out, self.buf = self.buf, ""
            self.prose += out
            return out
        return ""


def split_prose(text: str) -> str:
    """从**完整输出**里取 prose（无依赖、不抛异常；解析器不可用时的兜底）。

    - 有分隔符：分隔符之前就是 prose；
    - 没有分隔符且整段像 JSON：返回空串（prose 只能靠解析 `reply` 字段拿到）；
    - 其它：整段就是 prose。
    """
    t = text or ""
    i = t.find(SPEC_SEP)
    if i >= 0:
        return t[:i].rstrip("\n")
    if t.lstrip().startswith(_JSON_HEADS):
        return ""
    return t


def split_tail(text: str) -> str:
    """从完整输出里取分隔符之后的 JSON 文本（没有分隔符 = 空串）。"""
    t = text or ""
    i = t.find(SPEC_SEP)
    return t[i + len(SPEC_SEP):].strip() if i >= 0 else ""
