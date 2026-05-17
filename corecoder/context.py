"""
多层上下文压缩模块 - 智能管理对话历史的 token 消耗。

设计灵感来源于 Claude Code 的 4 层压缩策略：
  1. HISTORY_SNIP   - 将旧的工具输出截断为一行摘要
  2. Microcompact   - LLM 驱动的旧对话摘要（带缓存）
  3. CONTEXT_COLLAPSE - 接近硬限制时的激进压缩
  4. Autocompact    - 定期后台压缩

CoreCoder 实现了类似的 3 层压缩策略：
  第 1 层 (tool_snip)     - 截断冗长的工具输出，保留首尾各 3 行
  第 2 层 (summarize)     - 使用 LLM 生成旧对话的摘要
  第 3 层 (hard_collapse) - 最后手段：仅保留摘要 + 最近几条消息

压缩阈值（占 max_tokens 的比例）：
  50% -> 触发第 1 层（截断工具输出）
  70% -> 触发第 2 层（LLM 摘要）
  90% -> 触发第 3 层（硬压缩）
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM


def _approx_tokens(text: str) -> int:
    """
    粗略估算文本的 token 数量。

    输入:
        text (str): 需要估算的文本

    输出: int - 估算的 token 数量

    关键步骤:
        使用简单的字符数除以 3 作为近似值（约 3.5 字符/token，
        针对中英文混合内容取稍保守的值 3）
    """
    return len(text) // 3


def estimate_tokens(messages: list[dict]) -> int:
    """
    估算消息列表的总 token 数量。

    输入:
        messages (list[dict]): OpenAI 格式的消息列表

    输出: int - 估算的总 token 数量

    关键步骤:
        1. 遍历所有消息
        2. 累加 content 字段的 token 数
        3. 累加 tool_calls 字段的 token 数（转为字符串后估算）
    """
    total = 0
    for m in messages:
        # 累加文本内容的 token
        if m.get("content"):
            total += _approx_tokens(m["content"])
        # 累加工具调用信息的 token
        if m.get("tool_calls"):
            total += _approx_tokens(str(m["tool_calls"]))
    return total


class ContextManager:
    """
    上下文管理器 - 自动管理对话历史的 token 消耗。

    采用 3 层渐进式压缩策略，确保对话不会超出 LLM 的上下文窗口：
      - 第 1 层：截断冗长的工具输出（快速、无损）
      - 第 2 层：LLM 生成对话摘要（中等、有损）
      - 第 3 层：硬压缩，仅保留摘要和最近消息（激进、有损）

    属性:
        max_tokens (int): 上下文窗口最大 token 数
        _snip_at (int): 第 1 层触发阈值（50% of max_tokens）
        _summarize_at (int): 第 2 层触发阈值（70% of max_tokens）
        _collapse_at (int): 第 3 层触发阈值（90% of max_tokens）
    """

    def __init__(self, max_tokens: int = 128_000):
        """
        初始化上下文管理器。

        输入:
            max_tokens (int): 上下文窗口最大 token 数，默认 128,000

        输出: 无

        关键步骤:
            设置各层压缩的触发阈值（按 max_tokens 的比例）
        """
        self.max_tokens = max_tokens
        # 各层触发阈值（占 max_tokens 的比例）
        self._snip_at = int(max_tokens * 0.50)    # 50% -> 截断工具输出
        self._summarize_at = int(max_tokens * 0.70)  # 70% -> LLM 摘要
        self._collapse_at = int(max_tokens * 0.90)   # 90% -> 硬压缩

    def maybe_compress(self, messages: list[dict], llm: LLM | None = None) -> bool:
        """
        根据需要应用压缩层。返回是否执行了压缩。

        输入:
            messages (list[dict]): 消息列表（会被原地修改）
            llm (LLM | None): LLM 实例，用于第 2、3 层压缩时生成摘要

        输出: bool - 是否执行了任何压缩操作

        关键步骤:
            1. 估算当前 token 数
            2. 如果超过 50% 阈值，执行第 1 层（截断工具输出）
            3. 如果超过 70% 阈值且消息数 > 10，执行第 2 层（LLM 摘要）
            4. 如果超过 90% 阈值且消息数 > 4，执行第 3 层（硬压缩）
            5. 每层执行后重新估算 token 数，避免不必要的后续压缩
        """
        # 步骤1: 估算当前 token
        current = estimate_tokens(messages)
        compressed = False

        # 步骤2: 第 1 层 - 截断冗长的工具输出
        if current > self._snip_at:
            if self._snip_tool_outputs(messages):
                compressed = True
                current = estimate_tokens(messages)  # 重新估算

        # 步骤3: 第 2 层 - LLM 驱动的对话摘要
        if current > self._summarize_at and len(messages) > 10:
            if self._summarize_old(messages, llm, keep_recent=8):
                compressed = True
                current = estimate_tokens(messages)  # 重新估算

        # 步骤4: 第 3 层 - 硬压缩（最后手段）
        if current > self._collapse_at and len(messages) > 4:
            self._hard_collapse(messages, llm)
            compressed = True

        return compressed

    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        """
        第 1 层压缩：截断超过 1500 字符的工具输出，保留首尾各 3 行。

        输入:
            messages (list[dict]): 消息列表（会被原地修改）

        输出: bool - 是否有任何消息被截断

        关键步骤:
            1. 遍历所有 role="tool" 的消息
            2. 如果内容超过 1500 字符且超过 6 行，进行截断
            3. 保留前 3 行 + 后 3 行，中间用 "... (N lines, snipped)" 替代
            4. 修改消息的 content 字段

        设计说明:
            对应 Claude Code 的 HISTORY_SNIP 策略，将旧的工具输出替换为
            一行摘要以回收上下文空间。工具输出通常是最占 token 的部分
            （如文件内容、命令输出等）。
        """
        changed = False
        for m in messages:
            # 只处理工具输出消息
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            # 长度阈值：1500 字符
            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            # 行数阈值：6 行
            if len(lines) <= 6:
                continue
            # 保留首尾各 3 行，中间截断
            snipped = (
                "\n".join(lines[:3])
                + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                + "\n".join(lines[-3:])
            )
            m["content"] = snipped
            changed = True
        return changed

    def _summarize_old(self, messages: list[dict], llm: LLM | None,
                       keep_recent: int = 8) -> bool:
        """
        第 2 层压缩：使用 LLM 摘要旧对话，保留最近的消息不变。

        输入:
            messages (list[dict]): 消息列表（会被原地修改）
            llm (LLM | None): LLM 实例，用于生成摘要
            keep_recent (int): 保留最近 N 条消息不压缩，默认 8

        输出: bool - 是否执行了压缩

        关键步骤:
            1. 将消息分为旧消息（需要压缩）和新消息（保留）
            2. 使用 LLM 生成旧消息的摘要
            3. 用摘要替换旧消息：
               - 一条 user 消息包含 "[Context compressed]" + 摘要
               - 一条 assistant 确认消息
               - 加上保留的新消息
        """
        # 步骤1: 分割消息
        if len(messages) <= keep_recent:
            return False

        old = messages[:-keep_recent]  # 需要压缩的旧消息
        tail = messages[-keep_recent:]  # 保留的最近消息

        # 步骤2: 生成摘要
        summary = self._get_summary(old, llm)

        # 步骤3: 替换消息列表
        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Context compressed - conversation summary]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Got it, I have the context from our earlier conversation.",
        })
        messages.extend(tail)
        return True

    def _hard_collapse(self, messages: list[dict], llm: LLM | None):
        """
        第 3 层压缩：紧急压缩，仅保留最后 4 条消息 + 摘要。

        输入:
            messages (list[dict]): 消息列表（会被原地修改）
            llm (LLM | None): LLM 实例，用于生成摘要

        输出: 无

        关键步骤:
            1. 保留最后 4 条消息（或最后 2 条，如果总数不足）
            2. 对其余所有消息生成摘要
            3. 用 "[Hard context reset]" + 摘要替换被压缩的消息
            4. 保留最后几条消息以维持上下文连续性
        """
        # 步骤1: 保留最后几条消息
        tail = messages[-4:] if len(messages) > 4 else messages[-2:]
        # 步骤2: 对其余消息生成摘要
        summary = self._get_summary(messages[:-len(tail)], llm)

        # 步骤3-4: 替换消息列表
        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Hard context reset]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Context restored. Continuing from where we left off.",
        })
        messages.extend(tail)

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """
        生成对话摘要。优先使用 LLM，失败时回退到关键信息提取。

        输入:
            messages (list[dict]): 需要摘要的消息列表
            llm (LLM | None): LLM 实例

        输出: str - 对话摘要文本

        关键步骤:
            1. 将消息扁平化为可读文本
            2. 尝试使用 LLM 生成结构化摘要：
               - 保留：编辑的文件路径、关键决策、遇到的错误、当前任务状态
               - 丢弃：冗长的命令输出、代码清单、重复对话
            3. 如果 LLM 调用失败，回退到关键信息提取（无需 LLM）
        """
        # 步骤1: 扁平化消息
        flat = self._flatten(messages)

        # 步骤2: 尝试 LLM 摘要
        if llm:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception:
                pass

        # 步骤3: 回退方案 - 无需 LLM 的关键信息提取
        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        """
        将消息列表扁平化为可读的纯文本。

        输入:
            messages (list[dict]): 消息列表

        输出: str - 扁平化后的文本，格式为 "[role] content"

        关键步骤:
            1. 遍历所有消息
            2. 提取 role 和 content 字段
            3. 截断每条消息的 content 到 400 字符
            4. 拼接为统一格式的文本
        """
        parts = []
        for m in messages:
            role = m.get("role", "?")
            text = m.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """
        回退方案：无需 LLM，直接从消息中提取关键信息。

        输入:
            messages (list[dict]): 消息列表

        输出: str - 提取的关键信息摘要

        关键步骤:
            1. 使用正则表达式提取文件路径
            2. 提取包含 "error" 的行
            3. 组装为简洁的摘要文本

        设计说明:
            当 LLM 不可用时（例如 API 调用失败），此方法作为降级方案，
            虽然不如 LLM 摘要智能，但能保留最重要的上下文信息。
        """
        import re
        files_seen = set()  # 出现过的文件路径
        errors = []         # 错误信息
        decisions = []      # 决策记录

        for m in messages:
            text = m.get("content", "") or ""
            # 提取文件路径（匹配 *.ext 格式）
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            # 提取错误行
            for line in text.splitlines():
                if 'error' in line.lower() or 'Error' in line:
                    errors.append(line.strip()[:150])

        # 组装摘要
        parts = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"
