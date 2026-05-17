"""
核心代理循环模块 - CoreCoder 的核心引擎。

这是整个系统的心脏，采用简洁的循环模式：

    用户消息 -> LLM（带工具描述）-> 有工具调用？-> 执行工具 -> 继续循环
                                -> 纯文本回复？-> 返回给用户

循环会持续进行，直到 LLM 返回纯文本响应（没有工具调用），
这意味着它已经完成工作并准备向用户汇报结果。

核心设计思想（灵感来源于 Claude Code 的 StreamingToolExecutor）：
    1. 流式输出：LLM 的响应实时流式推送给用户，提升交互体验
    2. 工具并行：当 LLM 一次返回多个工具调用时，并行执行以提高效率
    3. 上下文压缩：当对话过长时，自动压缩历史以控制 token 消耗
"""

import concurrent.futures
from .llm import LLM
from .tools import ALL_TOOLS, get_tool
from .tools.base import Tool
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager


class Agent:
    """
    核心代理类 - 管理用户与 LLM 之间的完整对话循环。

    职责：
        1. 维护对话历史（messages 列表）
        2. 调用 LLM 获取响应（包括文本和工具调用）
        3. 执行工具并将结果反馈给 LLM
        4. 管理上下文压缩，防止 token 超限
        5. 支持子代理（AgentTool）的能力注入
    """

    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
    ):
        """
        初始化代理实例。

        输入:
            llm (LLM): LLM 接口实例，负责与语言模型 API 通信
            tools (list[Tool] | None): 可用工具列表，None 则使用全部内置工具
            max_context_tokens (int): 上下文窗口最大 token 数，默认 128,000
            max_rounds (int): 单次对话最大工具调用轮数，防止无限循环，默认 50

        输出: 无

        关键步骤:
            1. 存储 LLM 实例和工具列表
            2. 初始化空对话历史 messages
            3. 创建上下文管理器，用于后续的自动压缩
            4. 生成系统提示词（包含环境信息和工具描述）
            5. 遍历工具列表，将自身注入到 AgentTool 中，使子代理能够回调主代理
        """
        self.llm = llm                                              # LLM 接口
        self.tools = tools if tools is not None else ALL_TOOLS      # 可用工具集
        self.messages: list[dict] = []                              # 对话历史
        self.context = ContextManager(max_tokens=max_context_tokens)  # 上下文管理器
        self.max_rounds = max_rounds                                # 最大工具调用轮数
        self._system = system_prompt(self.tools)                    # 系统提示词

        # 关键步骤：将主代理自身注入到 AgentTool 中，实现子代理回调能力
        # 这样子代理在需要时可以访问主代理的工具集和配置
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        """
        构建发送给 LLM 的完整消息列表。

        输入: 无（使用 self._system 和 self.messages）
        输出: list[dict] - 包含系统提示词 + 所有历史消息的完整列表

        关键步骤:
            1. 将系统提示词作为第一条消息（role="system"）
            2. 拼接所有历史对话消息
        """
        return [{"role": "system", "content": self._system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        """
        获取所有工具的 OpenAI 函数调用 schema。

        输入: 无（使用 self.tools）
        输出: list[dict] - 每个工具的 JSON Schema 描述列表

        关键步骤:
            遍历所有工具，调用各自的 schema() 方法生成符合 OpenAI 函数调用格式的描述
        """
        return [t.schema() for t in self.tools]

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """
        处理一条用户消息，可能涉及多轮 LLM 调用和工具执行。

        输入:
            user_input (str): 用户输入的文本消息
            on_token (callable | None): 回调函数，每收到一个流式 token 时调用
                                        签名: on_token(token_text: str)
            on_tool (callable | None): 回调函数，每次工具被调用前触发
                                       签名: on_tool(tool_name: str, arguments: dict)

        输出: str - LLM 最终的文本回复内容

        关键步骤:
            1. 将用户消息追加到对话历史
            2. 检查是否需要上下文压缩，必要时自动压缩
            3. 进入最大 max_rounds 轮的循环：
               a. 调用 LLM（传入完整历史 + 工具 schema）获取响应
               b. 如果 LLM 返回纯文本（无工具调用），说明工作完成，返回文本
               c. 如果有工具调用：
                  - 单个工具：直接执行
                  - 多个工具：并行执行（使用线程池，最多 8 个工作线程）
               d. 将工具执行结果追加到对话历史
               e. 再次检查上下文压缩
            4. 如果达到最大轮数仍未结束，返回提示信息
        """
        # 步骤1: 记录用户消息
        self.messages.append({"role": "user", "content": user_input})
        # 步骤2: 必要时压缩上下文
        self.context.maybe_compress(self.messages, self.llm)

        # 步骤3: 进入工具调用循环
        for _ in range(self.max_rounds):
            # 3a: 调用 LLM 获取响应
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(),
                on_token=on_token,
            )

            # 3b: 无工具调用 -> LLM 工作完成，返回文本
            if not resp.tool_calls:
                self.messages.append(resp.message)
                return resp.content

            # 3c: 有工具调用 -> 执行工具
            self.messages.append(resp.message)

            if len(resp.tool_calls) == 1:
                # 单个工具调用，直接串行执行
                tc = resp.tool_calls[0]
                if on_tool:
                    on_tool(tc.name, tc.arguments)
                result = self._exec_tool(tc)
                # 3d: 将工具结果追加到历史
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            else:
                # 多个工具调用，并行执行（类似 Claude Code 的 StreamingToolExecutor）
                results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                for tc, result in zip(resp.tool_calls, results):
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            # 3e: 工具输出可能很大，再次压缩
            self.context.maybe_compress(self.messages, self.llm)

        # 步骤4: 达到最大轮数
        return "(reached maximum tool-call rounds)"

    def _exec_tool(self, tc) -> str:
        """
        执行单个工具调用。

        输入:
            tc: 工具调用对象，包含以下属性：
                - tc.name (str): 工具名称
                - tc.id (str): 工具调用的唯一标识符
                - tc.arguments (dict): 工具的参数字典

        输出: str - 工具执行结果文本（成功时）或错误信息（失败时）

        关键步骤:
            1. 通过工具名称查找对应的工具实例
            2. 如果工具不存在，返回错误信息
            3. 调用工具的 execute 方法，传入参数
            4. 捕获 TypeError（参数错误）和其他异常，返回友好的错误信息
        """
        # 步骤1: 查找工具
        tool = get_tool(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # 步骤2-4: 执行工具并处理异常
        try:
            return tool.execute(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        except Exception as e:
            return f"Error executing {tc.name}: {e}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """
        使用线程池并行执行多个工具调用。

        输入:
            tool_calls: 工具调用对象列表，每个包含 name/id/arguments 属性
            on_tool (callable | None): 工具调用前的回调函数

        输出: list[str] - 每个工具调用的执行结果列表（顺序与输入一致）

        关键步骤:
            1. 遍历所有工具调用，触发 on_tool 回调通知
            2. 创建线程池（最多 8 个工作线程）
            3. 为每个工具调用提交一个任务到线程池
            4. 等待所有任务完成并收集结果
            5. 返回按原始顺序排列的结果列表

        设计说明:
            受 Claude Code 的 StreamingToolExecutor 启发，当 LLM 一次返回
            多个工具调用时并行执行，而不是串行等待，显著提升执行效率。
            例如：同时读取多个文件、同时搜索多个目录等场景。
        """
        # 步骤1: 触发所有工具的回调通知
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # 步骤2-4: 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._exec_tool, tc) for tc in tool_calls]
            return [f.result() for f in futures]

    def reset(self):
        """
        清空对话历史，重置代理状态。

        输入: 无
        输出: 无

        关键步骤:
            清空 messages 列表，使代理回到初始状态
        """
        self.messages.clear()
