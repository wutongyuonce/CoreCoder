"""
LLM 提供者层 - OpenAI 兼容 API 的轻量封装。

本模块提供两种 LLM 后端：

1. LLM（标准后端）：
   - 直接使用 openai SDK 调用所有 OpenAI 兼容 API
   - 支持的提供者：OpenAI、DeepSeek、Qwen、Kimi、GLM、Ollama 等
   - 切换提供者只需更改 OPENAI_BASE_URL + OPENAI_API_KEY

2. LiteLLM（通用后端）：
   - 通过 litellm 库路由到 100+ 提供者
   - 适用于非 OpenAI 兼容的提供者（AWS Bedrock、Google Vertex 等）
   - 设置 CORECODER_PROVIDER=litellm 启用

流式响应处理流程：
    LLM API (stream) -> 逐 chunk 解析 -> 累积文本 + 工具调用 -> 返回 LLMResponse

关键数据结构：
    ToolCall:     单个工具调用（id, name, arguments）
    LLMResponse:  LLM 的完整响应（文本内容 + 工具调用列表 + token 统计）
"""

import json
import time
from dataclasses import dataclass, field

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError


@dataclass
class ToolCall:
    """
    工具调用数据结构 - 表示 LLM 请求执行的一个工具调用。

    属性:
        id (str): 工具调用的唯一标识符（用于将结果与调用关联）
        name (str): 工具名称（如 "bash", "read_file", "edit_file"）
        arguments (dict): 工具的参数字典
    """
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """
    LLM 响应数据结构 - 封装 LLM 返回的完整信息。

    属性:
        content (str): LLM 返回的文本内容
        tool_calls (list[ToolCall]): LLM 请求的工具调用列表（可能为空）
        prompt_tokens (int): 本次请求消耗的输入 token 数
        completion_tokens (int): 本次请求生成的输出 token 数
    """
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def message(self) -> dict:
        """
        将响应转换为 OpenAI 消息格式，用于追加到对话历史。

        输入: 无（使用自身的 content 和 tool_calls 属性）
        输出: dict - 符合 OpenAI API 格式的消息字典

        关键步骤:
            1. 构造基础消息（role="assistant", content=文本内容）
            2. 如果有工具调用，转换为 OpenAI 的 function calling 格式
            3. 返回可直接追加到 messages 列表的字典
        """
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


# 价格表：每百万 token 的价格，格式为 (输入价格, 输出价格)，单位：美元
# 数据来源：openai.com/api/pricing, api-docs.deepseek.com, platform.claude.com,
#           platform.moonshot.ai, alibabacloud.com/help/en/model-studio
_PRICING = {
    # OpenAI - 最新旗舰模型
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    # OpenAI - 上一代（仍广泛使用）
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen（通义千问）
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi（月之暗面）
    "kimi-k2.5": (0.6, 3),
}


class LLM:
    """
    标准 LLM 后端 - 基于 OpenAI SDK 的通用实现。

    适用于所有 OpenAI 兼容 API，通过修改 base_url 和 api_key
    即可切换到不同的提供者。

    属性:
        model (str): 模型名称
        client (OpenAI): OpenAI 客户端实例
        extra (dict): 额外参数（temperature, max_tokens 等）
        total_prompt_tokens (int): 累计输入 token 数
        total_completion_tokens (int): 累计输出 token 数
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs,
    ):
        """
        初始化 LLM 实例。

        输入:
            model (str): 模型名称（如 "gpt-4o", "deepseek-chat"）
            api_key (str): API 密钥
            base_url (str | None): API 基础 URL（None 使用 OpenAI 默认）
            **kwargs: 其他参数（temperature, max_tokens 等）

        输出: 无

        关键步骤:
            1. 存储模型名称
            2. 创建 OpenAI 客户端实例
            3. 存储额外参数
            4. 初始化 token 计数器
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.extra = kwargs  # temperature, max_tokens 等额外参数
        self.total_prompt_tokens = 0      # 累计输入 token
        self.total_completion_tokens = 0  # 累计输出 token

    @property
    def estimated_cost(self) -> float | None:
        """
        估算累计 API 调用费用（美元）。

        输入: 无（使用自身的 token 计数器和 _PRICING 价格表）
        输出: float | None - 估算的费用（美元），如果模型不在价格表中返回 None

        关键步骤:
            1. 从价格表中查找当前模型的价格
            2. 如果模型不在价格表中，返回 None
            3. 计算：输入token数 * 输入单价 / 1M + 输出token数 * 输出单价 / 1M
        """
        pricing = _PRICING.get(self.model)
        if not pricing:
            return None
        input_rate, output_rate = pricing
        return (
            self.total_prompt_tokens * input_rate / 1_000_000
            + self.total_completion_tokens * output_rate / 1_000_000
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """
        发送消息给 LLM，流式接收响应，处理工具调用。

        输入:
            messages (list[dict]): OpenAI 格式的消息列表
            tools (list[dict] | None): 工具的 JSON Schema 列表（可选）
            on_token (callable | None): 流式 token 回调函数

        输出: LLMResponse - 包含文本内容、工具调用和 token 统计的响应

        关键步骤:
            1. 构造请求参数（model, messages, stream=True, 其他额外参数）
            2. 添加工具 schema（如果提供）
            3. 尝试启用 stream_options 获取 usage 统计（OpenAI 扩展）
            4. 带重试的流式调用 LLM API
            5. 逐 chunk 处理流式响应：
               a. 从最后一个 chunk 提取 usage 信息
               b. 累积文本内容（实时通过 on_token 回调推送）
               c. 跨 chunk 累积工具调用信息（id, name, arguments）
            6. 解析累积的工具调用（JSON 反序列化 arguments）
            7. 更新累计 token 计数器
            8. 返回 LLMResponse 对象
        """
        # 步骤1-2: 构造请求参数
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            params["tools"] = tools

        # 步骤3-4: 流式调用（带重试）
        # stream_options 是 OpenAI 扩展，不是所有提供者都支持
        try:
            params["stream_options"] = {"include_usage": True}
            stream = self._call_with_retry(params)
        except Exception:
            params.pop("stream_options", None)
            stream = self._call_with_retry(params)

        # 步骤5: 逐 chunk 处理流式响应
        content_parts: list[str] = []         # 累积的文本片段
        tc_map: dict[int, dict] = {}          # 工具调用累积映射：index -> {id, name, args}
        prompt_tok = 0                        # 本次输入 token
        completion_tok = 0                    # 本次输出 token

        for chunk in stream:
            # 5a: 从最后一个 chunk 提取 usage 信息
            if chunk.usage:
                prompt_tok = chunk.usage.prompt_tokens
                completion_tok = chunk.usage.completion_tokens

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 5b: 累积文本内容，实时推送
            if delta.content:
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            # 5c: 跨 chunk 累积工具调用信息
            # 工具调用可能跨多个 chunk 到达（流式传输的特性）
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # 步骤6: 解析累积的工具调用
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        # 步骤7: 更新累计 token 计数器
        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok

        # 步骤8: 返回响应
        return LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """
        带指数退避重试的 API 调用。

        输入:
            params (dict): API 请求参数
            max_retries (int): 最大重试次数，默认 3

        输出: Stream - OpenAI 的流式响应对象

        关键步骤:
            1. 尝试调用 API
            2. 如果遇到瞬态错误（限流、超时、连接问题），指数退避重试
            3. 如果遇到 5xx 服务端错误，指数退避重试
            4. 如果遇到 4xx 客户端错误，直接抛出（不重试）
            5. 退避策略：等待 2^attempt 秒（1s, 2s, 4s）
        """
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                # 瞬态错误：限流、超时、连接问题
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
            except APIError as e:
                # 5xx = 服务端错误，重试；4xx = 客户端错误，不重试
                if e.status_code and e.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


class LiteLLM(LLM):
    """
    LiteLLM 后端 - 通过 litellm 库支持 100+ 提供者。

    当目标提供者不兼容 OpenAI API 时使用此后端（如 AWS Bedrock、
    Google Vertex、Cohere 等），或希望通过统一接口在任意提供者间切换。

    使用方式：
        设置 CORECODER_PROVIDER=litellm
        使用 LiteLLM 模型字符串，如：
        - anthropic/claude-3-haiku
        - bedrock/anthropic.claude-v2
        - vertex_ai/gemini-pro
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        """
        初始化 LiteLLM 实例。

        输入:
            model (str): LiteLLM 格式的模型名称
            api_key (str | None): API 密钥（可选）
            base_url (str | None): API 基础 URL（可选）
            **kwargs: 其他参数

        输出: 无

        关键步骤:
            跳过父类 LLM.__init__（因为它会创建 OpenAI 客户端），
            直接初始化必要的属性
        """
        # 跳过 LLM.__init__（不创建 OpenAI 客户端）
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """
        通过 litellm 发送消息，流式接收响应，处理工具调用。

        输入:
            messages (list[dict]): OpenAI 格式的消息列表
            tools (list[dict] | None): 工具的 JSON Schema 列表（可选）
            on_token (callable | None): 流式 token 回调函数

        输出: LLMResponse - 包含文本内容、工具调用和 token 统计的响应

        关键步骤:
            与父类 LLM.chat() 类似，但使用 litellm.completion() 作为 API 调用
            使用 getattr 安全访问属性（litellm 的返回格式可能略有差异）
        """
        # 构造请求参数
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            params["tools"] = tools

        # 流式调用
        stream = self._call_with_retry(params)

        # 逐 chunk 处理流式响应
        content_parts: list[str] = []
        tc_map: dict[int, dict] = {}
        prompt_tok = 0
        completion_tok = 0

        for chunk in stream:
            # 使用 getattr 安全访问（litellm 兼容性）
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or 0

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            # 累积文本
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            # 累积工具调用
            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # 解析工具调用
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        # 更新 token 计数器
        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """
        带指数退避重试的 litellm API 调用。

        输入:
            params (dict): API 请求参数
            max_retries (int): 最大重试次数，默认 3

        输出: Generator - litellm 的流式响应生成器

        关键步骤:
            1. 设置 drop_params=True（自动丢弃不支持的参数）
            2. 传入 api_key 和 api_base（如果提供）
            3. 调用 litellm.completion()
            4. 遇到瞬态或服务端错误时指数退避重试
        """
        import litellm

        # 设置 litellm 特定参数
        params["drop_params"] = True  # 自动丢弃不支持的参数
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        for attempt in range(max_retries):
            try:
                return litellm.completion(**params)
            except Exception as e:
                err = str(e).lower()
                # 检测瞬态错误关键词
                is_transient = any(
                    kw in err
                    for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"]
                )
                # 检测服务端错误
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                if (is_transient or is_server) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
