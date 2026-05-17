"""
配置管理模块 - 环境变量加载与默认值设定。

本模块负责从环境变量和 .env 文件中加载 CoreCoder 的配置参数。
支持的配置项包括：
  - 模型名称、API 密钥、API 基础 URL
  - 最大 token 数、温度参数
  - 上下文窗口大小、LLM 提供者类型

配置优先级：CLI 参数 > 环境变量 > .env 文件 > 默认值
"""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    """
    加载 .env 文件，从当前目录向上搜索到用户主目录。

    输入: 无
    输出: 无（副作用：将 .env 中的变量加载到 os.environ）

    关键步骤:
        1. 尝试导入 python-dotenv 库（可选依赖）
        2. 从当前目录开始向上搜索 .env 文件
        3. 找到后加载，但不覆盖已存在的环境变量（override=False）
        4. 如果 python-dotenv 未安装，静默跳过
    """
    try:
        from dotenv import load_dotenv
        # 从当前目录开始，向上逐级搜索 .env 文件
        env_path = Path(".env")
        if not env_path.exists():
            cur = Path.cwd()
            home = Path.home()
            # 逐级向上，直到到达主目录
            while cur != home and cur != cur.parent:
                candidate = cur / ".env"
                if candidate.exists():
                    env_path = candidate
                    break
                cur = cur.parent
        # 加载 .env 文件，不覆盖已存在的环境变量
        load_dotenv(env_path, override=False)
    except ImportError:
        pass  # python-dotenv 未安装，静默跳过


@dataclass
class Config:
    """
    CoreCoder 配置类 - 存储所有运行时配置参数。

    属性:
        model (str): LLM 模型名称，默认 "gpt-4o"
        api_key (str): API 密钥
        base_url (str | None): API 基础 URL（用于自定义端点）
        max_tokens (int): 单次 LLM 响应最大 token 数，默认 4096
        temperature (float): 生成温度，0 表示确定性输出，默认 0.0
        max_context_tokens (int): 上下文窗口最大 token 数，默认 128,000
        provider (str): LLM 提供者类型，默认 "openai"，可选 "litellm"
    """
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"

    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量创建配置实例（类方法）。

        输入: 无（读取 os.environ）
        输出: Config - 配置实例

        关键步骤:
            1. 先尝试加载 .env 文件（如果有）
            2. 读取 API Key，按优先级尝试：
               CORECODER_API_KEY > OPENAI_API_KEY > DEEPSEEK_API_KEY
            3. 读取其他环境变量，使用默认值作为后备
            4. 构造并返回 Config 实例

        环境变量映射:
            CORECODER_MODEL        -> model         (默认 "gpt-4o")
            CORECODER_API_KEY      -> api_key       (最高优先级)
            OPENAI_API_KEY         -> api_key       (次优先级)
            DEEPSEEK_API_KEY       -> api_key       (第三优先级)
            OPENAI_BASE_URL        -> base_url
            CORECODER_BASE_URL     -> base_url      (备选)
            CORECODER_MAX_TOKENS   -> max_tokens    (默认 4096)
            CORECODER_TEMPERATURE  -> temperature   (默认 0.0)
            CORECODER_MAX_CONTEXT  -> max_context_tokens (默认 128000)
            CORECODER_PROVIDER     -> provider      (默认 "openai")
        """
        # 步骤1: 加载 .env 文件
        _load_dotenv()
        # 步骤2: 按优先级读取 API Key
        api_key = (
            os.getenv("CORECODER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        # 步骤3-4: 构造配置实例
        return cls(
            model=os.getenv("CORECODER_MODEL", "gpt-4o"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("CORECODER_BASE_URL"),
            max_tokens=int(os.getenv("CORECODER_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("CORECODER_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("CORECODER_MAX_CONTEXT", "128000")),
            provider=os.getenv("CORECODER_PROVIDER", "openai"),
        )
