"""
CoreCoder - 极简 AI 编程代理，受 Claude Code 架构启发。

本包是 CoreCoder 的顶层入口，负责统一导出核心组件：
  - Agent:  核心代理循环（用户消息 -> LLM -> 工具调用 -> 执行 -> 循环）
  - LLM:    LLM 提供者层（OpenAI 兼容 API 封装）
  - Config: 配置管理（环境变量 + 默认值）
  - ALL_TOOLS: 所有内置工具列表

典型用法:
    from corecoder import Agent, LLM, Config
"""

__version__ = "0.3.0"

from corecoder.agent import Agent       # 核心代理类，管理对话循环与工具执行
from corecoder.llm import LLM           # LLM 接口封装，支持流式输出与工具调用
from corecoder.config import Config     # 配置管理，从环境变量加载设置
from corecoder.tools import ALL_TOOLS   # 所有内置工具实例列表

# 公开 API：外部使用者只需 import 这些即可
__all__ = ["Agent", "LLM", "Config", "ALL_TOOLS", "__version__"]
