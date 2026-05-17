"""
工具注册表 - 管理所有内置工具的实例和查找。

本模块是工具系统的入口，负责：
  1. 导入所有内置工具类
  2. 创建全局工具实例列表（ALL_TOOLS）
  3. 提供按名称查找工具的函数（get_tool）

内置工具列表：
  - BashTool:     执行 Shell 命令
  - ReadFileTool: 读取文件内容
  - WriteFileTool: 写入/创建文件
  - EditFileTool: 精确查找替换编辑
  - GlobTool:     文件模式匹配
  - GrepTool:     正则内容搜索
  - AgentTool:    生成子代理处理子任务
"""

from .bash import BashTool
from .read import ReadFileTool
from .write import WriteFileTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .agent import AgentTool

# 全局工具实例列表 - Agent 启动时默认使用这些工具
ALL_TOOLS = [
    BashTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
    GrepTool(),
    AgentTool(),
]


def get_tool(name: str):
    """
    按名称查找工具实例。

    输入:
        name (str): 工具名称（如 "bash", "read_file"）

    输出: Tool | None - 对应的工具实例，未找到返回 None

    关键步骤:
        遍历 ALL_TOOLS 列表，匹配名称并返回对应的工具实例
    """
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
