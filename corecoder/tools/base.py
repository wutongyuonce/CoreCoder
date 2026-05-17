"""
工具基类模块 - 所有工具的抽象接口。

定义了 Tool 抽象基类，所有具体工具（BashTool、ReadFileTool 等）
都必须继承此类并实现 execute 方法。

工具接口设计遵循 OpenAI 的函数调用规范：
  - name:        工具名称（用于 LLM 识别）
  - description: 工具描述（告诉 LLM 何时使用该工具）
  - parameters:  JSON Schema 格式的参数描述
  - schema():    生成 OpenAI 函数调用格式的 schema
  - execute():   实际执行工具逻辑
"""

from abc import ABC, abstractmethod


class Tool(ABC):
    """
    工具抽象基类 - 最小化的工具接口。

    继承此类并实现 execute 方法即可添加新工具。

    属性:
        name (str): 工具名称（子类必须定义）
        description (str): 工具描述（子类必须定义）
        parameters (dict): JSON Schema 格式的参数描述（子类必须定义）

    方法:
        execute(**kwargs) -> str: 执行工具，返回文本结果（抽象方法，子类必须实现）
        schema() -> dict: 生成 OpenAI 函数调用格式的 schema
    """

    name: str
    description: str
    parameters: dict  # JSON Schema 格式的函数参数描述

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        执行工具（抽象方法）。

        输入: **kwargs - 工具参数（由 parameters schema 定义）
        输出: str - 执行结果的文本描述

        子类必须实现此方法。
        """
        ...

    def schema(self) -> dict:
        """
        生成 OpenAI 函数调用格式的 schema。

        输入: 无（使用自身的 name, description, parameters 属性）
        输出: dict - 符合 OpenAI 函数调用规范的 schema 字典

        关键步骤:
            将工具的 name、description、parameters 包装为
            OpenAI 要求的 {"type": "function", "function": {...}} 格式
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
