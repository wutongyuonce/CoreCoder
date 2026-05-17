"""
子代理生成工具 - 受 Claude Code 的 AgentTool（1397 行）启发。

核心思想：对于复杂的子任务，生成一个独立的代理实例来处理。
子代理拥有自己的对话历史和工具访问权限，不会污染主代理的上下文窗口。

典型使用场景：
  - 研究一个代码库并报告发现
  - 在隔离环境中实现一个多步骤的变更
  - 任何受益于独立上下文窗口的任务

子代理运行到完成后返回文本摘要。
"""

from .base import Tool


class AgentTool(Tool):
    """
    子代理工具 - 生成独立的代理实例来处理子任务。

    属性:
        name (str): "agent"
        description (str): 工具描述
        parameters (dict): 参数 schema（仅需 task 参数）
        _parent_agent: 父代理实例（由 Agent.__init__ 注入）
    """
    name = "agent"
    description = (
        "Spawn a sub-agent to handle a complex sub-task independently. "
        "The sub-agent has its own context and tool access. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
        },
        "required": ["task"],
    }

    # 由 Agent.__init__ 在构造后注入
    _parent_agent = None

    def execute(self, task: str) -> str:
        """
        生成子代理并执行子任务。

        输入:
            task (str): 子代理需要完成的任务描述

        输出: str - 子代理的执行结果（文本摘要）

        关键步骤:
            1. 检查父代理是否已初始化
            2. 从父代理创建子代理实例：
               - 使用相同的 LLM 接口
               - 继承父代理的工具集（排除 agent 自身，防止递归）
               - 使用父代理的上下文窗口大小
               - 限制最大 20 轮（避免子代理运行过久）
            3. 调用子代理的 chat() 方法执行任务
            4. 如果结果超过 5000 字符，截断以避免撑爆父代理的上下文
            5. 返回格式化的结果

        异常处理:
            捕获所有异常并返回错误信息，不向上传播
        """
        # 步骤1: 检查初始化
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        # 延迟导入以避免循环依赖
        from ..agent import Agent

        # 步骤2: 创建子代理
        parent = self._parent_agent
        sub = Agent(
            llm=parent.llm,                                             # 共享 LLM 接口
            tools=[t for t in parent.tools if t.name != "agent"],       # 排除自身，防止递归
            max_context_tokens=parent.context.max_tokens,               # 继承上下文窗口大小
            max_rounds=20,                                              # 限制最大轮数
        )

        try:
            # 步骤3: 执行任务
            result = sub.chat(task)
            # 步骤4: 截断过长的结果
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return f"[Sub-agent completed]\n{result}"
        except Exception as e:
            return f"Sub-agent error: {e}"
