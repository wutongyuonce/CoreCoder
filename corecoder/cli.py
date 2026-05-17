"""
交互式终端界面（REPL）- CoreCoder 的用户接口层。

本模块是用户直接交互的界面，提供：
  1. 命令行参数解析（模型选择、API 配置等）
  2. 交互式 REPL（Read-Eval-Print Loop）循环
  3. 单次执行模式（-p 参数传入 prompt 后直接运行并退出）
  4. 内置命令系统（/help, /reset, /model, /save 等）

依赖的第三方库：
  - rich: 终端富文本渲染（Markdown、Panel 等）
  - prompt_toolkit: 高级终端输入（历史记录、快捷键绑定、多行输入）
"""

import sys
import os
import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from .agent import Agent
from .llm import LLM, LiteLLM
from .config import Config
from .session import save_session, load_session, list_sessions
from . import __version__

# 全局 Rich 控制台实例，用于带格式的终端输出
console = Console()


def _parse_args():
    """
    解析命令行参数。

    输入: 无（读取 sys.argv）
    输出: argparse.Namespace - 解析后的参数对象，包含以下属性：
        - model (str | None): 模型名称
        - base_url (str | None): API 基础 URL
        - api_key (str | None): API 密钥
        - prompt (str | None): 单次执行的 prompt（非交互模式）
        - resume (str | None): 要恢复的已保存会话 ID
        - version: 版本信息标志

    关键步骤:
        1. 创建 argparse.ArgumentParser
        2. 定义所有支持的命令行参数
        3. 解析并返回参数对象
    """
    p = argparse.ArgumentParser(
        prog="corecoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $CORECODER_MODEL or gpt-4o)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


def main():
    """
    主入口函数 - CoreCoder 的启动流程。

    输入: 无（通过 _parse_args() 获取命令行参数）
    输出: 无

    关键步骤:
        1. 解析命令行参数
        2. 从环境变量加载配置（Config.from_env()）
        3. 命令行参数覆盖环境变量配置
        4. 验证 API Key 是否存在，不存在则显示帮助信息并退出
        5. 根据 provider 类型选择 LLM 实现（标准 OpenAI 或 LiteLLM）
        6. 创建 Agent 实例
        7. 如果指定了 -r 参数，加载已保存的会话并恢复对话历史
        8. 如果指定了 -p 参数，进入单次执行模式
        9. 否则进入交互式 REPL 模式
    """
    # 步骤1: 解析命令行参数
    args = _parse_args()
    # 步骤2: 从环境变量加载默认配置
    config = Config.from_env()

    # 步骤3: CLI 参数覆盖环境变量
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    # 步骤4: API Key 验证
    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or CORECODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    # 步骤5: 选择 LLM 实现
    # litellm 支持 100+ 提供者（AWS Bedrock、Google Vertex 等）
    # 标准 LLM 适用于所有 OpenAI 兼容 API
    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    # 步骤6: 创建 Agent 实例
    agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)

    # 步骤7: 恢复已保存的会话（如果指定了 -r 参数）
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # 恢复会话时使用保存的模型，除非 CLI 明确指定了新模型
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # 步骤8: 单次执行模式
    if args.prompt:
        _run_once(agent, args.prompt)
        return

    # 步骤9: 交互式 REPL 模式
    _repl(agent, config)


def _run_once(agent: Agent, prompt: str):
    """
    非交互式单次执行模式 - 运行一条 prompt 后退出。

    输入:
        agent (Agent): 已初始化的代理实例
        prompt (str): 用户的单条指令

    输出: 无（直接打印到标准输出）

    关键步骤:
        1. 定义 on_token 回调：实时打印每个流式 token
        2. 定义 on_tool 回调：打印工具调用信息（dim 样式）
        3. 调用 agent.chat() 执行对话
    """
    def on_token(tok):
        """流式 token 回调 - 直接打印到终端"""
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        """工具调用回调 - 显示工具名称和简要参数"""
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    # 执行对话并输出结果
    agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    print()


def _repl(agent: Agent, config: Config):
    """
    交互式读取-求值-打印循环（REPL）- CoreCoder 的主交互界面。

    输入:
        agent (Agent): 已初始化的代理实例
        config (Config): 当前配置对象

    输出: 无（持续运行直到用户退出）

    关键步骤:
        1. 显示欢迎面板（版本号、模型名、基础使用提示）
        2. 初始化输入历史（存储在 ~/.corecoder_history）
        3. 配置快捷键：
           - Enter: 提交输入
           - Escape+Enter: 插入换行（用于粘贴多行代码）
        4. 进入主循环：
           a. 读取用户输入（支持多行、历史记录）
           b. 处理内置命令（/help, /reset, /model, /tokens 等）
           c. 非内置命令则调用 agent.chat() 处理
           d. 流式打印 LLM 响应或渲染 Markdown
        5. 处理 Ctrl+C 和 EOF（优雅退出）

    内置命令列表：
        /help      - 显示帮助信息
        /reset     - 清空对话历史
        /model     - 显示/切换当前模型
        /tokens    - 显示 token 使用量和估算费用
        /compact   - 手动压缩对话上下文
        /diff      - 显示本次会话修改过的文件
        /save      - 保存当前会话到磁盘
        /sessions  - 列出已保存的会话
        quit/exit  - 退出程序
    """
    # 步骤1: 显示欢迎面板
    console.print(Panel(
        f"[bold]CoreCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    # 步骤2: 初始化输入历史（持久化到磁盘）
    hist_path = os.path.expanduser("~/.corecoder_history")
    history = FileHistory(hist_path)

    # 步骤3: 配置快捷键
    # Enter 提交输入，Escape+Enter 插入换行（方便粘贴多行代码）
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        """Enter 键：提交当前输入"""
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        """Escape+Enter 键：插入换行符（用于多行输入）"""
        event.current_buffer.insert_text("\n")

    # 步骤4: 主循环
    while True:
        # 4a: 读取用户输入
        try:
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            # 步骤5: 优雅退出
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # 4b: 内置命令处理

        # 退出命令
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break

        # /help - 显示帮助
        if user_input == "/help":
            _show_help()
            continue

        # /reset - 重置对话
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue

        # /tokens - 显示 token 使用统计
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue

        # /model [name] - 查看或切换模型
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                # 切换到新模型
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                # 显示当前模型
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue

        # /compact - 手动触发上下文压缩
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue

        # /save - 保存当前会话
        if user_input == "/save":
            sid = save_session(agent.messages, config.model)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: corecoder -r {sid}")
            continue

        # /diff - 显示本次会话修改过的文件
        if user_input == "/diff":
            from .tools.edit import _changed_files
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue

        # /sessions - 列出已保存的会话
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue

        # 4c: 非内置命令 -> 调用 Agent 处理
        streamed: list[str] = []

        def on_token(tok):
            """流式 token 回调 - 实时打印并收集"""
            streamed.append(tok)
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            """工具调用回调 - 显示正在执行的工具"""
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        # 4d: 执行对话并显示结果
        try:
            response = agent.chat(user_input, on_token=on_token, on_tool=on_tool)
            if streamed:
                # 响应已通过流式打印，只需换行
                print()  # newline after streamed tokens
            else:
                # 响应未被流式打印（例如工具调用后的文本回复），用 Markdown 渲染
                console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help():
    """
    显示帮助信息面板。

    输入: 无
    输出: 无（直接打印到终端）

    关键步骤:
        使用 Rich Panel 渲染格式化的帮助文本，包含所有内置命令和输入方式说明
    """
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  quit           Exit CoreCoder\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="CoreCoder Help",
        border_style="dim",
    ))


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    """
    将工具参数字典格式化为简短的显示字符串。

    输入:
        kwargs (dict): 工具的参数字典
        maxlen (int): 最大显示长度，默认 80 字符

    输出: str - 格式化后的简短参数描述

    关键步骤:
        1. 遍历参数字典，格式化为 key=value 的形式
        2. 每个值最多显示 40 个字符
        3. 如果总长度超过 maxlen，截断并添加 "..."
    """
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
