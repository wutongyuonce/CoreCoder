"""
Shell 命令执行工具 - 带安全检查的终端命令执行器。

Claude Code 的 BashTool 有 1,143 行代码，这是其精简版本，保留了核心功能：
  - 输出捕获与截断（保留头部和尾部）
  - 超时支持（默认 120 秒）
  - 危险命令检测（防止误操作）
  - 工作目录跟踪（cd 命令感知）
"""

import os
import re
import subprocess
from .base import Tool

# 跨命令跟踪当前工作目录（Claude Code 也采用这种方式）
_cwd: str | None = None

# 危险命令模式列表 - 可能破坏文件系统或泄露密钥的命令
_DANGEROUS_PATTERNS = [
    (r"\brm\s+(-\w*)?-r\w*\s+(/|~|\$HOME)", "recursive delete on home/root"),       # 递归删除主目录/根目录
    (r"\brm\s+(-\w*)?-rf\s", "force recursive delete"),                              # 强制递归删除
    (r"\bmkfs\b", "format filesystem"),                                              # 格式化文件系统
    (r"\bdd\s+.*of=/dev/", "raw disk write"),                                        # 原始磁盘写入
    (r">\s*/dev/sd[a-z]", "overwrite block device"),                                 # 覆写块设备
    (r"\bchmod\s+(-R\s+)?777\s+/", "chmod 777 on root"),                             # 根目录 777 权限
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),                                          # Fork 炸弹
    (r"\bcurl\b.*\|\s*(sudo\s+)?bash", "pipe curl to bash"),                         # curl 管道到 bash
    (r"\bwget\b.*\|\s*(sudo\s+)?bash", "pipe wget to bash"),                         # wget 管道到 bash
]


class BashTool(Tool):
    """
    Shell 命令执行工具。

    属性:
        name (str): "bash"
        description (str): 工具描述
        parameters (dict): 参数 schema（command 必填，timeout 可选）
    """
    name = "bash"
    description = (
        "Execute a shell command. Returns stdout, stderr, and exit code. "
        "Use this for running tests, installing packages, git operations, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str, timeout: int = 120) -> str:
        """
        执行 Shell 命令并返回输出。

        输入:
            command (str): 要执行的 Shell 命令
            timeout (int): 超时时间（秒），默认 120

        输出: str - 命令输出（stdout + stderr + exit code），或错误信息

        关键步骤:
            1. 安全检查：检测危险命令模式，拦截可能的破坏性操作
            2. 获取当前工作目录（使用跨命令跟踪的 _cwd）
            3. 执行命令（subprocess.run, shell=True, capture_output=True）
            4. 如果命令成功执行且包含 cd，更新工作目录跟踪
            5. 组合输出（stdout + stderr + exit code）
            6. 截断过长输出（保留前 6000 + 后 3000 字符）
            7. 处理超时和其他异常
        """
        global _cwd
        # 步骤1: 安全检查
        warning = _check_dangerous(command)
        if warning:
            return f"⚠ Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."

        # 步骤2: 获取工作目录
        cwd = _cwd or os.getcwd()

        try:
            # 步骤3: 执行命令
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )

            # 步骤4: 跟踪 cd 命令
            if proc.returncode == 0:
                _update_cwd(command, cwd)

            # 步骤5: 组合输出
            out = proc.stdout
            if proc.stderr:
                out += f"\n[stderr]\n{proc.stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"

            # 步骤6: 截断过长输出
            if len(out) > 15_000:
                out = (
                    out[:6000]
                    + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
                    + out[-3000:]
                )
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            # 步骤7: 超时处理
            return f"Error: timed out after {timeout}s"
        except Exception as e:
            return f"Error running command: {e}"


def _check_dangerous(cmd: str) -> str | None:
    """
    检测命令是否包含危险模式。

    输入:
        cmd (str): 待检测的命令字符串

    输出: str | None - 如果检测到危险模式，返回警告描述；否则返回 None

    关键步骤:
        遍历 _DANGEROUS_PATTERNS 列表，使用正则表达式匹配命令
    """
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return reason
    return None


def _update_cwd(command: str, current_cwd: str):
    """
    跟踪 cd 命令以更新工作目录。

    输入:
        command (str): 刚执行的命令字符串
        current_cwd (str): 当前工作目录

    输出: 无（副作用：更新全局 _cwd 变量）

    关键步骤:
        1. 按 && 分割命令链
        2. 查找以 "cd " 开头的部分
        3. 提取目标目录路径
        4. 规范化路径（处理 ~ 和相对路径）
        5. 验证目录是否存在，存在则更新 _cwd
    """
    global _cwd
    # 步骤1: 按 && 分割命令链
    parts = command.split("&&")
    for part in parts:
        part = part.strip()
        # 步骤2: 查找 cd 命令
        if part.startswith("cd "):
            # 步骤3: 提取目标目录
            target = part[3:].strip().strip("'\"")
            if target:
                # 步骤4: 规范化路径
                new_dir = os.path.normpath(os.path.join(current_cwd, os.path.expanduser(target)))
                # 步骤5: 验证并更新
                if os.path.isdir(new_dir):
                    _cwd = new_dir
