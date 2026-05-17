"""
文件内容搜索工具 - 支持正则表达式的全文搜索。

功能类似 grep 命令，但更智能：
  - 自动跳过常见的无关目录（.git, node_modules 等）
  - 支持 include 参数过滤文件类型
  - 最多返回 200 条匹配（防止输出过大）
  - 最多扫描 5000 个文件（防止搜索过慢）
"""

import re
from pathlib import Path
from .base import Tool

# 跳过的目录列表 - 避免搜索无关文件产生噪音
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}


class GrepTool(Tool):
    """
    文件内容搜索工具 - 使用正则表达式搜索文件内容。

    属性:
        name (str): "grep"
        description (str): 工具描述
        parameters (dict): 参数 schema（pattern 必填，path 和 include 可选）
    """
    name = "grep"
    description = (
        "Search file contents with regex. "
        "Returns matching lines with file path and line number."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: cwd)",
            },
            "include": {
                "type": "string",
                "description": "Only search files matching this glob (e.g. '*.py')",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", include: str | None = None) -> str:
        """
        使用正则表达式搜索文件内容。

        输入:
            pattern (str): 正则表达式模式
            path (str): 搜索路径（文件或目录），默认当前目录 "."
            include (str | None): 文件过滤 glob 模式（如 "*.py"），None 搜索所有文件

        输出: str - 匹配结果列表（格式：文件路径:行号: 内容），或错误信息

        关键步骤:
            1. 编译正则表达式（失败则返回错误）
            2. 解析搜索路径
            3. 确定要搜索的文件列表：
               - 如果 path 是文件，只搜索该文件
               - 如果 path 是目录，使用 _walk() 递归遍历
            4. 逐文件逐行搜索匹配
            5. 累积匹配结果（最多 200 条）
            6. 返回结果或 "No matches found."
        """
        # 步骤1: 编译正则
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"

        # 步骤2: 解析路径
        base = Path(path).expanduser().resolve()
        if not base.exists():
            return f"Error: {path} not found"

        # 步骤3: 确定文件列表
        if base.is_file():
            files = [base]
        else:
            files = self._walk(base, include)

        # 步骤4-5: 逐文件搜索
        matches = []
        for fp in files:
            try:
                text = fp.read_text(errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{fp}:{lineno}: {line.rstrip()}")
                    # 步骤5: 达到上限则停止
                    if len(matches) >= 200:
                        matches.append("... (200 match limit reached)")
                        return "\n".join(matches)

        return "\n".join(matches) if matches else "No matches found."

    @staticmethod
    def _walk(root: Path, include: str | None) -> list[Path]:
        """
        递归遍历目录树，跳过无关目录。

        输入:
            root (Path): 根目录
            include (str | None): 文件过滤 glob 模式

        输出: list[Path] - 匹配的文件路径列表

        关键步骤:
            1. 使用 rglob() 递归搜索
            2. 跳过 _SKIP_DIRS 中列出的目录
            3. 只保留文件（排除目录）
            4. 最多返回 5000 个文件（防止搜索过慢）
        """
        results = []
        for item in root.rglob(include or "*"):
            # 步骤2: 跳过无关目录
            if any(part in _SKIP_DIRS for part in item.parts):
                continue
            # 步骤3: 只保留文件
            if item.is_file():
                results.append(item)
            # 步骤4: 达到上限则停止
            if len(results) >= 5000:
                break
        return results
