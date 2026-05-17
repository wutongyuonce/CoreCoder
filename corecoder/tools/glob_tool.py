"""
文件模式匹配工具 - 使用 glob 模式查找文件。

支持标准 glob 语法，包括 ** 递归匹配。
结果按修改时间倒序排列（最新修改的文件优先显示）。
"""

from pathlib import Path
from .base import Tool


class GlobTool(Tool):
    """
    文件模式匹配工具。

    属性:
        name (str): "glob"
        description (str): 工具描述
        parameters (dict): 参数 schema（pattern 必填，path 可选）
    """
    name = "glob"
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".") -> str:
        """
        使用 glob 模式查找匹配的文件。

        输入:
            pattern (str): glob 模式字符串（如 "**/*.py", "src/**/*.ts"）
            path (str): 搜索的根目录，默认为当前目录 "."

        输出: str - 匹配的文件路径列表（每行一个），或错误信息

        关键步骤:
            1. 解析并验证搜索目录
            2. 使用 Path.glob() 执行模式匹配
            3. 按修改时间倒序排列（最新的文件优先）
            4. 限制最多显示 100 个结果
            5. 如果结果超过 100，显示截断提示
        """
        try:
            # 步骤1: 解析搜索目录
            base = Path(path).expanduser().resolve()
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            # 步骤2: 执行模式匹配
            hits = list(base.glob(pattern))
            # 步骤3: 按修改时间倒序排列
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            # 步骤4-5: 限制显示数量
            total = len(hits)
            shown = hits[:100]
            lines = [str(h) for h in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return result or "No files matched."
        except Exception as e:
            return f"Error: {e}"
