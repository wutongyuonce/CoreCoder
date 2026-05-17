"""
文件读取工具 - 带行号的文件内容读取。

读取文件时自动添加行号，便于 LLM 引用具体行。
支持 offset（起始行）和 limit（最大行数）参数。
"""

from pathlib import Path
from .base import Tool


class ReadFileTool(Tool):
    """
    文件读取工具。

    属性:
        name (str): "read_file"
        description (str): 工具描述
        parameters (dict): 参数 schema（file_path 必填，offset 和 limit 可选）
    """
    name = "read_file"
    description = (
        "Read a file's contents with line numbers. "
        "Always read a file before editing it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file",
            },
            "offset": {
                "type": "integer",
                "description": "Start line (1-based). Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read. Default 2000.",
            },
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        """
        读取文件内容并添加行号。

        输入:
            file_path (str): 文件路径
            offset (int): 起始行号（1-based），默认 1
            limit (int): 最大读取行数，默认 2000

        输出: str - 带行号的文件内容，或错误信息

        关键步骤:
            1. 解析并验证文件路径（存在性、是否为文件）
            2. 读取文件内容（使用 errors="replace" 处理编码问题）
            3. 按行分割
            4. 计算截取范围（基于 offset 和 limit）
            5. 为每行添加行号（格式：行号\t内容）
            6. 如果文件还有更多行，显示截断提示
        """
        try:
            # 步骤1: 验证路径
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"
            if not p.is_file():
                return f"Error: {file_path} is a directory, not a file"

            # 步骤2-3: 读取并分割
            text = p.read_text(errors="replace")
            lines = text.splitlines()
            total = len(lines)

            # 步骤4: 截取指定范围
            start = max(0, offset - 1)
            chunk = lines[start : start + limit]

            # 步骤5: 添加行号
            numbered = [f"{start + i + 1}\t{ln}" for i, ln in enumerate(chunk)]
            result = "\n".join(numbered)

            # 步骤6: 截断提示
            if total > start + limit:
                result += f"\n... ({total} lines total, showing {start+1}-{start+len(chunk)})"
            return result or "(empty file)"
        except Exception as e:
            return f"Error: {e}"
