"""
文件写入工具 - 创建新文件或完全覆盖现有文件。

注意：对于现有文件的小改动，应优先使用 edit_file（查找替换），
而不是 write_file（完全重写）。write_file 主要用于：
  - 创建新文件
  - 完全重写现有文件
"""

from pathlib import Path
from .base import Tool
from .edit import _changed_files


class WriteFileTool(Tool):
    """
    文件写入工具。

    属性:
        name (str): "write_file"
        description (str): 工具描述
        parameters (dict): 参数 schema（file_path 和 content 均必填）
    """
    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing one. "
        "For small edits to existing files, prefer edit_file instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the file",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    def execute(self, file_path: str, content: str) -> str:
        """
        写入文件内容。

        输入:
            file_path (str): 目标文件路径
            content (str): 要写入的完整文件内容

        输出: str - 写入结果（包含行数）或错误信息

        关键步骤:
            1. 解析文件路径
            2. 创建父目录（如果不存在，mkdir -p）
            3. 写入内容到文件
            4. 记录文件到 _changed_files 集合（用于 /diff 命令）
            5. 计算写入的行数
            6. 返回写入结果
        """
        try:
            # 步骤1-2: 解析路径并创建目录
            p = Path(file_path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            # 步骤3: 写入内容
            p.write_text(content)
            # 步骤4: 记录修改
            _changed_files.add(str(p))
            # 步骤5-6: 计算行数并返回
            n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Wrote {n_lines} lines to {file_path}"
        except Exception as e:
            return f"Error: {e}"
