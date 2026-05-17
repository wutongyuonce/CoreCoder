"""
文件编辑工具 - 精确的查找替换编辑（Claude Code 的核心创新）。

核心思想：LLM 不需要重写整个文件，也不需要指定行号，
而是指定一个*精确的子串*来查找，然后替换它。

关键特性：
  - old_string 必须在文件中恰好出现一次（消除歧义）
  - 自动生成 unified diff 显示变更
  - 跟踪修改过的文件列表（用于 /diff 命令）
  - 提供足够的上下文以确保唯一性（由 LLM 负责）
"""

import difflib
from pathlib import Path

from .base import Tool

# 跟踪本次会话中修改过的文件（用于 /diff 命令）
_changed_files: set[str] = set()


class EditFileTool(Tool):
    """
    文件编辑工具 - 通过精确的查找替换来修改文件。

    属性:
        name (str): "edit_file"
        description (str): 工具描述
        parameters (dict): 参数 schema（file_path, old_string, new_string 均必填）
    """
    name = "edit_file"
    description = (
        "Edit a file by replacing an exact string match. "
        "old_string must appear exactly once in the file for safety. "
        "Include enough surrounding context to ensure uniqueness."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find (must be unique in file)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def execute(self, file_path: str, old_string: str, new_string: str) -> str:
        """
        执行文件编辑：查找并替换指定的文本。

        输入:
            file_path (str): 要编辑的文件路径
            old_string (str): 要查找的精确文本（必须在文件中恰好出现一次）
            new_string (str): 替换文本

        输出: str - 编辑结果（包含 unified diff）或错误信息

        关键步骤:
            1. 解析并验证文件路径
            2. 读取文件内容
            3. 检查 old_string 的出现次数：
               - 0 次：返回错误 + 文件开头预览（帮助 LLM 调整）
               - 多次：返回错误 + 出现次数（要求 LLM 增加上下文）
               - 恰好 1 次：继续执行
            4. 执行替换（只替换第一次出现）
            5. 写入新内容到文件
            6. 记录文件到 _changed_files 集合
            7. 生成 unified diff 显示变更
            8. 返回结果
        """
        try:
            # 步骤1: 解析路径
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"

            # 步骤2: 读取内容
            content = p.read_text()
            occurrences = content.count(old_string)

            # 步骤3: 检查出现次数
            if occurrences == 0:
                # 未找到：返回文件开头预览帮助 LLM 调整
                preview = content[:500] + ("..." if len(content) > 500 else "")
                return (
                    f"Error: old_string not found in {file_path}.\n"
                    f"File starts with:\n{preview}"
                )
            if occurrences > 1:
                # 多次出现：要求增加上下文
                return (
                    f"Error: old_string appears {occurrences} times in {file_path}. "
                    f"Include more surrounding lines to make it unique."
                )

            # 步骤4-5: 执行替换并写入
            new_content = content.replace(old_string, new_string, 1)
            p.write_text(new_content)

            # 步骤6: 记录修改过的文件
            _changed_files.add(str(p))

            # 步骤7-8: 生成 diff 并返回
            diff = _unified_diff(content, new_content, str(p))
            return f"Edited {file_path}\n{diff}"
        except Exception as e:
            return f"Error: {e}"


def _unified_diff(old: str, new: str, filename: str, context: int = 3) -> str:
    """
    生成旧内容和新内容之间的 compact unified diff。

    输入:
        old (str): 旧文件内容
        new (str): 新文件内容
        filename (str): 文件名（用于 diff 头部）
        context (int): 上下文行数，默认 3

    输出: str - unified diff 格式的变更文本

    关键步骤:
        1. 将内容按行分割
        2. 使用 difflib.unified_diff 生成 diff
        3. 如果 diff 超过 3000 字符，截断到 2500 字符
    """
    # 步骤1: 按行分割
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    # 步骤2: 生成 diff
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        n=context,
    )
    result = "".join(diff)
    # 步骤3: 截断过大的 diff
    if len(result) > 3000:
        result = result[:2500] + "\n... (diff truncated)\n"
    return result
