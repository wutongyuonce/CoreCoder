"""
会话持久化模块 - 保存和恢复对话会话。

Claude Code 通过 QueryEngine（1295 行代码）维护会话状态。
CoreCoder 将其简化为：JSON 格式的消息转储 + 模型配置。

存储位置：~/.corecoder/sessions/*.json
每个会话文件包含：id, model, saved_at, messages
"""

import json
import re
import time
from pathlib import Path

# 会话文件存储目录
SESSIONS_DIR = Path.home() / ".corecoder" / "sessions"
# 会话 ID 安全字符正则（仅允许字母、数字、点、下划线、连字符）
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _normalize_session_id(session_id: str | None) -> str:
    """
    规范化会话 ID，确保安全性。

    输入:
        session_id (str | None): 原始会话 ID

    输出: str - 规范化后的安全会话 ID

    关键步骤:
        1. 如果未提供 ID，生成基于时间戳的默认 ID
        2. 提取路径中的文件名部分（防止路径穿越攻击）
        3. 将不安全字符替换为连字符
        4. 去除首尾的连字符、点、下划线
        5. 如果结果为空，使用时间戳作为默认值
    """
    if not session_id:
        return f"session_{int(time.time())}"

    # 步骤2: 提取文件名，防止路径穿越
    name = session_id.strip().replace("\\", "/").split("/")[-1]
    # 步骤3-4: 替换不安全字符
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    return name or f"session_{int(time.time())}"


def _session_path(session_id: str) -> Path:
    """
    获取会话文件的完整路径（带安全校验）。

    输入:
        session_id (str): 会话 ID

    输出: Path - 会话 JSON 文件的绝对路径

    关键步骤:
        1. 规范化会话 ID
        2. 拼接为 SESSIONS_DIR/id.json 格式
        3. resolve() 解析符号链接和相对路径
        4. 安全校验：确保路径仍在 SESSIONS_DIR 内（防止路径穿越）

    异常:
        ValueError: 如果路径不在 SESSIONS_DIR 内
    """
    path = (SESSIONS_DIR / f"{_normalize_session_id(session_id)}.json").resolve()
    root = SESSIONS_DIR.resolve()
    # 安全校验：防止路径穿越攻击
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def save_session(messages: list[dict], model: str, session_id: str | None = None) -> str:
    """
    保存对话会话到磁盘。

    输入:
        messages (list[dict]): 对话消息列表
        model (str): 当前使用的模型名称
        session_id (str | None): 自定义会话 ID（可选）

    输出: str - 实际使用的会话 ID

    关键步骤:
        1. 确保会话目录存在（mkdir -p）
        2. 规范化会话 ID
        3. 构造会话数据字典（id, model, saved_at, messages）
        4. 序列化为 JSON 并写入文件
        5. 返回会话 ID
    """
    # 步骤1: 创建目录
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    # 步骤2: 规范化 ID
    session_id = _normalize_session_id(session_id)
    # 步骤3: 构造数据
    data = {
        "id": session_id,
        "model": model,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }
    # 步骤4: 写入文件
    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return session_id


def load_session(session_id: str) -> tuple[list[dict], str] | None:
    """
    加载已保存的会话。

    输入:
        session_id (str): 会话 ID

    输出: tuple[list[dict], str] | None - (消息列表, 模型名称) 元组，
                                         如果会话不存在返回 None

    关键步骤:
        1. 获取会话文件路径
        2. 检查文件是否存在
        3. 解析 JSON 文件
        4. 返回 messages 和 model 字段
    """
    path = _session_path(session_id)
    # 步骤2: 检查文件是否存在
    if not path.exists():
        return None
    # 步骤3-4: 解析并返回
    data = json.loads(path.read_text())
    return data["messages"], data["model"]


def list_sessions() -> list[dict]:
    """
    列出所有已保存的会话（按时间倒序）。

    输入: 无
    输出: list[dict] - 会话摘要列表，每个包含：
        - id (str): 会话 ID
        - model (str): 使用的模型
        - saved_at (str): 保存时间
        - preview (str): 第一条用户消息的预览（最多 80 字符）

    关键步骤:
        1. 检查会话目录是否存在
        2. 遍历目录下的所有 .json 文件
        3. 按文件名（时间戳）倒序排列
        4. 解析每个文件，提取摘要信息
        5. 提取第一条用户消息作为预览
        6. 最多返回 20 条会话
    """
    # 步骤1: 检查目录
    if not SESSIONS_DIR.exists():
        return []

    # 步骤2-5: 遍历并解析
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            # 步骤5: 提取第一条用户消息作为预览
            preview = ""
            for m in data.get("messages", []):
                if m.get("role") == "user" and m.get("content"):
                    preview = m["content"][:80]
                    break
            sessions.append({
                "id": data.get("id", f.stem),
                "model": data.get("model", "?"),
                "saved_at": data.get("saved_at", "?"),
                "preview": preview,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    # 步骤6: 限制返回数量
    return sessions[:20]
