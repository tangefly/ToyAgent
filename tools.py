"""工具定义与内置工具集（文件读写 / 目录列举 / 模式搜索）。

Tool: 一个可被 LLM 调用的工具 = 可调用对象 + OpenAI function schema。

内置文件工具让主 agent 和子 agent 都能读文件、写文件、列目录、按模式搜索，
使 ToyAgent 具备一个「基本智能体」该有的文件操作能力（不再是 main 独占）。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional


class Tool:
    """一个可被 LLM 调用的工具：可调用对象 + OpenAI function schema。

    aliases:   额外的调用名（如 sub agent 名）。模型常用别名直接调用
               （writer({...}) 而非 call_sub_agent({name: "writer", ...})），
               Agent._run_tool 会把别名解析到本工具，并把别名补进缺省的参数。
    root_only: 只允许 root(main) agent 调用（如 call_sub_agent）。该标记不进
               OpenAI schema；非 root agent 构造时自动剥离，_run_tool 另有运行时兜底。
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
        aliases: Optional[List[str]] = None,
        root_only: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.aliases = list(aliases or [])
        self.root_only = root_only

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: Dict[str, Any]) -> Any:
        return self.func(**arguments)


# ---------- 内置文件工具的实现 ----------


def _tool_read_file(path: str, line_start: int = 1, line_end: Optional[int] = None) -> str:
    """读取文本文件（带行号，可指定行范围）；失败时返回 ERROR 文本让模型自行调整。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"ERROR: 文件不存在: {path}"
    except IsADirectoryError:
        return f"ERROR: {path} 是目录，请改用 list_directory"
    except OSError as exc:
        return f"ERROR: 读取 {path} 失败: {exc!r}"
    if not lines:
        return "(空文件)"
    start = max(1, line_start)
    end = len(lines) if line_end is None else min(len(lines), line_end)
    if start > end:
        return f"ERROR: line_start({start}) > line_end({end})，文件共 {len(lines)} 行"
    body = "".join(f"{i:>5} | {line}" for i, line in enumerate(lines[start - 1:end], start=start))
    if end < len(lines):
        body += f"\n...(共 {len(lines)} 行，已按 line_end 截断)"
    return body


def _tool_write_file(path: str, content: str) -> str:
    """写入文件（自动创建父目录、覆盖已存在文件），返回写入结果。"""
    try:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        return f"ERROR: 写入 {path} 失败: {exc!r}"
    return f"已写入 {path}（{len(content)} 字符）"


def _tool_list_directory(path: str = ".") -> str:
    """列出目录条目（文件/子目录），失败返回 ERROR。"""
    try:
        entries = sorted(os.listdir(path))
    except FileNotFoundError:
        return f"ERROR: 目录不存在: {path}"
    except NotADirectoryError:
        return f"ERROR: {path} 不是目录"
    except OSError as exc:
        return f"ERROR: 列举 {path} 失败: {exc!r}"
    lines = []
    for name in entries:
        full = os.path.join(path, name)
        kind = "dir" if os.path.isdir(full) else "file"
        lines.append(f"{kind:<5}{name}")
    return "\n".join(lines) if lines else "(空目录)"


def _tool_search_files(pattern: str, path: str = ".") -> str:
    """按 glob 通配符模式递归搜索文件（只返回文件路径）。"""
    try:
        matches = sorted(
            p for p in glob.glob(os.path.join(path, "**", pattern), recursive=True)
            if os.path.isfile(p)
        )
    except OSError as exc:
        return f"ERROR: 搜索失败: {exc!r}"
    if not matches:
        return f"（没有匹配 {pattern!r} 的文件）"
    return "\n".join(matches)


def build_file_tools() -> List[Tool]:
    """内置文件工具集：主/子 agent 通用（read_file / write_file / list_directory / search_files）。"""
    return [
        Tool(
            name="read_file",
            description=(
                "读取文本文件内容（带行号，可指定行范围），用于查看代码、文档、配置等本地文件。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要读取的文件路径"},
                    "line_start": {"type": "integer", "description": "起始行号（从 1 开始，默认 1）"},
                    "line_end": {"type": "integer", "description": "结束行号（默认读到底）"},
                },
                "required": ["path"],
            },
            func=_tool_read_file,
        ),
        Tool(
            name="write_file",
            description="把内容写入文件（自动创建父目录、覆盖已存在文件），用于产出代码、文档等。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要写入的文件路径"},
                    "content": {"type": "string", "description": "完整的文件内容"},
                },
                "required": ["path", "content"],
            },
            func=_tool_write_file,
        ),
        Tool(
            name="list_directory",
            description="列出目录下的条目（文件/子目录），用于了解项目结构。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认当前目录"},
                },
                "required": [],
            },
            func=_tool_list_directory,
        ),
        Tool(
            name="search_files",
            description="按 glob 通配符模式递归搜索文件路径（如 *.py、**/test_*.py），用于定位文件。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob 模式，如 *.py"},
                    "path": {"type": "string", "description": "搜索起点目录，默认当前目录"},
                },
                "required": ["pattern"],
            },
            func=_tool_search_files,
        ),
    ]
