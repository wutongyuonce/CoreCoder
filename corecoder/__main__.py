"""
__main__.py - 包的直接执行入口。

当用户通过 `python -m corecoder` 或 `corecoder` 命令行工具运行时，
此文件被 Python 解释器自动加载。

功能：
    直接调用 cli 模块的 main() 函数，启动交互式 REPL 或单次执行模式。

执行流程：
    1. 导入 cli 模块的 main 函数
    2. 调用 main() 解析命令行参数并启动程序
"""

from corecoder.cli import main

# 启动 CoreCoder 的主入口
main()
