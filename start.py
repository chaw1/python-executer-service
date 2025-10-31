#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python Executor Service 启动脚本
"""
import os
import sys
import subprocess

def main():
    print("=" * 50)
    print("Python Executor Service 启动器")
    print("=" * 50)
    print()

    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 检查依赖
    print("[信息] 检查依赖包...")
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import matplotlib
        print("[成功] 所有依赖已安装")
    except ImportError as e:
        print(f"[警告] 缺少依赖: {e}")
        print("[信息] 正在安装依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # 启动服务
    print()
    print("[信息] 启动服务...")
    print("[信息] 服务地址: http://localhost:8001")
    print("[信息] API文档: http://localhost:8001/docs")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    print()

    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8001"
        ])
    except KeyboardInterrupt:
        print("\n[信息] 服务已停止")

if __name__ == "__main__":
    main()
