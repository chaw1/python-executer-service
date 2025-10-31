@echo off
chcp 65001 >nul
echo ========================================
echo Python Executor Service 启动器
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    pause
    exit /b 1
)

echo [信息] 检查依赖包...
python -c "import fastapi, uvicorn, pandas, numpy, matplotlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 缺少依赖包，正在安装...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

echo [信息] 启动服务...
echo [信息] 服务地址: http://localhost:8001
echo [信息] API文档: http://localhost:8001/docs
echo.
echo 按 Ctrl+C 停止服务
echo ========================================
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

pause
