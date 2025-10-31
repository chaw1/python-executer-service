"""
FastAPI 主应用
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from typing import Dict

from app.models import (
    ExecuteRequest,
    ExecuteResponse,
    HealthResponse
)
from app.executor import CodeExecutor
from app.sandbox import CODE_TEMPLATES

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Python Executor Service",
    description="安全的 Python 代码执行微服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    return response


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "内部服务器错误",
            "detail": str(exc)
        }
    )


@app.get("/", tags=["健康检查"])
async def root():
    """根路径"""
    return {
        "service": "Python Executor Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        available_libraries=[
            "numpy",
            "pandas",
            "matplotlib",
            "plotly"
        ]
    )


@app.post("/execute", response_model=ExecuteResponse, tags=["代码执行"])
async def execute_code(request: ExecuteRequest):
    """
    执行 Python 代码

    Args:
        request: 执行请求

    Returns:
        执行结果

    Raises:
        HTTPException: 执行失败时抛出
    """
    logger.info(f"收到执行请求，代码长度: {len(request.code)} 字符")

    try:
        # 创建执行器
        executor = CodeExecutor(timeout=request.timeout)

        # 执行代码
        result = executor.execute(request.code)

        logger.info(
            f"执行完成 - 状态: {result.status}, "
            f"耗时: {result.execution_time}ms"
        )

        return result

    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"执行失败: {str(e)}"
        )


@app.get("/templates", tags=["代码模板"])
async def get_templates() -> Dict[str, str]:
    """
    获取预定义代码模板

    Returns:
        模板字典
    """
    return CODE_TEMPLATES


@app.get("/templates/{template_name}", tags=["代码模板"])
async def get_template(template_name: str) -> Dict[str, str]:
    """
    获取指定代码模板

    Args:
        template_name: 模板名称

    Returns:
        模板内容

    Raises:
        HTTPException: 模板不存在时抛出
    """
    if template_name not in CODE_TEMPLATES:
        raise HTTPException(
            status_code=404,
            detail=f"模板 '{template_name}' 不存在"
        )

    return {
        "name": template_name,
        "code": CODE_TEMPLATES[template_name]
    }


@app.post("/validate", tags=["代码验证"])
async def validate_code(request: ExecuteRequest):
    """
    验证代码是否安全（不执行）

    Args:
        request: 执行请求

    Returns:
        验证结果
    """
    from app.sandbox import SafeExecutionEnvironment

    env = SafeExecutionEnvironment()
    is_valid, error_msg = env.validate_code(request.code)

    return {
        "valid": is_valid,
        "error": error_msg if not is_valid else None
    }


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    logger.info("Python Executor Service 启动成功")
    logger.info("文档地址: http://localhost:8000/docs")


# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info("Python Executor Service 关闭")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式
        log_level="info"
    )
