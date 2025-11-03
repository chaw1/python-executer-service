"""
数据模型定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ExecuteRequest(BaseModel):
    """代码执行请求"""
    code: str = Field(..., description="要执行的Python代码")
    timeout: int = Field(default=30, ge=1, le=60, description="超时时间（秒）")
    output_format: str = Field(default="json", alias="outputFormat", description="输出格式：json 或 html")
    datasets: Optional[Dict[str, str]] = Field(default=None, description="数据集内容，key为文件名，value为文件内容")
    preloaded_variables: Optional[Dict[str, Any]] = Field(default=None, alias="preloadedVariables", description="预加载的变量，key为变量名，value为变量值")

    class Config:
        populate_by_name = True  # 允许使用字段名或别名
        json_schema_extra = {
            "example": {
                "code": "import pandas as pd\ndf = pd.DataFrame({\"x\": [1,2,3]})\nprint(df)",
                "timeout": 30,
                "output_format": "json",
                "datasets": {
                    "data.csv": "A,B,C\n1,2,3\n4,5,6"
                },
                "preloadedVariables": {
                    "user_id": 123,
                    "config": {"debug": True}
                }
            }
        }


class ChartOutput(BaseModel):
    """图表输出"""
    type: str = Field(..., description="图表类型：matplotlib 或 plotly")
    format: str = Field(..., description="格式：png, svg, json")
    data: str = Field(..., description="图表数据（base64 或 JSON字符串）")
    width: Optional[int] = Field(None, description="图表宽度")
    height: Optional[int] = Field(None, description="图表高度")


class DataFrameOutput(BaseModel):
    """DataFrame输出"""
    name: str = Field(..., description="变量名")
    html: str = Field(..., description="HTML表格")
    rows: int = Field(..., description="行数")
    columns: int = Field(..., description="列数")


class ExecutionOutput(BaseModel):
    """执行输出"""
    stdout: str = Field(default="", description="标准输出")
    stderr: str = Field(default="", description="标准错误")
    charts: List[ChartOutput] = Field(default_factory=list, description="图表列表")
    dataframes: List[DataFrameOutput] = Field(default_factory=list, description="DataFrame列表")
    variables: Dict[str, Any] = Field(default_factory=dict, description="变量信息")


class ExecuteResponse(BaseModel):
    """代码执行响应"""
    status: str = Field(..., description="执行状态：success, error, timeout")
    execution_time: int = Field(..., description="执行时间（毫秒）")
    output: Optional[ExecutionOutput] = Field(None, description="执行输出")
    error: Optional[str] = Field(None, description="错误信息")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "execution_time": 1234,
                "output": {
                    "stdout": "执行成功\n",
                    "stderr": "",
                    "charts": [],
                    "dataframes": []
                },
                "error": None
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="版本号")
    available_libraries: List[str] = Field(..., description="可用库列表")
