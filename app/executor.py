"""
代码执行引擎
"""
import sys
import io
import time
import traceback
import signal
from contextlib import redirect_stdout, redirect_stderr
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

from app.sandbox import SafeExecutionEnvironment
from app.visualizer import ChartCapture, DataFrameCapture
from app.models import ExecuteResponse, ExecutionOutput


class TimeoutException(Exception):
    """超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时处理函数"""
    raise TimeoutException("代码执行超时")


class CodeExecutor:
    """代码执行器"""

    def __init__(self, timeout: int = 30):
        """
        初始化执行器

        Args:
            timeout: 超时时间（秒）
        """
        self.timeout = timeout
        self.safe_env = SafeExecutionEnvironment()

    def execute(self, code: str) -> ExecuteResponse:
        """
        执行 Python 代码

        Args:
            code: 要执行的代码

        Returns:
            执行结果
        """
        import logging
        import textwrap
        import re
        logger = logging.getLogger(__name__)

        start_time = time.time()

        # 清理代码：智能移除不一致的缩进
        # 处理混合缩进的情况（比如第一行没缩进，后面行有缩进）
        lines = code.split('\n')

        # 找到所有非空行的缩进
        indents = []
        for line in lines:
            if line.strip():  # 非空行
                # 计算前导空格数
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                indents.append(indent)

        # 如果有缩进，智能处理
        if indents:
            min_indent = min(indents)

            # 如果最小缩进是0（有些行没缩进）但也有缩进的行
            # 则移除最小的非零缩进
            if min_indent == 0 and len(set(indents)) > 1:
                # 找到最小的非零缩进
                non_zero_indents = [i for i in indents if i > 0]
                if non_zero_indents:
                    min_indent = min(non_zero_indents)

            # 移除缩进
            if min_indent > 0:
                cleaned_lines = []
                for line in lines:
                    if line.strip():  # 非空行
                        cleaned_lines.append(line[min_indent:] if len(line) >= min_indent else line)
                    else:  # 空行
                        cleaned_lines.append('')
                code = '\n'.join(cleaned_lines)
        else:
            # 如果所有行都是空行，使用原代码
            code = textwrap.dedent(code)

        # 验证代码
        is_valid, error_msg = self.safe_env.validate_code(code)
        if not is_valid:
            logger.error(f"代码验证失败: {error_msg}")
            logger.error(f"失败的代码:\n{code}")
            return ExecuteResponse(
                status="error",
                execution_time=0,
                output=None,
                error=f"代码验证失败: {error_msg}"
            )

        # 准备执行环境
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # 获取安全的全局环境
        global_vars = self.safe_env.get_safe_globals()
        local_vars = {}

        try:
            # 使用普通compile编译代码
            compiled_code = self.safe_env.compile_code(code)

            # 设置超时（仅在 Unix 系统上有效）
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)

            # 执行代码并捕获输出
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled_code, global_vars, local_vars)

            # 取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            # 捕获图表
            chart_capture = ChartCapture()
            charts = chart_capture.capture_all(local_vars)

            # 捕获 DataFrames
            dataframes = DataFrameCapture.capture_dataframes(local_vars)

            # 捕获变量信息
            variables = self._extract_variables(local_vars)

            # 清理
            chart_capture.clear_all()

            # 计算执行时间
            execution_time = int((time.time() - start_time) * 1000)

            # 获取 stdout 输出
            final_stdout = stdout_capture.getvalue()

            return ExecuteResponse(
                status="success",
                execution_time=execution_time,
                output=ExecutionOutput(
                    stdout=final_stdout,
                    stderr=stderr_capture.getvalue(),
                    charts=charts,
                    dataframes=dataframes,
                    variables=variables
                ),
                error=None
            )

        except TimeoutException:
            # 超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            execution_time = int((time.time() - start_time) * 1000)
            return ExecuteResponse(
                status="timeout",
                execution_time=execution_time,
                output=ExecutionOutput(
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue()
                ),
                error=f"执行超时（限制 {self.timeout} 秒）"
            )

        except Exception as e:
            # 执行错误
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            execution_time = int((time.time() - start_time) * 1000)
            error_traceback = traceback.format_exc()

            return ExecuteResponse(
                status="error",
                execution_time=execution_time,
                output=ExecutionOutput(
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue() + "\n" + error_traceback
                ),
                error=str(e)
            )

        finally:
            # 清理资源
            plt.close('all')
            stdout_capture.close()
            stderr_capture.close()

    def _extract_variables(self, local_vars: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取变量信息

        Args:
            local_vars: 局部变量字典

        Returns:
            变量信息字典
        """
        import numpy as np
        import pandas as pd

        variables = {}

        for var_name, var_value in local_vars.items():
            # 跳过私有变量和模块
            if (var_name.startswith('_') or
                var_name.startswith('__') or
                hasattr(var_value, '__module__')):
                continue

            try:
                # 基础类型
                if isinstance(var_value, (int, float, str, bool, list, dict, tuple)):
                    variables[var_name] = {
                        "type": type(var_value).__name__,
                        "value": str(var_value)[:200]  # 限制长度
                    }
                # NumPy 数组
                elif isinstance(var_value, np.ndarray):
                    variables[var_name] = {
                        "type": "numpy.ndarray",
                        "shape": var_value.shape,
                        "dtype": str(var_value.dtype),
                        "preview": str(var_value)[:200]
                    }
                # Pandas DataFrame
                elif isinstance(var_value, pd.DataFrame):
                    variables[var_name] = {
                        "type": "pandas.DataFrame",
                        "shape": var_value.shape,
                        "columns": var_value.columns.tolist()
                    }
                # Pandas Series
                elif isinstance(var_value, pd.Series):
                    variables[var_name] = {
                        "type": "pandas.Series",
                        "length": len(var_value),
                        "dtype": str(var_value.dtype)
                    }
            except Exception:
                # 跳过无法序列化的变量
                pass

        return variables


# 单例执行器
_executor_instance = None


def get_executor(timeout: int = 30) -> CodeExecutor:
    """
    获取执行器实例

    Args:
        timeout: 超时时间

    Returns:
        执行器实例
    """
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = CodeExecutor(timeout)
    return _executor_instance
