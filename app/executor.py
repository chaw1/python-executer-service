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

    def _normalize_code_indentation(self, code: str) -> str:
        """
        标准化代码缩进

        这个函数会智能处理代码的整体缩进，同时保留代码内部的相对缩进结构。
        例如：
        - 如果整个代码块被缩进了（从编辑器复制粘贴），会移除整体缩进
        - 保留函数、类、循环等结构内部的相对缩进

        Args:
            code: 原始代码

        Returns:
            标准化后的代码
        """
        import textwrap
        import re

        if not code or not code.strip():
            return code

        lines = code.split('\n')

        # 过滤掉完全空白的行来找最小缩进
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            return code

        # 检测制表符和空格混用
        has_tabs = any('\t' in line for line in non_empty_lines)
        has_spaces = any(line.startswith(' ') and not line.startswith('\t') for line in non_empty_lines)

        # 如果混用了制表符和空格，统一转换为4个空格
        if has_tabs and has_spaces:
            code = code.replace('\t', '    ')
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
        elif has_tabs:
            # 全是制表符，也转换为空格以便统一处理
            code = code.replace('\t', '    ')
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]

        # 找到所有非空行的前导空格数
        indents = []
        for line in non_empty_lines:
            # 计算前导空格数
            stripped = line.lstrip(' ')
            indent = len(line) - len(stripped)
            indents.append(indent)

        if not indents:
            return code

        # 找到最小缩进（这是整体的"偏移量"）
        min_indent = min(indents)

        # 如果最小缩进大于0，说明整个代码块都有统一的偏移，移除这个偏移
        if min_indent > 0:
            cleaned_lines = []
            for line in lines:
                if line.strip():  # 非空行
                    # 移除统一的偏移量，保留相对缩进
                    if len(line) >= min_indent:
                        cleaned_lines.append(line[min_indent:])
                    else:
                        # 防止某些行的缩进小于最小缩进（理论上不应该发生）
                        cleaned_lines.append(line.lstrip())
                else:
                    # 保留空行
                    cleaned_lines.append('')
            return '\n'.join(cleaned_lines)
        else:
            # 最小缩进是0，代码已经是标准格式了
            return code

    def _prepare_datasets(self, datasets: Dict[str, str], global_vars: Dict[str, Any]) -> None:
        """
        准备数据集，将文件内容注入到执行环境

        Args:
            datasets: 数据集字典，key为文件名，value为文件内容
            global_vars: 全局变量字典
        """
        import logging
        import pandas as pd
        import os

        logger = logging.getLogger(__name__)

        if not datasets:
            return

        logger.info(f"准备注入 {len(datasets)} 个数据集")

        # 存储原始内容和预处理后的DataFrame
        dataset_contents = {}  # 原始内容
        dataset_dataframes = {}  # 预处理的DataFrame

        for filename, content in datasets.items():
            # 存储原始内容
            dataset_contents[filename] = content

            # 尝试预读取为DataFrame
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(content))
                    dataset_dataframes[filename] = df
                    logger.debug(f"成功预读取 {filename} 为 DataFrame: {df.shape}")
                elif filename.endswith('.json'):
                    df = pd.read_json(io.StringIO(content))
                    dataset_dataframes[filename] = df
                    logger.debug(f"成功预读取 {filename} 为 DataFrame: {df.shape}")
            except Exception as e:
                logger.debug(f"无法预读取 {filename}: {e}")

        # 创建自定义的 pd.read_csv 函数，优先从内存读取
        original_read_csv = pd.read_csv
        original_read_json = pd.read_json

        def custom_read_csv(filepath_or_buffer, *args, **kwargs):
            """自定义 read_csv，优先从注入的数据集读取"""
            # 如果是字符串路径
            if isinstance(filepath_or_buffer, str):
                # 移除 {{dataset_path}}/ 前缀
                clean_path = filepath_or_buffer.replace('{{dataset_path}}/', '').replace('{{dataset_path}}', '')
                # 获取文件名
                filename = os.path.basename(clean_path)

                # 优先返回预处理的DataFrame
                if filename in dataset_dataframes:
                    logger.debug(f"从内存返回 {filename} 的 DataFrame")
                    return dataset_dataframes[filename].copy()

                # 如果有原始内容，从StringIO读取
                if filename in dataset_contents:
                    logger.debug(f"从 StringIO 读取 {filename}")
                    return original_read_csv(io.StringIO(dataset_contents[filename]), *args, **kwargs)

            # 否则使用原始函数
            return original_read_csv(filepath_or_buffer, *args, **kwargs)

        def custom_read_json(filepath_or_buffer, *args, **kwargs):
            """自定义 read_json，优先从注入的数据集读取"""
            if isinstance(filepath_or_buffer, str):
                clean_path = filepath_or_buffer.replace('{{dataset_path}}/', '').replace('{{dataset_path}}', '')
                filename = os.path.basename(clean_path)

                if filename in dataset_dataframes:
                    logger.debug(f"从内存返回 {filename} 的 DataFrame")
                    return dataset_dataframes[filename].copy()

                if filename in dataset_contents:
                    logger.debug(f"从 StringIO 读取 {filename}")
                    return original_read_json(io.StringIO(dataset_contents[filename]), *args, **kwargs)

            return original_read_json(filepath_or_buffer, *args, **kwargs)

        # 注入自定义函数和数据
        global_vars['__datasets_content__'] = dataset_contents
        global_vars['__datasets_df__'] = dataset_dataframes

        # 获取 pandas 模块并替换读取函数
        if 'pd' in global_vars:
            global_vars['pd'].read_csv = custom_read_csv
            global_vars['pd'].read_json = custom_read_json
            logger.debug("已覆盖 pd.read_csv 和 pd.read_json")

        logger.info(f"已注入 {len(dataset_dataframes)} 个预处理 DataFrame，{len(dataset_contents)} 个原始内容")

    def execute(self, code: str, datasets: Dict[str, str] = None) -> ExecuteResponse:
        """
        执行 Python 代码

        Args:
            code: 要执行的代码
            datasets: 数据集字典，key为文件名，value为文件内容（可选）

        Returns:
            执行结果
        """
        import logging
        logger = logging.getLogger(__name__)

        start_time = time.time()

        # 标准化代码缩进
        original_code = code
        code = self._normalize_code_indentation(code)

        # 记录缩进处理
        if code != original_code:
            logger.debug("代码缩进已标准化")
            logger.debug(f"原始代码:\n{original_code}")
            logger.debug(f"标准化后:\n{code}")

        # 验证代码
        is_valid, error_msg = self.safe_env.validate_code(code)
        if not is_valid:
            logger.warning(f"代码验证失败: {error_msg}")
            logger.debug(f"验证失败的代码:\n{code}")
            return ExecuteResponse(
                status="error",
                execution_time=0,
                output=ExecutionOutput(
                    stdout="",
                    stderr=error_msg
                ),
                error=error_msg
            )

        # 准备执行环境
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # 获取安全的全局环境
        global_vars = self.safe_env.get_safe_globals()
        local_vars = {}

        # 注入数据集（如果提供）
        if datasets:
            self._prepare_datasets(datasets, global_vars)

        try:
            # 编译代码
            logger.debug("正在编译代码...")
            compiled_code = self.safe_env.compile_code(code)
            logger.debug("代码编译成功")

            # 设置超时（仅在 Unix 系统上有效）
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
                logger.debug(f"已设置超时: {self.timeout}秒")

            # 执行代码并捕获输出
            logger.debug("开始执行代码...")
            exec_start = time.time()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled_code, global_vars, local_vars)
            exec_time = time.time() - exec_start
            logger.debug(f"代码执行完成，耗时: {exec_time:.3f}秒")

            # 取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            # 捕获图表
            logger.debug("正在捕获图表...")
            chart_capture = ChartCapture()
            charts = chart_capture.capture_all(local_vars)
            if charts:
                logger.info(f"捕获到 {len(charts)} 个图表")

            # 捕获 DataFrames
            logger.debug("正在捕获 DataFrames...")
            dataframes = DataFrameCapture.capture_dataframes(local_vars)
            if dataframes:
                logger.info(f"捕获到 {len(dataframes)} 个 DataFrame")

            # 捕获变量信息
            variables = self._extract_variables(local_vars)
            if variables:
                logger.debug(f"捕获到 {len(variables)} 个变量")

            # 清理
            chart_capture.clear_all()

            # 计算执行时间
            execution_time = int((time.time() - start_time) * 1000)

            # 获取 stdout 输出
            final_stdout = stdout_capture.getvalue()

            logger.info(f"执行成功 - 总耗时: {execution_time}ms")

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
            logger.warning(f"代码执行超时 - 限制: {self.timeout}秒, 已用时: {execution_time}ms")

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

            # 记录错误
            logger.error(f"代码执行失败: {str(e)}")
            logger.debug(f"错误堆栈:\n{error_traceback}")

            # 提供更友好的错误信息
            error_message = str(e)
            if "name" in str(e).lower() and "is not defined" in str(e).lower():
                error_message += "\n提示: 请检查变量名是否拼写正确，或该变量是否已定义。"
            elif "module" in str(e).lower() and "has no attribute" in str(e).lower():
                error_message += "\n提示: 请检查模块或对象的方法/属性名是否正确。"

            return ExecuteResponse(
                status="error",
                execution_time=execution_time,
                output=ExecutionOutput(
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue() + "\n" + error_traceback
                ),
                error=error_message
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
