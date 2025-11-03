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
        标准化代码缩进（增强版）

        智能处理各种缩进问题，包括：
        - 统一Tab和空格
        - 移除公共前导空格
        - 自动修复不一致的缩进
        - 智能识别代码块结构

        Args:
            code: 原始代码

        Returns:
            标准化后的代码
        """
        import textwrap
        import re

        if not code or not code.strip():
            return code

        # 统一处理制表符，转换为4个空格
        code = code.replace('\t', '    ')

        # 先用 dedent 移除公共前导空格
        dedented_code = textwrap.dedent(code)

        # 进一步智能处理：修复不一致的缩进
        lines = dedented_code.split('\n')
        normalized_lines = []
        indent_stack = [0]  # 缩进栈，用于跟踪缩进层级

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            # 空行或纯注释行保持原样
            if not stripped or stripped.startswith('#'):
                normalized_lines.append(line)
                continue

            # 计算原始缩进
            original_indent = len(line) - len(stripped)

            # 检查是否是顶层语句（不应该有缩进）
            is_top_level = any(stripped.startswith(kw) for kw in [
                'import ', 'from ', 'def ', 'class ', '@'
            ])

            if is_top_level and i < 3:  # 文件开头的顶层语句
                current_indent = 0
                indent_stack = [0]
            else:
                # 检查是否是代码块开始（以冒号结尾）
                if stripped.rstrip().endswith(':'):
                    current_indent = indent_stack[-1]
                    # 下一行应该增加缩进
                    indent_stack.append(current_indent + 4)
                else:
                    # 使用当前缩进栈的最后一个值
                    current_indent = indent_stack[-1]

                    # 如果缩进减少了，弹出栈
                    if original_indent < indent_stack[-1] and len(indent_stack) > 1:
                        # 找到匹配的缩进层级
                        while len(indent_stack) > 1 and indent_stack[-1] > original_indent:
                            indent_stack.pop()
                        current_indent = indent_stack[-1]

            # 应用标准化的缩进
            normalized_line = ' ' * current_indent + stripped
            normalized_lines.append(normalized_line)

        return '\n'.join(normalized_lines)

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

        # 创建 selected_files 变量（用户可以直接使用）
        selected_files = []
        for filename, content in dataset_contents.items():
            selected_files.append({
                'name': filename,
                'path': filename,  # 目前使用文件名作为路径
                'content': content
            })

        global_vars['selected_files'] = selected_files
        logger.info(f"已创建 selected_files 变量，包含 {len(selected_files)} 个文件")

        # 获取 pandas 模块并替换读取函数
        if 'pd' in global_vars:
            global_vars['pd'].read_csv = custom_read_csv
            global_vars['pd'].read_json = custom_read_json
            logger.debug("已覆盖 pd.read_csv 和 pd.read_json")

        logger.info(f"已注入 {len(dataset_dataframes)} 个预处理 DataFrame，{len(dataset_contents)} 个原始内容")

    def execute(self, code: str, datasets: Dict[str, str] = None, preloaded_variables: Dict[str, Any] = None) -> ExecuteResponse:
        """
        执行 Python 代码

        Args:
            code: 要执行的代码
            datasets: 数据集字典，key为文件名，value为文件内容（可选）
            preloaded_variables: 预加载的变量字典，key为变量名，value为变量值（可选）

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

        # 注入预加载变量（如果提供）
        if preloaded_variables:
            logger.info(f"注入 {len(preloaded_variables)} 个预加载变量")
            for var_name, var_value in preloaded_variables.items():
                logger.debug(f"注入变量: {var_name} = {type(var_value).__name__}")
                global_vars[var_name] = var_value

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
            import matplotlib.pyplot as plt_check
            fignums_before = plt_check.get_fignums()
            logger.info(f"[图表捕获] 执行后的 figure 数量: {len(fignums_before)}, fignums={fignums_before}")

            chart_capture = ChartCapture()
            charts = chart_capture.capture_all(local_vars)
            logger.info(f"[图表捕获] 捕获结果: {len(charts) if charts else 0} 个图表")
            if charts:
                for i, chart in enumerate(charts):
                    logger.info(f"[图表捕获] 图表 {i+1}: type={chart.type}, format={chart.format}, data_length={len(chart.data) if chart.data else 0}")

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
