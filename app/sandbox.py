"""
安全沙箱配置
"""
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px

# 配置 matplotlib 使用非交互式后端
matplotlib.use('Agg')


class SafeExecutionEnvironment:
    """安全执行环境"""

    # 允许的库白名单（预导入）
    ALLOWED_MODULES = {
        'numpy': np,
        'np': np,
        'pandas': pd,
        'pd': pd,
        'matplotlib': matplotlib,
        'plt': plt,
        'plotly': plotly,
        'go': go,
        'px': px,
    }

    # 禁止的操作和模块
    FORBIDDEN_NAMES = [
        'open', 'file', 'input', 'raw_input',
        'compile', 'reload', '__import__',
        'execfile', 'eval', 'exec',
        'os', 'sys', 'subprocess', 'socket',
        'urllib', 'requests', 'http', 'httpx',
        'pathlib', 'shutil', 'glob',
    ]

    # 禁止的 import 模块
    FORBIDDEN_IMPORTS = [
        'os', 'sys', 'subprocess', 'socket',
        'urllib', 'requests', 'http', 'httpx',
        'pathlib', 'shutil', 'glob', 'pickle',
        'multiprocessing', 'threading', 'asyncio',
    ]

    @classmethod
    def safe_import(cls, name, globals=None, locals=None, fromlist=(), level=0):
        """
        安全的 import 函数
        只允许导入白名单中的模块，并返回预先导入的对象
        允许内部依赖的导入，但阻止禁止的模块
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"safe_import called: name={name}, fromlist={fromlist}, level={level}")

        base_module = name.split('.')[0]

        # 检查是否是禁止的模块
        if base_module in cls.FORBIDDEN_IMPORTS or name in cls.FORBIDDEN_IMPORTS:
            raise ImportError(f"不允许导入模块: {name}")

        # 允许的模块映射（返回已经导入的对象）
        allowed_mapping = {
            'numpy': np,
            'pandas': pd,
            'matplotlib': matplotlib,
            'matplotlib.pyplot': plt,
            'plotly': plotly,
            'plotly.graph_objects': go,
            'plotly.express': px,
        }

        # 处理允许的库
        if base_module in ['matplotlib', 'plotly', 'numpy', 'pandas']:
            # 对于 "import matplotlib.pyplot as plt" 这种情况
            # fromlist 为空，需要返回顶层模块（matplotlib）
            # Python 会自动处理 matplotlib.pyplot 的访问
            if not fromlist:
                # 没有 fromlist，返回顶层模块
                logger.debug(f"Returning top-level module for {base_module}")
                if base_module in allowed_mapping:
                    return allowed_mapping[base_module]

            # 对于 "from matplotlib import pyplot" 或 "from matplotlib.pyplot import figure"
            # fromlist 不为空，需要返回请求的模块
            else:
                logger.debug(f"Returning module {name} with fromlist {fromlist}")
                # 如果完整名称在映射中，返回它
                if name in allowed_mapping:
                    return allowed_mapping[name]
                # 否则返回基础模块，Python 会从中提取 fromlist
                if base_module in allowed_mapping:
                    return allowed_mapping[base_module]

            # 如果上面没有返回，尝试实际导入
            try:
                return __import__(name, globals, locals, fromlist, level)
            except ImportError:
                # 如果无法导入，返回基础模块（如果有的话）
                if base_module in allowed_mapping:
                    return allowed_mapping[base_module]
                raise

        # 对于其他模块（包括库的内部依赖），允许正常导入
        # 但要确保不是禁止的模块
        try:
            return __import__(name, globals, locals, fromlist, level)
        except ImportError:
            # 如果导入失败，返回一个空的模块对象（避免中断执行）
            import types
            return types.ModuleType(name)

    @classmethod
    def get_safe_globals(cls) -> Dict[str, Any]:
        """获取安全的全局命名空间"""
        # 创建安全的 builtins
        safe_builtins = {
            # 基础函数
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'frozenset': frozenset,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'bytes': bytes,
            # 类型检查
            'type': type,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            # 其他安全函数
            'any': any,
            'all': all,
            'ord': ord,
            'chr': chr,
            'hex': hex,
            'oct': oct,
            'bin': bin,
            'pow': pow,
            'divmod': divmod,
            # 异常
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'RuntimeError': RuntimeError,
            'StopIteration': StopIteration,
            # 控制 import
            '__import__': cls.safe_import,
            # 常量
            'True': True,
            'False': False,
            'None': None,
        }

        safe_dict = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }

        # 预先添加允许的模块（无需 import 即可使用）
        safe_dict.update(cls.ALLOWED_MODULES)

        return safe_dict

    @classmethod
    def validate_code(cls, code: str) -> tuple[bool, str]:
        """
        验证代码是否安全

        Returns:
            (is_valid, error_message)
        """
        import re

        if not code or not code.strip():
            return False, "代码不能为空"

        # 检查代码长度
        if len(code) > 100000:  # 100KB
            return False, "代码过长（最大100KB）"

        # 检查禁止的关键字（使用正则表达式更精确地匹配）
        for forbidden in cls.FORBIDDEN_NAMES:
            # 使用正则表达式匹配函数调用，避免误报
            # 例如：匹配 "open(" 但不匹配 "reopen("
            pattern = r'\b' + re.escape(forbidden) + r'\s*\('
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"检测到禁止的操作: {forbidden}()\n安全策略不允许使用此函数。"

        # 检查禁止的 import（改进的检测）
        lines = code.split('\n')
        for line_no, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # 跳过注释
            if line_stripped.startswith('#'):
                continue

            # 检查 import 语句
            if line_stripped.startswith('import ') or line_stripped.startswith('from '):
                if line_stripped.startswith('import '):
                    # 处理 import 语句: import os, sys
                    import_part = line_stripped[7:].split('#')[0].strip()  # 移除注释
                    modules = [m.strip().split()[0].split('.')[0] for m in import_part.split(',')]

                    for module in modules:
                        if module in cls.FORBIDDEN_IMPORTS:
                            return False, f"第{line_no}行：禁止导入模块 '{module}'\n安全策略不允许使用此模块。"

                elif line_stripped.startswith('from '):
                    # 处理 from 语句: from os import path
                    parts = line_stripped[5:].split('#')[0].strip().split()
                    if parts:
                        module = parts[0].split('.')[0]
                        if module in cls.FORBIDDEN_IMPORTS:
                            return False, f"第{line_no}行：禁止导入模块 '{module}'\n安全策略不允许使用此模块。"

        # 尝试编译
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            # 提供更友好的语法错误信息
            error_msg = f"语法错误"
            if e.lineno:
                error_msg += f"（第{e.lineno}行）"
            if e.msg:
                error_msg += f": {e.msg}"
            if e.text:
                error_msg += f"\n问题代码: {e.text.strip()}"
                if e.offset:
                    error_msg += f"\n{' ' * (e.offset - 1)}^"

            return False, error_msg
        except IndentationError as e:
            # 缩进错误
            error_msg = f"缩进错误"
            if e.lineno:
                error_msg += f"（第{e.lineno}行）"
            error_msg += f": {e.msg}"
            return False, error_msg
        except Exception as e:
            return False, f"编译错误: {str(e)}"

    @classmethod
    def compile_code(cls, code: str):
        """编译代码"""
        return compile(code, '<string>', 'exec')


# 预定义的代码模板
CODE_TEMPLATES = {
    "matplotlib_basic": """import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
""",

    "plotly_scatter": """import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 1, 5, 3],
    'category': ['A', 'B', 'A', 'B', 'A']
})

fig = px.scatter(df, x='x', y='y', color='category', title='Scatter Plot')
fig.show()
""",

    "pandas_analysis": """import pandas as pd
import numpy as np

# 创建示例数据
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'Score': [85, 92, 78, 88]
})

print("数据概览:")
print(df)
print("\\n统计信息:")
print(df.describe())
"""
}
