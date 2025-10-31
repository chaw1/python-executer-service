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

        # 如果是用户代码中导入的允许模块，返回预先导入的对象
        if name in allowed_mapping:
            return allowed_mapping[name]
        elif base_module in allowed_mapping:
            # 对于子模块，返回基础模块
            return allowed_mapping[base_module]

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
        # 检查禁止的关键字
        code_lower = code.lower()
        for forbidden in cls.FORBIDDEN_NAMES:
            # 更精确的检查，避免误报
            if f' {forbidden}(' in code_lower or f'\n{forbidden}(' in code_lower or code_lower.startswith(f'{forbidden}('):
                return False, f"禁止使用: {forbidden}"

        # 检查禁止的 import
        lines = code.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('import ') or line_stripped.startswith('from '):
                # 提取模块名
                if line_stripped.startswith('import '):
                    parts = line_stripped[7:].split()
                    if parts:
                        module = parts[0].split('.')[0].strip(',')
                        if module in cls.FORBIDDEN_IMPORTS:
                            return False, f"禁止导入模块: {module}"
                elif line_stripped.startswith('from '):
                    parts = line_stripped[5:].split()
                    if parts:
                        module = parts[0].split('.')[0]
                        if module in cls.FORBIDDEN_IMPORTS:
                            return False, f"禁止导入模块: {module}"

        # 尝试编译
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"语法错误: {str(e)}"
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
