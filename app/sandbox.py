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
import sklearn
from sklearn import preprocessing, model_selection, metrics, linear_model, ensemble, tree
import seaborn as sns
import scipy
from scipy import stats

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
        'sklearn': sklearn,
        'seaborn': sns,
        'sns': sns,
        'scipy': scipy,
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
            'sklearn': sklearn,
            'seaborn': sns,
            'scipy': scipy,
        }

        # 处理允许的库
        if base_module in ['matplotlib', 'plotly', 'numpy', 'pandas', 'sklearn', 'seaborn', 'scipy']:
            # 对于 "import matplotlib.pyplot as plt" 这种情况
            # fromlist 为空，需要返回顶层模块（matplotlib）
            # Python 会自动处理 matplotlib.pyplot 的访问
            if not fromlist:
                # 没有 fromlist，返回顶层模块
                logger.debug(f"Returning top-level module for {base_module}")
                if base_module in allowed_mapping:
                    return allowed_mapping[base_module]

            # 对于 "from matplotlib import pyplot" 或 "from sklearn.preprocessing import StandardScaler"
            # fromlist 不为空，需要返回请求的模块
            else:
                logger.debug(f"Returning module {name} with fromlist {fromlist}")
                # 如果完整名称在映射中，返回它
                if name in allowed_mapping:
                    return allowed_mapping[name]

            # 尝试实际导入（支持子模块，如 sklearn.preprocessing）
            try:
                logger.debug(f"Attempting real import of {name}")
                result = __import__(name, globals, locals, fromlist, level)
                logger.debug(f"Import successful, returning module")
                return result
            except ImportError as e:
                logger.debug(f"Import failed: {e}")
                # 如果无法导入，返回基础模块（如果有的话）
                if base_module in allowed_mapping:
                    logger.debug(f"Fallback to base module {base_module}")
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
""",

    "sklearn_preprocessing": """from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 创建示例数据
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

# 分离特征和标签
X = data[['feature1', 'feature2']]
y = data['target']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("原始数据:")
print(X_train.head())
print("\\n标准化后:")
print(pd.DataFrame(X_train_scaled, columns=X.columns).head())
""",

    "sklearn_linear_regression": """from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + 1.5 + np.random.randn(100, 1) * 2

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 评估
print(f"系数: {model.coef_[0][0]:.2f}")
print(f"截距: {model.intercept_[0]:.2f}")
print(f"R² 分数: {r2_score(y, y_pred):.3f}")
print(f"均方误差: {mean_squared_error(y, y_pred):.3f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='数据点')
plt.plot(X, y_pred, 'r-', linewidth=2, label='拟合线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归示例')
plt.legend()
plt.grid(True)
plt.show()
""",

    "sklearn_classification": """from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# 创建分类数据
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练随机森林
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.3f}")
print("\\n分类报告:")
print(classification_report(y_test, y_pred))
print("\\n特征重要性:")
for i, importance in enumerate(clf.feature_importances_):
    print(f"特征 {i}: {importance:.3f}")
""",

    "seaborn_visualization": """import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置样式
sns.set_theme(style="whitegrid")

# 创建示例数据
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'value': np.random.rand(100) * 100
})

# 创建多子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 散点图
sns.scatterplot(data=df, x='x', y='y', hue='category', size='value',
                ax=axes[0, 0], alpha=0.6)
axes[0, 0].set_title('散点图')

# 箱线图
sns.boxplot(data=df, x='category', y='value', ax=axes[0, 1])
axes[0, 1].set_title('箱线图')

# 小提琴图
sns.violinplot(data=df, x='category', y='value', ax=axes[1, 0])
axes[1, 0].set_title('小提琴图')

# 直方图
sns.histplot(data=df, x='value', hue='category', ax=axes[1, 1], kde=True)
axes[1, 1].set_title('直方图')

plt.tight_layout()
plt.show()
""",

    "scipy_stats": """from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
data1 = np.random.normal(100, 15, 100)
data2 = np.random.normal(105, 15, 100)

# 描述性统计
print("数据1统计:")
print(f"均值: {np.mean(data1):.2f}")
print(f"标准差: {np.std(data1):.2f}")
print(f"偏度: {stats.skew(data1):.2f}")
print(f"峰度: {stats.kurtosis(data1):.2f}")

# t检验
t_stat, p_value = stats.ttest_ind(data1, data2)
print(f"\\nt检验结果:")
print(f"t统计量: {t_stat:.3f}")
print(f"p值: {p_value:.3f}")
print(f"结论: {'显著差异' if p_value < 0.05 else '无显著差异'}")

# 正态性检验
stat, p = stats.normaltest(data1)
print(f"\\n正态性检验:")
print(f"p值: {p:.3f}")
print(f"结论: {'符合正态分布' if p > 0.05 else '不符合正态分布'}")

# 可视化
plt.figure(figsize=(10, 6))
plt.hist(data1, bins=20, alpha=0.5, label='数据1', density=True)
plt.hist(data2, bins=20, alpha=0.5, label='数据2', density=True)
plt.xlabel('值')
plt.ylabel('频率')
plt.title('数据分布对比')
plt.legend()
plt.grid(True)
plt.show()
"""
}
