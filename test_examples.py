#!/usr/bin/env python
"""
Python Executor Service - 完整测试示例
运行此脚本以测试所有功能
"""
from app.executor import CodeExecutor
import json


def print_result(test_name, result):
    """打印测试结果"""
    print(f"\n{'='*80}")
    print(f"测试: {test_name}")
    print(f"{'='*80}")
    print(f"状态: {result.status}")
    print(f"执行时间: {result.execution_time}ms")

    if result.output:
        if result.output.stdout:
            stdout = result.output.stdout
            if len(stdout) > 1000:
                stdout = stdout[:1000] + "\n... (输出过长，已截断)"
            print(f"\n输出:\n{stdout}")

        if result.output.charts:
            print(f"\n图表数量: {len(result.output.charts)}")

        if result.output.dataframes:
            print(f"DataFrame数量: {len(result.output.dataframes)}")

        if result.output.variables:
            print(f"变量数量: {len(result.output.variables)}")

    if result.error:
        print(f"\n错误: {result.error}")

    if result.output and result.output.stderr and result.output.stderr.strip():
        stderr = result.output.stderr
        if len(stderr) > 500:
            stderr = stderr[:500] + "\n... (错误信息过长，已截断)"
        print(f"\n错误详情: {stderr}")

    return result.status == "success"


# 测试用例
TESTS = {
    "1. 基础 - Hello World": """
print("Hello, Python Executor Service!")
x = 10
y = 20
print(f"x + y = {x + y}")
""",

    "2. Pandas - 数据分析": """
import pandas as pd
import numpy as np

# 创建数据
df = pd.DataFrame({
    '产品': ['A', 'B', 'C', 'A', 'B', 'C'],
    '销量': [100, 150, 80, 120, 160, 90],
    '价格': [10, 15, 20, 10, 15, 20]
})
df['营收'] = df['销量'] * df['价格']

print("数据概览:")
print(df)

print("\\n按产品汇总:")
print(df.groupby('产品')[['销量', '营收']].sum())

print(f"\\n总营收: {df['营收'].sum()}")
""",

    "3. NumPy - 矩阵运算": """
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("矩阵 A:")
print(A)

print("\\n矩阵 B:")
print(B)

print("\\n矩阵乘法 A @ B:")
print(A @ B)

print(f"\\nA的行列式: {np.linalg.det(A):.2f}")
""",

    "4. Matplotlib - 可视化": """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('三角函数')
plt.legend()
plt.grid(True)
plt.show()

print("图表已生成")
""",

    "5. Scikit-learn - 线性回归": """
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# 生成数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

print(f"系数: {model.coef_[0]:.2f}")
print(f"截距: {model.intercept_:.2f}")
print(f"R² 分数: {r2_score(y, y_pred):.3f}")

# 预测新值
new_X = [[6], [7]]
predictions = model.predict(new_X)
print(f"\\n预测 X=6: {predictions[0]:.2f}")
print(f"预测 X=7: {predictions[1]:.2f}")
""",

    "6. Scikit-learn - 分类": """
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"准确率: {accuracy:.3f}")
print(f"特征重要性: {clf.feature_importances_}")
print(f"\\n分类样本: {y_pred[:10]}")
""",

    "7. Scikit-learn - 数据预处理": """
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

# 创建数据
data = np.array([[1, 10, 100],
                 [2, 20, 200],
                 [3, 30, 300],
                 [4, 40, 400],
                 [5, 50, 500]])

df = pd.DataFrame(data, columns=['A', 'B', 'C'])

print("原始数据:")
print(df)

# Z-Score 标准化
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

print("\\nZ-Score 标准化后:")
print(pd.DataFrame(scaled, columns=['A', 'B', 'C']))

# Min-Max 归一化
minmax = MinMaxScaler()
normalized = minmax.fit_transform(df)

print("\\nMin-Max 归一化后:")
print(pd.DataFrame(normalized, columns=['A', 'B', 'C']))
""",

    "8. Seaborn - 统计图表": """
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'value': np.random.randn(100) * 10 + 50
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='value')
sns.swarmplot(data=df, x='category', y='value', color='black', alpha=0.3, size=3)
plt.title('分类数据分布')
plt.show()

print("Seaborn 图表已生成")
""",

    "9. SciPy - 统计检验": """
from scipy import stats
import numpy as np

np.random.seed(42)
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(105, 15, 50)

# t检验
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"组1 均值: {group1.mean():.2f}")
print(f"组2 均值: {group2.mean():.2f}")
print(f"\\nt统计量: {t_stat:.4f}")
print(f"p值: {p_value:.4f}")

if p_value < 0.05:
    print("\\n结论: 两组存在显著差异")
else:
    print("\\n结论: 两组无显著差异")

# 正态性检验
stat, p = stats.normaltest(group1)
print(f"\\n组1 正态性检验 p值: {p:.4f}")
print(f"结论: {'符合正态分布' if p > 0.05 else '不符合正态分布'}")
""",

    "10. Plotly - 交互式图表": """
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50),
    'category': np.random.choice(['A', 'B', 'C'], 50),
    'size': np.random.randint(10, 100, 50)
})

fig = px.scatter(df, x='x', y='y', color='category', size='size',
                 title='交互式散点图')
fig.show()

print("Plotly 图表已生成")
""",

    "11. 综合 - 完整工作流": """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. 生成数据
np.random.seed(42)
n = 100
X = np.random.randn(n, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(n)*0.5

# 2. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. 训练模型
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 5. 预测和评估
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== 完整机器学习工作流 ===")
print(f"数据集大小: {n}")
print(f"特征数量: 3")
print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print(f"\\n模型性能:")
print(f"  R² 分数: {r2:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"\\n特征重要性: {model.feature_importances_}")
""",

    "12. 用户场景 - 数据预处理": """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 创建示例数据（模拟用户的 CSV 数据）
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'feature3': [100, 200, 300, 400, 500]
})

# 选择数值列
numeric_cols = df.select_dtypes(include=[np.number]).columns

print("=== 原始数据统计 ===")
print(df[numeric_cols].describe())

# Z-Score 标准化
scaler_standard = StandardScaler()
df_standard = df.copy()
df_standard[numeric_cols] = scaler_standard.fit_transform(df[numeric_cols])

print("\\n=== Z-Score 标准化后 ===")
print(df_standard[numeric_cols].describe())

# Min-Max 归一化
scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numeric_cols] = scaler_minmax.fit_transform(df[numeric_cols])

print("\\n=== Min-Max 归一化后 ===")
print(df_minmax[numeric_cols].describe())

print("\\n数据预处理完成！")
""",
}


def run_all_tests():
    """运行所有测试"""
    print("="*80)
    print(" "*20 + "Python Executor Service 功能测试")
    print("="*80)

    executor = CodeExecutor(timeout=30)
    results = {}

    for test_name, code in TESTS.items():
        try:
            result = executor.execute(code)
            success = print_result(test_name, result)
            results[test_name] = success

            if success:
                print(f"\n✓ {test_name} - 成功")
            else:
                print(f"\n✗ {test_name} - 失败")

        except Exception as e:
            print(f"\n✗ {test_name} - 异常: {e}")
            results[test_name] = False

        print()

    # 总结
    print("\n" + "="*80)
    print(" "*30 + "测试总结")
    print("="*80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")

    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 所有测试通过！服务运行正常！")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查。")

    print("="*80)


if __name__ == "__main__":
    run_all_tests()
