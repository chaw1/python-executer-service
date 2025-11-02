# 可用算子清单

## 📋 概述

本文档列出了 Python Executor Service 中所有可用的数据治理算子（操作）。这些算子按功能分类，方便在智能数据标注平台中选择使用。

---

## 🎯 算子分类

### 📊 一、数据分析类

#### 1.1 基础统计分析
```python
# pandas 描述性统计
df.describe()           # 数值列统计摘要
df.info()              # 数据类型和缺失值信息
df.value_counts()      # 频数统计
df.corr()              # 相关性矩阵
```

**预置模板**: `pandas_analysis`

**功能**:
- 数据概览
- 统计摘要（均值、标准差、四分位数等）
- 数据类型检查
- 缺失值统计

**使用场景**: 数据上传后的初步质量检查

---

#### 1.2 高级统计分析
```python
# scipy 统计检验
from scipy import stats

stats.ttest_ind(data1, data2)    # t检验
stats.normaltest(data)            # 正态性检验
stats.skew(data)                  # 偏度
stats.kurtosis(data)              # 峰度
stats.pearsonr(x, y)              # 皮尔逊相关系数
```

**预置模板**: `scipy_stats`

**功能**:
- 假设检验（t检验、卡方检验等）
- 分布检验（正态性、偏度、峰度）
- 相关性分析
- 统计推断

**使用场景**: 数据质量深度分析、A/B测试

---

### 🧹 二、数据清洗类

#### 2.1 缺失值处理
```python
# 检测缺失值
df.isnull().sum()          # 统计缺失值
df.isnull().any()          # 是否有缺失

# 填充缺失值
df.fillna(0)               # 填充0
df.fillna(method='ffill')  # 前向填充
df.fillna(method='bfill')  # 后向填充
df.fillna(df.mean())       # 均值填充
df.fillna(df.median())     # 中位数填充

# 删除缺失值
df.dropna()                # 删除含缺失值的行
df.dropna(axis=1)          # 删除含缺失值的列
```

**使用场景**: 数据质量提升、补全不完整数据

---

#### 2.2 重复值处理
```python
# 检测重复
df.duplicated()            # 标记重复行
df.duplicated().sum()      # 统计重复数

# 删除重复
df.drop_duplicates()                    # 删除所有重复
df.drop_duplicates(subset=['col'])      # 基于指定列删除
df.drop_duplicates(keep='first')        # 保留第一个
df.drop_duplicates(keep='last')         # 保留最后一个
```

**使用场景**: 数据去重、保证数据唯一性

---

#### 2.3 异常值处理
```python
# IQR 方法检测异常值
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 过滤异常值
df_clean = df[(df['column'] >= lower_bound) &
              (df['column'] <= upper_bound)]

# Z-Score 方法
from scipy import stats
z_scores = np.abs(stats.zscore(df['column']))
df_clean = df[z_scores < 3]
```

**使用场景**: 数据质量控制、异常数据清理

---

#### 2.4 数据类型转换
```python
# 类型转换
df['col'].astype('int')        # 转整数
df['col'].astype('float')      # 转浮点
df['col'].astype('str')        # 转字符串
pd.to_datetime(df['date'])     # 转日期时间
pd.to_numeric(df['col'])       # 转数值（自动推断）

# 分类类型
df['category'].astype('category')
```

**使用场景**: 数据类型规范化、提高存储效率

---

### 🔧 三、数据预处理类

#### 3.1 数据标准化（Z-Score）
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
# 均值0，标准差1
```

**预置模板**: `sklearn_preprocessing`

**功能**:
- Z-Score 标准化
- 消除量纲影响
- 适用于正态分布数据

**使用场景**: 机器学习模型训练前的特征缩放

---

#### 3.2 数据归一化（Min-Max）
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)
# 缩放到 [0, 1] 范围
```

**预置模板**: `sklearn_preprocessing`

**功能**:
- Min-Max 归一化
- 缩放到指定范围
- 保持数据分布形状

**使用场景**: 神经网络输入、特征缩放

---

#### 3.3 标签编码
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 标签编码（序号）
encoder = LabelEncoder()
df['category_encoded'] = encoder.fit_transform(df['category'])

# 独热编码（One-Hot）
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['category']])

# pandas 独热编码
pd.get_dummies(df['category'])
```

**使用场景**: 分类特征转数值、机器学习模型输入

---

#### 3.4 数据分割
```python
from sklearn.model_selection import train_test_split

# 训练集/测试集分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 测试集比例
    random_state=42     # 随机种子
)
```

**预置模板**: `sklearn_preprocessing`

**使用场景**: 机器学习模型训练和评估

---

### 🤖 四、机器学习类

#### 4.1 线性回归
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 评估
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
```

**预置模板**: `sklearn_linear_regression`

**功能**:
- 线性回归模型
- R²、MSE 评估
- 可视化拟合线

**使用场景**: 数值预测、趋势分析

---

#### 4.2 分类模型
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 随机森林分类
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
```

**预置模板**: `sklearn_classification`

**功能**:
- 随机森林分类器
- 准确率、精确率、召回率
- 特征重要性分析

**使用场景**: 数据分类、辅助标注

---

#### 4.3 聚类分析
```python
from sklearn.cluster import KMeans

# K-Means 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 聚类中心
centers = kmeans.cluster_centers_
```

**使用场景**: 数据分组、无监督标注辅助

---

### 📈 五、数据可视化类

#### 5.1 Matplotlib 基础图表
```python
import matplotlib.pyplot as plt

# 折线图
plt.plot(x, y)

# 散点图
plt.scatter(x, y)

# 柱状图
plt.bar(categories, values)

# 直方图
plt.hist(data, bins=20)

# 饼图
plt.pie(sizes, labels=labels)

plt.show()
```

**预置模板**: `matplotlib_basic`

**功能**:
- 折线图、散点图、柱状图
- 直方图、饼图
- 基础配置（标题、坐标轴、网格）

**使用场景**: 快速数据可视化、报告生成

---

#### 5.2 Seaborn 高级可视化
```python
import seaborn as sns

# 散点图（带分类）
sns.scatterplot(data=df, x='x', y='y', hue='category')

# 箱线图
sns.boxplot(data=df, x='category', y='value')

# 小提琴图
sns.violinplot(data=df, x='category', y='value')

# 热力图（相关性）
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# 分布图
sns.histplot(data=df, x='value', hue='category', kde=True)

# 成对关系图
sns.pairplot(df, hue='category')
```

**预置模板**: `seaborn_visualization`

**功能**:
- 统计图表（箱线图、小提琴图）
- 相关性热力图
- 分布可视化
- 多变量关系图

**使用场景**: 数据探索性分析、特征关系发现

---

#### 5.3 Plotly 交互式图表
```python
import plotly.express as px
import plotly.graph_objects as go

# 交互式散点图
fig = px.scatter(df, x='x', y='y', color='category')

# 交互式折线图
fig = px.line(df, x='date', y='value')

# 3D 散点图
fig = px.scatter_3d(df, x='x', y='y', z='z', color='category')

# 动态气泡图
fig = px.scatter(df, x='x', y='y', size='size',
                 animation_frame='time')

fig.show()
```

**预置模板**: `plotly_scatter`

**功能**:
- 交互式图表（缩放、悬停、选择）
- 3D 可视化
- 动画图表
- 仪表盘

**使用场景**: 交互式数据展示、复杂关系可视化

---

### 🔍 六、数据转换类

#### 6.1 分组聚合
```python
# 分组统计
df.groupby('category')['value'].mean()
df.groupby('category')['value'].sum()
df.groupby('category')['value'].count()

# 多重聚合
df.groupby('category').agg({
    'value1': ['mean', 'sum'],
    'value2': ['min', 'max']
})

# 透视表
pd.pivot_table(df,
               values='value',
               index='row_category',
               columns='col_category',
               aggfunc='mean')
```

**使用场景**: 数据汇总、多维度分析

---

#### 6.2 数据合并
```python
# 横向合并
pd.merge(df1, df2, on='key')              # 内连接
pd.merge(df1, df2, on='key', how='left')  # 左连接
pd.merge(df1, df2, on='key', how='outer') # 外连接

# 纵向合并
pd.concat([df1, df2], axis=0)             # 上下拼接
pd.concat([df1, df2], axis=1)             # 左右拼接
```

**使用场景**: 多数据源整合、数据拼接

---

#### 6.3 数据透视
```python
# 长转宽
df.pivot(index='id', columns='category', values='value')

# 宽转长
pd.melt(df, id_vars=['id'],
        value_vars=['col1', 'col2'])
```

**使用场景**: 数据格式转换、报表制作

---

## 📦 预置代码模板（通过 API 获取）

### 可用模板列表

| 模板名称 | 功能描述 | 主要库 |
|---------|---------|--------|
| `matplotlib_basic` | Matplotlib 基础图表绘制 | matplotlib, numpy |
| `plotly_scatter` | Plotly 交互式散点图 | plotly, pandas |
| `pandas_analysis` | Pandas 数据分析 | pandas, numpy |
| `sklearn_preprocessing` | 数据预处理（标准化/归一化） | sklearn, pandas |
| `sklearn_linear_regression` | 线性回归模型 | sklearn, matplotlib |
| `sklearn_classification` | 随机森林分类 | sklearn |
| `seaborn_visualization` | Seaborn 统计可视化 | seaborn, matplotlib |
| `scipy_stats` | 统计检验和分析 | scipy, matplotlib |

### 获取模板

**API 调用**:
```bash
# 获取所有模板
GET http://localhost:8000/templates

# 获取指定模板
GET http://localhost:8000/templates/pandas_analysis
```

**Python 代码**:
```python
import requests

# 获取所有模板
response = requests.get('http://localhost:8000/templates')
templates = response.json()

# 获取指定模板
response = requests.get('http://localhost:8000/templates/pandas_analysis')
template = response.json()
code = template['code']
```

---

## 🔧 自定义算子开发

用户可以自由编写自定义的数据治理代码，只要符合安全规范：

### ✅ 允许的操作

**数据处理**:
```python
import numpy as np
import pandas as pd
from scipy import stats

# 所有 pandas/numpy/scipy 操作
```

**机器学习**:
```python
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.model_selection import *
```

**可视化**:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
```

### ❌ 禁止的操作

```python
# 文件系统访问
open('file.txt')          # ❌

# 网络访问
import requests           # ❌
import urllib             # ❌

# 系统操作
import os                 # ❌
import sys                # ❌
import subprocess         # ❌

# 动态代码执行
eval('code')              # ❌
exec('code')              # ❌
```

---

## 🎯 按业务场景选择算子

### 场景 1: 数据质量检查
**推荐算子**:
- 基础统计分析 (`df.describe()`)
- 缺失值检测 (`df.isnull().sum()`)
- 重复值检测 (`df.duplicated().sum()`)
- 数据类型检查 (`df.dtypes`)

**模板**: `pandas_analysis`

---

### 场景 2: 数据清洗
**推荐算子**:
- 缺失值处理 (`fillna`, `dropna`)
- 重复值处理 (`drop_duplicates`)
- 异常值处理 (IQR, Z-Score)
- 类型转换 (`astype`)

**自定义代码**

---

### 场景 3: 数据标准化
**推荐算子**:
- StandardScaler (Z-Score 标准化)
- MinMaxScaler (Min-Max 归一化)

**模板**: `sklearn_preprocessing`

---

### 场景 4: 数据可视化
**推荐算子**:
- Matplotlib 基础图表
- Seaborn 统计图表
- Plotly 交互式图表

**模板**: `matplotlib_basic`, `seaborn_visualization`, `plotly_scatter`

---

### 场景 5: 辅助标注
**推荐算子**:
- K-Means 聚类
- 随机森林分类
- 特征重要性分析

**模板**: `sklearn_classification` + 自定义聚类代码

---

### 场景 6: 统计分析
**推荐算子**:
- 描述性统计 (`describe`)
- 相关性分析 (`corr`)
- 假设检验 (scipy.stats)
- 分布检验

**模板**: `scipy_stats`

---

## 📊 算子能力矩阵

| 算子类别 | 数据清洗 | 数据分析 | 数据转换 | 可视化 | 机器学习 |
|---------|---------|---------|---------|-------|---------|
| pandas 基础 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |
| numpy | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
| sklearn 预处理 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| sklearn 模型 | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| matplotlib | ⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐ |
| seaborn | ⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| plotly | ⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| scipy | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 🚀 快速使用

### 方式 1: 使用预置模板

```python
import requests

# 获取模板
response = requests.get('http://localhost:8000/templates/pandas_analysis')
code = response.json()['code']

# 执行模板
response = requests.post('http://localhost:8000/execute', json={
    'code': code,
    'datasets': {
        'data.csv': csv_content
    }
})
```

### 方式 2: 自定义代码

```python
custom_code = """
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('data.csv')

# 数据清洗
df = df.drop_duplicates()
df = df.fillna(df.mean())

# 统计分析
print("数据质量报告:")
print(f"总行数: {len(df)}")
print(f"缺失值: {df.isnull().sum().sum()}")
print(f"重复行: {df.duplicated().sum()}")

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
df.hist(bins=20)
plt.tight_layout()
plt.show()
"""

# 执行
response = requests.post('http://localhost:8000/execute', json={
    'code': custom_code,
    'datasets': {'data.csv': csv_content}
})
```

---

## 📚 相关文档

- **README.md** - 项目总览
- **DATASETS_USAGE.md** - 数据集传递功能详解
- **TEST_EXAMPLES.md** - 测试示例
- **PROJECT_SUMMARY.md** - 项目总结
- **API 文档** - http://localhost:8000/docs

---

**版本**: v1.2.0
**最后更新**: 2025-10-31
**总算子数量**: 60+ 个数据治理算子
