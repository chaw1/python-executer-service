# Python Executor Service - 测试代码集合

## 如何使用

### 方法 1：通过 API 测试（推荐）

```bash
# 启动服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 在浏览器访问 Swagger UI 测试
http://localhost:8000/docs
```

### 方法 2：直接运行 Python 脚本测试

```bash
python test_examples.py
```

---

## 测试代码示例

### 1. 基础测试 - Hello World

```python
print("Hello, Python Executor Service!")
print("当前时间:", "2025-10-31")

# 简单计算
x = 10
y = 20
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")
```

**预期输出**：
```
Hello, Python Executor Service!
当前时间: 2025-10-31
x + y = 30
x * y = 200
```

---

### 2. Pandas 数据分析

```python
import pandas as pd
import numpy as np

# 创建销售数据
data = {
    '日期': pd.date_range('2024-01-01', periods=10, freq='D'),
    '产品': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    '销量': [100, 150, 120, 80, 160, 110, 90, 140, 130, 95],
    '价格': [10, 15, 10, 20, 15, 10, 20, 15, 10, 20]
}

df = pd.DataFrame(data)
df['营收'] = df['销量'] * df['价格']

print("=== 销售数据概览 ===")
print(df)

print("\n=== 按产品分组统计 ===")
summary = df.groupby('产品').agg({
    '销量': 'sum',
    '营收': 'sum'
}).reset_index()
summary['平均单价'] = summary['营收'] / summary['销量']
print(summary)

print("\n=== 总体统计 ===")
print(f"总销量: {df['销量'].sum()}")
print(f"总营收: {df['营收'].sum()}")
print(f"平均日销量: {df['销量'].mean():.2f}")
```

---

### 3. NumPy 数学计算

```python
import numpy as np

# 创建矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print("矩阵 A:")
print(A)

print("\n矩阵 B:")
print(B)

print("\n矩阵加法 A + B:")
print(A + B)

print("\n矩阵乘法 A @ B:")
print(A @ B)

print("\n矩阵 A 的特征值:")
eigenvalues = np.linalg.eigvals(A.astype(float))
print(eigenvalues)

print("\n矩阵 A 的统计信息:")
print(f"均值: {A.mean():.2f}")
print(f"标准差: {A.std():.2f}")
print(f"最大值: {A.max()}")
print(f"最小值: {A.min()}")
```

---

### 4. Matplotlib 数据可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图表
plt.figure(figsize=(12, 5))

# 子图1: 正弦和余弦曲线
plt.subplot(1, 2, 1)
plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('三角函数')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: 散点图
np.random.seed(42)
x_scatter = np.random.randn(50)
y_scatter = 2 * x_scatter + np.random.randn(50) * 0.5

plt.subplot(1, 2, 2)
plt.scatter(x_scatter, y_scatter, alpha=0.6, s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('线性关系散点图')
plt.grid(True, alpha=0.3)

# 添加拟合线
z = np.polyfit(x_scatter, y_scatter, 1)
p = np.poly1d(z)
plt.plot(x_scatter, p(x_scatter), "r-", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
plt.legend()

plt.tight_layout()
plt.show()
```

---

### 5. Plotly 交互式图表

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 创建示例数据
np.random.seed(42)
df = pd.DataFrame({
    '月份': ['1月', '2月', '3月', '4月', '5月', '6月'] * 3,
    '产品': ['产品A'] * 6 + ['产品B'] * 6 + ['产品C'] * 6,
    '销售额': np.random.randint(50, 200, 18)
})

# 创建分组柱状图
fig = px.bar(df,
             x='月份',
             y='销售额',
             color='产品',
             barmode='group',
             title='各产品月度销售额对比',
             labels={'销售额': '销售额（万元）'},
             template='plotly_white')

fig.update_layout(
    xaxis_title="月份",
    yaxis_title="销售额（万元）",
    legend_title="产品类别",
    font=dict(size=12)
)

fig.show()

print("图表已生成！")
```

---

### 6. Scikit-learn 机器学习 - 回归

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据：房价预测
np.random.seed(42)
n_samples = 100

# 特征：房屋面积（平方米）
area = np.random.uniform(50, 200, n_samples)

# 目标：房价（万元）= 0.5 * 面积 + 噪声
price = 0.5 * area + 10 + np.random.normal(0, 5, n_samples)

# 准备数据
X = area.reshape(-1, 1)
y = price

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print("=== 模型性能 ===")
print(f"系数（每平米价格）: {model.coef_[0]:.2f} 万元/m²")
print(f"截距（基础价格）: {model.intercept_:.2f} 万元")
print(f"R² 分数: {r2_score(y_test, y_pred):.3f}")
print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"均方根误差 (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# 示例预测
print("\n=== 预测示例 ===")
test_areas = [80, 120, 150]
for area_val in test_areas:
    predicted_price = model.predict([[area_val]])[0]
    print(f"{area_val}m² 房屋预测价格: {predicted_price:.2f} 万元")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.6, label='实际价格', s=50)
plt.scatter(X_test, y_pred, alpha=0.6, label='预测价格', s=50, marker='x')
plt.plot(X_test, y_pred, 'r-', linewidth=2, label='回归线')
plt.xlabel('房屋面积 (m²)')
plt.ylabel('价格（万元）')
plt.title('房价预测模型')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 7. Scikit-learn 机器学习 - 分类

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

# 生成示例数据：客户流失预测
np.random.seed(42)
n_samples = 500

# 特征
age = np.random.randint(18, 70, n_samples)
tenure = np.random.randint(1, 120, n_samples)  # 月数
monthly_charges = np.random.uniform(20, 200, n_samples)
total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_samples)

# 目标：流失概率（基于一些简单规则 + 噪声）
churn_prob = (
    (age < 30) * 0.3 +
    (tenure < 12) * 0.4 +
    (monthly_charges > 150) * 0.2 +
    np.random.uniform(0, 0.3, n_samples)
)
churn = (churn_prob > 0.6).astype(int)

# 创建 DataFrame
df = pd.DataFrame({
    '年龄': age,
    '使用月数': tenure,
    '月费用': monthly_charges,
    '总费用': total_charges,
    '是否流失': churn
})

print("=== 数据概览 ===")
print(df.head(10))
print(f"\n流失客户比例: {churn.mean():.2%}")

# 准备数据
X = df[['年龄', '使用月数', '月费用', '总费用']]
y = df['是否流失']

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 训练随机森林模型
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估
print("\n=== 模型性能 ===")
print(f"准确率: {accuracy_score(y_test, y_pred):.3f}")

print("\n交叉验证分数:")
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print(f"平均准确率: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

print("\n分类报告:")
print(classification_report(y_test, y_pred,
                          target_names=['未流失', '流失']))

print("\n特征重要性:")
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': clf.feature_importances_
}).sort_values('重要性', ascending=False)
print(feature_importance)

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

---

### 8. Scikit-learn 数据预处理完整流程

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 创建混合类型的数据集
np.random.seed(42)
n = 200

df = pd.DataFrame({
    '年龄': np.random.randint(20, 60, n),
    '收入': np.random.uniform(3000, 20000, n),
    '信用分数': np.random.randint(300, 850, n),
    '城市': np.random.choice(['北京', '上海', '广州', '深圳'], n),
    '教育程度': np.random.choice(['高中', '本科', '硕士', '博士'], n),
    '贷款批准': np.random.choice([0, 1], n)
})

print("=== 原始数据 ===")
print(df.head(10))
print(f"\n数据形状: {df.shape}")
print(f"\n数据类型:\n{df.dtypes}")

# 1. 分离数值和类别特征
numeric_features = ['年龄', '收入', '信用分数']
categorical_features = ['城市', '教育程度']

print("\n=== 数值特征统计 ===")
print(df[numeric_features].describe())

# 2. 类别编码
print("\n=== 类别特征编码 ===")
df_encoded = df.copy()

for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col + '_编码'] = le.fit_transform(df[col])
    print(f"\n{col} 映射:")
    for i, label in enumerate(le.classes_):
        print(f"  {label} -> {i}")

# 3. 数值特征标准化
print("\n=== 数值特征标准化 ===")

# Z-Score 标准化
scaler_standard = StandardScaler()
df_standard = df_encoded.copy()
df_standard[numeric_features] = scaler_standard.fit_transform(
    df_encoded[numeric_features]
)

print("\nZ-Score 标准化后的统计:")
print(df_standard[numeric_features].describe())

# Min-Max 归一化
scaler_minmax = MinMaxScaler()
df_minmax = df_encoded.copy()
df_minmax[numeric_features] = scaler_minmax.fit_transform(
    df_encoded[numeric_features]
)

print("\nMin-Max 归一化后的统计:")
print(df_minmax[numeric_features].describe())

# 4. 准备建模数据
feature_cols = numeric_features + [col + '_编码' for col in categorical_features]
X = df_standard[feature_cols]
y = df_encoded['贷款批准']

print("\n=== 最终特征矩阵 ===")
print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"\n特征列: {list(X.columns)}")
print(f"\n前5行数据:")
print(X.head())
```

---

### 9. Seaborn 高级统计可视化

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置样式
sns.set_theme(style="whitegrid", palette="husl")

# 生成示例数据：学生成绩分析
np.random.seed(42)
n_students = 150

data = []
for subject in ['数学', '英语', '物理']:
    for class_name in ['A班', 'B班', 'C班']:
        n = n_students // 3
        if subject == '数学':
            scores = np.random.normal(75, 15, n)
        elif subject == '英语':
            scores = np.random.normal(80, 12, n)
        else:
            scores = np.random.normal(70, 18, n)

        # 添加班级差异
        if class_name == 'A班':
            scores += 5
        elif class_name == 'C班':
            scores -= 5

        scores = np.clip(scores, 0, 100)

        for score in scores:
            data.append({
                '科目': subject,
                '班级': class_name,
                '分数': score
            })

df = pd.DataFrame(data)

print("=== 数据概览 ===")
print(df.groupby(['科目', '班级'])['分数'].agg(['mean', 'std', 'count']))

# 创建多个可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 箱线图
sns.boxplot(data=df, x='科目', y='分数', hue='班级', ax=axes[0, 0])
axes[0, 0].set_title('各科目各班级分数分布（箱线图）', fontsize=14, pad=10)
axes[0, 0].set_ylabel('分数', fontsize=12)
axes[0, 0].legend(title='班级')

# 2. 小提琴图
sns.violinplot(data=df, x='科目', y='分数', hue='班级',
               split=False, ax=axes[0, 1])
axes[0, 1].set_title('各科目各班级分数分布（小提琴图）', fontsize=14, pad=10)
axes[0, 1].set_ylabel('分数', fontsize=12)
axes[0, 1].legend(title='班级')

# 3. 分数分布直方图
sns.histplot(data=df, x='分数', hue='科目', kde=True,
             ax=axes[1, 0], bins=20, alpha=0.6)
axes[1, 0].set_title('各科目分数分布', fontsize=14, pad=10)
axes[1, 0].set_xlabel('分数', fontsize=12)
axes[1, 0].set_ylabel('频数', fontsize=12)

# 4. 分组条形图（平均分）
avg_scores = df.groupby(['科目', '班级'])['分数'].mean().reset_index()
sns.barplot(data=avg_scores, x='科目', y='分数', hue='班级', ax=axes[1, 1])
axes[1, 1].set_title('各科目各班级平均分对比', fontsize=14, pad=10)
axes[1, 1].set_ylabel('平均分', fontsize=12)
axes[1, 1].legend(title='班级')

plt.tight_layout()
plt.show()

print("\n图表已生成！")
```

---

### 10. SciPy 统计分析

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# 生成两组数据：A/B 测试
np.random.seed(42)

# 对照组（原版本）
group_a = np.random.normal(100, 15, 100)  # 平均转化率 100
# 实验组（新版本）
group_b = np.random.normal(105, 15, 100)  # 平均转化率 105

print("=== A/B 测试数据分析 ===")
print(f"\n对照组 (A):")
print(f"  样本数: {len(group_a)}")
print(f"  均值: {group_a.mean():.2f}")
print(f"  标准差: {group_a.std():.2f}")
print(f"  中位数: {np.median(group_a):.2f}")

print(f"\n实验组 (B):")
print(f"  样本数: {len(group_b)}")
print(f"  均值: {group_b.mean():.2f}")
print(f"  标准差: {group_b.std():.2f}")
print(f"  中位数: {np.median(group_b):.2f}")

# 1. 正态性检验
print("\n=== 正态性检验 ===")
stat_a, p_a = stats.normaltest(group_a)
stat_b, p_b = stats.normaltest(group_b)

print(f"对照组 p值: {p_a:.4f} - {'符合正态分布' if p_a > 0.05 else '不符合正态分布'}")
print(f"实验组 p值: {p_b:.4f} - {'符合正态分布' if p_b > 0.05 else '不符合正态分布'}")

# 2. 方差齐性检验
print("\n=== 方差齐性检验 ===")
stat_var, p_var = stats.levene(group_a, group_b)
print(f"p值: {p_var:.4f} - {'方差齐性' if p_var > 0.05 else '方差不齐'}")

# 3. t检验（独立样本）
print("\n=== 独立样本 t 检验 ===")
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t统计量: {t_stat:.4f}")
print(f"p值: {p_value:.4f}")

if p_value < 0.05:
    print(f"结论: 两组存在显著差异 (p < 0.05)")
    improvement = ((group_b.mean() - group_a.mean()) / group_a.mean()) * 100
    print(f"提升幅度: {improvement:.2f}%")
else:
    print(f"结论: 两组无显著差异 (p >= 0.05)")

# 4. 效应量（Cohen's d）
cohens_d = (group_b.mean() - group_a.mean()) / np.sqrt(
    (group_a.std()**2 + group_b.std()**2) / 2
)
print(f"\nCohen's d (效应量): {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    print("效应量: 小")
elif abs(cohens_d) < 0.5:
    print("效应量: 中")
else:
    print("效应量: 大")

# 5. 置信区间
print("\n=== 95% 置信区间 ===")
ci_a = stats.t.interval(0.95, len(group_a)-1,
                        loc=group_a.mean(),
                        scale=stats.sem(group_a))
ci_b = stats.t.interval(0.95, len(group_b)-1,
                        loc=group_b.mean(),
                        scale=stats.sem(group_b))

print(f"对照组: [{ci_a[0]:.2f}, {ci_a[1]:.2f}]")
print(f"实验组: [{ci_b[0]:.2f}, {ci_b[1]:.2f}]")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 分布对比
axes[0].hist(group_a, bins=20, alpha=0.6, label='对照组 A', density=True)
axes[0].hist(group_b, bins=20, alpha=0.6, label='实验组 B', density=True)
axes[0].axvline(group_a.mean(), color='blue', linestyle='--',
                linewidth=2, label=f'A均值: {group_a.mean():.1f}')
axes[0].axvline(group_b.mean(), color='orange', linestyle='--',
                linewidth=2, label=f'B均值: {group_b.mean():.1f}')
axes[0].set_xlabel('值')
axes[0].set_ylabel('密度')
axes[0].set_title('分布对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 箱线图对比
bp = axes[1].boxplot([group_a, group_b],
                      labels=['对照组 A', '实验组 B'],
                      patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
axes[1].set_ylabel('值')
axes[1].set_title('箱线图对比')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

---

### 11. 综合案例：完整的数据分析工作流

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# 设置随机种子
np.random.seed(42)
sns.set_theme(style="whitegrid")

print("="*70)
print("完整数据分析工作流示例：销售预测")
print("="*70)

# 第1步：数据生成
print("\n【第1步】生成示例数据...")
n = 200
dates = pd.date_range('2023-01-01', periods=n, freq='D')

# 模拟销售数据，包含季节性和趋势
trend = np.linspace(100, 150, n)
seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 30)
noise = np.random.normal(0, 10, n)

df = pd.DataFrame({
    '日期': dates,
    '销售额': trend + seasonality + noise,
    '广告支出': np.random.uniform(10, 50, n),
    '促销活动': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    '天气': np.random.choice(['晴', '阴', '雨'], n),
    '工作日': [1 if d.weekday() < 5 else 0 for d in dates]
})

print(f"数据集大小: {df.shape}")
print(f"\n前5行数据:")
print(df.head())

# 第2步：探索性数据分析
print("\n【第2步】探索性数据分析...")
print("\n基本统计信息:")
print(df[['销售额', '广告支出']].describe())

print("\n销售额按促销活动分组:")
print(df.groupby('促销活动')['销售额'].agg(['mean', 'std', 'count']))

print("\n销售额按天气分组:")
print(df.groupby('天气')['销售额'].agg(['mean', 'std', 'count']))

# 第3步：特征工程
print("\n【第3步】特征工程...")

# 添加时间特征
df['月份'] = df['日期'].dt.month
df['周几'] = df['日期'].dt.dayofweek

# 天气编码
weather_map = {'晴': 2, '阴': 1, '雨': 0}
df['天气_编码'] = df['天气'].map(weather_map)

# 选择特征
feature_cols = ['广告支出', '促销活动', '工作日', '天气_编码', '月份', '周几']
X = df[feature_cols]
y = df['销售额']

print(f"特征列: {feature_cols}")
print(f"特征矩阵形状: {X.shape}")

# 第4步：数据标准化
print("\n【第4步】数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

print("标准化完成")
print(f"标准化后的均值: {X_scaled.mean().mean():.6f}")
print(f"标准化后的标准差: {X_scaled.std().mean():.6f}")

# 第5步：模型训练
print("\n【第5步】训练预测模型...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 第6步：模型评估
print("\n【第6步】模型评估...")
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_test - y_pred))

print(f"R² 分数: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# 特征重要性
print("\n特征重要性排序:")
feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)
print(feature_importance)

# 第7步：可视化
print("\n【第7步】生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 时间序列趋势
axes[0, 0].plot(df['日期'], df['销售额'], linewidth=1.5, alpha=0.7)
axes[0, 0].set_xlabel('日期')
axes[0, 0].set_ylabel('销售额')
axes[0, 0].set_title('销售额时间序列')
axes[0, 0].grid(True, alpha=0.3)

# 2. 预测 vs 实际
axes[0, 1].scatter(y_test, y_pred, alpha=0.6, s=30)
axes[0, 1].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='理想预测')
axes[0, 1].set_xlabel('实际销售额')
axes[0, 1].set_ylabel('预测销售额')
axes[0, 1].set_title(f'预测效果 (R²={r2:.3f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 特征重要性
axes[1, 0].barh(feature_importance['特征'], feature_importance['重要性'])
axes[1, 0].set_xlabel('重要性')
axes[1, 0].set_title('特征重要性')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. 销售额分布
axes[1, 1].hist(df['销售额'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(df['销售额'].mean(), color='r',
                   linestyle='--', linewidth=2,
                   label=f'均值: {df["销售额"].mean():.1f}')
axes[1, 1].set_xlabel('销售额')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title('销售额分布')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("分析完成！")
print("="*70)
```

---

## 快速 API 测试（curl 命令）

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 执行简单代码
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(\"Hello World\")\nprint(2 + 2)",
    "timeout": 30
  }'

# 3. 获取代码模板
curl http://localhost:8000/templates

# 4. 获取特定模板
curl http://localhost:8000/templates/sklearn_preprocessing

# 5. 验证代码
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import pandas as pd\nprint(\"test\")"
  }'
```

---

## 批量测试脚本

见 `test_examples.py` 文件，可以一次性运行所有测试。
