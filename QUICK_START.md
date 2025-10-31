# Python Executor Service - 快速开始

## 🚀 立即测试

### 方法 1：运行测试脚本（最简单）

```bash
# 运行完整测试套件
python test_examples.py
```

**预期结果**：所有 12 个测试通过 ✓

---

### 方法 2：启动服务并使用 API

```bash
# 1. 启动服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 2. 在浏览器打开
http://localhost:8000/docs

# 3. 使用 Swagger UI 测试
```

---

### 方法 3：curl 命令行测试

```bash
# 健康检查
curl http://localhost:8000/health

# 执行代码
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({\"x\": [1,2,3], \"y\": [4,5,6]})\nprint(df)",
    "timeout": 30
  }'

# 获取所有代码模板
curl http://localhost:8000/templates
```

---

## 📚 常用代码示例

### 1. 数据分析（Pandas）

```python
import pandas as pd
import numpy as np

# 创建数据
df = pd.DataFrame({
    '产品': ['A', 'B', 'C'],
    '销量': [100, 150, 80],
    '价格': [10, 15, 20]
})

# 计算营收
df['营收'] = df['销量'] * df['价格']

# 统计分析
print(df.describe())
print(f"总营收: {df['营收'].sum()}")
```

---

### 2. 机器学习（Scikit-learn）

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# 生成数据
X = np.random.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# 预测
accuracy = clf.score(X_test, y_test)
print(f"准确率: {accuracy:.3f}")
```

---

### 3. 数据预处理（你的场景）

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

# 创建数据
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'feature3': [100, 200, 300, 400, 500]
})

# Z-Score 标准化
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

# Min-Max 归一化
minmax = MinMaxScaler()
df_normalized = pd.DataFrame(
    minmax.fit_transform(df),
    columns=df.columns
)

print("标准化后:")
print(df_scaled.describe())
```

---

### 4. 数据可视化（Matplotlib）

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.title('正弦曲线')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
```

---

### 5. 统计分析（SciPy）

```python
from scipy import stats
import numpy as np

# A/B 测试
group_a = np.random.normal(100, 15, 50)
group_b = np.random.normal(105, 15, 50)

# t检验
t_stat, p_value = stats.ttest_ind(group_a, group_b)

print(f"p值: {p_value:.4f}")
if p_value < 0.05:
    print("结论: 存在显著差异")
else:
    print("结论: 无显著差异")
```

---

### 6. 高级可视化（Seaborn）

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 创建数据
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'value': np.random.randn(100) * 10 + 50
})

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='value')
plt.title('分类数据分布')
plt.show()
```

---

## 🎯 支持的库列表

### 数据处理
- ✅ **NumPy** - 数值计算
- ✅ **Pandas** - 数据分析
- ✅ **SciPy** - 科学计算

### 机器学习
- ✅ **Scikit-learn** - 完整的机器学习工作流
  - 预处理、模型训练、评估

### 可视化
- ✅ **Matplotlib** - 基础绘图
- ✅ **Plotly** - 交互式图表
- ✅ **Seaborn** - 统计可视化

---

## 📖 完整文档

- **TEST_EXAMPLES.md** - 详细测试代码（11个完整示例）
- **test_examples.py** - 可运行的测试脚本
- **UPGRADE_TODO.md** - 未来优化计划
- **UPGRADE_SUMMARY.md** - 升级完成总结
- **README.md** - 完整使用文档

---

## 🔥 获取代码模板

服务提供 9 个预定义模板：

```bash
# 获取所有模板
curl http://localhost:8000/templates

# 模板列表
# 1. matplotlib_basic - Matplotlib 基础图表
# 2. plotly_scatter - Plotly 散点图
# 3. pandas_analysis - Pandas 数据分析
# 4. sklearn_preprocessing - Sklearn 数据预处理
# 5. sklearn_linear_regression - Sklearn 线性回归
# 6. sklearn_classification - Sklearn 分类
# 7. seaborn_visualization - Seaborn 可视化
# 8. scipy_stats - SciPy 统计分析
```

---

## ⚡ 快速 API 测试

### Python requests

```python
import requests
import json

url = "http://localhost:8000/execute"

code = """
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print(df)
"""

response = requests.post(url, json={
    "code": code,
    "timeout": 30
})

result = response.json()
print(result['output']['stdout'])
```

### JavaScript fetch

```javascript
const url = "http://localhost:8000/execute";

const code = `
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print(df)
`;

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ code: code, timeout: 30 })
})
.then(res => res.json())
.then(data => console.log(data.output.stdout));
```

---

## ⚠️ 限制和注意事项

### ❌ 不支持的功能

1. **文件系统访问**
   ```python
   # ❌ 不支持
   df = pd.read_csv('data.csv')

   # ✅ 替代方案：使用内存数据
   df = pd.DataFrame({...})
   ```

2. **网络请求**
   ```python
   # ❌ 不支持
   import requests
   response = requests.get('http://...')
   ```

3. **系统调用**
   ```python
   # ❌ 不支持
   import os
   os.system('ls')
   ```

### ✅ 解决方案

- 查看 **UPGRADE_TODO.md** 了解文件上传功能计划
- 使用内存数据替代文件读取
- 所有数据处理在内存中完成

---

## 🆘 遇到问题？

### 1. 代码缩进问题
服务会自动处理缩进，直接粘贴即可！

### 2. 库不支持
检查库列表，或查看 UPGRADE_TODO.md 计划

### 3. 执行超时
- 调整 timeout 参数（默认30秒，最大60秒）
- 优化代码性能

### 4. 查看详细日志
```bash
python -m uvicorn app.main:app --log-level debug
```

---

## 📞 帮助

- **详细示例**：见 `TEST_EXAMPLES.md`
- **API 文档**：http://localhost:8000/docs
- **升级计划**：见 `UPGRADE_TODO.md`

---

**版本**：v1.1.0
**状态**：✅ 生产就绪
**测试通过率**：100% (12/12)
