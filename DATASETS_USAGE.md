# 数据集传递功能使用指南

## 概述

Python Executor Service 现在支持通过 API 传递数据集内容，无需依赖文件系统即可使用 `pd.read_csv()` 等文件读取函数。

## 问题背景

由于安全沙箱限制，服务不允许直接的文件系统访问：
- ❌ `pd.read_csv('/path/to/file.csv')` - 文件不存在
- ❌ `open('file.txt', 'r')` - 禁止的操作

## 解决方案

通过 API 请求传递数据集内容，服务会自动注入到执行环境中。

## API 使用方法

### 请求格式

```json
{
  "code": "你的 Python 代码",
  "timeout": 30,
  "datasets": {
    "文件名1": "文件内容1",
    "文件名2": "文件内容2"
  }
}
```

### 示例 1：基本 CSV 读取

**请求**：
```json
{
  "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
  "datasets": {
    "data.csv": "name,age,score\nAlice,25,95\nBob,30,87\nCharlie,22,92"
  }
}
```

**代码中使用**：
```python
import pandas as pd

# 直接使用文件名
df = pd.read_csv('data.csv')
print(df.head())
```

### 示例 2：支持 {{dataset_path}} 占位符

**代码**：
```python
import pandas as pd

# 兼容带路径占位符的代码
df = pd.read_csv('{{dataset_path}}/data.csv')
print(df.shape)
```

服务会自动识别并移除 `{{dataset_path}}/` 前缀，直接读取对应的数据集。

### 示例 3：数据预处理（完整示例）

**请求**：
```json
{
  "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\n\ndf = pd.read_csv('data.csv')\nnumeric_cols = df.select_dtypes(include=[np.number]).columns\n\nscaler = StandardScaler()\ndf_scaled = df.copy()\ndf_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])\n\nprint('标准化后:')\nprint(df_scaled.describe())",
  "datasets": {
    "data.csv": "feature1,feature2,feature3\n1,10,100\n2,20,200\n3,30,300\n4,40,400\n5,50,500"
  }
}
```

## 支持的文件格式

### CSV 文件
```python
df = pd.read_csv('data.csv')
```

### JSON 文件
```python
df = pd.read_json('data.json')
```

数据集会自动根据文件扩展名识别并预处理为 DataFrame。

## 工作原理

1. **数据注入**：服务接收 `datasets` 参数后，会将文件内容存储在内存中
2. **预处理**：对于 CSV/JSON 文件，会提前读取为 DataFrame
3. **函数覆盖**：自动覆盖 `pd.read_csv()` 和 `pd.read_json()`，优先从内存读取
4. **路径处理**：自动识别并移除 `{{dataset_path}}` 占位符

## Python 代码示例

### 使用 requests 库调用

```python
import requests

# 准备数据
csv_content = """name,age,score
Alice,25,95
Bob,30,87
Charlie,22,92
"""

# 准备代码
code = """
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')
print("原始数据:")
print(df)

# 数据预处理
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\\n标准化后:")
print(df)
"""

# 发送请求
response = requests.post(
    'http://localhost:8000/execute',
    json={
        'code': code,
        'datasets': {
            'data.csv': csv_content
        }
    }
)

result = response.json()
print(result['output']['stdout'])
```

### 使用 curl 调用

```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import pandas as pd\ndf = pd.read_csv(\"data.csv\")\nprint(df.head())",
    "datasets": {
      "data.csv": "name,age,score\nAlice,25,95\nBob,30,87"
    }
  }'
```

## 多数据集支持

可以同时传递多个数据集：

```json
{
  "code": "import pandas as pd\ndf1 = pd.read_csv('users.csv')\ndf2 = pd.read_csv('orders.csv')\nmerged = df1.merge(df2, on='user_id')\nprint(merged)",
  "datasets": {
    "users.csv": "user_id,name,age\n1,Alice,25\n2,Bob,30",
    "orders.csv": "order_id,user_id,amount\n101,1,99.99\n102,2,49.99"
  }
}
```

## 限制说明

### ✅ 支持的操作
- `pd.read_csv('filename.csv')`
- `pd.read_csv('{{dataset_path}}/filename.csv')`
- `pd.read_json('filename.json')`
- 任意数据处理和机器学习操作

### ❌ 不支持的操作
- `open('filename.txt', 'r')` - 仍然被安全策略禁止
- 直接文件系统访问 - 出于安全考虑

### 替代方案
如需读取非结构化文本文件，可以：
1. 通过 datasets 传递内容
2. 在代码中直接使用变量访问：

```python
# 在全局环境中可以访问
file_content = __datasets_content__['data.txt']
print(file_content)
```

## 性能优化

- **预处理**：CSV/JSON 文件会在执行前预读为 DataFrame，避免重复解析
- **内存共享**：多次读取同一文件会返回预处理的副本，性能更高
- **自动缓存**：DataFrame 自动缓存在执行环境中

## 安全说明

1. **内存隔离**：每次执行都是独立的，数据不会在请求间共享
2. **大小限制**：建议单个数据集不超过 10MB
3. **沙箱保护**：数据集仍在安全沙箱内处理，无文件系统访问

## 常见问题

### Q: 如何传递大文件？
A: 对于大数据集，建议：
1. 压缩数据（CSV 格式已经很紧凑）
2. 只传递必要的列和行
3. 考虑数据采样

### Q: 支持 Excel 文件吗？
A: 目前支持 CSV 和 JSON。Excel 文件可以先转换为 CSV 格式。

### Q: 可以保存处理后的数据吗？
A: 处理后的 DataFrame 会在响应中返回（通过 `dataframes` 字段），可以在客户端保存。

### Q: 错误提示 "文件不存在" 怎么办？
A: 检查：
1. `datasets` 参数是否正确传递
2. 代码中的文件名是否与 datasets 的 key 匹配
3. 文件扩展名是否正确（.csv, .json）

## 完整示例

完整的 Python 客户端示例：

```python
import requests
import json

class PythonExecutorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def execute_with_datasets(self, code, datasets=None, timeout=30):
        """执行代码并传递数据集"""
        payload = {
            "code": code,
            "timeout": timeout
        }
        if datasets:
            payload["datasets"] = datasets

        response = requests.post(
            f"{self.base_url}/execute",
            json=payload
        )
        return response.json()

# 使用示例
client = PythonExecutorClient()

# 准备数据
csv_data = """name,age,score
Alice,25,95
Bob,30,87
Charlie,22,92
David,28,88
"""

# 准备代码
code = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 读取数据
df = pd.read_csv('{{dataset_path}}/data.csv')

# 选择数值列
numeric_cols = df.select_dtypes(include=[np.number]).columns

print("原始数据统计:")
print(df[numeric_cols].describe())

# Z-Score 标准化
scaler = StandardScaler()
df_standard = df.copy()
df_standard[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\\nZ-Score 标准化后:")
print(df_standard[numeric_cols].describe())

# Min-Max 归一化
scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numeric_cols] = scaler_minmax.fit_transform(df[numeric_cols])

print("\\nMin-Max 归一化后:")
print(df_minmax[numeric_cols].describe())
"""

# 执行
result = client.execute_with_datasets(
    code=code,
    datasets={"data.csv": csv_data}
)

# 打印结果
if result['status'] == 'success':
    print("✓ 执行成功")
    print(result['output']['stdout'])
else:
    print("✗ 执行失败")
    print(result['error'])
```

## 更新日志

- **v1.2.0** (2025-10-31): 新增数据集传递功能
  - 支持通过 API 传递文件内容
  - 自动覆盖 pd.read_csv/read_json
  - 支持 {{dataset_path}} 占位符
  - 预处理和缓存优化

---

**相关文档**：
- [README.md](README.md) - 服务总体介绍
- [QUICK_START.md](QUICK_START.md) - 快速开始指南
- [API 文档](http://localhost:8000/docs) - 完整 API 文档
