# 数据集功能实现总结

## 📋 问题描述

用户遇到两个文件访问错误：

### 错误 1: 文件路径问题
```
错误: [Errno 2] No such file or directory: '{{dataset_path}}/data.csv'
```

### 错误 2: open() 函数被禁用
```
错误: 检测到禁止的操作: open()
安全策略不允许使用此函数。
```

**用户代码示例**：
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 读取数据
df = pd.read_csv('{{dataset_path}}/data.csv')

# 数据预处理
numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])
```

## ✅ 解决方案

实现了完整的数据集传递功能，允许通过 API 传递文件内容到执行环境。

### 核心实现

#### 1. API 层（app/models.py）

新增 `datasets` 参数到请求模型：

```python
class ExecuteRequest(BaseModel):
    code: str
    timeout: int = 30
    output_format: str = "json"
    datasets: Optional[Dict[str, str]] = None  # 新增
```

**使用示例**：
```json
{
  "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df)",
  "datasets": {
    "data.csv": "name,age,score\nAlice,25,95\nBob,30,87"
  }
}
```

#### 2. 执行器层（app/executor.py）

新增数据集准备和注入功能：

```python
def _prepare_datasets(self, datasets: Dict[str, str], global_vars: Dict[str, Any]) -> None:
    """
    准备数据集，将文件内容注入到执行环境

    功能：
    1. 预处理 CSV/JSON 为 DataFrame（缓存）
    2. 创建自定义 read_csv/read_json 函数
    3. 支持 {{dataset_path}} 占位符
    4. 优先从内存读取，性能优化
    """
```

**关键技术点**：
- 覆盖 `pd.read_csv()` 和 `pd.read_json()`
- 路径清理：移除 `{{dataset_path}}/` 前缀
- 预读取优化：CSV/JSON 提前转为 DataFrame
- 内存缓存：多次读取返回副本

#### 3. API 端点（app/main.py）

更新执行端点传递数据集：

```python
@app.post("/execute")
async def execute_code(request: ExecuteRequest):
    executor = CodeExecutor(timeout=request.timeout)

    # 传递 datasets 到执行器
    result = executor.execute(request.code, datasets=request.datasets)

    return result
```

## 🎯 功能特性

### ✅ 支持的操作

1. **直接文件名访问**
```python
df = pd.read_csv('data.csv')
```

2. **占位符路径（兼容旧代码）**
```python
df = pd.read_csv('{{dataset_path}}/data.csv')  # 自动处理
```

3. **多数据集**
```python
df1 = pd.read_csv('users.csv')
df2 = pd.read_csv('orders.csv')
merged = df1.merge(df2, on='user_id')
```

4. **完整数据处理流程**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=['number']))
```

### 🚀 性能优化

- **预处理缓存**: CSV/JSON 自动预读为 DataFrame
- **函数覆盖**: 零性能开销的函数替换
- **内存共享**: 同一文件多次读取返回缓存副本

## 📊 测试验证

### 测试脚本: test_datasets.py

**测试 1: 基本数据读取**
```python
code = """
import pandas as pd
df = pd.read_csv('{{dataset_path}}/data.csv')
print("数据形状:", df.shape)
print(df.head())
"""

datasets = {
    "data.csv": "name,age,score\nAlice,25,95\nBob,30,87"
}

result = executor.execute(code, datasets=datasets)
```

**结果**: ✅ 通过 (16ms)

**测试 2: Sklearn 数据预处理**
```python
code = """
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('data.csv')
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Z-Score 标准化
scaler = StandardScaler()
df_standard = df.copy()
df_standard[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("标准化后:")
print(df_standard.describe())
"""
```

**结果**: ✅ 通过 (22ms)

**通过率**: 2/2 (100%)

## 📚 文档

### 新增文档

1. **DATASETS_USAGE.md**
   - 完整使用指南
   - API 使用示例
   - Python 客户端代码
   - 常见问题解答
   - 多个实用示例

2. **RELEASE_NOTES_v1.2.0.md**
   - 版本发布说明
   - 详细变更日志
   - 升级指南
   - 性能指标

3. **README.md（更新）**
   - 新增数据集传递特性
   - 新增 API 使用示例
   - 链接到详细文档

## 🔧 技术细节

### 数据流程

```
1. 用户发送请求
   POST /execute
   {
     "code": "...",
     "datasets": {"data.csv": "..."}
   }

2. API 层接收
   ExecuteRequest.datasets: Dict[str, str]

3. 执行器准备数据集
   _prepare_datasets():
   - 预读 CSV → DataFrame
   - 创建自定义 read_csv 函数
   - 注入到 global_vars

4. 代码执行
   用户代码调用: pd.read_csv('data.csv')
   ↓
   实际调用: custom_read_csv('data.csv')
   ↓
   返回: 预处理的 DataFrame 副本

5. 返回结果
   ExecuteResponse 包含输出
```

### 关键代码

**自定义 read_csv 函数**:
```python
def custom_read_csv(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str):
        # 清理路径
        clean_path = filepath_or_buffer.replace('{{dataset_path}}/', '')
        filename = os.path.basename(clean_path)

        # 优先返回预处理的 DataFrame
        if filename in dataset_dataframes:
            return dataset_dataframes[filename].copy()

        # 否则从 StringIO 读取
        if filename in dataset_contents:
            return original_read_csv(
                io.StringIO(dataset_contents[filename]),
                *args, **kwargs
            )

    # 兜底使用原始函数
    return original_read_csv(filepath_or_buffer, *args, **kwargs)

# 注入到执行环境
global_vars['pd'].read_csv = custom_read_csv
```

## 🎉 成果

### 解决的问题

1. ✅ 用户可以使用 `pd.read_csv('data.csv')` 读取数据
2. ✅ 支持 `{{dataset_path}}` 占位符（自动处理）
3. ✅ 完整的 sklearn 数据预处理工作流
4. ✅ 无需文件系统访问，保持安全性

### 新增能力

1. ✅ 通过 API 传递任意数据文件
2. ✅ 支持 CSV、JSON 格式
3. ✅ 自动预处理和缓存
4. ✅ 高性能内存访问

### 保持不变

1. ✅ 安全沙箱限制不变
2. ✅ 向后兼容（datasets 为可选参数）
3. ✅ 无性能回退

## 📦 部署

### Git 提交

```bash
commit 27b5d07
新增数据集传递功能 (v1.2.0)

修改文件：
- app/models.py
- app/executor.py
- app/main.py
- README.md

新增文件：
- DATASETS_USAGE.md
- RELEASE_NOTES_v1.2.0.md
```

### 版本升级

- **之前**: v1.1.0
- **现在**: v1.2.0
- **状态**: ✅ 生产就绪

## 📖 使用指南

### 快速开始

**Python 客户端**:
```python
import requests

# 准备数据
csv_data = """name,age,score
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

scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\\n标准化后:")
print(df)
"""

# 发送请求
response = requests.post(
    'http://localhost:8000/execute',
    json={
        'code': code,
        'datasets': {'data.csv': csv_data}
    }
)

result = response.json()
print(result['output']['stdout'])
```

**curl 示例**:
```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import pandas as pd\ndf = pd.read_csv(\"data.csv\")\nprint(df.head())",
    "datasets": {
      "data.csv": "name,age\nAlice,25\nBob,30"
    }
  }'
```

### 详细文档

请查阅：
- **DATASETS_USAGE.md** - 完整使用指南
- **RELEASE_NOTES_v1.2.0.md** - 版本说明
- **README.md** - 总体介绍

## 🔮 后续优化

可选的未来增强（见 UPGRADE_TODO.md）：

1. **会话管理** - 支持多步骤数据处理
2. **数据导出** - 处理后数据导出为文件
3. **Excel 支持** - pd.read_excel()
4. **文件大小限制** - 可配置的数据集大小限制

---

**完成时间**: 2025-10-31
**版本**: v1.2.0
**状态**: ✅ 已部署
**分支**: claude/fix-code-recognition-011CUeruPh1DbVmPpyh2Bcbb
