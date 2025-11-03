# Python Executor Service - 升级建议 v2.0

## 📊 当前状态评估（v1.2.0+）

### ✅ 已完成的功能

#### 1. 核心功能
- ✅ **安全沙箱**：RestrictedPython + 白名单机制
- ✅ **智能缩进**：使用栈算法处理代码块层级
- ✅ **数据集传递**：datasets + selected_files 双格式支持
- ✅ **预加载变量**：preloadedVariables 支持
- ✅ **自动函数覆盖**：pd.read_csv/read_json 内存读取

#### 2. 库支持（最新更新 ✨）
- ✅ **数据处理**：NumPy, Pandas, SciPy
- ✅ **机器学习**：Scikit-learn (完整)
- ✅ **可视化**：Matplotlib, Plotly, Seaborn
- ✅ **图像处理**：PIL/Pillow (Image, ImageEnhance, ImageFilter) 🆕
- ✅ **标准库**：io, base64, json, re, collections.Counter 🆕

#### 3. 缩进处理（最新算法 ✨）
- ✅ 使用缩进栈追踪代码块层级
- ✅ 智能识别顶层语句（import, def, class）
- ✅ 自动修复不一致缩进
- ✅ 正确处理代码块结构（冒号结尾）

---

## 🎯 升级方向建议

### 📈 优先级分类

```
P0 = 立即实施（1-2周）
P1 = 近期规划（1-2个月）
P2 = 中期规划（3-6个月）
P3 = 长期规划（6个月+）
```

---

## 🚀 一、图片处理增强（P0 - 立即可做）

### 现状
- ✅ PIL/Pillow 已导入
- ✅ Image, ImageEnhance, ImageFilter 已支持
- ✅ io, base64 已支持
- ⚠️ 缺少使用文档和示例

### 建议：完善图片处理功能

#### 1.1 创建图片处理模板

**新增代码模板**：

```python
# 模板 1: image_format_convert
"""批量图片格式转换"""
from PIL import Image
import io
import base64

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # 解码图片
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        # 转换为 PNG
        output = io.BytesIO()
        if img.mode == 'RGBA':
            img.save(output, format='PNG')
        else:
            img = img.convert('RGB')
            img.save(output, format='PNG')

        result = base64.b64encode(output.getvalue()).decode()
        print(f"✓ {file['name']} -> PNG")

# 模板 2: image_compress
"""批量图片压缩"""
from PIL import Image
import io
import base64

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        # 调整尺寸
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        # 压缩
        output = io.BytesIO()
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output, format='JPEG', quality=85, optimize=True)

        original_size = len(img_data)
        new_size = output.tell()
        saved = (1 - new_size/original_size) * 100

        print(f"✓ {file['name']}: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB (省{saved:.1f}%)")

# 模板 3: image_analysis
"""图片数据集分析"""
from PIL import Image
import io
import base64

image_stats = {
    'count': 0,
    'formats': {},
    'total_size': 0,
    'avg_width': 0,
    'avg_height': 0
}

widths = []
heights = []

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        image_stats['count'] += 1
        image_stats['formats'][img.format] = image_stats['formats'].get(img.format, 0) + 1
        image_stats['total_size'] += len(img_data)

        widths.append(img.width)
        heights.append(img.height)

if widths:
    image_stats['avg_width'] = sum(widths) / len(widths)
    image_stats['avg_height'] = sum(heights) / len(heights)

print("=" * 60)
print("图片数据集分析报告")
print("=" * 60)
print(f"总图片数: {image_stats['count']}")
print(f"格式分布: {image_stats['formats']}")
print(f"平均尺寸: {image_stats['avg_width']:.0f} x {image_stats['avg_height']:.0f}")
print(f"总大小: {image_stats['total_size']/1024/1024:.2f} MB")

# 模板 4: image_enhance
"""图片增强处理"""
from PIL import Image, ImageEnhance
import io
import base64

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        # 增强亮度
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)

        # 增强对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)

        # 锐化
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)

        print(f"✓ {file['name']} 已增强")
```

**实施步骤**：
1. 在 `app/sandbox.py` 的 `CODE_TEMPLATES` 添加这4个模板
2. 测试每个模板
3. 更新 API 文档

**预计工作量**：2-3小时

---

#### 1.2 创建图片处理使用指南

**文件名**：`IMAGE_PROCESSING_GUIDE.md`

**内容要点**：
- 图片 base64 编码方式
- 前端如何准备图片数据
- 完整的代码示例
- 常见问题解答

**预计工作量**：2小时

---

#### 1.3 前端适配建议

**需要前端配合**：

```javascript
// 前端读取图片并转 base64
function readImageAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            // 获取 base64（移除 data:image/xxx;base64, 前缀）
            const base64 = e.target.result.split(',')[1];
            resolve({
                name: file.name,
                content: base64,
                content_type: file.type,
                encoding: 'base64'
            });
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// 批量处理
async function prepareImages(files) {
    const datasets = {};
    for (const file of files) {
        if (file.type.startsWith('image/')) {
            const imageData = await readImageAsBase64(file);
            datasets[file.name] = imageData.content;
        }
    }
    return datasets;
}
```

**预计工作量**：前端 0.5天

---

## 📊 二、Excel 功能增强（P0）

### 现状
- ✅ 基础 Excel 读取（单 sheet）
- ❌ 不支持多 sheet
- ❌ 不支持 Excel 写入

### 建议：完整 Excel 支持

#### 2.1 添加 openpyxl 依赖

**修改 requirements.txt**：
```
openpyxl>=3.1.2
```

#### 2.2 更新 sandbox.py

```python
# 新增到 ALLOWED_MODULES
import openpyxl
from openpyxl import Workbook

ALLOWED_MODULES = {
    # ... 现有的
    'openpyxl': openpyxl,
    'Workbook': Workbook,
}
```

#### 2.3 创建 Excel 处理模板

```python
# 模板: excel_multi_sheet
"""读取多 sheet Excel"""
import pandas as pd
import io
import base64

for file in selected_files:
    if file['name'].endswith(('.xlsx', '.xls')):
        # 解码
        excel_bytes = base64.b64decode(file['content'])

        # 读取所有 sheet
        excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))

        print(f"\n文件: {file['name']}")
        print(f"Sheet 数量: {len(excel_file.sheet_names)}")

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            print(f"\nSheet: {sheet_name}")
            print(f"  形状: {df.shape}")
            print(f"  列名: {list(df.columns)}")
            print(df.head())
```

**预计工作量**：0.5天（包括测试）

---

## 📝 三、文本处理增强（P1）

### 现状
- ✅ 基础文本读取
- ⚠️ 缺少高级文本处理功能

### 建议：增强文本分析能力

#### 3.1 添加文本处理库

**requirements.txt 新增**：
```
jieba>=0.42.1          # 中文分词
wordcloud>=1.9.3       # 词云生成（可选）
```

#### 3.2 更新 sandbox.py

```python
import jieba

ALLOWED_MODULES = {
    # ... 现有的
    'jieba': jieba,
}
```

#### 3.3 创建文本分析模板

```python
# 模板: text_word_frequency
"""文本词频统计（支持中文）"""
import jieba
from collections import Counter

for file in selected_files:
    if file['name'].endswith('.txt'):
        text = file['content']

        # 中文分词
        words = jieba.lcut(text)

        # 过滤停用词和标点
        words = [w for w in words if len(w) > 1 and w.isalnum()]

        # 词频统计
        word_freq = Counter(words)

        print(f"\n文件: {file['name']}")
        print(f"总词数: {len(words)}")
        print(f"不重复词数: {len(word_freq)}")
        print("\n高频词 Top 20:")
        for word, count in word_freq.most_common(20):
            print(f"  {word}: {count}")

# 模板: text_cleaning
"""文本清洗"""
import re

for file in selected_files:
    if file['name'].endswith('.txt'):
        text = file['content']

        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)

        # 去除特殊字符（保留中英文、数字）
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

        # 去除空行
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = '\n'.join(lines)

        print(f"\n文件: {file['name']}")
        print(f"原始长度: {len(file['content'])} 字符")
        print(f"清洗后长度: {len(cleaned_text)} 字符")
        print(f"节省: {(1 - len(cleaned_text)/len(file['content']))*100:.1f}%")
```

**预计工作量**：1天（包括测试）

---

## 🔍 四、代码执行监控和诊断（P1）

### 建议：增强可观测性

#### 4.1 执行统计信息

**在 ExecuteResponse 中新增字段**：

```python
class ExecutionStats(BaseModel):
    """执行统计信息"""
    peak_memory_mb: Optional[float] = None      # 峰值内存
    cpu_time_ms: Optional[int] = None           # CPU 时间
    chart_count: int = 0                         # 生成图表数
    dataframe_count: int = 0                     # DataFrame 数量
    dataset_count: int = 0                       # 数据集数量
    code_lines: int = 0                          # 代码行数

class ExecuteResponse(BaseModel):
    status: str
    execution_time: int
    output: Optional[ExecutionOutput] = None
    error: Optional[str] = None
    stats: Optional[ExecutionStats] = None  # 新增
```

#### 4.2 性能监控

```python
import psutil
import os

def execute(self, code: str, datasets=None, preloaded_variables=None):
    # 开始监控
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 执行代码...

    # 记录峰值内存
    peak_memory = process.memory_info().rss / 1024 / 1024

    # 返回统计信息
    stats = ExecutionStats(
        peak_memory_mb=peak_memory,
        chart_count=len(charts),
        dataframe_count=len(dataframes),
        dataset_count=len(datasets) if datasets else 0,
        code_lines=len(code.splitlines())
    )
```

**依赖**：
```
psutil>=5.9.8
```

**预计工作量**：1-2天

---

## 🔒 五、安全性增强（P1）

### 5.1 代码复杂度限制

**目的**：防止过于复杂的代码消耗资源

```python
def validate_code_complexity(code: str) -> tuple[bool, str]:
    """验证代码复杂度"""

    # 限制代码行数
    lines = code.splitlines()
    if len(lines) > 500:
        return False, f"代码行数超限（{len(lines)}/500）"

    # 限制嵌套层级
    max_indent = 0
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent // 4)

    if max_indent > 10:
        return False, f"代码嵌套层级过深（{max_indent}/10）"

    # 限制循环数量
    loop_count = code.count('for ') + code.count('while ')
    if loop_count > 20:
        return False, f"循环数量超限（{loop_count}/20）"

    return True, ""
```

**预计工作量**：0.5天

---

### 5.2 执行日志审计

**记录所有执行**：

```python
class ExecutionLog(BaseModel):
    timestamp: datetime
    user_id: Optional[str] = None
    code_hash: str                     # 代码的 hash
    status: str                        # success/error/timeout
    execution_time: int
    error: Optional[str] = None

# 记录到数据库或日志文件
def log_execution(log: ExecutionLog):
    # 可以存储到 SQLite/PostgreSQL 或日志文件
    pass
```

**预计工作量**：1天

---

## 📈 六、性能优化（P2）

### 6.1 结果缓存

**场景**：相同代码和数据的重复执行

```python
import hashlib
from functools import lru_cache

class CodeExecutor:
    def __init__(self):
        self.cache = {}  # 或使用 Redis

    def _get_cache_key(self, code: str, datasets: Dict) -> str:
        """生成缓存键"""
        content = f"{code}:{sorted(datasets.items())}"
        return hashlib.sha256(content.encode()).hexdigest()

    def execute(self, code: str, datasets=None, use_cache=True):
        if use_cache:
            cache_key = self._get_cache_key(code, datasets or {})
            if cache_key in self.cache:
                logger.info("使用缓存结果")
                return self.cache[cache_key]

        # 执行代码...
        result = ...

        if use_cache:
            self.cache[cache_key] = result

        return result
```

**依赖**：
```
redis>=5.0.1  # 可选，用于分布式缓存
```

**预计工作量**：2天

---

### 6.2 异步执行

**场景**：长时间运行的任务

```python
from fastapi import BackgroundTasks

@app.post("/execute/async")
async def execute_code_async(request: ExecuteRequest, background_tasks: BackgroundTasks):
    """异步执行代码"""

    # 生成任务 ID
    task_id = str(uuid.uuid4())

    # 添加到后台任务
    background_tasks.add_task(execute_in_background, task_id, request)

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "任务已提交"
    }

@app.get("/execute/status/{task_id}")
async def get_execution_status(task_id: str):
    """查询任务状态"""
    # 从缓存或数据库获取任务状态
    return task_status
```

**预计工作量**：2-3天

---

## 🎨 七、用户体验增强（P2）

### 7.1 代码自动补全API

```python
@app.post("/autocomplete")
async def autocomplete(request: AutocompleteRequest):
    """代码自动补全"""

    code = request.code
    cursor_position = request.cursor_position

    # 简单的补全逻辑
    suggestions = []

    # 如果在输入 pd.
    if code.endswith('pd.'):
        suggestions = ['read_csv', 'read_json', 'DataFrame', 'Series']

    # 如果在输入 df.
    elif code.endswith('df.'):
        suggestions = ['head', 'tail', 'describe', 'info', 'shape', 'columns']

    return {
        "suggestions": suggestions
    }
```

**预计工作量**：1-2天

---

### 7.2 代码格式化

```python
@app.post("/format")
async def format_code(request: FormatRequest):
    """格式化代码"""
    import black

    try:
        formatted = black.format_str(request.code, mode=black.Mode())
        return {
            "formatted_code": formatted,
            "changed": formatted != request.code
        }
    except Exception as e:
        return {
            "error": str(e)
        }
```

**依赖**：
```
black>=24.0.0
```

**预计工作量**：0.5天

---

## 🌐 八、国际化支持（P3）

### 建议：多语言错误信息

```python
# app/i18n.py
MESSAGES = {
    'zh': {
        'timeout': '代码执行超时',
        'syntax_error': '语法错误',
        'forbidden_operation': '检测到禁止的操作',
    },
    'en': {
        'timeout': 'Code execution timeout',
        'syntax_error': 'Syntax error',
        'forbidden_operation': 'Forbidden operation detected',
    }
}

def get_message(key: str, lang: str = 'zh') -> str:
    return MESSAGES.get(lang, MESSAGES['zh']).get(key, key)
```

**预计工作量**：1天

---

## 📦 九、部署和运维增强（P2）

### 9.1 健康检查增强

```python
@app.get("/health/detailed")
async def detailed_health_check():
    """详细健康检查"""

    import psutil

    return {
        "status": "healthy",
        "version": "1.3.0",
        "uptime_seconds": get_uptime(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "metrics": {
            "total_executions": execution_count,
            "success_rate": success_rate,
            "avg_execution_time": avg_time
        }
    }
```

**预计工作量**：1天

---

### 9.2 Prometheus 指标

```python
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
execution_counter = Counter('python_executor_executions_total', 'Total executions')
execution_duration = Histogram('python_executor_duration_seconds', 'Execution duration')
active_executions = Gauge('python_executor_active_executions', 'Active executions')

@app.get("/metrics")
async def metrics():
    """Prometheus 指标"""
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")
```

**依赖**：
```
prometheus-client>=0.19.0
```

**预计工作量**：1天

---

## 🧪 十、测试覆盖增强（P1）

### 10.1 单元测试

```python
# tests/test_executor.py
import pytest
from app.executor import CodeExecutor

def test_simple_execution():
    executor = CodeExecutor()
    result = executor.execute("print('hello')")
    assert result.status == "success"
    assert "hello" in result.output.stdout

def test_dataset_injection():
    executor = CodeExecutor()
    result = executor.execute(
        "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(len(df))",
        datasets={"data.csv": "a,b\n1,2\n3,4"}
    )
    assert result.status == "success"
    assert "2" in result.output.stdout

def test_image_processing():
    executor = CodeExecutor()
    # ... 测试图片处理
```

**测试覆盖目标**：80%+

**预计工作量**：3-5天

---

### 10.2 集成测试

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_execute_endpoint():
    response = client.post("/execute", json={
        "code": "print('test')"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_with_datasets():
    response = client.post("/execute", json={
        "code": "print(len(selected_files))",
        "datasets": {"test.csv": "a,b\n1,2"}
    })
    assert response.status_code == 200
```

**预计工作量**：2-3天

---

## 📊 升级路线图总结

### Phase 1: 立即实施（1-2周）

| 任务 | 优先级 | 工作量 | 价值 |
|-----|-------|-------|-----|
| 图片处理模板 | P0 | 2-3h | ⭐⭐⭐⭐⭐ |
| Excel 多 sheet | P0 | 0.5天 | ⭐⭐⭐⭐ |
| 图片处理文档 | P0 | 2h | ⭐⭐⭐⭐ |
| 前端图片适配 | P0 | 0.5天 | ⭐⭐⭐⭐⭐ |

**总工作量**：约 2-3 天
**ROI**：高（立即解决用户需求）

---

### Phase 2: 近期规划（1个月）

| 任务 | 优先级 | 工作量 | 价值 |
|-----|-------|-------|-----|
| 文本处理增强 | P1 | 1天 | ⭐⭐⭐⭐ |
| 执行监控统计 | P1 | 1-2天 | ⭐⭐⭐⭐ |
| 安全性增强 | P1 | 1.5天 | ⭐⭐⭐⭐⭐ |
| 单元测试 | P1 | 3-5天 | ⭐⭐⭐⭐ |

**总工作量**：约 1-2 周
**ROI**：中高（提升稳定性和安全性）

---

### Phase 3: 中期规划（2-3个月）

| 任务 | 优先级 | 工作量 | 价值 |
|-----|-------|-------|-----|
| 结果缓存 | P2 | 2天 | ⭐⭐⭐ |
| 异步执行 | P2 | 2-3天 | ⭐⭐⭐⭐ |
| 代码补全 | P2 | 1-2天 | ⭐⭐⭐ |
| 监控指标 | P2 | 1天 | ⭐⭐⭐ |

**总工作量**：约 1-2 周
**ROI**：中（性能和体验提升）

---

### Phase 4: 长期规划（3-6个月）

| 任务 | 优先级 | 工作量 | 价值 |
|-----|-------|-------|-----|
| 国际化 | P3 | 1天 | ⭐⭐ |
| 高级缓存 | P3 | 3天 | ⭐⭐⭐ |
| 分布式执行 | P3 | 5天+ | ⭐⭐⭐ |

---

## 🎯 推荐实施顺序

### Week 1-2（立即）
1. ✅ 创建图片处理模板（4个）
2. ✅ 编写图片处理文档
3. ✅ Excel 多 sheet 支持
4. ✅ 前端图片 base64 适配

### Week 3-4
1. 文本处理增强（jieba 分词）
2. 执行统计信息
3. 代码复杂度限制

### Month 2
1. 单元测试覆盖
2. 集成测试
3. 性能基准测试

### Month 3
1. 结果缓存
2. 异步执行支持
3. Prometheus 监控

---

## 📝 具体行动建议

### 今天就可以做的（1小时内）

1. **添加图片处理模板**
   - 在 `app/sandbox.py` 的 `CODE_TEMPLATES` 添加 4 个图片处理模板
   - 立即可用，无需其他改动

2. **更新 README**
   - 添加图片处理功能说明
   - 更新功能列表

3. **创建图片处理示例文档**
   - 示例代码
   - 使用说明

### 本周可以完成的

1. **Excel 多 sheet 支持**
   - 添加 openpyxl 依赖
   - 更新 sandbox.py
   - 创建测试用例

2. **前端适配指导**
   - 编写前端图片处理文档
   - 提供 JavaScript 示例代码
   - 与前端团队沟通

---

## 💡 创新性建议

### 1. AI 代码助手集成（未来）

```python
@app.post("/ai/suggest")
async def ai_code_suggestion(request: AISuggestRequest):
    """AI 代码建议"""
    # 基于用户需求，生成代码建议
    # 可以集成 GPT-4 或本地模型
    pass
```

### 2. 代码模板市场

- 用户可以分享自己的代码模板
- 评分和评论系统
- 模板分类和搜索

### 3. 可视化工作流

- 拖拽式数据处理流程
- 自动生成 Python 代码
- 类似 Orange 或 KNIME

---

## ✅ 总结

### 当前优势
- ✅ 核心功能完善
- ✅ 安全性良好
- ✅ 图片库已支持
- ✅ 智能缩进算法
- ✅ 双格式数据传递

### 主要改进方向
1. **图片处理**：添加模板和文档（P0）
2. **Excel 增强**：多 sheet 支持（P0）
3. **文本分析**：jieba 分词（P1）
4. **监控统计**：性能指标（P1）
5. **测试覆盖**：单元+集成测试（P1）

### 预期效果
- 🎯 功能完整度：90%+
- 🔒 安全性：95%+
- ⚡ 性能：优秀
- 📊 可观测性：良好
- 🧪 测试覆盖：80%+

---

**文档版本**: v2.0
**创建日期**: 2025-11-03
**状态**: 建议中
**预计完成时间**: Phase 1-2 约 1-1.5 个月
