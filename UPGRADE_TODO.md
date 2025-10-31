# Python Executor Service - 升级优化 TODO 列表

## 当前支持情况

### ✅ 已支持的功能（6/8 场景）

1. **Pandas 基础操作** - 内存数据处理
2. **Pandas 高级操作** - 筛选、分组、排序、聚合等
3. **NumPy 数学运算** - 数组操作、线性代数、统计
4. **手动数据处理** - 标准化、归一化（纯 Python/NumPy 实现）
5. **Matplotlib 可视化** - 静态图表生成
6. **Plotly 可视化** - 交互式图表生成

### ❌ 不支持的功能（2/8 场景）

1. **文件系统访问** - 无法读取/写入 CSV、Excel 等文件
2. **Scikit-learn 库** - 缺少机器学习和数据预处理功能

---

## 🎯 升级优化 TODO

### 优先级 P0 - 核心功能增强

#### 1. 支持数据文件上传和访问
**问题**：用户无法使用 `pd.read_csv()` 读取数据文件

**解决方案选项**：

**方案 A：临时文件系统（推荐）**
- [ ] 为每个执行会话创建临时隔离目录
- [ ] 添加文件上传 API endpoint (`/upload`)
- [ ] 限制文件大小（如 10MB）和类型（CSV, Excel, JSON）
- [ ] 执行完成后自动清理临时文件
- [ ] 修改沙箱配置，允许受限的文件操作

```python
# 新增 API
@app.post("/upload")
async def upload_file(file: UploadFile):
    """上传数据文件到临时目录"""
    pass

@app.post("/execute-with-files")
async def execute_with_files(code: str, files: List[str]):
    """执行代码并提供文件访问"""
    pass
```

**方案 B：内存数据传递**
- [ ] 添加 API 直接传递数据（JSON 格式）
- [ ] 在执行环境中预加载数据到 DataFrame
- [ ] 用户代码直接使用预定义的 DataFrame 变量

```python
@app.post("/execute-with-data")
async def execute_with_data(code: str, datasets: Dict[str, Any]):
    """执行代码并提供预加载的数据集"""
    pass
```

**优先级**：⭐⭐⭐⭐⭐ 高
**工作量**：2-3 天
**影响范围**：API、沙箱、安全策略

---

#### 2. 添加 Scikit-learn 支持
**问题**：用户无法使用 sklearn 进行机器学习和数据预处理

**实施步骤**：
- [ ] 安装 scikit-learn 依赖
- [ ] 添加到 requirements.txt
- [ ] 更新沙箱白名单配置
- [ ] 预导入常用模块（preprocessing, model_selection, metrics）
- [ ] 添加代码模板

```python
# sandbox.py 更新
ALLOWED_MODULES = {
    'numpy': np,
    'pandas': pd,
    'matplotlib': matplotlib,
    'plt': plt,
    'plotly': plotly,
    'sklearn': sklearn,  # 新增
}

# 代码模板
CODE_TEMPLATES = {
    "sklearn_preprocessing": """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 数据预处理示例
...
""",
}
```

**优先级**：⭐⭐⭐⭐⭐ 高
**工作量**：1 天
**影响范围**：依赖、沙箱配置

---

### 优先级 P1 - 功能扩展

#### 3. 添加更多数据科学库

- [ ] **Seaborn** - 高级统计可视化
  - 安装 seaborn
  - 添加到白名单
  - 提供模板

- [ ] **SciPy** - 科学计算
  - 支持统计分析、优化、信号处理
  - 添加到白名单

- [ ] **Statsmodels** - 统计建模
  - 时间序列、回归分析
  - 添加到白名单

```python
ALLOWED_MODULES = {
    # 现有的...
    'seaborn': seaborn,
    'sns': seaborn,
    'scipy': scipy,
    'statsmodels': statsmodels,
}
```

**优先级**：⭐⭐⭐⭐ 中高
**工作量**：2 天
**影响范围**：依赖、沙箱配置

---

#### 4. 数据持久化和会话管理

**功能描述**：
- [ ] 支持跨请求的数据持久化
- [ ] 实现会话管理机制
- [ ] 支持多步骤数据处理流程

**API 设计**：
```python
@app.post("/session/create")
async def create_session():
    """创建新的执行会话"""
    return {"session_id": "xxx"}

@app.post("/session/{session_id}/execute")
async def execute_in_session(session_id: str, code: str):
    """在会话中执行代码，保留变量状态"""
    pass

@app.get("/session/{session_id}/variables")
async def get_session_variables(session_id: str):
    """获取会话中的变量"""
    pass

@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    """关闭会话并清理资源"""
    pass
```

**优先级**：⭐⭐⭐⭐ 中高
**工作量**：3-4 天
**影响范围**：API、执行器、缓存

---

#### 5. 增强的数据导出功能

**功能描述**：
- [ ] 支持导出处理后的数据
- [ ] 支持多种格式（CSV, Excel, JSON, Parquet）
- [ ] 生成下载链接或返回文件

**API 设计**：
```python
@app.post("/execute-and-export")
async def execute_and_export(
    code: str,
    export_format: str = "csv",
    export_variables: List[str] = None
):
    """执行代码并导出指定变量"""
    pass
```

**优先级**：⭐⭐⭐ 中
**工作量**：2 天
**影响范围**：API、执行器

---

### 优先级 P2 - 性能和体验优化

#### 6. 执行结果缓存

**功能描述**：
- [ ] 对相同代码的执行结果进行缓存
- [ ] 使用 Redis 或内存缓存
- [ ] 设置合理的过期时间
- [ ] 添加缓存控制选项

**优先级**：⭐⭐⭐ 中
**工作量**：2-3 天
**影响范围**：执行器、缓存系统

---

#### 7. 代码自动补全和提示

**功能描述**：
- [ ] 提供代码自动补全 API
- [ ] 基于上下文的智能提示
- [ ] 函数签名和文档提示

**API 设计**：
```python
@app.post("/autocomplete")
async def autocomplete(code: str, cursor_position: int):
    """提供代码自动补全建议"""
    pass

@app.post("/inspect")
async def inspect_symbol(symbol: str):
    """获取符号的文档和签名"""
    pass
```

**优先级**：⭐⭐⭐ 中
**工作量**：3-4 天
**影响范围**：新模块

---

#### 8. 代码格式化和 Lint

**功能描述**：
- [ ] 集成 Black 自动格式化代码
- [ ] 集成 Flake8 或 Pylint 进行代码检查
- [ ] 提供代码质量建议

**API 设计**：
```python
@app.post("/format")
async def format_code(code: str):
    """格式化代码"""
    pass

@app.post("/lint")
async def lint_code(code: str):
    """检查代码质量"""
    pass
```

**优先级**：⭐⭐ 低
**工作量**：2 天
**影响范围**：新模块

---

#### 9. 增强的图表功能

**功能描述**：
- [ ] 支持更多图表库（Altair, Bokeh）
- [ ] 支持多图表组合
- [ ] 自定义图表大小和分辨率
- [ ] 图表导出为 SVG、PDF

**优先级**：⭐⭐ 低
**工作量**：2-3 天
**影响范围**：可视化模块

---

### 优先级 P3 - 安全和监控

#### 10. 资源使用监控和限制

**功能描述**：
- [ ] 实时监控内存使用
- [ ] 监控 CPU 使用率
- [ ] 自动终止超限执行
- [ ] 提供资源使用报告

**优先级**：⭐⭐⭐⭐ 中高
**工作量**：3 天
**影响范围**：执行器、监控系统

---

#### 11. 审计日志和安全扫描

**功能描述**：
- [ ] 记录所有代码执行历史
- [ ] 检测可疑代码模式
- [ ] 生成安全报告
- [ ] 支持代码审查

**优先级**：⭐⭐⭐ 中
**工作量**：3-4 天
**影响范围**：安全模块

---

#### 12. 并发控制和限流

**功能描述**：
- [ ] 实现令牌桶或漏桶算法
- [ ] 基于用户的请求限流
- [ ] 队列管理
- [ ] 优先级调度

**优先级**：⭐⭐⭐ 中
**工作量**：2-3 天
**影响范围**：中间件

---

## 📊 优先级矩阵

| 功能 | 优先级 | 工作量 | 用户价值 | 技术复杂度 |
|------|--------|--------|----------|------------|
| 1. 文件上传访问 | P0 | 3天 | ⭐⭐⭐⭐⭐ | 中 |
| 2. Scikit-learn 支持 | P0 | 1天 | ⭐⭐⭐⭐⭐ | 低 |
| 3. 更多数据科学库 | P1 | 2天 | ⭐⭐⭐⭐ | 低 |
| 4. 会话管理 | P1 | 4天 | ⭐⭐⭐⭐ | 中 |
| 5. 数据导出 | P1 | 2天 | ⭐⭐⭐⭐ | 低 |
| 6. 结果缓存 | P2 | 3天 | ⭐⭐⭐ | 中 |
| 7. 代码补全 | P2 | 4天 | ⭐⭐⭐ | 高 |
| 8. 格式化和 Lint | P2 | 2天 | ⭐⭐ | 低 |
| 9. 增强图表 | P2 | 3天 | ⭐⭐⭐ | 中 |
| 10. 资源监控 | P3 | 3天 | ⭐⭐⭐⭐ | 中 |
| 11. 审计日志 | P3 | 4天 | ⭐⭐⭐ | 中 |
| 12. 并发限流 | P3 | 3天 | ⭐⭐⭐ | 中 |

---

## 🚀 推荐实施路线

### 第一阶段（1-2周）- 核心功能
1. ✅ 添加 Scikit-learn 支持（1天）
2. ✅ 实现文件上传和临时存储（3天）
3. ✅ 添加更多数据科学库（2天）

**目标**：支持 95% 的数据科学用例

### 第二阶段（2-3周）- 功能增强
4. ✅ 实现会话管理（4天）
5. ✅ 添加数据导出功能（2天）
6. ✅ 实现结果缓存（3天）

**目标**：提供完整的数据处理工作流

### 第三阶段（3-4周）- 体验优化
7. ✅ 添加资源监控（3天）
8. ✅ 实现代码补全（4天）
9. ✅ 增强图表功能（3天）

**目标**：提升用户体验和安全性

### 第四阶段（按需）- 生产级别
10. ✅ 实现并发限流（3天）
11. ✅ 添加审计日志（4天）
12. ✅ 代码格式化（2天）

**目标**：生产环境部署就绪

---

## 📝 测试场景覆盖

### 当前测试覆盖率：75% (6/8)

需要新增测试场景：
- [ ] 文件读取和写入
- [ ] Scikit-learn 数据预处理
- [ ] Scikit-learn 模型训练
- [ ] 会话状态管理
- [ ] 数据导出
- [ ] 并发执行
- [ ] 资源限制
- [ ] 错误恢复

**目标覆盖率**：95%+

---

## 🔧 技术债务清理

- [ ] 移除 .bak 和 .old 备份文件
- [ ] 统一日志格式
- [ ] 添加类型注解（Type Hints）
- [ ] 增加单元测试覆盖率
- [ ] 优化错误处理流程
- [ ] 文档完善（API 文档、开发文档）

---

## 📦 依赖升级

当前依赖：
```
fastapi>=0.109.0
uvicorn>=0.27.0
matplotlib>=3.8.2
plotly>=5.18.0
pandas>=2.1.4
numpy>=1.26.3
```

计划新增：
```
scikit-learn>=1.4.0      # P0
seaborn>=0.13.0          # P1
scipy>=1.12.0            # P1
statsmodels>=0.14.0      # P1
redis>=5.0.0             # P2 (缓存)
jedi>=0.19.0             # P2 (代码补全)
black>=24.0.0            # P2 (格式化)
flake8>=7.0.0            # P2 (Lint)
```

---

## 🎯 成功指标

1. **功能完整性**：支持 95%+ 的数据科学使用场景
2. **性能**：平均响应时间 < 2秒
3. **可用性**：99.9% uptime
4. **安全性**：0 安全漏洞
5. **用户满意度**：4.5+ 星评分

---

## 📞 下一步行动

建议优先实施：

**本周（快速见效）**：
1. 添加 Scikit-learn 支持 ✅
2. 创建基础测试用例 ✅
3. 更新文档 ✅

**下周（核心功能）**：
1. 设计文件上传 API
2. 实现临时文件系统
3. 添加 Seaborn、SciPy 支持

需要确认：
- [ ] 是否需要支持文件上传？
- [ ] 优先级排序是否合理？
- [ ] 预算和时间限制？
- [ ] 团队资源配置？
