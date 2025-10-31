# Python Executor Service

安全的 Python 代码执行微服务，支持在线运行 Python 代码并生成可视化结果。

## 特性

- ✅ **安全沙箱**：使用 RestrictedPython 限制危险操作
- ✅ **图表支持**：自动捕获 matplotlib 和 plotly 图表
- ✅ **数据表格**：自动识别并渲染 pandas DataFrame
- ✅ **超时控制**：防止长时间运行的代码占用资源
- ✅ **并发限制**：控制资源使用
- ✅ **Docker 部署**：容器化部署，资源隔离

## 快速开始

### 1. 本地运行（开发模式）

#### 前置要求

- Python 3.11+
- pip

#### 安装依赖

```bash
cd D:\PythonProjects\python-executor-service
pip install -r requirements.txt
```

#### 启动服务

```bash
# 方式 1：直接运行
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 方式 2：使用 Python
python app/main.py
```

#### 访问 API 文档

启动后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. Docker 部署（推荐）

#### 构建镜像

```bash
cd D:\PythonProjects\python-executor-service
docker build -t python-executor-service:latest .
```

#### 运行容器

```bash
docker run -d \
  --name python-executor \
  -p 8000:8000 \
  -e MAX_WORKERS=4 \
  -e TIMEOUT_SECONDS=30 \
  --cpus=1.0 \
  --memory=512m \
  python-executor-service:latest
```

#### 使用 Docker Compose（推荐）

```bash
docker-compose up -d
```

### 3. 生产环境部署

#### 编辑 docker-compose.yml

```yaml
# 修改资源限制和环境变量
environment:
  - MAX_WORKERS=4
  - TIMEOUT_SECONDS=30
  - LOG_LEVEL=info

deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 1G
```

#### 启动服务

```bash
docker-compose -f docker-compose.yml up -d
```

## API 使用示例

### 1. 执行 Python 代码

**请求**：

```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(\"Hello World\")",
    "timeout": 30
  }'
```

**响应**：

```json
{
  "status": "success",
  "execution_time": 123,
  "output": {
    "stdout": "Hello World\n",
    "stderr": "",
    "charts": [],
    "dataframes": [],
    "variables": {}
  },
  "error": null
}
```

### 2. 生成 Matplotlib 图表

**请求**：

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.show()
```

**响应**：

```json
{
  "status": "success",
  "execution_time": 1234,
  "output": {
    "stdout": "",
    "stderr": "",
    "charts": [
      {
        "type": "matplotlib",
        "format": "png",
        "data": "base64_encoded_image_string",
        "width": 800,
        "height": 600
      }
    ]
  }
}
```

### 3. 获取代码模板

```bash
curl http://localhost:8000/templates
```

### 4. 验证代码

```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"test\")"}'
```

## 安全配置

### 允许的库

- ✅ numpy
- ✅ pandas
- ✅ matplotlib
- ✅ plotly

### 禁止的操作

- ❌ 文件系统访问（open, file）
- ❌ 网络访问（requests, urllib, socket）
- ❌ 系统调用（os, sys, subprocess）
- ❌ 动态执行（eval, exec, compile）

### 资源限制

- **CPU**：单核 50-100%
- **内存**：512MB - 1GB
- **超时**：默认 30 秒，最大 60 秒
- **代码长度**：最大 50KB

## 监控和日志

### 健康检查

```bash
curl http://localhost:8000/health
```

### 查看日志

```bash
# Docker 日志
docker logs python-executor-service -f

# 本地日志（如果挂载了 logs 目录）
tail -f logs/app.log
```

## 性能优化

### 1. 增加工作进程

```bash
docker run -e MAX_WORKERS=8 ...
```

### 2. 调整资源限制

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 2G
```

### 3. 使用负载均衡

```nginx
upstream python_executor {
    server python-executor-1:8000;
    server python-executor-2:8000;
    server python-executor-3:8000;
}
```

## 故障排查

### 1. 容器无法启动

```bash
# 查看日志
docker logs python-executor-service

# 检查端口占用
netstat -ano | findstr :8000
```

### 2. 代码执行超时

- 检查超时配置
- 优化代码性能
- 增加资源限制

### 3. 内存不足

- 增加容器内存限制
- 减少并发数
- 优化代码

## 开发指南

### 项目结构

```
python-executor-service/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI 应用入口
│   ├── models.py         # 数据模型
│   ├── executor.py       # 代码执行引擎
│   ├── sandbox.py        # 沙箱配置
│   └── visualizer.py     # 图表处理
├── requirements.txt      # Python 依赖
├── Dockerfile           # Docker 镜像配置
├── docker-compose.yml   # Docker Compose 配置
└── README.md           # 说明文档
```

### 添加新的允许库

编辑 `app/sandbox.py`：

```python
ALLOWED_MODULES = {
    'numpy': np,
    'pandas': pd,
    # 添加新库
    'sklearn': sklearn,
}
```

### 添加代码模板

编辑 `app/sandbox.py`：

```python
CODE_TEMPLATES = {
    "my_template": """
import numpy as np
# Your template code here
"""
}
```

## 测试

### 单元测试

```bash
pytest tests/
```

### 手动测试

访问 http://localhost:8000/docs 使用 Swagger UI 测试。

## 许可证

MIT License

## 联系方式

如有问题，请联系：SolarSense Team
