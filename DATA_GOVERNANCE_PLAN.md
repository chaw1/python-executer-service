# 数据治理中心功能建议 - 完整方案

## 📋 当前实现分析

### 1. 实现差异对比

**已实现的 datasets 格式**:
```python
{
  "datasets": {
    "data.csv": "列1,列2\n值1,值2"  # Dict[文件名, 内容]
  }
}
```

**您实际使用的 selected_files 格式**:
```python
{
  "selected_files": [
    {
      "id": "9011",
      "name": "data.csv",
      "path": "dataset/data.csv",
      "content": "列1,列2\n值1,值2"
    }
  ]
}
```

**建议**: 两种格式都保留支持，增加兼容性

---

## 🎯 数据治理中心功能规划

### 一、文件类型支持矩阵

| 文件类型 | 优先级 | 主要操作 | 技术实现 |
|---------|-------|---------|---------|
| **表格数据** | P0 | ✅ 已完成 | pandas |
| **图片数据** | P0 | 🔨 需实现 | Pillow |
| **文本数据** | P0 | ✅ 已完成 | 原生 Python |
| **JSON数据** | P0 | ✅ 已完成 | json/pandas |
| **Excel** | P1 | 🔨 需增强 | openpyxl |
| **音频** | P2 | 待规划 | librosa |
| **视频** | P3 | 待规划 | opencv-python |

---

## 📊 一、表格数据治理（已完成 ✅）

### 当前支持的操作

#### 1. 数据清洗
- ✅ 缺失值处理（填充、删除）
- ✅ 重复值删除
- ✅ 异常值检测（IQR、Z-Score）
- ✅ 数据类型转换

#### 2. 数据转换
- ✅ 标准化/归一化
- ✅ 编码（Label、OneHot）
- ✅ 分组聚合
- ✅ 数据透视

#### 3. 数据分析
- ✅ 描述性统计
- ✅ 相关性分析
- ✅ 统计检验

#### 4. 数据可视化
- ✅ 各类图表（折线、散点、柱状、热力图等）

### 读取方式（已适配）
```python
import pandas as pd
import io

# 方式1: 使用 datasets 参数（我实现的）
df = pd.read_csv('data.csv')  # 自动从内存读取

# 方式2: 使用 selected_files（您的实现）
for file in selected_files:
    if file['name'].endswith('.csv'):
        df = pd.read_csv(io.StringIO(file['content']))
```

---

## 🖼️ 二、图片数据治理（重点新增 🔨）

### 2.1 基础图片操作

#### ✅ 已有库支持
- **Pillow** (PIL) - 已安装在 requirements.txt

#### 📦 需要新增的操作

##### A. 格式转换
```python
from PIL import Image
import io
import base64

def convert_image_format(image_content, target_format='PNG'):
    """
    图片格式转换

    支持格式: PNG, JPEG, BMP, GIF, TIFF, WebP
    """
    # 从 base64 解码
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    # 转换格式
    output = io.BytesIO()
    img.save(output, format=target_format)

    # 返回 base64
    return base64.b64encode(output.getvalue()).decode()

# 使用示例
for file in selected_files:
    if file['name'].endswith(('.jpg', '.png', '.bmp')):
        # 统一转为 PNG
        png_content = convert_image_format(file['content'], 'PNG')
        print(f"✓ {file['name']} 已转为 PNG 格式")
```

**支持的转换**:
- JPG/JPEG → PNG
- PNG → JPEG
- BMP → PNG/JPEG
- GIF → PNG
- TIFF → PNG/JPEG
- WebP → PNG/JPEG

---

##### B. 图片压缩
```python
from PIL import Image
import io
import base64

def compress_image(image_content, quality=85, max_size=None):
    """
    图片压缩

    Args:
        quality: JPEG 质量 (1-100)
        max_size: 最大尺寸 (width, height)
    """
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    # 调整尺寸
    if max_size:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    # 压缩
    output = io.BytesIO()
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(output, format='JPEG', quality=quality, optimize=True)

    original_size = len(image_data)
    compressed_size = output.tell()
    compression_ratio = (1 - compressed_size / original_size) * 100

    print(f"压缩比: {compression_ratio:.1f}%")
    print(f"原始: {original_size/1024:.1f}KB → 压缩后: {compressed_size/1024:.1f}KB")

    return base64.b64encode(output.getvalue()).decode()

# 批量压缩
for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
        compressed = compress_image(
            file['content'],
            quality=85,           # 压缩质量
            max_size=(1920, 1080) # 最大分辨率
        )
```

---

##### C. 图片信息提取
```python
from PIL import Image
import io
import base64

def get_image_info(image_content):
    """提取图片元数据"""
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    info = {
        "format": img.format,           # 格式
        "mode": img.mode,               # 颜色模式
        "size": img.size,               # (宽, 高)
        "width": img.width,
        "height": img.height,
        "file_size_kb": len(image_data) / 1024
    }

    return info

# 批量分析
for file in selected_files:
    if is_image_file(file['name']):
        info = get_image_info(file['content'])
        print(f"\n文件: {file['name']}")
        print(f"  格式: {info['format']}")
        print(f"  尺寸: {info['width']}x{info['height']}")
        print(f"  大小: {info['file_size_kb']:.1f}KB")
```

---

##### D. 图片尺寸调整
```python
def resize_image(image_content, target_size, keep_aspect=True):
    """
    调整图片尺寸

    Args:
        target_size: (width, height)
        keep_aspect: 是否保持宽高比
    """
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    if keep_aspect:
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
    else:
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    output = io.BytesIO()
    img.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()

# 批量调整为统一尺寸（用于模型训练）
target_size = (224, 224)  # 常见的模型输入尺寸

for file in selected_files:
    if is_image_file(file['name']):
        resized = resize_image(file['content'], target_size)
        print(f"✓ {file['name']} 已调整为 {target_size}")
```

---

##### E. 图片旋转和翻转
```python
def rotate_image(image_content, angle):
    """旋转图片"""
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    rotated = img.rotate(angle, expand=True)

    output = io.BytesIO()
    rotated.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()

def flip_image(image_content, direction='horizontal'):
    """翻转图片"""
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    if direction == 'horizontal':
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:  # vertical
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)

    output = io.BytesIO()
    flipped.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()
```

---

##### F. 图片滤镜和增强
```python
from PIL import ImageFilter, ImageEnhance

def apply_filter(image_content, filter_type='BLUR'):
    """应用滤镜"""
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    filters = {
        'BLUR': ImageFilter.BLUR,
        'SHARPEN': ImageFilter.SHARPEN,
        'EDGE_ENHANCE': ImageFilter.EDGE_ENHANCE,
        'SMOOTH': ImageFilter.SMOOTH,
    }

    filtered = img.filter(filters.get(filter_type, ImageFilter.BLUR))

    output = io.BytesIO()
    filtered.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()

def adjust_brightness(image_content, factor=1.5):
    """调整亮度"""
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    enhancer = ImageEnhance.Brightness(img)
    enhanced = enhancer.enhance(factor)

    output = io.BytesIO()
    enhanced.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()

def adjust_contrast(image_content, factor=1.5):
    """调整对比度"""
    image_data = base64.b64decode(image_content)
    img = Image.open(io.BytesIO(image_data))

    enhancer = ImageEnhance.Contrast(img)
    enhanced = enhancer.enhance(factor)

    output = io.BytesIO()
    enhanced.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()
```

---

##### G. 批量图片处理
```python
from PIL import Image
import io
import base64

def batch_process_images(selected_files, operations):
    """
    批量处理图片

    operations = {
        'resize': (800, 600),
        'format': 'PNG',
        'quality': 85,
        'compress': True
    }
    """
    results = []

    for file in selected_files:
        if not is_image_file(file['name']):
            continue

        try:
            image_data = base64.b64decode(file['content'])
            img = Image.open(io.BytesIO(image_data))

            # 调整尺寸
            if 'resize' in operations:
                img.thumbnail(operations['resize'], Image.Resampling.LANCZOS)

            # 格式转换
            target_format = operations.get('format', 'PNG')

            # 保存
            output = io.BytesIO()
            if img.mode == 'RGBA' and target_format == 'JPEG':
                img = img.convert('RGB')

            save_kwargs = {'format': target_format}
            if target_format == 'JPEG':
                save_kwargs['quality'] = operations.get('quality', 85)
                save_kwargs['optimize'] = operations.get('compress', True)

            img.save(output, **save_kwargs)

            result_content = base64.b64encode(output.getvalue()).decode()

            results.append({
                'original_name': file['name'],
                'new_name': f"{file['name'].rsplit('.', 1)[0]}.{target_format.lower()}",
                'content': result_content,
                'original_size': len(image_data),
                'new_size': output.tell()
            })

            print(f"✓ {file['name']}: {len(image_data)/1024:.1f}KB → {output.tell()/1024:.1f}KB")

        except Exception as e:
            print(f"✗ {file['name']}: {e}")

    return results

# 使用示例
results = batch_process_images(selected_files, {
    'resize': (1024, 1024),
    'format': 'JPEG',
    'quality': 85,
    'compress': True
})
```

---

### 2.2 图片数据标注相关

##### H. 图片统计分析
```python
import numpy as np
from PIL import Image
import io
import base64

def analyze_image_dataset(selected_files):
    """分析图片数据集"""

    stats = {
        'count': 0,
        'formats': {},
        'sizes': [],
        'file_sizes': [],
        'color_modes': {}
    }

    for file in selected_files:
        if not is_image_file(file['name']):
            continue

        image_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(image_data))

        stats['count'] += 1

        # 统计格式
        fmt = img.format
        stats['formats'][fmt] = stats['formats'].get(fmt, 0) + 1

        # 统计尺寸
        stats['sizes'].append(img.size)

        # 统计文件大小
        stats['file_sizes'].append(len(image_data) / 1024)

        # 统计颜色模式
        mode = img.mode
        stats['color_modes'][mode] = stats['color_modes'].get(mode, 0) + 1

    # 输出报告
    print("=" * 60)
    print("图片数据集分析报告")
    print("=" * 60)
    print(f"\n总图片数: {stats['count']}")

    print(f"\n格式分布:")
    for fmt, count in stats['formats'].items():
        print(f"  {fmt}: {count} ({count/stats['count']*100:.1f}%)")

    print(f"\n颜色模式:")
    for mode, count in stats['color_modes'].items():
        print(f"  {mode}: {count}")

    if stats['sizes']:
        widths = [s[0] for s in stats['sizes']]
        heights = [s[1] for s in stats['sizes']]

        print(f"\n尺寸统计:")
        print(f"  宽度: 最小={min(widths)}, 最大={max(widths)}, 平均={np.mean(widths):.0f}")
        print(f"  高度: 最小={min(heights)}, 最大={max(heights)}, 平均={np.mean(heights):.0f}")

    if stats['file_sizes']:
        print(f"\n文件大小:")
        print(f"  最小: {min(stats['file_sizes']):.1f}KB")
        print(f"  最大: {max(stats['file_sizes']):.1f}KB")
        print(f"  平均: {np.mean(stats['file_sizes']):.1f}KB")
        print(f"  总计: {sum(stats['file_sizes'])/1024:.1f}MB")

    return stats

# 执行分析
stats = analyze_image_dataset(selected_files)
```

---

## 📝 三、文本数据治理（基础支持 ✅）

### 当前支持

#### 3.1 基础文本操作
```python
# 读取文本文件
for file in selected_files:
    if file['name'].endswith('.txt'):
        lines = file['content'].splitlines()
        print(f"总行数: {len(lines)}")
```

#### 3.2 需要增强的功能

##### A. 文本清洗
```python
import re

def clean_text(text):
    """文本清洗"""
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 去除空行
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)

# 批量清洗
for file in selected_files:
    if file['name'].endswith('.txt'):
        cleaned = clean_text(file['content'])
        print(f"✓ {file['name']}: {len(file['content'])} → {len(cleaned)} 字符")
```

##### B. 文本统计
```python
def analyze_text(content):
    """文本统计分析"""
    lines = content.splitlines()
    words = content.split()

    stats = {
        'total_chars': len(content),
        'total_lines': len(lines),
        'total_words': len(words),
        'avg_line_length': np.mean([len(line) for line in lines]),
        'empty_lines': sum(1 for line in lines if not line.strip())
    }

    return stats
```

##### C. 文本编码转换
```python
def convert_encoding(content, from_encoding='gbk', to_encoding='utf-8'):
    """编码转换"""
    try:
        decoded = content.encode(from_encoding).decode(to_encoding)
        return decoded
    except Exception as e:
        print(f"编码转换失败: {e}")
        return content
```

---

## 📄 四、Excel 文件支持（需增强 🔨）

### 当前状态
- ✅ 基础读取支持（通过 pandas）
- ❌ 需要 openpyxl 支持多 sheet

### 需要新增

```python
# 需要添加到 requirements.txt
# openpyxl>=3.1.0

import pandas as pd
import io

def read_excel_file(file_content):
    """读取 Excel 文件（多 sheet）"""
    # 注意：需要 bytes 而不是 string
    excel_bytes = base64.b64decode(file_content)  # 如果是 base64

    # 读取所有 sheet
    excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))

    sheets = {}
    for sheet_name in excel_file.sheet_names:
        sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)

    return sheets

# 使用
for file in selected_files:
    if file['name'].endswith(('.xlsx', '.xls')):
        sheets = read_excel_file(file['content'])
        print(f"文件: {file['name']}")
        print(f"Sheet 数量: {len(sheets)}")
        for name, df in sheets.items():
            print(f"  - {name}: {df.shape}")
```

---

## 🎯 五、完整的数据治理操作矩阵

| 数据类型 | 清洗 | 转换 | 分析 | 可视化 | 导出 |
|---------|-----|-----|-----|-------|-----|
| **CSV/表格** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **图片** | 🔨 | 🔨 | 🔨 | ✅ | 🔨 |
| **文本** | 🔨 | ✅ | 🔨 | ✅ | ✅ |
| **JSON** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Excel** | ✅ | ✅ | ✅ | ✅ | 🔨 |

图例：
- ✅ 已完成
- 🔨 需实现/增强
- ❌ 暂不支持

---

## 📦 六、实现优先级建议

### Phase 1: 图片基础处理（P0）

**必须实现**:
1. ✅ 图片格式转换（PNG, JPEG, BMP）
2. ✅ 图片压缩（质量调整）
3. ✅ 尺寸调整（resize, thumbnail）
4. ✅ 图片信息提取
5. ✅ 批量处理功能

**预计工作量**: 2-3 天

**依赖**:
- Pillow（已安装）
- base64 处理逻辑

---

### Phase 2: 图片高级处理（P1）

**功能**:
1. ✅ 旋转和翻转
2. ✅ 滤镜应用
3. ✅ 亮度/对比度调整
4. ✅ 图片数据集统计分析
5. ✅ 水印添加

**预计工作量**: 2 天

---

### Phase 3: Excel 增强（P1）

**功能**:
1. 🔨 多 sheet 读取
2. 🔨 Excel 写入
3. 🔨 格式保留

**预计工作量**: 1-2 天

**依赖**: openpyxl

---

### Phase 4: 文本增强（P2）

**功能**:
1. 🔨 文本清洗增强
2. 🔨 编码检测和转换
3. 🔨 分词和词频统计
4. 🔨 情感分析（可选）

**预计工作量**: 2-3 天

---

## 🔧 七、技术实现建议

### 7.1 API 兼容性

**建议同时支持两种格式**:

```python
class ExecuteRequest(BaseModel):
    code: str
    timeout: int = 30

    # 方式1: datasets（我实现的）
    datasets: Optional[Dict[str, str]] = None

    # 方式2: selected_files（您使用的）
    selected_files: Optional[List[Dict[str, Any]]] = None

# 在 executor 中统一处理
def prepare_data(self, datasets=None, selected_files=None):
    if selected_files:
        # 转换为统一格式
        datasets = {f['name']: f['content'] for f in selected_files}

    # 后续统一处理
    self._inject_datasets(datasets)
```

---

### 7.2 图片处理的特殊考虑

**Base64 编码**:
```python
# 图片需要 base64 编码传输
import base64

# 前端发送
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# 后端接收
image_bytes = base64.b64decode(image_base64)
```

**内容类型标记**:
```python
selected_files = [
    {
        "name": "photo.jpg",
        "content": "base64_encoded_string",
        "content_type": "image/jpeg",  # 新增类型标记
        "encoding": "base64"
    }
]
```

---

### 7.3 预置模板建议

**新增图片处理模板**:

1. `image_batch_convert` - 批量格式转换
2. `image_compress` - 批量压缩
3. `image_resize` - 批量调整尺寸
4. `image_analysis` - 图片数据集分析
5. `image_enhance` - 图片增强

---

## 📚 八、完整示例代码

### 示例 1: 图片批量处理完整流程

```python
from PIL import Image, ImageEnhance
import io
import base64

print(f"开始处理 {len(selected_files)} 个文件...")

# 筛选图片文件
image_files = [f for f in selected_files
               if f['name'].lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

print(f"找到 {len(image_files)} 个图片文件\n")

# 批量处理
results = []

for file in image_files:
    print(f"{'='*60}")
    print(f"处理: {file['name']}")

    try:
        # 解码
        image_data = base64.b64decode(file['content'])
        original_size = len(image_data)
        img = Image.open(io.BytesIO(image_data))

        print(f"  原始尺寸: {img.size}")
        print(f"  原始大小: {original_size/1024:.1f}KB")
        print(f"  格式: {img.format}")

        # 1. 调整尺寸
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        print(f"  调整后尺寸: {img.size}")

        # 2. 增强（可选）
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)

        # 3. 转换格式并压缩
        output = io.BytesIO()
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output, format='JPEG', quality=85, optimize=True)

        new_size = output.tell()
        compression_ratio = (1 - new_size / original_size) * 100

        print(f"  压缩后大小: {new_size/1024:.1f}KB")
        print(f"  压缩比: {compression_ratio:.1f}%")

        # 保存结果
        result_content = base64.b64encode(output.getvalue()).decode()
        results.append({
            'name': file['name'].rsplit('.', 1)[0] + '.jpg',
            'content': result_content,
            'size': new_size
        })

        print("  ✓ 处理成功")

    except Exception as e:
        print(f"  ✗ 处理失败: {e}")

print(f"\n{'='*60}")
print(f"处理完成! 成功: {len(results)}/{len(image_files)}")
print(f"总节省空间: {(sum(original_size) - sum(r['size'] for r in results))/1024/1024:.1f}MB")
```

---

## 🚀 九、立即可用的功能

### 当前就能使用的图片处理（基于已有的 Pillow）

```python
# 示例代码 - 直接可用
from PIL import Image
import io
import base64

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# 检查选中的文件
if not selected_files:
    print("⚠️ 请先选择文件")
else:
    image_count = sum(1 for f in selected_files if is_image_file(f['name']))
    print(f"✓ 已选择 {len(selected_files)} 个文件，其中 {image_count} 个图片")

    # 遍历处理图片
    for file in selected_files:
        if is_image_file(file['name']):
            # 解码图片（假设是 base64）
            try:
                image_data = base64.b64decode(file['content'])
                img = Image.open(io.BytesIO(image_data))

                print(f"\n文件: {file['name']}")
                print(f"  尺寸: {img.size}")
                print(f"  格式: {img.format}")
                print(f"  模式: {img.mode}")
                print(f"  大小: {len(image_data)/1024:.1f}KB")
            except Exception as e:
                print(f"\n文件: {file['name']}")
                print(f"  ✗ 无法解析: {e}")
```

---

## 📋 十、需要后端配合的改动

### 10.1 图片文件的编码

**问题**: 图片是二进制数据，需要 base64 编码

**建议**:
```javascript
// 前端上传图片时
const file = event.target.files[0];
const reader = new FileReader();

reader.onload = (e) => {
  const base64Content = btoa(
    new Uint8Array(e.target.result)
      .reduce((data, byte) => data + String.fromCharCode(byte), '')
  );

  // 发送到后端
  selectedFiles.push({
    id: file.id,
    name: file.name,
    path: file.path,
    content: base64Content,
    encoding: 'base64',  // 标记编码方式
    contentType: file.type
  });
};

reader.readAsArrayBuffer(file);
```

### 10.2 内容类型识别

**建议在 selected_files 中添加字段**:
```python
{
  "id": "9011",
  "name": "photo.jpg",
  "path": "dataset/photo.jpg",
  "content": "base64_string",
  "content_type": "image/jpeg",    # 新增
  "encoding": "base64",            # 新增
  "size": 102400                   # 新增（可选）
}
```

---

## ✅ 总结与行动计划

### 立即可做的（无需改动）

1. ✅ 表格数据治理 - 完全支持
2. ✅ 文本基础处理 - 基本支持
3. ✅ JSON 处理 - 完全支持
4. ✅ 图片信息查看 - 可以做（如果图片是 base64）

### 需要小改动

1. 🔨 图片格式转换 - 需要 base64 编码支持
2. 🔨 图片压缩 - 需要 base64 编码支持
3. 🔨 图片尺寸调整 - 需要 base64 编码支持

### 需要新增依赖

1. 🔨 Excel 多 sheet - 需要 openpyxl
2. 🔨 文本高级处理 - 可选

---

**建议优先级**:

1. **P0**: 确保图片能正确传递（base64 编码）
2. **P0**: 实现图片基础处理（格式转换、压缩、调整）
3. **P1**: 图片批量处理和统计分析
4. **P1**: Excel 多 sheet 支持
5. **P2**: 文本高级处理

**预计总工作量**: 5-7 天（含测试）

---

**文档版本**: v1.0
**创建日期**: 2025-10-31
**状态**: 规划中
