# 图片处理功能使用指南

## 📋 概述

Python Executor Service 现已完整支持图片处理功能，基于 **PIL/Pillow** 库提供强大的图像处理能力。

---

## ✅ 支持的功能

### 1. 格式转换
- PNG ↔ JPEG ↔ BMP ↔ GIF
- 自动颜色模式转换（RGBA → RGB）
- 批量转换

### 2. 图片压缩
- 智能尺寸调整（保持宽高比）
- 质量控制（1-100）
- 优化压缩算法

### 3. 图片分析
- 格式统计
- 尺寸分布
- 文件大小统计
- 颜色模式分析

### 4. 图片增强
- 亮度调整
- 对比度调整
- 锐化处理
- 滤镜应用

---

## 🎯 快速开始

### 前提条件

**图片数据必须使用 base64 编码传递**

---

## 📤 前端准备图片数据

### JavaScript 示例

```javascript
/**
 * 读取图片文件并转换为 base64
 */
async function readImageAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (e) => {
            // 移除 "data:image/xxx;base64," 前缀
            const base64String = e.target.result.split(',')[1];

            resolve({
                name: file.name,
                content: base64String,
                content_type: file.type,
                encoding: 'base64'
            });
        };

        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * 批量处理图片文件
 */
async function prepareImageDatasets(files) {
    const datasets = {};

    for (const file of files) {
        if (file.type.startsWith('image/')) {
            const imageData = await readImageAsBase64(file);
            datasets[file.name] = imageData.content;
        }
    }

    return datasets;
}

/**
 * 完整示例：上传图片并处理
 */
async function processImages() {
    const fileInput = document.getElementById('imageInput');
    const files = fileInput.files;

    // 准备数据集
    const datasets = await prepareImageDatasets(files);

    // 调用执行服务
    const response = await fetch('http://localhost:8000/execute', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            code: `
from PIL import Image
import io
import base64

print(f"收到 {len(selected_files)} 个图片文件")

for file in selected_files:
    img_data = base64.b64decode(file['content'])
    img = Image.open(io.BytesIO(img_data))
    print(f"{file['name']}: {img.size}, {img.format}")
            `,
            datasets: datasets
        })
    });

    const result = await response.json();
    console.log(result.output.stdout);
}
```

---

## 📊 API 使用示例

### 示例 1: 图片格式转换

**API 请求**：
```json
{
  "code": "# 使用预置模板\n# 模板会自动转换所有图片为 PNG 格式",
  "datasets": {
    "photo1.jpg": "base64_encoded_content_here",
    "photo2.bmp": "base64_encoded_content_here"
  }
}
```

或使用模板：
```bash
# 获取模板
GET http://localhost:8000/templates/image_format_convert

# 执行模板
POST http://localhost:8000/execute
{
  "code": "从上面获取的模板代码",
  "datasets": { ... }
}
```

**输出示例**：
```
============================================================
批量图片格式转换
============================================================
✓ photo1.jpg
  JPEG -> PNG
  245.3KB -> 198.7KB
✓ photo2.bmp
  BMP -> PNG
  512.0KB -> 156.4KB

成功转换 2 个文件
```

---

### 示例 2: 图片压缩

**Python 代码**：
```python
from PIL import Image
import io
import base64

# 配置参数
MAX_SIZE = (1920, 1080)
QUALITY = 85

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        # 调整尺寸
        img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)

        # 压缩
        output = io.BytesIO()
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output, format='JPEG', quality=QUALITY, optimize=True)

        print(f"✓ {file['name']}: 已压缩")
```

或使用预置模板：
```bash
GET http://localhost:8000/templates/image_compress
```

---

### 示例 3: 图片数据集分析

**使用预置模板**：
```python
# 获取模板
GET http://localhost:8000/templates/image_analysis
```

**输出示例**：
```
============================================================
图片数据集分析报告
============================================================

📊 总图片数: 15

📁 格式分布:
  JPEG: 10 (66.7%)
  PNG: 4 (26.7%)
  BMP: 1 (6.7%)

🎨 颜色模式:
  RGB: 12
  RGBA: 3

📐 尺寸统计:
  宽度: 最小=640, 最大=4096, 平均=1920
  高度: 最小=480, 最大=3072, 平均=1080

💾 文件大小:
  最小: 45.3KB
  最大: 2345.6KB
  平均: 512.8KB
  总计: 7.51MB
```

---

### 示例 4: 图片增强

**使用预置模板**：
```python
GET http://localhost:8000/templates/image_enhance
```

**输出示例**：
```
============================================================
批量图片增强处理
============================================================
✓ photo1.jpg
  亮度: +20%
  对比度: +10%
  锐化: +50%
✓ photo2.png
  亮度: +20%
  对比度: +10%
  锐化: +50%

成功增强 2 个文件
```

---

## 🔧 自定义图片处理

### 完整示例：批量处理工作流

```python
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64

print("=" * 60)
print("自定义图片处理工作流")
print("=" * 60)

processed_count = 0
total_original_size = 0
total_new_size = 0

for file in selected_files:
    if not file['name'].lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    try:
        # 1. 解码图片
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        original_size = len(img_data)
        total_original_size += original_size

        print(f"\n处理: {file['name']}")
        print(f"  原始: {img.size}, {img.format}, {original_size/1024:.1f}KB")

        # 2. 调整尺寸（如果过大）
        if img.width > 2048 or img.height > 2048:
            img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            print(f"  → 调整尺寸: {img.size}")

        # 3. 增强处理
        # 亮度
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)

        # 对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.05)

        # 锐化
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)

        print(f"  → 应用增强效果")

        # 4. 可选：应用滤镜
        # img = img.filter(ImageFilter.SHARPEN)

        # 5. 保存（压缩）
        output = io.BytesIO()
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output, format='JPEG', quality=90, optimize=True)

        new_size = output.tell()
        total_new_size += new_size
        saved = original_size - new_size
        saved_percent = (saved / original_size) * 100

        print(f"  → 压缩: {new_size/1024:.1f}KB (节省 {saved_percent:.1f}%)")

        # 6. 结果可以通过 base64 返回
        result_base64 = base64.b64encode(output.getvalue()).decode()
        # 这里可以存储或进一步处理

        processed_count += 1
        print(f"  ✓ 处理完成")

    except Exception as e:
        print(f"  ✗ 处理失败: {e}")

# 汇总统计
print(f"\n{'='*60}")
print(f"处理完成统计")
print(f"{'='*60}")
print(f"成功处理: {processed_count} 个文件")
print(f"原始总大小: {total_original_size/1024/1024:.2f}MB")
print(f"处理后总大小: {total_new_size/1024/1024:.2f}MB")
print(f"节省空间: {(total_original_size - total_new_size)/1024/1024:.2f}MB")
print(f"压缩比: {(1 - total_new_size/total_original_size)*100:.1f}%")
```

---

## 🎨 高级图片操作

### 旋转和翻转

```python
from PIL import Image
import io
import base64

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        # 旋转 90 度
        rotated = img.rotate(90, expand=True)

        # 水平翻转
        flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 垂直翻转
        flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)

        print(f"✓ {file['name']}: 已旋转和翻转")
```

### 裁剪

```python
from PIL import Image
import io
import base64

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        # 裁剪中心区域 (left, top, right, bottom)
        width, height = img.size
        left = width // 4
        top = height // 4
        right = width * 3 // 4
        bottom = height * 3 // 4

        cropped = img.crop((left, top, right, bottom))

        print(f"✓ {file['name']}: {img.size} → {cropped.size}")
```

### 滤镜效果

```python
from PIL import Image, ImageFilter
import io
import base64

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        # 模糊
        blurred = img.filter(ImageFilter.BLUR)

        # 锐化
        sharpened = img.filter(ImageFilter.SHARPEN)

        # 边缘增强
        edge_enhanced = img.filter(ImageFilter.EDGE_ENHANCE)

        # 轮廓检测
        contour = img.filter(ImageFilter.CONTOUR)

        # 浮雕效果
        embossed = img.filter(ImageFilter.EMBOSS)

        print(f"✓ {file['name']}: 应用了 5 种滤镜")
```

### 颜色调整

```python
from PIL import Image, ImageEnhance
import io
import base64

for file in selected_files:
    if file['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        # 亮度 (0.0 = 黑色, 1.0 = 原始, 2.0 = 更亮)
        brightness = ImageEnhance.Brightness(img)
        img_bright = brightness.enhance(1.5)

        # 对比度
        contrast = ImageEnhance.Contrast(img)
        img_contrast = contrast.enhance(1.3)

        # 颜色饱和度
        color = ImageEnhance.Color(img)
        img_color = color.enhance(1.2)

        # 锐度
        sharpness = ImageEnhance.Sharpness(img)
        img_sharp = sharpness.enhance(2.0)

        print(f"✓ {file['name']}: 颜色调整完成")
```

---

## 📊 预置模板列表

| 模板名称 | 功能 | 获取方式 |
|---------|------|---------|
| `image_format_convert` | 批量格式转换为 PNG | `GET /templates/image_format_convert` |
| `image_compress` | 批量压缩（调整尺寸+质量） | `GET /templates/image_compress` |
| `image_analysis` | 数据集统计分析 | `GET /templates/image_analysis` |
| `image_enhance` | 批量增强（亮度+对比度+锐化） | `GET /templates/image_enhance` |

---

## ⚙️ 常用配置参数

### 图片质量

```python
# JPEG 质量（1-100）
quality = 85  # 推荐：85-95（高质量）
quality = 75  # 推荐：75-85（平衡）
quality = 60  # 推荐：60-75（小文件）
```

### 尺寸调整

```python
# 保持宽高比
img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)

# 固定尺寸（可能变形）
img = img.resize((800, 600), Image.Resampling.LANCZOS)

# 常见尺寸
THUMBNAIL = (150, 150)      # 缩略图
WEB_SMALL = (640, 480)      # 网页小图
WEB_MEDIUM = (1280, 720)    # 网页中图
WEB_LARGE = (1920, 1080)    # 网页大图
PRINT = (3000, 2000)        # 打印
```

### 重采样算法

```python
# 最高质量（推荐）
Image.Resampling.LANCZOS

# 其他选项
Image.Resampling.BICUBIC   # 双三次插值
Image.Resampling.BILINEAR  # 双线性插值
Image.Resampling.NEAREST   # 最近邻（最快，质量最低）
```

---

## 🔍 调试和问题排查

### 检查图片是否正确接收

```python
import base64

print(f"选中文件数: {len(selected_files)}")

for file in selected_files:
    print(f"\n文件: {file['name']}")
    print(f"  内容长度: {len(file['content'])} 字符")
    print(f"  前20字符: {file['content'][:20]}...")

    # 尝试解码
    try:
        img_data = base64.b64decode(file['content'])
        print(f"  ✓ Base64 解码成功，{len(img_data)} 字节")
    except Exception as e:
        print(f"  ✗ Base64 解码失败: {e}")
```

### 查看图片详细信息

```python
from PIL import Image
import io
import base64

for file in selected_files:
    try:
        img_data = base64.b64decode(file['content'])
        img = Image.open(io.BytesIO(img_data))

        print(f"\n{file['name']}:")
        print(f"  格式: {img.format}")
        print(f"  模式: {img.mode}")
        print(f"  尺寸: {img.size}")
        print(f"  文件大小: {len(img_data)/1024:.1f}KB")

        # 如果有 EXIF 信息
        if hasattr(img, '_getexif') and img._getexif():
            print(f"  EXIF: 有")

    except Exception as e:
        print(f"\n{file['name']}: ✗ {e}")
```

---

## 💡 最佳实践

### 1. 批量处理时的错误处理

```python
success_count = 0
error_count = 0
errors = []

for file in selected_files:
    try:
        # 处理逻辑...
        success_count += 1
    except Exception as e:
        error_count += 1
        errors.append(f"{file['name']}: {e}")

print(f"\n成功: {success_count}, 失败: {error_count}")
if errors:
    print("\n错误详情:")
    for error in errors:
        print(f"  - {error}")
```

### 2. 渐进式质量调整

```python
# 逐步降低质量直到满足目标大小
target_size_kb = 500

for quality in range(95, 50, -5):
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=quality)

    if output.tell() / 1024 <= target_size_kb:
        print(f"使用质量 {quality} 达到目标大小")
        break
```

### 3. 保持原始宽高比

```python
def resize_keep_ratio(img, max_width, max_height):
    """调整尺寸但保持宽高比"""
    width, height = img.size
    ratio = min(max_width / width, max_height / height)

    if ratio < 1:  # 只有图片太大时才缩小
        new_size = (int(width * ratio), int(height * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    return img
```

---

## 🚀 性能优化建议

### 1. 批量处理

对于大量图片，考虑分批处理：

```python
BATCH_SIZE = 10

for i in range(0, len(selected_files), BATCH_SIZE):
    batch = selected_files[i:i+BATCH_SIZE]
    print(f"处理批次 {i//BATCH_SIZE + 1}...")

    for file in batch:
        # 处理...
        pass
```

### 2. 质量预设

根据用途选择合适的质量：

```python
QUALITY_PRESETS = {
    'thumbnail': (150, 150, 70),      # (宽, 高, 质量)
    'web': (1280, 720, 85),
    'print': (3000, 2000, 95),
    'storage': (1920, 1080, 75),
}

preset = QUALITY_PRESETS['web']
img.thumbnail((preset[0], preset[1]), Image.Resampling.LANCZOS)
img.save(output, quality=preset[2])
```

---

## 📚 相关资源

- **PIL/Pillow 官方文档**: https://pillow.readthedocs.io/
- **图片处理最佳实践**: 参考 UPGRADE_RECOMMENDATIONS_V2.md
- **API 文档**: http://localhost:8000/docs
- **代码模板**: `GET /templates`

---

## ❓ 常见问题

### Q: 如何从前端发送图片？
A: 使用 FileReader API 读取为 base64，详见"前端准备图片数据"章节。

### Q: 支持哪些图片格式？
A: JPEG, PNG, BMP, GIF, TIFF, WebP 等常见格式。

### Q: 图片太大怎么办？
A: 使用 `image_compress` 模板，或自定义压缩参数。

### Q: 如何保存处理后的图片？
A: 图片在内存中处理，结果通过 base64 返回到前端，由前端负责保存。

### Q: 可以批量处理多少张图片？
A: 建议单次不超过 20 张，每张不超过 5MB（base64 编码前）。

---

**文档版本**: v1.0
**创建日期**: 2025-11-03
**最后更新**: 2025-11-03
**状态**: ✅ 完成
