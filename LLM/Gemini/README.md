# Gemini 树种识别

使用 Google Gemini API 对图片进行树种识别。

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install google-generativeai Pillow
```

## 配置 API Key

### 方法1: 环境变量（推荐）

```bash
export GEMINI_API_KEY='your-api-key-here'
```

### 方法2: 在代码中设置

编辑 `identify_trees.py`，修改：

```python
API_KEY = "your-api-key-here"
```

## 获取 Gemini API Key

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 登录你的 Google 账号
3. 创建新的 API Key
4. 复制 API Key 并保存

## 使用方法

### 1. 测试 API Key

```bash
python test_key.py
```

**注意**: 如果遇到模型找不到的错误（404），可以尝试使用 REST API 版本（见下方）。

### 2. 运行识别脚本

**方法1: 使用 google.generativeai 包（推荐，如果可用）**

```bash
python identify_trees.py
```

**方法2: 使用 REST API（如果方法1遇到模型找不到的问题）**

```bash
python identify_trees_rest.py
```

REST API 版本不依赖 `google.generativeai` 包，直接使用 HTTP 请求调用 Gemini API，兼容性更好。

脚本会：
- 读取 `/home/yjc/Project/plant_classfication/LLM/folder_names.csv` 中的树种列表
- 处理 `/home/yjc/Project/plant_classfication/LLM/images` 文件夹中的所有图片
- 为每张图片生成最有可能的树种预测
- 将结果保存为 JSON 和 CSV 格式

### 3. 查看结果

结果文件保存在 `Gemini/` 目录下，文件名格式为：
- `results_YYYYMMDD_HHMMSS.json` - JSON 格式（包含完整信息）
- `results_YYYYMMDD_HHMMSS.csv` - CSV 格式（便于查看）

## 模型选择

默认使用 `gemini-pro`（基础模型，兼容性最好）

脚本会自动尝试以下模型（按顺序）：
1. 环境变量 `GEMINI_MODEL` 指定的模型
2. `gemini-pro`（默认）
3. `gemini-1.5-pro`（更准确但较慢）
4. `gemini-1.5-flash`（更快，适合批量处理）

可以通过环境变量指定优先使用的模型：

```bash
export GEMINI_MODEL="gemini-1.5-pro"  # 更准确但较慢
python identify_trees.py
```

**注意**：如果指定的模型不可用，脚本会自动尝试其他可用模型。

## 输出格式

### JSON 格式示例

```json
[
  {
    "image": "example.jpg",
    "image_path": "/path/to/image.jpg",
    "prediction": "sylvestris",
    "timestamp": "2024-01-01T12:00:00"
  }
]
```

### CSV 格式示例

```csv
image,prediction,timestamp
example.jpg,sylvestris,2024-01-01T12:00:00
```

## 注意事项

1. **API 配额**: Gemini API 有免费配额限制，大量图片可能需要付费
2. **处理速度**: 每张图片需要几秒到几十秒，取决于模型和网络
3. **错误处理**: 脚本会自动重试失败的请求，并保存中间结果
4. **图片格式**: 支持 JPG, PNG, BMP, GIF, WEBP 等常见格式

## 故障排除

### 模型找不到（404 错误）

如果遇到 "404 models/xxx is not found" 错误：

1. **使用 REST API 版本**（推荐）:
   ```bash
   python identify_trees_rest.py
   ```

2. **查看可用模型**:
   ```bash
   python list_models.py
   ```

3. **检查 API Key 权限**: 确保 API Key 有访问相应模型的权限

### API Key 无效

- 检查 API Key 是否正确
- 确认 API Key 已启用
- 运行 `python test_key.py` 测试

### 配额不足

- 检查 [Google Cloud Console](https://console.cloud.google.com/) 中的配额设置
- 考虑升级到付费计划

### 识别结果不在列表中

- 模型可能返回了不在树种列表中的名称
- 检查 CSV 文件中的树种列表是否完整
- 查看 JSON 输出中的原始预测结果

