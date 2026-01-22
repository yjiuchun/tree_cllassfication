# 基于 timm iNaturalist 预训练模型的微调

本目录包含用于在 iNaturalist 预训练模型基础上进行微调的完整代码，支持自定义数据集和类别数。

## 数据集结构

数据集应按以下结构组织（ImageFolder 格式）：

```
dataset/
├── 0_0656_tree1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── 1_0654_tree2/
│   ├── img1.jpg
│   └── ...
└── 2_8877_tree3/
    └── ...
```

### 文件夹命名规则

- **格式**: `{label}_{其他信息}`
- **第一个下划线前的数字**代表分类标签（0, 1, 2, ...）
- 例如：
  - `0_0656_tree1` → 标签为 0
  - `1_0654_tree2` → 标签为 1
  - `2_8877_tree3` → 标签为 2

**注意**：
- 标签会自动重新映射为连续的整数（0, 1, 2, ...）
- 即使原始标签不连续（如 0, 5, 10），也会被映射为 0, 1, 2
- 文件夹内的图片即为该类的样本

## 快速开始

### 1. 基本使用（iNaturalist 模型）

使用 iNaturalist 预训练的 EVA02 模型进行微调：

```bash
python train.py \
    --dataset dataset \
    --model hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.0001 \
    --use-inat-norm
```

**注意**：脚本会自动检测 iNaturalist 模型并设置：
- 图像尺寸为 336x336
- 使用 iNaturalist 归一化参数

### 2. 使用其他 timm 模型

```bash
python train.py \
    --dataset dataset \
    --model resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

### 3. 冻结骨干网络（只训练分类头）

适用于小数据集或快速实验：

```bash
python train.py \
    --dataset dataset \
    --model hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 \
    --freeze-backbone \
    --epochs 20 \
    --lr 0.01 \
    --use-inat-norm
```

### 4. 恢复训练

```bash
python train.py \
    --dataset dataset \
    --model hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 \
    --resume outputs/checkpoint_epoch_10.pth \
    --use-inat-norm
```

## 主要参数说明

### 数据参数
- `--dataset`: 数据集路径（默认: `dataset`）
- `--batch-size`: 批次大小（默认: 32）
- `--image-size`: 输入图像尺寸（默认: 224，iNaturalist 模型自动设为 336）
- `--val-split`: 验证集比例（默认: 0.2）
- `--test-split`: 测试集比例（默认: 0.1）

### 模型参数
- `--model`: timm 模型名称
  - iNaturalist 模型示例: `hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21`
  - 其他模型: `resnet50`, `vit_base_patch16_224`, `efficientnet_b0` 等
- `--pretrained`: 使用预训练权重（默认: True）
- `--freeze-backbone`: 冻结整个骨干网络，只训练分类头
- `--freeze-layers`: 冻结前 N 层（默认: 0）
- `--use-inat-norm`: 使用 iNaturalist 模型的归一化参数（CLIP/EVA02 系列）

### 训练参数
- `--epochs`: 训练轮数（默认: 50）
- `--lr`: 初始学习率（默认: 0.001）
- `--optimizer`: 优化器选择 `sgd`, `adam`, `adamw`（默认: `adamw`）
- `--scheduler`: 学习率调度器 `cosine`, `step`, `plateau`（默认: `cosine`）
- `--weight-decay`: 权重衰减（默认: 1e-4）

### 其他参数
- `--device`: 设备 `cuda` 或 `cpu`（默认: 自动检测）
- `--output-dir`: 输出目录（默认: `outputs`）
- `--resume`: 恢复训练的检查点路径
- `--seed`: 随机种子（默认: 42）

## 模型微调策略

### 1. 全模型微调（推荐用于中等/大数据集）

```bash
python train.py \
    --dataset dataset \
    --model hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.0001 \
    --use-inat-norm
```

**特点**：
- 所有参数都可训练
- 需要更多计算资源和时间
- 通常获得更好的性能

### 2. 只训练分类头（推荐用于小数据集）

```bash
python train.py \
    --dataset dataset \
    --model hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 \
    --freeze-backbone \
    --epochs 20 \
    --lr 0.01 \
    --use-inat-norm
```

**特点**：
- 只训练分类头，骨干网络冻结
- 训练速度快，显存占用少
- 适合小数据集（< 1000 样本）

### 3. 渐进式解冻（手动控制）

先冻结训练分类头，然后逐步解冻：

```bash
# 第一步：只训练分类头
python train.py --freeze-backbone --epochs 10 --lr 0.01

# 第二步：解冻最后几层（需要修改代码或使用 --freeze-layers）
# 第三步：全模型微调
python train.py --epochs 30 --lr 0.0001
```

## 输出文件

训练完成后，在 `outputs/` 目录下会生成：

- `checkpoint_epoch_N.pth`: 每个 epoch 的检查点
- `best_model.pth`: 验证集上表现最好的模型
- 检查点包含：
  - 模型权重
  - 优化器状态
  - 训练历史
  - 类别名称
  - 训练参数

## 使用微调后的模型进行推理

```python
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

# 加载模型
checkpoint = torch.load('outputs/best_model.pth')
model = timm.create_model(
    'hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21',
    pretrained=False,
    num_classes=checkpoint['args'].num_classes  # 使用训练时的类别数
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预处理（使用 iNaturalist 归一化）
transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

# 推理
img = Image.open('test.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    pred_class_id = torch.argmax(output, dim=1).item()
    class_name = checkpoint['class_names'][pred_class_id]
    print(f"预测类别: {class_name}")
```

## 常见问题

### Q: 如何选择合适的模型？

- **小数据集（< 1000 样本）**: 使用较小的模型如 `resnet50` 或 `efficientnet_b0`
- **中等数据集（1000-10000 样本）**: 使用 `vit_base_patch16_224` 或 `eva02_base`
- **大数据集（> 10000 样本）**: 可以使用 `eva02_large` 等大模型

### Q: 如何选择学习率？

- **全模型微调**: 较小的学习率（1e-4 到 1e-3）
- **只训练分类头**: 较大的学习率（1e-2 到 1e-1）
- **iNaturalist 模型**: 建议从 1e-4 开始

### Q: 显存不足怎么办？

- 减小 `--batch-size`
- 使用 `--freeze-backbone` 只训练分类头
- 使用较小的模型
- 减小 `--image-size`（如果不是 iNaturalist 模型）

### Q: 训练很慢怎么办？

- 增加 `--num-workers`（但不要超过 CPU 核心数）
- 使用 GPU（确保 `--device cuda`）
- 使用较小的模型或冻结更多层

## 参考

- [timm 文档](https://github.com/huggingface/pytorch-image-models)
- [iNaturalist 数据集](https://www.inaturalist.org/)
- [EVA02 模型](https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b_ft_inat21)
