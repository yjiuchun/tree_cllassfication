#!/usr/bin/python3
# Copyright 2023 WolkenVision AG. All rights reserved.
"""a simple plant classification script using timm, for test only"""

import timm
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
# 1. 加载你的预训练模型

vit_large_14_336 = "hf_hub:timm/vit_large_patch14_clip_336.datacompxl_ft_augreg_inat21"
eva_large_14_336 = "hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"
model = timm.create_model(vit_large_14_336, pretrained=True)
name_model = timm.create_model(eva_large_14_336, pretrained=True)
# 关键：模型设置为「推理模式」，关闭Dropout/BatchNorm的训练行为
model.eval()
name_model.eval()

# 2. 官方标准预处理流程（固定写法，适配该模型，不能修改尺寸/标准化参数）
transform = transforms.Compose([
    transforms.Resize((336, 336)),  # 强制缩放到模型要求的336x336
    transforms.ToTensor(),          # 转tensor：0-255 → 0-1，shape [H,W,C] → [C,H,W]
    # timm官方标准化均值和方差（适配所有CLIP/EVA02系列模型）
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

# 3. 获取类别名称列表（list格式，索引=类别ID，值=物种名称）
inat21_class_names = name_model.default_cfg["label_names"]

# 4. 加载图片+预处理+推理

img_path = "/root/autodl-fs/dataset_150/0_05823_humilis/00bfc468-ea01-46bd-96c4-a6f9b95b13dd.jpg"

img = Image.open(img_path).convert('RGB')  # 强制转RGB，避免灰度图报错
img_tensor = transform(img).unsqueeze(0)   # 增加batch维度：[3,336,336] → [1,3,336,336]

# 4. 无梯度推理（加速+省显存）
with torch.no_grad():
    output = model(img_tensor)  # output shape: [1, 1000] 对应inat21的1000个类别

# 5. 输出预测结果（取概率最大的类别ID）
pred_class_id = torch.argmax(output, dim=1).item()
print(output.shape)


# 6. 查询ID=8863对应的物种
# target_id = pred_class_id
# if 0 <= target_id < len(inat21_class_names):
#     print(f"类别ID {target_id} 对应的物种：{inat21_class_names[target_id]}")
# else:
#     print(f"错误：类别ID {target_id} 超出范围（iNat2021共{len(inat21_class_names)}类）")