#!/usr/bin/python3
# Copyright 2023 WolkenVision AG. All rights reserved.
"""a simple plant classification script using timm,处理文件夹下每一类别下的所有图片，并保存结果"""
"""
待测试数据集文件夹结构：
folder
├── class1
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── class2
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...


"""

import timm
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import csv
# 1. 加载你的预训练模型

vit_large_14_336 = "hf_hub:timm/vit_large_patch14_clip_336.datacompxl_ft_augreg_inat21"
eva_large_14_336 = "hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"




model_name = eva_large_14_336
model = timm.create_model(model_name, pretrained=True)
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

data_path = "/home/yjc/Project/plant_classfication/yunnan360/dataset"
result_path = "/home/yjc/Project/plant_classfication/yunnan360"

# 确保结果目录存在
os.makedirs(result_path, exist_ok=True)

# 遍历每个文件夹
for folder in os.listdir(data_path):
    image_folder = os.path.join(data_path, folder)
    
    # 跳过非目录文件
    if not os.path.isdir(image_folder):
        continue
    
    # 为每个文件夹创建单独的结果文件
    # 清理模型名称中的特殊字符，用于文件名
    model_name_clean = model_name.replace("/", "_").replace(":", "_")
    result_file = os.path.join(result_path, f"{model_name_clean}_{folder}.csv")
    
    # 打开CSV文件准备写入
    with open(result_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['图片路径', '图片名称', '预测类别ID', '预测类别名称', '置信度']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"\n处理文件夹: {folder}")
        print(f"结果将保存到: {result_file}")
        
        # 遍历文件夹下的所有图片
        for img_name in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_name)
            
            # 跳过非图片文件
            if not os.path.isfile(img_path) or not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            try:
                img = Image.open(img_path).convert('RGB')  # 强制转RGB，避免灰度图报错
                img_tensor = transform(img).unsqueeze(0)   # 增加batch维度：[3,336,336] → [1,3,336,336]

                # 无梯度推理（加速+省显存）
                with torch.no_grad():
                    output = model(img_tensor)  # output shape: [1, 1000] 对应inat21的1000个类别

                # 计算概率（softmax）
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
                # 获取预测结果（取概率最大的类别ID）
                pred_class_id = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][pred_class_id].item()

                # 查询对应的物种名称
                if 0 <= pred_class_id < len(inat21_class_names):
                    pred_class_name = inat21_class_names[pred_class_id]
                    print(f"  {img_name}: 类别ID {pred_class_id} - {pred_class_name} (置信度: {confidence:.4f})")
                else:
                    pred_class_name = f"未知类别（超出范围）"
                    print(f"  错误：{img_name} - 类别ID {pred_class_id} 超出范围（iNat2021共{len(inat21_class_names)}类）")

                # 写入CSV文件
                writer.writerow({
                    '图片路径': img_path,
                    '图片名称': img_name,
                    '预测类别ID': pred_class_id,
                    '预测类别名称': pred_class_name,
                    '置信度': f"{confidence:.6f}"
                })
                
            except Exception as e:
                print(f"  处理图片 {img_name} 时出错: {str(e)}")
                # 即使出错也记录到CSV
                writer.writerow({
                    '图片路径': img_path,
                    '图片名称': img_name,
                    '预测类别ID': 'N/A',
                    '预测类别名称': f'处理错误: {str(e)}',
                    '置信度': 'N/A'
                })
        
        print(f"文件夹 {folder} 处理完成，结果已保存到: {result_file}")

print(f"\n所有文件夹处理完成！结果文件保存在: {result_path}")