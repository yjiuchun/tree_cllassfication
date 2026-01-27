#!/usr/bin/env python3
"""
从 dataset_val 中随机选择 100 个文件夹，每个文件夹随机选择一张图片
将图片命名为子文件夹第一个下划线之前的 ID，保存到 LLM/images
"""

import random
import shutil
from pathlib import Path

# 配置路径
DATASET_VAL_DIR = Path("/home/yjc/Project/plant_classfication/timm/tune_inaturalist/dataset_val")
OUTPUT_DIR = Path("/home/yjc/Project/plant_classfication/LLM/images")

# 要选择的文件夹数量
N_FOLDERS = 100

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

def get_all_folders(dataset_dir):
    """获取所有子文件夹"""
    folders = []
    for item in dataset_dir.iterdir():
        if item.is_dir():
            folders.append(item)
    return sorted(folders)

def get_valid_images(folder_path):
    """获取文件夹中所有有效的图片文件（排除以 ._ 开头的）"""
    images = []
    for file in folder_path.iterdir():
        if file.is_file():
            # 排除以 ._ 开头的文件
            if file.name.startswith('._'):
                continue
            # 检查是否是图片文件
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(file)
    return images

def extract_folder_id(folder_name):
    """从文件夹名中提取第一个下划线之前的 ID"""
    # 例如: "0_05823_humilis" -> "0"
    if '_' in folder_name:
        return folder_name.split('_')[0]
    return folder_name

def main():
    print("=" * 60)
    print("随机选择图片并重命名")
    print("=" * 60)
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 获取所有文件夹
    print(f"\n正在扫描文件夹: {DATASET_VAL_DIR}")
    all_folders = get_all_folders(DATASET_VAL_DIR)
    print(f"找到 {len(all_folders)} 个文件夹")
    
    if len(all_folders) < N_FOLDERS:
        print(f"⚠️  警告: 只有 {len(all_folders)} 个文件夹，少于请求的 {N_FOLDERS} 个")
        print(f"将选择所有 {len(all_folders)} 个文件夹")
        selected_folders = all_folders
    else:
        # 随机选择 N_FOLDERS 个文件夹
        selected_folders = random.sample(all_folders, N_FOLDERS)
        print(f"随机选择了 {len(selected_folders)} 个文件夹")
    
    # 处理每个文件夹
    success_count = 0
    skip_count = 0
    error_count = 0
    
    print(f"\n开始处理...")
    print("-" * 60)
    
    for idx, folder in enumerate(selected_folders, 1):
        folder_name = folder.name
        folder_id = extract_folder_id(folder_name)
        
        # 获取文件夹中的有效图片
        images = get_valid_images(folder)
        
        if not images:
            print(f"[{idx}/{len(selected_folders)}] {folder_name}: ❌ 没有找到有效图片")
            skip_count += 1
            continue
        
        # 随机选择一张图片
        selected_image = random.choice(images)
        
        # 目标文件名
        target_name = f"{folder_id}.jpg"
        target_path = OUTPUT_DIR / target_name
        
        # 如果目标文件已存在，跳过或覆盖（这里选择跳过）
        if target_path.exists():
            print(f"[{idx}/{len(selected_folders)}] {folder_name}: ⚠️  {target_name} 已存在，跳过")
            skip_count += 1
            continue
        
        try:
            # 复制图片到目标目录并重命名
            shutil.copy2(selected_image, target_path)
            print(f"[{idx}/{len(selected_folders)}] {folder_name}: ✅ {selected_image.name} -> {target_name}")
            success_count += 1
        except Exception as e:
            print(f"[{idx}/{len(selected_folders)}] {folder_name}: ❌ 错误: {e}")
            error_count += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"成功: {success_count} 个")
    print(f"跳过: {skip_count} 个（无图片或文件已存在）")
    print(f"错误: {error_count} 个")
    print(f"\n所有图片已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

