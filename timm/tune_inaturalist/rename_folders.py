#!/usr/bin/env python3
"""
重命名数据集文件夹：从 "0_05823_humilis" 格式改为 "0"
并将原始信息保存到 JSON 文件中
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple


def parse_folder_name(folder_name: str) -> Tuple[str, str, str]:
    """
    解析文件夹名称，格式：数字_数字_名称
    例如: "0_05823_humilis" -> ("0", "05823", "humilis")
    """
    parts = folder_name.split('_', 2)  # 最多分割2次，保留名称中的下划线
    if len(parts) != 3:
        raise ValueError(f"文件夹名称格式不正确: {folder_name}，应为: 数字_数字_名称")
    return parts[0], parts[1], parts[2]


def rename_folders_in_dir(directory: Path, mapping: Dict[str, Dict[str, str]], dry_run: bool = False):
    """
    重命名目录中的所有文件夹
    
    Args:
        directory: 要处理的目录（train 或 val）
        mapping: 用于存储映射关系的字典
        dry_run: 如果为 True，只打印不实际重命名
    """
    if not directory.exists():
        print(f"警告: 目录不存在: {directory}")
        return
    
    folders = [f for f in directory.iterdir() if f.is_dir()]
    print(f"\n处理目录: {directory}")
    print(f"找到 {len(folders)} 个文件夹")
    
    for folder in sorted(folders):
        old_name = folder.name
        
        # 跳过已经重命名过的文件夹（只包含数字）
        if old_name.isdigit():
            print(f"  跳过（已重命名）: {old_name}")
            continue
        
        try:
            id_num, code, name = parse_folder_name(old_name)
            new_name = id_num
            new_path = directory / new_name
            
            # 检查新名称是否已存在（可能是其他文件夹）
            if new_path.exists() and new_path != folder:
                print(f"  警告: 目标文件夹已存在，跳过: {old_name} -> {new_name}")
                continue
            
            # 保存映射关系（以 id_num 为 key）
            if id_num not in mapping:
                mapping[id_num] = {
                    "id": id_num,
                    "code": code,
                    "name": name
                }
            
            if dry_run:
                print(f"  [DRY RUN] {old_name} -> {new_name}")
            else:
                folder.rename(new_path)
                print(f"  ✓ {old_name} -> {new_name}")
                
        except ValueError as e:
            print(f"  ✗ 错误: {old_name} - {e}")
            continue


def main():
    dataset_dir = Path(__file__).parent / "dataset"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    output_json = dataset_dir / "folder_mapping.json"
    
    print("=" * 60)
    print("数据集文件夹重命名工具")
    print("=" * 60)
    print(f"数据集目录: {dataset_dir}")
    print(f"输出 JSON: {output_json}")
    
    # 先进行 dry run 检查
    print("\n" + "=" * 60)
    print("步骤 1: 预览模式（不实际重命名）")
    print("=" * 60)
    mapping = {}
    rename_folders_in_dir(train_dir, mapping, dry_run=True)
    rename_folders_in_dir(val_dir, mapping, dry_run=True)
    
    print(f"\n将创建/更新 {len(mapping)} 个映射关系")
    
    # 询问用户是否继续
    response = input("\n是否继续执行实际重命名？(yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("已取消")
        return
    
    # 实际执行重命名
    print("\n" + "=" * 60)
    print("步骤 2: 执行重命名")
    print("=" * 60)
    mapping = {}  # 重新初始化
    rename_folders_in_dir(train_dir, mapping, dry_run=False)
    rename_folders_in_dir(val_dir, mapping, dry_run=False)
    
    # 保存映射关系到 JSON
    print("\n" + "=" * 60)
    print("步骤 3: 保存映射关系到 JSON")
    print("=" * 60)
    
    # 按 id 排序
    sorted_mapping = dict(sorted(mapping.items(), key=lambda x: int(x[0])))
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(sorted_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 映射关系已保存到: {output_json}")
    print(f"  共 {len(sorted_mapping)} 个类别")
    
    # 显示前几个示例
    print("\n前 5 个映射示例:")
    for i, (id_num, info) in enumerate(list(sorted_mapping.items())[:5]):
        print(f"  {id_num}: {info['name']} (code: {info['code']})")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()




