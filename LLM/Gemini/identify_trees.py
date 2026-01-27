#!/usr/bin/env python3
"""
使用 Gemini API 进行树种识别
识别 /home/yjc/Project/plant_classfication/LLM/images 文件夹中的图片
"""

import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from PIL import Image

# 配置
# 优先从环境变量读取API Key
API_KEY = os.getenv("GEMINI_API_KEY", "")
FOLDER_NAMES_CSV = "/home/yjc/Project/plant_classfication/LLM/folder_names.csv"
IMAGES_DIR = "/home/yjc/Project/plant_classfication/LLM/images"
OUTPUT_DIR = "/home/yjc/Project/plant_classfication/LLM/Gemini"

# 模型选择
# "gemini-pro" - 基础模型
# "gemini-1.5-pro" - 更准确但较慢
# "gemini-1.5-flash" - 更快，适合批量处理（推荐）
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-pro")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

def load_species_list(csv_path):
    """从CSV文件加载树种列表"""
    species = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            if row and row[0].strip():
                species.append(row[0].strip())
    return species

def get_image_files(folder_path):
    """获取文件夹中的所有图片文件"""
    image_files = []
    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return image_files
    
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file)
    return sorted(image_files)

def identify_tree_species(model, image_path, species_list, max_retries=3):
    """使用Gemini API识别树种"""
    # 构建提示词
    # 每行显示多个名称以节省空间（每行5个）
    species_lines = []
    for i in range(0, len(species_list), 5):
        line_species = species_list[i:i+5]
        species_lines.append(" | ".join(line_species))
    species_text = "\n".join(species_lines)
    
    prompt = f"""你是植物分类专家。分析图片中的树木，从树种列表中选择最匹配的拉丁学名种加词。

**规则：**
1. 列表中的名称是拉丁学名种加词（如 sylvestris, pendula, nigra）或中文名称（如 银杏、水杉）
2. 根据叶形、树皮、树形、果实/球果等特征判断
3. 必须返回列表中完全匹配的名称，不要其他文字
4. 即使不确定，也要选择最接近的匹配，不要返回"未知"

**树种列表（共{len(species_list)}种）：**
{species_text}

**只返回最匹配的树种名称：**"""

    for attempt in range(max_retries):
        try:
            # 读取图片
            img = Image.open(image_path)
            
            # 调用Gemini API
            response = model.generate_content([prompt, img])
            
            prediction = response.text.strip()
            
            # 后处理：清理预测结果，确保匹配列表中的名称
            # 移除可能的标点符号和多余文字
            prediction_clean = prediction.strip().rstrip('.,;:!?')
            
            # 检查是否完全匹配列表中的某个名称
            if prediction_clean in species_list:
                return prediction_clean
            
            # 如果不完全匹配，尝试模糊匹配（忽略大小写和空格）
            prediction_lower = prediction_clean.lower()
            for species in species_list:
                if species.lower() == prediction_lower or species.lower() in prediction_lower or prediction_lower in species.lower():
                    return species
            
            # 如果还是找不到匹配，尝试提取可能的名称
            words = prediction_clean.split()
            for word in words:
                word_clean = word.strip().rstrip('.,;:!?')
                if word_clean in species_list:
                    return word_clean
            
            # 如果完全无法匹配，返回"未知"但保留原始预测用于调试
            print(f"  警告: 预测结果 '{prediction_clean}' 不在树种列表中")
            return "未知"
            
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # 如果是配额不足错误，不重试
            if "quota" in error_str or "rate limit" in error_str:
                print(f"  ❌ API 配额/频率限制: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    return f"错误: API配额/频率限制"
            
            # 其他错误可以重试
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  识别出错 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  识别失败 (已重试 {max_retries} 次): {e}")
                return f"错误: {str(e)}"

def main():
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f"results_{timestamp}.json")
    
    # 检查API Key
    if not API_KEY:
        print("错误: 请设置Gemini API Key!")
        print("方法1: 在代码中修改 API_KEY 变量")
        print("方法2: 设置环境变量 GEMINI_API_KEY")
        print("例如: export GEMINI_API_KEY='your-key-here'")
        return
    
    # 配置Gemini API
    genai.configure(api_key=API_KEY)
    
    # 初始化模型 - 尝试多个模型名称和格式
    model = None
    used_model = None
    
    # 尝试不同的模型名称格式
    model_names_to_try = [
        MODEL_NAME,
        "gemini-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "models/gemini-pro",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
    ]
    
    for model_name in model_names_to_try:
        try:
            print(f"尝试使用模型: {model_name}...")
            model = genai.GenerativeModel(model_name)
            # 简单测试调用
            test_response = model.generate_content("test")
            used_model = model_name
            print(f"✅ 成功使用模型: {used_model}")
            break
        except Exception as e:
            error_msg = str(e)
            # 如果是 404 错误，继续尝试下一个
            if "404" in error_msg or "not found" in error_msg.lower():
                print(f"  ❌ {model_name} 不可用: 模型不存在")
            else:
                print(f"  ❌ {model_name} 不可用: {error_msg[:100]}")
            continue
    
    if model is None:
        print("❌ 错误: 无法找到可用的模型")
        print("\n建议:")
        print("1. 检查 API Key 是否正确")
        print("2. 尝试使用 REST API 版本: python identify_trees_rest.py")
        print("3. 查看可用模型列表: python list_models.py")
        print("\n尝试列出可用模型...")
        try:
            models = genai.list_models()
            print("可用模型列表:")
            found_any = False
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    print(f"  - {m.name}")
                    found_any = True
            if not found_any:
                print("  (未找到支持 generateContent 的模型)")
        except Exception as e2:
            print(f"无法列出模型: {e2}")
        return
    
    # 加载树种列表
    print("正在加载树种列表...")
    species_list = load_species_list(FOLDER_NAMES_CSV)
    print(f"已加载 {len(species_list)} 个树种")
    
    # 获取所有图片文件
    print(f"\n正在扫描图片文件夹: {IMAGES_DIR}")
    image_files = get_image_files(IMAGES_DIR)
    
    if not image_files:
        print(f"错误: 在 {IMAGES_DIR} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 存储结果
    results = []
    start_time = time.time()
    
    # 遍历每张图片
    for idx, image_path in enumerate(image_files, 1):
        image_name = image_path.name
        print(f"\n[{idx}/{len(image_files)}] 处理图片: {image_name}")
        
        # 调用API识别
        print(f"  正在识别...")
        prediction = identify_tree_species(model, image_path, species_list)
        
        # 记录结果
        result = {
            "image": image_name,
            "image_path": str(image_path),
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        print(f"  预测结果: {prediction}")
        
        # 每处理10张图片保存一次（防止数据丢失）
        if idx % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / idx
            remaining = (len(image_files) - idx) * avg_time
            print(f"  已保存中间结果到 {output_file}")
            print(f"  进度: {idx}/{len(image_files)} ({idx/len(image_files)*100:.1f}%), 预计剩余时间: {remaining/60:.1f} 分钟")
    
    # 保存最终结果
    print(f"\n正在保存最终结果...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 同时保存为CSV格式便于查看
    csv_output = output_file.replace('.json', '.csv')
    with open(csv_output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'prediction', 'timestamp'])
        writer.writeheader()
        for result in results:
            writer.writerow({
                'image': result['image'],
                'prediction': result['prediction'],
                'timestamp': result['timestamp']
            })
    
    total_time = time.time() - start_time
    print(f"\n完成！共处理 {len(results)} 张图片")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"平均每张图片: {total_time/len(results):.1f} 秒")
    print(f"结果已保存到:")
    print(f"  JSON: {output_file}")
    print(f"  CSV: {csv_output}")

if __name__ == "__main__":
    main()

