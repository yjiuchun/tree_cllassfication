#!/usr/bin/env python3
"""
使用 Gemini REST API 进行树种识别（不依赖 google.generativeai 包）
识别 /home/yjc/Project/plant_classfication/LLM/images 文件夹中的图片
"""

import os
import csv
import json
import time
import base64
import requests
from pathlib import Path
from datetime import datetime
from PIL import Image
import io

# 配置
API_KEY = os.getenv("GEMINI_API_KEY", "")
FOLDER_NAMES_CSV = "/home/yjc/Project/plant_classfication/LLM/folder_names.csv"
IMAGES_DIR = "/home/yjc/Project/plant_classfication/LLM/images"
OUTPUT_DIR = "/home/yjc/Project/plant_classfication/LLM/Gemini"

# 模型选择 - 尝试不同的模型
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

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

def encode_image_to_base64(image_path):
    """将图片编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_mime_type(image_path):
    """根据文件扩展名获取 MIME 类型"""
    ext = image_path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')

def identify_tree_species_rest(image_path, species_list, model_name="gemini-1.5-flash", max_retries=3):
    """使用Gemini REST API识别树种"""
    # 构建提示词
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

    # 编码图片
    image_base64 = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    
    # 构建请求
    url = f"{API_BASE_URL}/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_base64
                    }
                }
            ]
        }]
    }
    
    for attempt in range(max_retries):
        try:
            # 发送请求
            response = requests.post(
                url,
                headers=headers,
                params={"key": API_KEY},
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    prediction = result['candidates'][0]['content']['parts'][0]['text'].strip()
                else:
                    raise Exception("API 响应中没有找到预测结果")
            elif response.status_code == 404:
                # 模型不存在，尝试其他模型
                raise Exception(f"模型 {model_name} 不存在 (404)")
            else:
                error_msg = response.text
                raise Exception(f"API 错误 ({response.status_code}): {error_msg[:200]}")
            
            # 后处理：清理预测结果
            prediction_clean = prediction.strip().rstrip('.,;:!?')
            
            # 检查是否完全匹配列表中的某个名称
            if prediction_clean in species_list:
                return prediction_clean
            
            # 模糊匹配
            prediction_lower = prediction_clean.lower()
            for species in species_list:
                if species.lower() == prediction_lower or species.lower() in prediction_lower or prediction_lower in species.lower():
                    return species
            
            # 尝试提取可能的名称
            words = prediction_clean.split()
            for word in words:
                word_clean = word.strip().rstrip('.,;:!?')
                if word_clean in species_list:
                    return word_clean
            
            print(f"  警告: 预测结果 '{prediction_clean}' 不在树种列表中")
            return "未知"
            
        except Exception as e:
            error_str = str(e).lower()
            
            # 如果是模型不存在，尝试其他模型
            if "404" in str(e) or "not found" in error_str:
                if attempt == 0:
                    # 尝试其他模型名称
                    alternative_models = ["gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
                    for alt_model in alternative_models:
                        if alt_model != model_name:
                            try:
                                print(f"  尝试使用模型: {alt_model}...")
                                return identify_tree_species_rest(image_path, species_list, alt_model, max_retries=1)
                            except:
                                continue
                raise Exception(f"所有模型都不可用: {e}")
            
            # 其他错误可以重试
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  识别出错 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  识别失败 (已重试 {max_retries} 次): {e}")
                return f"错误: {str(e)[:100]}"

def main():
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f"results_rest_{timestamp}.json")
    
    # 检查API Key
    if not API_KEY:
        print("错误: 请设置Gemini API Key!")
        print("方法1: 在代码中修改 API_KEY 变量")
        print("方法2: 设置环境变量 GEMINI_API_KEY")
        print("例如: export GEMINI_API_KEY='your-key-here'")
        return
    
    print(f"使用 REST API 调用 Gemini")
    print(f"尝试模型: {MODEL_NAME}")
    
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
        prediction = identify_tree_species_rest(image_path, species_list, MODEL_NAME)
        
        # 记录结果
        result = {
            "image": image_name,
            "image_path": str(image_path),
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        print(f"  预测结果: {prediction}")
        
        # 每处理10张图片保存一次
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
    
    # 同时保存为CSV格式
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



