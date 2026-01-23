import os
import csv
import random
import json
import time
from pathlib import Path
from openai import OpenAI
from datetime import datetime

# 配置
# 优先从环境变量读取API Key
API_KEY = os.getenv("OPENAI_API_KEY", "")
FOLDER_NAMES_CSV = "/home/yjc/Project/plant_classfication/LLM/folder_names.csv"
VAL_DIR = "/home/yjc/Project/plant_classfication/timm/tune_inaturalist/dataset_val"
OUTPUT_DIR = "/home/yjc/Project/plant_classfication/LLM/chatGPT"

# 模型选择
# "gpt-4o-mini" - 便宜，支持图像识别（推荐，成本约为 gpt-4o 的 1/10）
# "gpt-4o" - 更准确但昂贵，适合高精度需求
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

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
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            image_files.append(file)
    return image_files

def encode_image(image_path):
    """将图片编码为base64"""
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def identify_tree_species(client, image_path, species_list, model_name="gpt-4o-mini", max_retries=3):
    """使用ChatGPT API识别树种"""
    # 构建提示词 - 改进版本，更好地处理拉丁学名
    # 每行显示多个名称以节省空间（每行5个）
    species_lines = []
    for i in range(0, len(species_list), 5):
        line_species = species_list[i:i+5]
        species_lines.append(" | ".join(line_species))
    species_text = "\n".join(species_lines)
    
    prompt = f"""你是植物分类专家。分析图片中的树木，从树种列表中选择最匹配的拉丁学名种加词。

**规则：**
1. 列表中的名称是拉丁学名种加词（如 sylvestris, pendula, nigra）
2. 根据叶形、树皮、树形、果实/球果等特征判断
3. 必须返回列表中完全匹配的名称，不要其他文字
4. 即使不确定，也要选择最接近的匹配，不要返回"未知"

**树种列表（共{len(species_list)}种）：**
{species_text}

**只返回最匹配的树种名称：**"""

    for attempt in range(max_retries):
        try:
            # 读取并编码图片
            base64_image = encode_image(image_path)
            
            # 调用OpenAI API (GPT-4 Vision)
            response = client.chat.completions.create(
                model=model_name,  # 使用配置的模型
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            prediction = response.choices[0].message.content.strip()
            
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
            
            # 如果还是找不到匹配，返回原始预测（可能包含额外信息）
            # 但先尝试提取可能的名称
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
            if "insufficient_quota" in error_str or ("quota" in error_str and "exceeded" in error_str):
                print(f"  ❌ API 配额不足: {e}")
                print("  💡 请检查:")
                print("     - 账户余额: https://platform.openai.com/account/billing")
                print("     - 是否已用完免费额度")
                print("     - 是否需要充值")
                return f"错误: API配额不足，请检查账户余额"
            
            # 如果是频率限制，可以重试
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避
                print(f"  识别出错 (尝试 {attempt + 1}/{max_retries}): {e}")
                if "rate limit" in error_str or "RateLimitError" in error_type:
                    wait_time = min(wait_time * 2, 60)  # 频率限制时等待更长时间
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  识别失败 (已重试 {max_retries} 次): {e}")
                return f"错误: {str(e)}"

def main():
    # 生成输出文件名（在函数内部生成，确保每次运行都有新的时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f"results_{timestamp}.json")
    
    # 检查API Key
    if API_KEY == "your-api-key-here" or not API_KEY:
        print("错误: 请设置OpenAI API Key!")
        print("方法1: 在代码中修改 API_KEY 变量")
        print("方法2: 设置环境变量 OPENAI_API_KEY")
        print("例如: export OPENAI_API_KEY='your-key-here'")
        return
    
    # 初始化OpenAI客户端
    client = OpenAI(api_key=API_KEY)
    
    # 显示使用的模型
    print(f"使用的模型: {MODEL_NAME}")
    if MODEL_NAME == "gpt-4o":
        print("⚠️  注意: gpt-4o 成本较高，如果余额不足建议改用 gpt-4o-mini")
    print()
    
    # 加载树种列表
    print("正在加载树种列表...")
    species_list = load_species_list(FOLDER_NAMES_CSV)
    print(f"已加载 {len(species_list)} 个树种")
    
    # 获取所有子文件夹
    val_path = Path(VAL_DIR)
    subfolders = sorted([f for f in val_path.iterdir() if f.is_dir()])
    print(f"找到 {len(subfolders)} 个子文件夹")
    
    # 存储结果
    results = []
    start_time = time.time()
    
    # 遍历每个子文件夹
    for idx, subfolder in enumerate(subfolders, 1):
        folder_name = subfolder.name
        print(f"\n[{idx}/{len(subfolders)}] 处理文件夹: {folder_name}")
        
        # 获取图片文件
        image_files = get_image_files(subfolder)
        
        if not image_files:
            print(f"  跳过: 没有找到图片文件")
            continue
        
        # 随机选择一张图片
        selected_image = random.choice(image_files)
        image_path = subfolder / selected_image
        
        print(f"  选择的图片: {selected_image}")
        
        # 调用API识别
        print(f"  正在识别...")
        prediction = identify_tree_species(client, image_path, species_list, MODEL_NAME)
        
        # 记录结果
        result = {
            "folder": folder_name,
            "image": selected_image,
            "image_path": str(image_path),
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        print(f"  预测结果: {prediction}")
        
        # 每处理10个文件夹保存一次（防止数据丢失）
        if idx % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / idx
            remaining = (len(subfolders) - idx) * avg_time
            print(f"  已保存中间结果到 {output_file}")
            print(f"  进度: {idx}/{len(subfolders)} ({idx/len(subfolders)*100:.1f}%), 预计剩余时间: {remaining/60:.1f} 分钟")
    
    # 保存最终结果
    print(f"\n正在保存最终结果...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 同时保存为CSV格式便于查看
    csv_output = output_file.replace('.json', '.csv')
    with open(csv_output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['folder', 'image', 'prediction', 'timestamp'])
        writer.writeheader()
        for result in results:
            writer.writerow({
                'folder': result['folder'],
                'image': result['image'],
                'prediction': result['prediction'],
                'timestamp': result['timestamp']
            })
    
    total_time = time.time() - start_time
    print(f"\n完成！共处理 {len(results)} 个文件夹")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"结果已保存到:")
    print(f"  JSON: {output_file}")
    print(f"  CSV: {csv_output}")

if __name__ == "__main__":
    main()
