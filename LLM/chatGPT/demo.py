import os
import csv
import random
import json
import time
from pathlib import Path
from openai import OpenAI
from datetime import datetime

# 配置
# 优先从环境变量读取API Key，如果没有则使用这里设置的值
API_KEY = os.getenv("OPENAI_API_KEY", "")  # 请替换为你的API Key或设置环境变量
FOLDER_NAMES_CSV = "/root/folder_names.csv"
VAL_DIR = "/root/autodl-fs/val"
OUTPUT_DIR = "/root/tree_cllassfication/LLM/chatGPT"

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

def identify_tree_species(client, image_path, species_list, max_retries=3):
    """使用ChatGPT API识别树种"""
    # 构建提示词
    species_text = "\n".join([f"- {species}" for species in species_list])
    prompt = f"""请识别这张图片中的树种。以下是我数据库中所有可能的树种列表：

{species_text}

请仔细分析图片，然后只返回你认为最匹配的树种名称（必须完全匹配列表中的某个名称）。如果无法确定，请返回"未知"。
只返回树种名称，不要返回其他内容。"""

    for attempt in range(max_retries):
        try:
            # 读取并编码图片
            base64_image = encode_image(image_path)
            
            # 调用OpenAI API (GPT-4 Vision)
            response = client.chat.completions.create(
                model="gpt-4o",  # 或使用 "gpt-4-vision-preview"
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
            return prediction
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避
                print(f"  识别出错 (尝试 {attempt + 1}/{max_retries}): {e}")
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
        prediction = identify_tree_species(client, image_path, species_list)
        
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
