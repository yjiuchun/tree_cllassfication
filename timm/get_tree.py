#!/usr/bin/python3
# Copyright 2023 WolkenVision AG. All rights reserved.
"""
终端交互：输入类别 id，返回对应的 name（来自 timm 模型 default_cfg["label_names"]）。
批量处理：从 CSV 读取 ID，获取 name 并翻译成中文，保存到 CSV。

用法：
  # 交互模式
  python timm/get_tree.py
  python timm/get_tree.py --model hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21
  
  # 批量处理 CSV
  python timm/get_tree.py --csv timm/plant_species_filtered.csv
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

import timm


vit_large_14_336 = "hf_hub:timm/vit_large_patch14_clip_336.datacompxl_ft_augreg_inat21"
eva_large_14_336 = "hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"


def _load_label_names(model_name: str) -> list[str]:
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    label_names = model.default_cfg.get("label_names")
    if not label_names:
        raise RuntimeError(
            f"Model {model_name!r} 没有在 default_cfg 里提供 label_names；"
            "请换一个带 inat21 label 的模型，或自行提供映射表。"
        )
    return list(label_names)


def _translate_to_chinese(text: str, max_retries: int = 3) -> str:
    """将英文文本翻译成中文，带重试机制"""
    if GoogleTranslator is None:
        print(
            "警告: 未安装 deep-translator，无法翻译。请运行: pip install deep-translator",
            file=sys.stderr,
        )
        return text
    
    import time
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source="en", target="zh-CN")
            translated = translator.translate(text)
            # 添加小延迟避免 API 限制
            time.sleep(0.2)
            return translated
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间：2s, 4s, 6s
                print(f"翻译失败 (尝试 {attempt + 1}/{max_retries}): {e}，{wait_time}秒后重试...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                print(f"翻译失败 (已重试 {max_retries} 次): {e}，返回原文", file=sys.stderr)
                return text
    return text


def _process_csv(csv_path: str, model_name: str, output_path: str | None = None) -> int:
    """从 CSV 读取 ID，获取 name 并翻译，保存到新 CSV（支持断点续传）"""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"错误: CSV 文件不存在: {csv_path}", file=sys.stderr)
        return 1

    print(f"加载模型 {model_name}...")
    label_names = _load_label_names(model_name)
    n = len(label_names)
    print(f"已加载 {n} 个标签")

    if output_path is None:
        output_path = csv_file.parent / f"{csv_file.stem}_with_names.csv"
    output_file = Path(output_path)

    print(f"读取 CSV: {csv_file}")
    print(f"输出 CSV: {output_file}")

    # 检查输出文件是否已存在（断点续传）
    existing_data = {}
    if output_file.exists():
        print(f"检测到已存在的输出文件，将从中恢复已翻译的内容...")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    id_str = row.get("ID", "").strip()
                    if id_str:
                        existing_data[id_str] = {
                            "Name_EN": row.get("Name_EN", "").strip(),
                            "Name_CN": row.get("Name_CN", "").strip(),
                        }
            print(f"已恢复 {len(existing_data)} 条已翻译的记录")
        except Exception as e:
            print(f"警告: 读取已存在文件失败: {e}，将重新开始", file=sys.stderr)
            existing_data = {}

    rows = []
    translated_count = 0
    skipped_count = 0
    
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            print("错误: CSV 文件没有表头", file=sys.stderr)
            return 1

        # 确保有 Name_EN 和 Name_CN 列
        if "Name_EN" not in fieldnames:
            fieldnames = list(fieldnames) + ["Name_EN"]
        if "Name_CN" not in fieldnames:
            fieldnames = list(fieldnames) + ["Name_CN"]

        for row in reader:
            id_str = row.get("ID", "").strip()
            if not id_str:
                continue

            # 将 ID 转换为整数索引（去掉前导零）
            try:
                idx = int(id_str)
            except ValueError:
                print(f"警告: 无法解析 ID '{id_str}'，跳过", file=sys.stderr)
                continue

            if idx < 0 or idx >= n:
                print(f"警告: ID {idx} 超出范围 [0, {n-1}]，跳过", file=sys.stderr)
                continue

            # 获取英文 name
            name_en = label_names[idx]
            row["Name_EN"] = name_en

            # 检查是否已翻译
            if id_str in existing_data:
                existing_cn = existing_data[id_str].get("Name_CN", "").strip()
                if existing_cn and existing_cn != name_en:  # 有翻译且不是原文
                    row["Name_CN"] = existing_cn
                    skipped_count += 1
                    if skipped_count % 10 == 0:
                        print(f"已跳过 {skipped_count} 条已翻译的记录...")
                else:
                    # 需要翻译
                    name_cn = _translate_to_chinese(name_en)
                    row["Name_CN"] = name_cn
                    translated_count += 1
                    print(f"ID {idx}: {name_en} -> {name_cn} ({translated_count} 条新翻译)")
            else:
                # 需要翻译
                name_cn = _translate_to_chinese(name_en)
                row["Name_CN"] = name_cn
                translated_count += 1
                print(f"ID {idx}: {name_en} -> {name_cn} ({translated_count} 条新翻译)")

            rows.append(row)
            
            # 每翻译 10 条保存一次（防止中断丢失进度）
            if translated_count > 0 and translated_count % 10 == 0:
                with open(output_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"已保存进度（{len(rows)} 行）...")

    # 最终保存
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n完成！已处理 {len(rows)} 行，其中新翻译 {translated_count} 条，跳过 {skipped_count} 条已翻译的记录")
    print(f"保存到: {output_file}")
    return 0


def _interactive_mode(model_name: str) -> int:
    """交互模式：输入 id，返回 name"""
    label_names = _load_label_names(model_name)
    n = len(label_names)
    print(f"Loaded {n} labels from {model_name}")
    print("输入 id (0-based)，或输入 q/quit/exit 退出。")

    while True:
        s = input("id> ").strip()
        if not s:
            continue
        if s.lower() in {"q", "quit", "exit"}:
            break
        try:
            idx = int(s)
        except ValueError:
            print("请输入整数 id（例如 0, 1, 2 ...），或 q 退出。")
            continue
        if idx < 0 or idx >= n:
            print(f"id 越界：有效范围 [0, {n - 1}]")
            continue
        print(label_names[idx])

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="输入类别 id，返回对应的 label name；或批量处理 CSV。"
    )
    parser.add_argument(
        "--model",
        default=eva_large_14_336,
        help=f"timm model name（默认: {eva_large_14_336}）",
    )
    parser.add_argument(
        "--csv",
        help="CSV 文件路径（如果提供，将批量处理 CSV 而不是进入交互模式）",
    )
    parser.add_argument(
        "--output",
        help="输出 CSV 文件路径（默认: 原文件名_with_names.csv）",
    )
    args = parser.parse_args()

    if args.csv:
        return _process_csv(args.csv, args.model, args.output)
    else:
        return _interactive_mode(args.model)


if __name__ == "__main__":
    raise SystemExit(main())