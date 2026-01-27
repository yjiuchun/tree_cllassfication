import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ============== 配置（你只需要粘贴 API_KEY） ==============
API_KEY = "2b10A8I66tA1eZ691WbrKQlbO"  # TODO: 粘贴你的 PlantNet API Key
PROJECT = "all"  # try "weurope" or "canada"

DATASET_VAL_DIR = Path("/home/yjc/Project/plant_classfication/timm/tune_inaturalist/dataset_val")
START_IDX = 120
END_IDX = 561

# 每个子文件夹抽样图片数
N_IMAGES_PER_FOLDER = 3

# 输出 CSV
OUTPUT_DIR = Path("/home/yjc/Project/plant_classfication/plantnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / f"plantnet_results_{START_IDX}_{END_IDX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# 请求超时 / 重试
REQUEST_TIMEOUT_SEC = 120
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 3

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_prefix_index(folder_name: str) -> Optional[int]:
    """
    dataset_val 子文件夹名形如: '100_07710_nevadensis'
    返回前缀数字 100；若不符合返回 None
    """
    try:
        prefix = folder_name.split("_", 1)[0]
        return int(prefix)
    except Exception:
        return None


def list_target_folders(root: Path, start_idx: int, end_idx: int) -> List[Path]:
    folders = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        idx = parse_prefix_index(p.name)
        if idx is None:
            continue
        if start_idx <= idx <= end_idx:
            folders.append(p)
    # 用前缀数字排序，保证 100..561 的顺序一致
    folders.sort(key=lambda x: parse_prefix_index(x.name) or 10**18)
    return folders


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def pick_images(images: List[Path], k: int) -> List[Path]:
    if len(images) <= k:
        return images
    return random.sample(images, k)


def extract_topk(json_result: Dict[str, Any], k: int = 5) -> List[Tuple[str, Optional[float]]]:
    """
    PlantNet 返回结构一般包含:
    {
      "results":[
        {"score":0.123, "species":{"scientificNameWithoutAuthor":"...","scientificName":"...","commonNames":[...]}}
      ]
    }
    这里返回 [(species_name, score), ...]，长度最多 k
    """
    results = json_result.get("results") or []
    topk: List[Tuple[str, Optional[float]]] = []
    for r in results[:k]:
        score = r.get("score")
        species = r.get("species") or {}
        name = (
            species.get("scientificNameWithoutAuthor")
            or species.get("scientificName")
            or (species.get("commonNames")[0] if isinstance(species.get("commonNames"), list) and species.get("commonNames") else None)
            or "UNKNOWN"
        )
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        topk.append((str(name), score_f))
    return topk


def call_plantnet(api_endpoint: str, image_paths: List[Path]) -> Dict[str, Any]:
    """
    发送 multipart 请求到 PlantNet。organs 数组长度需要与 images 数量一致。
    """
    # 注意：requests 需要传 file-like 对象，所以这里打开文件并在 finally 里关闭
    files = []
    opened = []
    try:
        for img in image_paths:
            f = open(img, "rb")
            opened.append(f)
            files.append(("images", (img.name, f)))

        data = {"organs": ["auto"] * len(image_paths)}
        resp = requests.post(api_endpoint, files=files, data=data, timeout=REQUEST_TIMEOUT_SEC)
        # 有些错误会返回非 JSON
        try:
            payload = resp.json()
        except Exception:
            payload = {"_raw_text": resp.text}
        payload["_status_code"] = resp.status_code
        return payload
    finally:
        for f in opened:
            try:
                f.close()
            except Exception:
                pass


def main() -> None:
    if not API_KEY.strip():
        raise SystemExit("请先在 plantnet/test.py 里粘贴 API_KEY")

    api_endpoint = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}"

    folders = list_target_folders(DATASET_VAL_DIR, START_IDX, END_IDX)
    if not folders:
        raise SystemExit(f"未找到符合范围 {START_IDX}-{END_IDX} 的子文件夹：{DATASET_VAL_DIR}")

    print(f"目标文件夹数量: {len(folders)} (idx {START_IDX}-{END_IDX})")
    print(f"输出 CSV: {OUTPUT_CSV}")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "folder_idx",
                "picked_images",
                "top1_species",
                "top1_score",
                "top5_species",
                "top5_scores",
                "status_code",
                "error",
            ],
        )
        writer.writeheader()

        for i, folder in enumerate(folders, 1):
            folder_idx = parse_prefix_index(folder.name)
            images = list_images(folder)
            if not images:
                writer.writerow(
                    {
                        "folder": folder.name,
                        "folder_idx": folder_idx,
                        "picked_images": "",
                        "top1_species": "",
                        "top1_score": "",
                        "top5_species": "",
                        "top5_scores": "",
                        "status_code": "",
                        "error": "no_images",
                    }
                )
                continue

            picked = pick_images(images, N_IMAGES_PER_FOLDER)
            picked_names = [p.name for p in picked]

            # 调用（带简单重试）
            last_err: Optional[str] = None
            payload: Optional[Dict[str, Any]] = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    payload = call_plantnet(api_endpoint, picked)
                    # 2xx 以外也先记录，但如果是临时错误可以重试
                    status = int(payload.get("_status_code", 0))
                    if status == 200:
                        last_err = None
                        break
                    last_err = f"http_{status}"
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP_SEC)

            status_code = payload.get("_status_code") if isinstance(payload, dict) else ""
            topk = extract_topk(payload or {}, k=5)
            top1_species, top1_score = (topk[0] if topk else ("", ""))
            top5_species = ";".join([t[0] for t in topk])
            top5_scores = ";".join(["" if t[1] is None else str(t[1]) for t in topk])

            writer.writerow(
                {
                    "folder": folder.name,
                    "folder_idx": folder_idx,
                    "picked_images": ";".join(picked_names),
                    "top1_species": top1_species,
                    "top1_score": "" if top1_score is None else str(top1_score),
                    "top5_species": top5_species,
                    "top5_scores": top5_scores,
                    "status_code": status_code,
                    "error": last_err or "",
                }
            )

            if i % 10 == 0:
                f.flush()
                print(f"进度: {i}/{len(folders)}  最近: {folder.name} top1={top1_species} score={top1_score}")


if __name__ == "__main__":
    main()