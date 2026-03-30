import os
import glob
import json
import shutil
import yaml
from tqdm import tqdm
import numpy as np
from config import CONFIG
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# train 이상치 데이터
BAD_FILES = {
    "K-001900-016548-019607-033009_0_2_0_2_70_000_200",
    "K-003351-013900-021325_0_2_0_2_70_000_200",
    "K-003351-013900-036637_0_2_0_2_70_000_200",
    "K-003351-016262-018357_0_2_0_2_75_000_200",
    "K-003351-020014-020238_0_2_0_2_75_000_200",
    "K-003351-020014-022074_0_2_0_2_90_000_200",
    "K-003351-020238-031863_0_2_0_2_70_000_200",
    "K-003351-021325-032310_0_2_0_2_90_000_200",
    "K-003351-029667-031863_0_2_0_2_70_000_200",
    "K-003351-032310-038162_0_2_0_2_70_000_200",
    "K-003351-033880-038162_0_2_0_2_75_000_200",
    "K-003351-035206-041768_0_2_0_2_70_000_200",
    "K-003544-004543-012247-016548_0_2_0_2_90_000_200"
}

def preprocess_data():
    train_img_dir = CONFIG["train_img_dir"]
    train_json_dir = CONFIG["train_json_dir"]
    dataset_dir = CONFIG["dataset_dir"]
    seed = CONFIG.get("seed", 42)
    val_size = CONFIG.get("split_ratio", [0.8, 0.2])[1]
    
    all_img_paths = glob.glob(os.path.join(train_img_dir, '**', '*.[jp][pn]g'), recursive=True)
    img_path_map = {os.path.basename(p): p for p in all_img_paths}

    json_files = glob.glob(os.path.join(train_json_dir, '**', '*.json'), recursive=True)

    parsed_annotations = []
    unique_classes = set()

    for json_path in tqdm(json_files, desc="JSON 파싱 및 클래스 수집 중"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images_data = data.get('images', {})
        img_info = images_data[0] if isinstance(images_data, list) and len(images_data) > 0 else images_data

        if not isinstance(img_info, dict):
            continue

        file_name = img_info.get('file_name')
        img_width = img_info.get('width')
        img_height = img_info.get('height')
        dl_idx_str = img_info.get('dl_idx')

        if not file_name or dl_idx_str is None:
            continue

        base_name = os.path.splitext(file_name)[0]
        if base_name in BAD_FILES:
            continue

        category_id_raw = int(dl_idx_str)
        unique_classes.add(category_id_raw)

        anns = data.get('annotations', [])
        for ann in anns:
            bbox = ann.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            
            parsed_annotations.append({
                'file_name': file_name,
                'img_width': img_width,
                'img_height': img_height,
                'category_id_raw': category_id_raw,
                'bbox': bbox
            })

    sorted_classes = sorted(list(unique_classes))
    class_mapping = {raw_id: idx for idx, raw_id in enumerate(sorted_classes)}
    inv_class_mapping = {idx: raw_id for idx, raw_id in enumerate(sorted_classes)}

    image_annotations = {}
    for ann_data in tqdm(parsed_annotations, desc="YOLO 포맷 변환 중"):
        file_name = ann_data['file_name']
        img_width = ann_data['img_width']
        img_height = ann_data['img_height']
        category_id_raw = ann_data['category_id_raw']
        x_min, y_min, w, h = ann_data['bbox']

        category_id_yolo = class_mapping[category_id_raw]

        x_center = max(0.0, min(1.0, (x_min + w / 2.0) / img_width))
        y_center = max(0.0, min(1.0, (y_min + h / 2.0) / img_height))
        norm_w = max(0.0, min(1.0, w / img_width))
        norm_h = max(0.0, min(1.0, h / img_height))

        yolo_line = f"{category_id_yolo} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

        if file_name not in image_annotations:
            image_annotations[file_name] = []
        image_annotations[file_name].append(yolo_line)

    valid_images = sorted(list(image_annotations.keys()))
    
    single_imgs = []
    combo_imgs = []
    combo_labels = [] 
    
    for img_name in valid_images:
        if len(image_annotations[img_name]) == 1:
            single_imgs.append(img_name)
        else:
            combo_imgs.append(img_name)
            
            classes_in_img = set()
            for yolo_line in image_annotations[img_name]:
                class_id = int(yolo_line.split()[0])
                classes_in_img.add(class_id)
            
            multi_hot = np.zeros(len(class_mapping))
            for cid in classes_in_img:
                multi_hot[cid] = 1
            combo_labels.append(multi_hot)

    combo_imgs = np.array(combo_imgs)
    combo_labels = np.array(combo_labels)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    
    for train_index, val_index in msss.split(combo_imgs, combo_labels):
        combo_train = combo_imgs[train_index].tolist()
        val_imgs = combo_imgs[val_index].tolist()

    train_imgs = single_imgs + combo_train

    def create_split(split_name, img_list):
        img_out_dir = os.path.join(dataset_dir, 'images', split_name)
        lbl_out_dir = os.path.join(dataset_dir, 'labels', split_name)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        for file_name in tqdm(img_list, desc=f"[{split_name.upper()}] 세트 구성 중"):
            src_img_path = img_path_map.get(file_name)
            if not src_img_path or not os.path.exists(src_img_path):
                continue

            shutil.copy(src_img_path, os.path.join(img_out_dir, file_name))
            
            base_name = os.path.splitext(file_name)[0]
            txt_path = os.path.join(lbl_out_dir, f"{base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(image_annotations[file_name]))

    create_split('train', train_imgs)
    create_split('val', val_imgs)
    
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    names_list = [str(inv_class_mapping[i]) for i in range(len(class_mapping))]
    
    yaml_content = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_mapping),
        "names": names_list
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)


if __name__ == "__main__":
    preprocess_data()