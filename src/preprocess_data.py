import os
import glob
import json
import shutil
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import CONFIG

def preprocess_data():
    train_img_dir = CONFIG["train_img_dir"]
    train_json_dir = CONFIG["train_json_dir"]
    dataset_dir = CONFIG["dataset_dir"]
    seed = CONFIG.get("seed", 42)
    val_size = CONFIG.get("split_ratio", [0.8, 0.2])[1]
    
    class_mapping = {}
    inv_class_mapping = {}

    all_img_paths = glob.glob(os.path.join(train_img_dir, '**', '*.[jp][pn]g'), recursive=True)
    img_path_map = {os.path.basename(p): p for p in all_img_paths}

    image_annotations = {}
    json_files = glob.glob(os.path.join(train_json_dir, '**', '*.json'), recursive=True)

    for json_path in tqdm(json_files, desc="JSON 파싱 중"):
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

        category_id_raw = int(dl_idx_str)
        if category_id_raw not in class_mapping:
            mapped_id = len(class_mapping)
            class_mapping[category_id_raw] = mapped_id
            inv_class_mapping[mapped_id] = category_id_raw

        category_id_yolo = class_mapping[category_id_raw]

        anns = data.get('annotations', [])
        for ann in anns:
            bbox = ann.get('bbox')
            if not bbox or len(bbox) != 4:
                continue

            x_min, y_min, w, h = bbox
            x_center = max(0.0, min(1.0, (x_min + w / 2.0) / img_width))
            y_center = max(0.0, min(1.0, (y_min + h / 2.0) / img_height))
            norm_w = max(0.0, min(1.0, w / img_width))
            norm_h = max(0.0, min(1.0, h / img_height))

            yolo_line = f"{category_id_yolo} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

            if file_name not in image_annotations:
                image_annotations[file_name] = []
            image_annotations[file_name].append(yolo_line)

    valid_images = list(image_annotations.keys())
    
    train_imgs, val_imgs = train_test_split(valid_images, test_size=val_size, random_state=seed)

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