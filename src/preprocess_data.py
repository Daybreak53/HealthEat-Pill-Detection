import os
import glob
import json
import shutil
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import CONFIG
import cv2
import numpy as np

def preprocess_data():
    # 1. 경로 설정 (쉘 스크립트에 맞춰 하드코딩 또는 config 활용)
    # 기존 원본 데이터 경로
    orig_img_dir = "./data/sprint_ai_project1_data/train_images"
    orig_json_dir = "./data/sprint_ai_project1_data/train_annotations"
    
    # 추가 단일 이미지 데이터 경로 (쉘 스크립트에서 압축 푼 경로)
    single_img_dir = "./data/single_pills/ai_hub_imgs"
    single_json_dir = "./data/single_pills/ai_hub_jsons"
    
    dataset_dir = CONFIG["dataset_dir"]
    seed = CONFIG.get("seed", 42)
    val_size = CONFIG.get("split_ratio", [0.8, 0.2])[1]
    
    class_mapping = {}
    inv_class_mapping = {}
    image_annotations = {}
    img_path_map = {}
    
    orig_files = []   # 232장 추적용
    single_files = [] # 3474장 추적용

    # 2. 데이터 처리 함수 (중복 코드를 줄이기 위해 함수화)
    def process_directory(img_dir, json_dir, is_single_data):
        img_paths = glob.glob(os.path.join(img_dir, '**', '*.[jp][pn]g'), recursive=True)
        for p in img_paths:
            img_path_map[os.path.basename(p)] = p

        json_files = glob.glob(os.path.join(json_dir, '**', '*.json'), recursive=True)
        desc_text = "단일 데이터 파싱" if is_single_data else "원본 데이터 파싱"
        
        for json_path in tqdm(json_files, desc=desc_text):
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

            # 유효한 파일명을 리스트에 추가하여 추적
            if file_name in image_annotations:
                if is_single_data:
                    if file_name not in single_files:
                        single_files.append(file_name)
                else:
                    if file_name not in orig_files:
                        orig_files.append(file_name)

    # 두 디렉토리 모두 처리
    process_directory(orig_img_dir, orig_json_dir, is_single_data=False)
    process_directory(single_img_dir, single_json_dir, is_single_data=True)

    # 3. 데이터 분할 전략 (핵심 로직)
    print(f"\n원본 이미지 수: {len(orig_files)} | 단일 이미지 수: {len(single_files)}")
    
    # 원본 데이터(232장)만 Train과 Val로 나눔
    train_orig, val_imgs = train_test_split(orig_files, test_size=val_size, random_state=seed)
    
    # 최종 Train 데이터는 [원본 Train 분할분 + 단일 이미지 전체]
    train_imgs = train_orig + single_files

    print(f"최종 Train 세트: {len(train_imgs)}장 | 최종 Val 세트: {len(val_imgs)}장\n")

    def remove_glare_and_save(src_path, dst_path):
        # 1. 이미지 읽기
        img = cv2.imread(src_path)
        if img is None:
            return
        
        # 2. 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. 임계값을 적용하여 빛 반사가 심한 하얀 영역(밝기 220 이상)을 마스크로 추출
        mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
        
        # 4. 빛이 번진 후광 영역까지 덮을 수 있도록 마스크 영역 팽창(Dilate)
        kernel = np.ones((5, 5), np.uint8)
        inpaint_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 5. Telea 알고리즘을 사용한 인페인팅(Inpainting)으로 손상된 픽셀 텍스처 복원
        result_img = cv2.inpaint(img, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
        # 6. 전처리된 이미지를 지정된 경로에 저장
        cv2.imwrite(dst_path, result_img)

    # 4. 파일 복사 및 txt 생성 로직 (기존과 동일)
    def create_split(split_name, img_list):
        img_out_dir = os.path.join(dataset_dir, 'images', split_name)
        lbl_out_dir = os.path.join(dataset_dir, 'labels', split_name)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        for file_name in tqdm(img_list, desc=f"[{split_name.upper()}] 세트 구성 중"):
            src_img_path = img_path_map.get(file_name)
            if not src_img_path or not os.path.exists(src_img_path):
                continue

            dst_img_path = os.path.join(img_out_dir, file_name)
            remove_glare_and_save(src_img_path, dst_img_path)
            
            base_name = os.path.splitext(file_name)[0]
            txt_path = os.path.join(lbl_out_dir, f"{base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(image_annotations[file_name]))

    create_split('train', train_imgs)
    create_split('val', val_imgs)
    
    # 5. YAML 파일 생성 (기존과 동일)
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