import os
import re
import pandas as pd
from ultralytics import YOLO
from config import CONFIG

def inference():
    # 학습이 완료된 최고 성능(best) 모델 가중치 경로 설정
    best_model_path = f"runs/detect/{CONFIG['wandb_project']}/{CONFIG['wandb_run_name']}/weights/best.pt"
    model = YOLO(best_model_path)

    submission_rows = []
    annotation_id_counter = 1
    
    # 테스트 이미지 디렉토리에서 추론(Predict) 수행
    results = model.predict(
        source=CONFIG["test_img_dir"],
        agnostic_nms=CONFIG["agnostic_nms"],
        max_det=CONFIG["max_det"],
        stream=True,
        verbose=False
    )

    # 각 이미지별 추론 결과 파싱
    for result in results:
        # 파일명에서 숫자로 된 image_id 추출
        filename = os.path.basename(result.path)
        num_matches = re.findall(r'\d+', filename)
        image_id = int("".join(num_matches)) if num_matches else 0

        if len(result.boxes) == 0:
            continue
        
        # 탐지된 바운딩 박스별 정보 추출
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_yolo_cls = int(box.cls.item())
            original_category_id = model.names[pred_yolo_cls]

            # 제출 형식(CSV)에 맞게 데이터 행 구성
            row = {
                "annotation_id": annotation_id_counter,
                "image_id": image_id,
                "category_id": original_category_id,
                "bbox_x": int(round(x1)),
                "bbox_y": int(round(y1)),
                "bbox_w": int(round(x2 - x1)),
                "bbox_h": int(round(y2 - y1)),
                "score": round(float(box.conf.item()), 5)
            }
            submission_rows.append(row)
            annotation_id_counter += 1

    # 수집된 데이터를 DataFrame으로 변환 후 CSV로 저장
    df = pd.DataFrame(submission_rows)
    if not df.empty:
        df = df[["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]]

    output_csv = CONFIG["submission_csv"]
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    inference()