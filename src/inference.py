import os
import re
import pandas as pd
from ultralytics import YOLO
from config import CONFIG

def inference():
    best_model_path = f"runs/detect/{CONFIG['wandb_project']}/{CONFIG['wandb_run_name']}/weights/best.pt"
    model = YOLO(best_model_path)
    submission_rows = []
    annotation_id_counter = 1
    
    results = model.predict(source=CONFIG["test_img_dir"], imgsz=CONFIG["imgsz"], stream=True, verbose=False)

    for result in results:
        filename = os.path.basename(result.path)
        num_matches = re.findall(r'\d+', filename)
        image_id = int("".join(num_matches)) if num_matches else 0

        if len(result.boxes) == 0:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_yolo_cls = int(box.cls.item())
            original_category_id = model.names[pred_yolo_cls]

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

    df = pd.DataFrame(submission_rows)
    if not df.empty:
        df = df[["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]]

    output_csv = CONFIG["submission_csv"]
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    inference()