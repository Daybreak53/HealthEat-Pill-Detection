from ultralytics import YOLO
from config import CONFIG
import numpy as np

def main():
    best_model_path = f"runs/detect/{CONFIG['wandb_project']}/{CONFIG['wandb_run_name']}/weights/best.pt"
    model = YOLO(best_model_path)

    metrics = model.val(
        data=CONFIG["data_yaml"],
        conf=CONFIG["conf"],
        imgsz=CONFIG["imgsz"]
    )
    
    raw_ap_tensor = metrics.box.all_ap 
    strict_ap_subtensor = raw_ap_tensor[:, 5:]
    map_75_to_95 = float(np.mean(strict_ap_subtensor))

    results = metrics.results_dict.copy()
    results["metrics/mAP@[0.75:0.95](B)"] = map_75_to_95

    print("\n" + "=" * 55)
    print(f"{'Validation Metrics':^55}")
    print("=" * 55)
    print(f"{'Metric Name':<35} | {'Value':>15}")
    print("-" * 55)
    
    for key, value in results.items():
        print(f"{key:<35} | {value:>15.5f}")
        
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()