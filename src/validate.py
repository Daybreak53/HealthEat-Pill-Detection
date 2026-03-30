from ultralytics import YOLO
from config import CONFIG


def main():
    best_model_path = f"runs/detect/{CONFIG['wandb_project']}/{CONFIG['wandb_run_name']}/weights/best.pt"
    model = YOLO(best_model_path)

    metrics = model.val(
        data=CONFIG["data_yaml"],
        agnostic_nms=CONFIG["agnostic_nms"],
        max_det=CONFIG["max_det"],
    )
    
    print(metrics.results_dict)


if __name__ == "__main__":
    main()