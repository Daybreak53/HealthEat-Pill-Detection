from ultralytics import YOLO
from config import MODEL_CONFIG


def main():
    model_path = f"{MODEL_CONFIG['wandb_project']}/{MODEL_CONFIG['wandb_run_name']}/weights/best.pt"
    model = YOLO(model_path)

    metrics = model.val(data=MODEL_CONFIG["data_yaml"])
    print(metrics.results_dict)


if __name__ == "__main__":
    main()