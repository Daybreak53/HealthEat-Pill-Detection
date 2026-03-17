from ultralytics import YOLO
from config import CONFIG

def main():
    model = YOLO(f"{CONFIG['wandb_project']}/{CONFIG['wandb_run_name']}/weights/best.pt")
    metrics = model.val(data=CONFIG["data_yaml"])
    print(metrics.results_dict)

if __name__ == "__main__":
    main()
