import os
import wandb
from ultralytics import settings
from ultralytics import YOLO
from config import CONFIG


def main():
    settings.update({"wandb": True})

    wandb.login()
    
    run = wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        name="yolov11l_finetune_highres"
        config=CONFIG,
    )

    model = YOLO("yolo11l-p2.yaml")

    best_weight_path = "/content/HealthEat-Pill-Detection/runs/detect/HealthEat-Pill-Detection/yolov11l_baseline/weights/best.pt"
    model.load(best_weight_path)

    model.train(
        data=CONFIG["data_yaml"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        device=CONFIG["device"],
        workers=CONFIG["workers"],
        seed=CONFIG["seed"],
        patience=CONFIG["patience"],
        lr0=CONFIG["lr0"], 
        lrf=CONFIG["lrf"], 
        cls=CONFIG["cls"],
        label_smoothing=CONFIG["label_smoothing"],
        optimizer=CONFIG["optimizer"],
        dfl=CONFIG["dfl"],
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run_name"],

        hsv_h=CONFIG["hsv_h"],
        hsv_s=CONFIG["hsv_s"],
        hsv_v=CONFIG["hsv_v"],
        degrees=CONFIG["degrees"],
        translate=CONFIG["translate"],
        scale=CONFIG["scale"],
        fliplr=CONFIG["fliplr"],
        flipud=CONFIG["flipud"],
        mosaic=CONFIG["mosaic"],
        mixup=CONFIG["mixup"],

        copy_paste=0.5,
        box=10.0,
        cos_lr=True,
        freeze=10
    )

    wandb.finish()


if __name__ == "__main__":
    main()