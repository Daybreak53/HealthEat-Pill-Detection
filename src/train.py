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
        name="yolov11l_finetune_highres",
        config=CONFIG,
    )

    model = YOLO(CONFIG["model_name"])

    best_weight_path = "best.pt"
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
    )

    wandb.finish()


if __name__ == "__main__":
    main()