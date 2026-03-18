import wandb
from ultralytics import YOLO
from config import MODEL_CONFIG


def main():
    wandb.login()

    run = wandb.init(
        project=MODEL_CONFIG["wandb_project"],
        entity=MODEL_CONFIG["wandb_entity"],
        name=MODEL_CONFIG["wandb_run_name"],
        config=MODEL_CONFIG
    )

    model = YOLO(MODEL_CONFIG["model_name"])

    model.train(
        data=MODEL_CONFIG["data_yaml"],
        epochs=MODEL_CONFIG["epochs"],
        imgsz=MODEL_CONFIG["imgsz"],
        batch=MODEL_CONFIG["batch"],
        device=MODEL_CONFIG["device"],
        workers=MODEL_CONFIG["workers"],
        seed=MODEL_CONFIG["seed"],
        optimizer=MODEL_CONFIG["optimizer"],
        lr0=MODEL_CONFIG["lr0"],
        patience=MODEL_CONFIG["patience"],
        project=MODEL_CONFIG["wandb_project"],
        name=MODEL_CONFIG["wandb_run_name"]
    )

    wandb.finish()
    print("YOLO11 training finished")


if __name__ == "__main__":
    main()