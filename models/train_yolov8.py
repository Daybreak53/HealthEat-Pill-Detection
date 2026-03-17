import wandb
from ultralytics import YOLO
from config import CONFIG

def main():
    wandb.login()

    run = wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        name=CONFIG["wandb_run_name"],
        config=CONFIG
    )

    model = YOLO(CONFIG["model_name"])

    model.train(
        data=CONFIG["data_yaml"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        device=CONFIG["device"],
        workers=CONFIG["workers"],
        seed=CONFIG["seed"],
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run_name"]
    )

    wandb.finish()
    print("training finished")

if __name__ == "__main__":
    main()
