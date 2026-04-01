import os
import wandb
from ultralytics import settings
from ultralytics import YOLO
from config import CONFIG


def main():
    # 학습 모니터링을 위한 wandb 연동 설정
    settings.update({"wandb": True})
    wandb.login()
    
    # wandb 프로젝트 및 실험 이름 초기화
    run = wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        name=CONFIG["wandb_run_name"],
        config=CONFIG,
    )

    # 지정된 학습 모델 불러오기
    model = YOLO(CONFIG["model_name"])

    # config.yaml의 설정값에 따라 모델 학습 진행
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

        # Augmentation 하이퍼파라미터
        hsv_h=CONFIG["hsv_h"],
        hsv_s=CONFIG["hsv_s"],
        hsv_v=CONFIG["hsv_v"],
        degrees=CONFIG["degrees"],
        translate=CONFIG["translate"],
        scale=CONFIG["scale"],
        fliplr=CONFIG["fliplr"],
        flipud=CONFIG["flipud"],
        mosaic=CONFIG["mosaic"],
        mixup=CONFIG["mixup"]
    )

    # 학습 종료 후 wandb 세션 종료
    wandb.finish()


if __name__ == "__main__":
    main()