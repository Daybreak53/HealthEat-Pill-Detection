# HealthEat-Pill-Detection
💊 [5팀] Health Eat - 경구약제 Object Detection 모델 개발

### 📌 프로젝트 개요
- **시나리오:** 작업자가 헬스케어 스타트업 헬스잇(Health Eat)의 AI 엔지니어링 팀원이라고 가정
- **수행 업무:** 모바일 애플리케이션의 유저가 업로드한 경구약 사진을 통해 해당 약에 대한 정보를 확인할 수 있는 이미지 인식 모델 구축
- **목표:** 미리 촬영된 약 이미지를 통해 최대 4개의 알약 이름(클래스)와 위치(bbox)를 검출하는 모델 구현
- **기대 효과:** 유저의 건강 상태 및 함께 복용하면 안되는 약 등 헬스케어 정보를 유저들에게 제공함으로써 비즈니스적 가치 생성

### 📁 디렉토리 구조
- **data/ :** 이미지 데이터셋 (.gitignore 처리)
- **models/ :** 모델 설계 코드
- **notebooks/ :** EDA 및 실험을 위한 주피터 노트북
- **utils/ :** 데이터 전처리, Augmentation 함수 등

### ⚙️ 기술 스택
- **Language**: Python
- **Library**: YOLO(Ultralytics), Pandas, Numpy, yaml, CONFIG, wandb 등
- **Tools**: GitHub(코드 공유), WandB(실험 관리)


### 🔗 소스 파일 
- **config.py :** YOLO 모델에서 사용되는 config.yaml의 파일 경로를 정의하고 불러옴
- **inference.py :** 학습된 YOLO 모델로 test 이미지에서 객체를 탐지하고 그 결과를 CSV 파일로 저장
- **preprocess_data.py :** Annotation 데이터를 모델에 맞게 전처리하고 데이터셋을 train 및 validation 세트로 분할
- **train.py :** 모델 파라미터를 사용자 config 값에 맞춰 train 데이터셋을 학습
- **validate.py :** 학습이 완료된 최적의 모델을 validation 데이터셋에 적용하고 mAP@[0.75:0.95]를 계산하여 모델 성능 평가


 
