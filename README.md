# HealthEat-Pill-Detection
**💊 [5팀] Health Eat - 경구약제 Object Detection 모델 개발**

### 📌 프로젝트 개요
- **시나리오:** 작업자가 헬스케어 스타트업 헬스잇(Health Eat)의 AI 엔지니어링 팀원이라고 가정
- **수행 업무:** 모바일 애플리케이션의 유저가 업로드한 경구약 사진을 통해 해당 약에 대한 정보를 확인할 수 있는 이미지 인식 모델 구축
- **목표:** 미리 촬영된 약 이미지를 통해 최대 4개의 알약 이름(클래스)와 위치(bbox)를 검출하는 모델 구현
- **기대 효과:** 유저의 건강 상태 및 함께 복용하면 안되는 약 등 헬스케어 정보를 유저들에게 제공함으로써 비즈니스적 가치 생성

### 📁 디렉토리 구조
- **data/ :** 이미지 데이터셋 (.gitignore 처리)
- **src/ :** 모델 학습 파이프라인 코드
- **notebook/ :** 실험 진행 방법을 정리한 주피터 노트북
- **utils/ :** 후처리 코드

### ⚙️ 기술 스택
- **Language**: Python
- **Library**: YOLO(Ultralytics), Pandas, Numpy, yaml, wandb 등
- **Tools**: GitHub(코드 공유), WandB(실험 관리)

### 🔗 소스 파일 
- **config.py :** YOLO 모델에서 사용되는 config.yaml의 파일 경로를 정의하고 불러옴
- **inference.py :** 학습된 YOLO 모델로 test 이미지에서 객체를 탐지하고 그 결과를 CSV 파일로 저장
- **preprocess_data.py :** Annotation 데이터를 모델에 맞게 전처리하고 데이터셋을 train 및 validation 세트로 분할
- **train.py :** 모델 파라미터를 사용자 config 값에 맞춰 train 데이터셋을 학습
- **validate.py :** 학습이 완료된 최적의 모델을 validation 데이터셋에 적용하고 mAP@[0.75:0.95]를 계산하여 모델 성능 평가

### 😎 팀원 소개 및 협업일지 링크  
- **천지연 : [ Project Manager ]** 프로젝트 총괄 및 협업 환경 세팅, 추가 데이터 정제, 코드 리뷰 및 실험  
[**협업일지**](https://www.notion.so/AI-9-80a3c6dfd7ee832d8bdd012e9c60bc4e)
- **김범수 : [ Model Architect ]** 모델 서치, 설계, 베이스라인 코드 작성, 튜닝  
[**협업일지**](https://www.notion.so/3286f05dfc10807284cae95eb0c027d9)
- **박채빈 : [ Experiment Lead ]** 모델 학습 하이퍼파라미터 튜닝, 성능 지표 비교  
[**협업일지**](https://www.notion.so/Daily-32d8278ae1208050aa99f0e11ab9103c)
- **신희정 : [ Data Engineer ]** 전처리 기법 탐색 및 적용, 전처리 하이퍼파라미터 튜닝  
[**협업일지**](https://www.notion.so/AI09-327bb18aad658062b305ca42814af444)
- **양기우 : [ Data Engineer ]** 전처리 기법 탐색 및 적용, 전처리 하이퍼파라미터 튜닝  
[**협업일지**](https://www.notion.so/335fe74d9d2d803ca976f2c6690931b3)

### 📃 최종 보고서
- **[다운로드](https://drive.google.com/file/d/1VPewr9iUtUNA8MUBX4EJzqedBgeAxi85/view?usp=sharing)**
