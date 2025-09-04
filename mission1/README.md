# 굴뚝 탐지 (Chimney Detection) - YOLOv8

Kompsat-3/3A 위성 이미지를 활용한 굴뚝 위치 탐지 프로젝트

## 📋 프로젝트 개요

- **목표**: 위성 이미지에서 굴뚝을 탐지하여 바운딩 박스로 위치 예측
- **모델**: YOLOv8 (Object Detection)
- **데이터**: Training 8,052개, Validation 1,006개
- **평가지표**: mAP@IoU=0.5

## 🗂️ 파일 구조

```
├── requirements.txt              # 필요 패키지 목록
├── data_preparation.py          # 데이터 전처리 및 YOLO 형식 변환
├── train_yolo.py               # YOLOv8 모델 훈련
├── evaluate.py                 # 모델 평가 및 mAP@IoU=0.5 계산
├── run_full_pipeline.py        # 전체 파이프라인 실행
├── README.md                   # 프로젝트 설명서
└── 3.개방데이터/               # 원본 데이터
    └── 1.데이터/
        ├── Training/
        │   ├── 01.원천데이터/TS_KS/        # 훈련 이미지
        │   └── 02.라벨링데이터/TL_KS_BBOX/ # 훈련 라벨
        └── Validation/
            ├── 01.원천데이터/VS_KS/        # 검증 이미지  
            └── 02.라벨링데이터/VL_KS_BBOX/ # 검증 라벨
```

## 🚀 사용법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행 (권장)
```bash
python run_full_pipeline.py
```

### 3. 단계별 실행

#### 3.1 데이터 전처리
```bash
python data_preparation.py
```
- JSON 라벨을 YOLO 형식으로 변환
- 훈련/검증 데이터를 YOLO 디렉터리 구조로 정리

#### 3.2 모델 훈련  
```bash
python train_yolo.py
```
- YOLOv8 모델로 굴뚝 탐지 훈련
- 100 에포크 기본 설정
- GPU 자동 감지 및 사용

#### 3.3 모델 평가
```bash
python evaluate.py
```
- mAP@IoU=0.5 계산
- 평가 결과를 JSON으로 저장
- 샘플 예측 시각화

## 📊 데이터 형식

### 입력 라벨 형식 (JSON)
```json
{
  "annotations": [
    {
      "shape_attributes": {
        "name": "rect",
        "x": 336,
        "y": 280, 
        "width": 45,
        "height": 100
      }
    }
  ]
}
```

### YOLO 라벨 형식 (변환 후)
```
0 0.503125 0.390625 0.070312 0.156250
```
- 형식: `class_id center_x center_y width height` (모두 정규화)
- class_id: 0 (굴뚝)

## 🎯 주요 기능

### data_preparation.py
- JSON 어노테이션을 YOLO 형식으로 변환
- 훈련/검증 데이터셋 분할
- YOLO 디렉터리 구조 생성
- dataset.yaml 설정 파일 생성

### train_yolo.py  
- YOLOv8n 모델 기반 훈련
- GPU/CPU 자동 선택
- 훈련 과정 시각화
- 모델 체크포인트 저장

### evaluate.py
- 정확한 mAP@IoU=0.5 계산
- IoU 기반 TP/FP/FN 분석  
- 예측 결과 시각화
- 평가 메트릭 JSON 저장

## 📈 결과 파일

훈련 완료 후 생성되는 주요 파일:

```
runs/detect/chimney_detection/
├── weights/
│   ├── best.pt                 # 최고 성능 모델
│   └── last.pt                 # 마지막 에포크 모델
├── results.png                 # 훈련 곡선
├── confusion_matrix.png        # 혼동 행렬
└── val_batch*.jpg             # 검증 예측 결과

evaluation_results.json         # mAP@IoU=0.5 점수
sample_prediction.png          # 샘플 예측 시각화
```

## ⚙️ 하이퍼파라미터 조정

train_yolo.py에서 다음 파라미터 조정 가능:

```python
detector.train_model(
    epochs=100,        # 훈련 에포크 수
    img_size=640,      # 입력 이미지 크기  
    batch_size=8       # 배치 크기
)
```

## 🔧 문제 해결

### 메모리 부족 시
- batch_size를 4 또는 2로 감소
- img_size를 512 또는 416으로 감소

### 데이터 로드 오류 시
- 데이터 경로 확인
- JSON 형식 검증
- 이미지와 라벨 파일명 일치 확인

## 📝 평가 지표

- **mAP@IoU=0.5**: IoU 임계값 0.5에서의 Mean Average Precision
- **Precision**: 예측된 굴뚝 중 실제 굴뚝 비율
- **Recall**: 실제 굴뚝 중 탐지된 굴뚝 비율

## 🎉 완료 체크리스트

- [x] 데이터 전처리 (JSON → YOLO)
- [x] YOLOv8 모델 훈련 
- [x] mAP@IoU=0.5 평가
- [x] 결과 시각화
- [x] 전체 파이프라인 자동화