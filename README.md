# 2025 DCC Data Creator Camp

이 저장소는 2025년 데이터 크리에이터 캠프의 프로젝트들을 포함하고 있습니다. 컴퓨터 비전과 머신러닝을 활용한 3가지 미션으로 구성되어 있습니다.

## 프로젝트 구조

```
DCC/
├── mission1/                  # 굴뚝 탐지 (YOLO)
├── mission2/                  # 높이 추정
├── mission3/                  # 산업시설 분할
├── requirements.txt           # 프로젝트 의존성
└── README.md                 # 이 파일
```

## 미션 개요

### Mission 1: 굴뚝 탐지 (Chimney Detection)
- **목표**: YOLO 모델을 사용한 위성 이미지에서의 굴뚝 탐지
- **기술**: YOLOv8, 객체 탐지, 컴퓨터 비전
- **파일**: `mission1/mission1_chimney_detection.ipynb`

### Mission 2: 높이 추정 (Height Estimation)
- **목표**: 건물이나 구조물의 높이를 추정하는 딥러닝 모델 개발
- **기술**: PyTorch, 회귀 모델, 이미지 분석
- **파일**: `mission2/mission2_height_estimation.ipynb`

### Mission 3: 산업시설 분할 (Industrial Segmentation)
- **목표**: 위성/항공 이미지에서 산업시설을 분할하는 세그멘테이션 모델
- **기술**: 시맨틱 세그멘테이션, PyTorch, segmentation-models-pytorch
- **파일**: `mission3/mission3_industrial_segmentation.ipynb`

## 설치 및 설정

### 1. 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. GPU 지원 (선택사항)
CUDA 지원을 위해서는 PyTorch CUDA 버전이 설치되어 있어야 합니다:
```bash
# CUDA 12.1 버전 (requirements.txt에 포함됨)
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

## 사용 방법

각 미션은 Jupyter 노트북으로 구성되어 있습니다:

1. Jupyter Lab 또는 Jupyter Notebook 실행:
   ```bash
   jupyter lab
   # 또는
   jupyter notebook
   ```

2. 원하는 미션의 노트북 파일을 열어서 실행

## 주요 의존성

- **PyTorch**: 딥러닝 프레임워크
- **Ultralytics**: YOLO 모델 구현
- **segmentation-models-pytorch**: 세그멘테이션 모델
- **OpenCV**: 이미지 처리
- **Rasterio**: 지리공간 데이터 처리
- **GeoPandas**: 지리공간 데이터 분석

## 결과 파일

각 미션의 결과는 다음과 같이 저장됩니다:
- `mission1/runs/`: YOLO 훈련 결과 및 모델
- `mission2/mission2_results.json`: 높이 추정 결과
- `mission3/mission3_results.json`: 분할 결과

## 데이터

프로젝트에서 사용하는 데이터:
- 대기오염 배출원 공간 분포 데이터 (`53.대기오염 배출원 공간 분포 데이터/`)
- 각 미션별 훈련된 모델 가중치

## 주의사항

- GPU 메모리가 부족한 경우 배치 크기를 줄여주세요
- 대용량 데이터셋으로 인해 충분한 저장 공간이 필요합니다
- CUDA 환경에서 최적의 성능을 얻을 수 있습니다

## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.