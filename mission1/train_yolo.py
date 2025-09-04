from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from pathlib import Path

class ChimneyDetector:
    def __init__(self, model_name="yolov8n.pt", data_config="yolo_dataset/dataset.yaml"):
        self.model_name = model_name
        self.data_config = data_config
        self.model = None
        
    def initialize_model(self):
        """YOLO 모델 초기화"""
        print(f"Initializing YOLOv8 model: {self.model_name}")
        self.model = YOLO(self.model_name)
        
        # GPU 사용 가능 시 GPU 사용
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
    def train_model(self, epochs=100, img_size=640, batch_size=16):
        """모델 훈련"""
        if self.model is None:
            self.initialize_model()
            
        print("Starting model training...")
        
        # 훈련 파라미터
        training_args = {
            'data': self.data_config,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'name': 'chimney_detection',
            'save_period': 10,  # 10 에포크마다 저장
            'patience': 20,     # Early stopping patience
            'save': True,
            'plots': True,
            'val': True
        }
        
        # 훈련 실행
        results = self.model.train(**training_args)
        
        print("Training completed!")
        return results
    
    def validate_model(self, model_path="runs/detect/chimney_detection/weights/best.pt"):
        """모델 검증"""
        print("Validating model...")
        
        # 최고 성능 모델 로드
        if Path(model_path).exists():
            self.model = YOLO(model_path)
        
        # 검증 실행
        results = self.model.val(data=self.data_config)
        
        # mAP@0.5 결과 출력
        map50 = results.box.map50
        print(f"mAP@IoU=0.5: {map50:.4f}")
        
        return results
    
    def predict_sample(self, image_path, model_path="runs/detect/chimney_detection/weights/best.pt", save_results=True):
        """샘플 이미지 예측"""
        if Path(model_path).exists():
            model = YOLO(model_path)
        else:
            model = self.model
            
        # 예측 실행
        results = model(image_path, save=save_results, conf=0.25)
        
        # 결과 출력
        for r in results:
            print(f"Detected {len(r.boxes)} chimneys in {image_path}")
            
        return results
    
    def export_model(self, model_path="runs/detect/chimney_detection/weights/best.pt", format="onnx"):
        """모델 내보내기"""
        if Path(model_path).exists():
            model = YOLO(model_path)
            model.export(format=format)
            print(f"Model exported to {format} format")

def main():
    # 훈련 실행
    detector = ChimneyDetector()
    
    # 데이터셋 설정 파일 확인
    if not Path("yolo_dataset/dataset.yaml").exists():
        print("Dataset configuration not found. Please run data_preparation.py first.")
        return
    
    # 모델 훈련
    print("Starting chimney detection training...")
    detector.train_model(epochs=100, img_size=640, batch_size=8)
    
    # 모델 검증
    validation_results = detector.validate_model()
    
    # 샘플 예측 (첫 번째 검증 이미지 사용)
    val_images = list(Path("yolo_dataset/images/val").glob("*.jpg"))
    if val_images:
        sample_img = val_images[0]
        detector.predict_sample(sample_img)
        print(f"Sample prediction saved for {sample_img}")

if __name__ == "__main__":
    main()