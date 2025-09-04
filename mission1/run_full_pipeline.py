import subprocess
import sys
from pathlib import Path
import os

def run_command(command, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # 마지막 500자만 출력
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr[-500:] if e.stderr else "No error message")
        return False

def check_requirements():
    """필요한 패키지 설치 확인"""
    print("Checking and installing required packages...")
    
    packages = [
        "ultralytics==8.0.196",
        "opencv-python==4.8.1.78", 
        "Pillow==10.0.0",
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "PyYAML==6.0.1",
        "tqdm==4.66.1",
        "pandas==2.0.3"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}")

def main():
    print("🚀 Starting Chimney Detection Pipeline")
    print("This will run the complete pipeline for chimney detection using YOLOv8")
    
    # 1. 패키지 설치
    check_requirements()
    
    # 2. 데이터 전처리
    if not run_command("python data_preparation.py", "Data Preparation"):
        print("❌ Data preparation failed. Please check your data structure.")
        return
    
    # 3. 데이터셋 구조 확인
    dataset_path = Path("yolo_dataset")
    if not dataset_path.exists():
        print("❌ Dataset directory not created. Please check data_preparation.py")
        return
        
    print(f"\n📊 Dataset Structure:")
    for split in ['train', 'val']:
        img_count = len(list((dataset_path / "images" / split).glob("*.jpg")))
        label_count = len(list((dataset_path / "labels" / split).glob("*.txt")))
        print(f"  {split}: {img_count} images, {label_count} labels")
    
    # 4. 모델 훈련
    print(f"\n🎯 Starting Model Training...")
    if not run_command("python train_yolo.py", "YOLOv8 Model Training"):
        print("❌ Training failed. Please check the training script.")
        return
    
    # 5. 모델 평가
    print(f"\n📈 Starting Model Evaluation...")
    if not run_command("python evaluate.py", "Model Evaluation (mAP@IoU=0.5)"):
        print("❌ Evaluation failed. Please check the evaluation script.")
        return
    
    # 6. 결과 요약
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    # 훈련 결과 경로
    results_dir = Path("runs/detect/chimney_detection")
    if results_dir.exists():
        print(f"\n📁 Training Results Location: {results_dir.absolute()}")
        print("   - weights/best.pt: Best model weights")
        print("   - weights/last.pt: Last epoch weights") 
        print("   - results.png: Training curves")
        print("   - confusion_matrix.png: Confusion matrix")
        print("   - val_batch*.jpg: Validation predictions")
    
    # 평가 결과
    eval_file = Path("evaluation_results.json")
    if eval_file.exists():
        print(f"\n📊 Evaluation Results: {eval_file.absolute()}")
        
    # 샘플 예측 결과
    sample_pred = Path("sample_prediction.png")  
    if sample_pred.exists():
        print(f"🖼️  Sample Prediction: {sample_pred.absolute()}")
    
    print(f"\n💡 Next Steps:")
    print("1. Check mAP@IoU=0.5 score in evaluation_results.json")
    print("2. Review training curves in runs/detect/chimney_detection/results.png")
    print("3. Examine sample predictions")
    print("4. Fine-tune hyperparameters if needed")
    
if __name__ == "__main__":
    main()