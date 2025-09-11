import os
import json
import shutil
from pathlib import Path
import cv2
import yaml

class DataPreparator:
    def __init__(self, data_root="3.개방데이터/1.데이터"):
        self.data_root = Path(data_root)
        self.output_dir = Path("yolo_dataset")
        
    def create_yolo_structure(self):
        """YOLO 데이터셋 구조 생성"""
        directories = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val", 
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def convert_bbox_to_yolo(self, bbox_info, img_width, img_height):
        """바운딩 박스를 YOLO 형식으로 변환"""
        x = bbox_info["x"]
        y = bbox_info["y"] 
        width = bbox_info["width"]
        height = bbox_info["height"]
        
        # YOLO 형식: center_x, center_y, width, height (모두 정규화)
        center_x = (x + width / 2) / img_width
        center_y = (y + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        
        return center_x, center_y, norm_width, norm_height
    
    def process_json_labels(self, json_file_path, img_path):
        """JSON 라벨 파일을 처리하여 YOLO 형식으로 변환"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img = cv2.imread(str(img_path))
            if img is None:
                return None
                
            img_height, img_width = img.shape[:2]
            
            yolo_labels = []
            
            # JSON 구조에 따라 어노테이션 추출
            if "annotations" in data:
                for annotation in data["annotations"]:
                    if "shape_attributes" in annotation:
                        bbox = annotation["shape_attributes"]
                        if bbox["name"] == "rect":
                            center_x, center_y, norm_width, norm_height = self.convert_bbox_to_yolo(
                                bbox, img_width, img_height
                            )
                            # class_id = 0 (굴뚝 클래스)
                            yolo_labels.append(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            return yolo_labels
            
        except Exception as e:
            print(f"Error processing {json_file_path}: {e}")
            return None
    
    def prepare_dataset(self):
        """전체 데이터셋 준비"""
        self.create_yolo_structure()
        
        # 훈련 데이터 처리
        train_img_dir = self.data_root / "Training" / "01.원천데이터" / "TS_KS"
        train_label_dir = self.data_root / "Training" / "02.라벨링데이터" / "TL_KS_BBOX"
        
        # 검증 데이터 처리  
        val_img_dir = self.data_root / "Validation" / "01.원천데이터" / "VS_KS"
        val_label_dir = self.data_root / "Validation" / "02.라벨링데이터" / "VL_KS_BBOX"
        
        self.process_split(train_img_dir, train_label_dir, "train")
        self.process_split(val_img_dir, val_label_dir, "val")
        
        # YAML 설정 파일 생성
        self.create_yaml_config()
        
    def process_split(self, img_dir, label_dir, split):
        """훈련/검증 분할 처리"""
        if not img_dir.exists():
            print(f"Image directory not found: {img_dir}")
            return
            
        img_files = list(img_dir.glob("*.jpg"))
        processed = 0
        
        for img_file in img_files:
            # 라벨 파일 찾기
            label_file = label_dir / f"{img_file.stem}.json"
            
            if not label_file.exists():
                continue
                
            # 이미지 복사
            dst_img = self.output_dir / "images" / split / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # 라벨 변환 및 저장
            yolo_labels = self.process_json_labels(label_file, img_file)
            if yolo_labels:
                dst_label = self.output_dir / "labels" / split / f"{img_file.stem}.txt"
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                    
                processed += 1
                
        print(f"Processed {processed} {split} samples")
        
    def create_yaml_config(self):
        """YOLO 설정 YAML 파일 생성"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,  # 클래스 수 (굴뚝 1개)
            'names': ['chimney']  # 클래스 이름
        }
        
        with open(self.output_dir / "dataset.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"Dataset configuration saved to {self.output_dir / 'dataset.yaml'}")

if __name__ == "__main__":
    preparator = DataPreparator()
    preparator.prepare_dataset()