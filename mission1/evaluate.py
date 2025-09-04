import torch
from ultralytics import YOLO
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class ChimneyEvaluator:
    def __init__(self, model_path, data_config="yolo_dataset/dataset.yaml"):
        self.model_path = model_path
        self.data_config = data_config
        self.model = YOLO(model_path) if Path(model_path).exists() else None
        
    def calculate_iou(self, box1, box2):
        """두 바운딩 박스 간의 IoU 계산"""
        # box format: [x1, y1, x2, y2]
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1]) 
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
            
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def load_ground_truth(self, label_dir):
        """Ground truth 라벨 로드"""
        gt_data = {}
        label_files = list(Path(label_dir).glob("*.txt"))
        
        for label_file in label_files:
            img_name = label_file.stem + ".jpg"
            boxes = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class_id, center_x, center_y, width, height
                            class_id, cx, cy, w, h = map(float, parts[:5])
                            boxes.append([cx, cy, w, h])
            
            gt_data[img_name] = boxes
            
        return gt_data
    
    def yolo_to_xyxy(self, yolo_box, img_width, img_height):
        """YOLO 형식을 xyxy 형식으로 변환"""
        cx, cy, w, h = yolo_box
        
        # 정규화된 좌표를 픽셀 좌표로 변환
        cx *= img_width
        cy *= img_height  
        w *= img_width
        h *= img_height
        
        # center, width, height -> x1, y1, x2, y2
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        return [x1, y1, x2, y2]
    
    def calculate_map_at_iou(self, predictions, ground_truths, iou_threshold=0.5):
        """특정 IoU 임계값에서의 mAP 계산"""
        all_precisions = []
        all_recalls = []
        
        for img_name in ground_truths.keys():
            if img_name not in predictions:
                continue
                
            pred_boxes = predictions[img_name]['boxes']
            pred_scores = predictions[img_name]['scores']
            gt_boxes = ground_truths[img_name]
            
            if len(gt_boxes) == 0:
                continue
                
            # 예측 결과를 신뢰도 순으로 정렬
            sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
            
            tp = np.zeros(len(pred_boxes))
            fp = np.zeros(len(pred_boxes))
            gt_matched = np.zeros(len(gt_boxes))
            
            for idx, pred_idx in enumerate(sorted_indices):
                pred_box = pred_boxes[pred_idx]
                max_iou = 0
                max_gt_idx = -1
                
                # 모든 GT 박스와 IoU 계산
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                        
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = gt_idx
                
                # IoU 임계값 이상이면 TP, 아니면 FP
                if max_iou >= iou_threshold and max_gt_idx >= 0:
                    tp[idx] = 1
                    gt_matched[max_gt_idx] = 1
                else:
                    fp[idx] = 1
            
            # Precision과 Recall 계산
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(tp_cumsum)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            
            all_precisions.append(precisions)
            all_recalls.append(recalls)
        
        # Average Precision 계산
        if not all_precisions:
            return 0.0
            
        # 전체 이미지에 대한 평균 precision-recall 계산
        max_recall = max([r.max() if len(r) > 0 else 0 for r in all_recalls])
        recall_points = np.linspace(0, max_recall, 101)
        
        interpolated_precisions = []
        for precisions, recalls in zip(all_precisions, all_recalls):
            if len(precisions) == 0:
                continue
            interp_prec = np.interp(recall_points, recalls, precisions)
            interpolated_precisions.append(interp_prec)
        
        if not interpolated_precisions:
            return 0.0
            
        mean_precision = np.mean(interpolated_precisions, axis=0)
        ap = np.trapz(mean_precision, recall_points) / max_recall if max_recall > 0 else 0.0
        
        return ap
    
    def evaluate_model(self, test_images_dir="yolo_dataset/images/val", 
                      test_labels_dir="yolo_dataset/labels/val", conf_threshold=0.25):
        """모델 평가 실행"""
        if self.model is None:
            print("Model not loaded. Please check model path.")
            return None
            
        print("Loading ground truth data...")
        ground_truths = self.load_ground_truth(test_labels_dir)
        
        print("Running inference on test images...")
        predictions = {}
        
        test_images = list(Path(test_images_dir).glob("*.jpg"))
        
        for img_path in tqdm(test_images):
            img_name = img_path.name
            
            # 이미지 크기 가져오기
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
            
            # 모델 예측
            results = self.model(img_path, conf=conf_threshold, verbose=False)
            
            pred_boxes = []
            pred_scores = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        pred_boxes.append(box.tolist())
                        pred_scores.append(score)
            
            predictions[img_name] = {
                'boxes': pred_boxes,
                'scores': pred_scores,
                'image_size': (img_width, img_height)
            }
            
            # Ground truth 박스를 xyxy 형식으로 변환
            if img_name in ground_truths:
                gt_xyxy = []
                for gt_box in ground_truths[img_name]:
                    xyxy = self.yolo_to_xyxy(gt_box, img_width, img_height)
                    gt_xyxy.append(xyxy)
                ground_truths[img_name] = gt_xyxy
        
        # mAP@IoU=0.5 계산
        print("Calculating mAP@IoU=0.5...")
        map_05 = self.calculate_map_at_iou(predictions, ground_truths, iou_threshold=0.5)
        
        print(f"\n=== Evaluation Results ===")
        print(f"mAP@IoU=0.5: {map_05:.4f}")
        
        # 추가 통계
        total_predictions = sum(len(pred['boxes']) for pred in predictions.values())
        total_ground_truths = sum(len(gt) for gt in ground_truths.values())
        
        print(f"Total predictions: {total_predictions}")
        print(f"Total ground truths: {total_ground_truths}")
        print(f"Images evaluated: {len(predictions)}")
        
        return {
            'map_05': map_05,
            'total_predictions': total_predictions,
            'total_ground_truths': total_ground_truths,
            'images_evaluated': len(predictions)
        }
    
    def visualize_predictions(self, image_path, save_path=None):
        """예측 결과 시각화"""
        if self.model is None:
            print("Model not loaded.")
            return
            
        results = self.model(image_path, conf=0.25)
        
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color='red', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(x1, y1-5, f'Chimney: {score:.2f}', 
                           color='red', fontsize=10, fontweight='bold')
        
        plt.title(f'Chimney Detection Results: {Path(image_path).name}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

def main():
    # 최고 성능 모델 경로
    model_path = "runs/detect/chimney_detection/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_yolo.py")
        return
    
    # 평가 실행
    evaluator = ChimneyEvaluator(model_path)
    results = evaluator.evaluate_model()
    
    if results:
        # 결과를 JSON으로 저장
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nEvaluation results saved to evaluation_results.json")
    
    # 샘플 이미지 시각화
    val_images = list(Path("yolo_dataset/images/val").glob("*.jpg"))
    if val_images:
        sample_img = val_images[0]
        evaluator.visualize_predictions(sample_img, "sample_prediction.png")

if __name__ == "__main__":
    main()