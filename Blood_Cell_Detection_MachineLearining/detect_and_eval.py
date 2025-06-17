import os, joblib
from utils.selective_search import get_multi_scale_candidates
from utils.feature_extraction import extract_feature_from_patch
from utils.evaluation import evaluate_detections
from utils.visualize import draw_boxes
from utils.nms import nms
import cv2
import json
from tqdm import tqdm

def run_detection_and_evaluation(data_dir, model_path, feature_type='hog', iou_thresh=0.5):
    class_names = ['RBC', 'WBC', 'Platelet']
    clf = joblib.load(model_path)
    image_dir = os.path.join(data_dir, 'JPEGImages')
    ann_dir = os.path.join(data_dir, 'Annotations')
    results = []

    for img_file in tqdm(list(os.listdir(image_dir))):
        if not img_file.endswith('.jpg'):
            continue
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        candidates = get_multi_scale_candidates(img)

        detections = []
        for (x1, y1, x2, y2) in candidates:
            patch = img[y1:y2, x1:x2]
            feat = extract_feature_from_patch(patch, feature_type)
            label = clf.predict([feat])[0]
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba([feat])[0].max()
            else:
                prob = 1.0
            if prob > 0.9:
                detections.append((x1, y1, x2, y2, label, prob))
                
        keep = nms(detections, iou_thresh)
        results.append({
        'image_id': img_file,
        'objects': [
            {
                'bbox': [x1, y1, x2, y2],
                'name': label,
                'confidence': float(prob)
            }
            for (x1, y1, x2, y2, label, prob) in keep
        ]
    })

        draw_boxes(img, keep, save_path=f"results/detections/{img_file}")
        # break

    metrics = evaluate_detections(results,class_names,ann_dir,iou_thresh)
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("[INFO] Evaluation complete. Metrics saved to results/metrics.json")
