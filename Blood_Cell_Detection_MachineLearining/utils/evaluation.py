import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score
import numpy as np
import os
def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        box = [int(bbox.find(x).text) for x in ['xmin', 'ymin', 'xmax', 'ymax']]
        objects.append({'name': name, 'bbox': box})
    return objects

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou_val

def evaluate_detections(detections, class_names, ann_dir, iou_thresh=0.5):
    ap_dict = {}
    tp_total, fp_total, fn_total = 0, 0, 0
    all_precisions = []
    all_recalls = []

    for cls in class_names:
        tp, fp, fn = 0, 0, 0

        for det in detections:
            img_file = det['image_id']
            pred_objs = [obj for obj in det['objects'] if obj['name'] == cls]

            xml_path = os.path.join(ann_dir, img_file.replace('.jpg', '.xml'))
            gt_objs = [obj for obj in parse_annotation(xml_path) if obj['name'] == cls]

            matched = set()
            for p in pred_objs:
                found_match = False
                for idx, g in enumerate(gt_objs):
                    if idx in matched:
                        continue
                    if compute_iou(p['bbox'], g['bbox']) >= iou_thresh:
                        tp += 1
                        matched.add(idx)
                        found_match = True
                        break
                if not found_match:
                    fp += 1
            fn += len(gt_objs) - len(matched)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        ap = precision * recall  # 简化版 AP = P * R，仅为估算，非真正 PR 曲线积分

        ap_dict[cls] = ap
        all_precisions.append(precision)
        all_recalls.append(recall)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    mAP = sum(ap_dict.values()) / len(class_names)
    overall_precision = tp_total / (tp_total + fp_total + 1e-6)
    overall_recall = tp_total / (tp_total + fn_total + 1e-6)

    return {
        'ap_per_class': ap_dict,
        'mean_ap': mAP,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall
    }