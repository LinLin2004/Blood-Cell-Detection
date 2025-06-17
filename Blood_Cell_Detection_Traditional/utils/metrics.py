from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.iou import compute_iou
from config import IOU_THRESHOLD, CLASS_NAMES


def compute_confusion(pred_types, gt_types):
    cm = pd.DataFrame(0, index=CLASS_NAMES, columns=CLASS_NAMES)
    for pred, gt in zip(pred_types, gt_types):
        if pred in CLASS_NAMES and gt in CLASS_NAMES:
            cm.loc[gt, pred] += 1
    return cm

def plot_confusion_matrix(cm, save_path='results/confusion_matrix.png'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_map(all_preds, all_gts):
    aps = []
    ap_dict = {}
    for cls in CLASS_NAMES:
        tp, fp, fn = 0, 0, 0
        for preds, gts in zip(all_preds, all_gts):
            pred_boxes = [p for p, t in preds if t == cls]
            gt_boxes = [g for g, t in gts if t == cls]
            matched = set()
            for pb in pred_boxes:
                match_found = False
                for i, gb in enumerate(gt_boxes):
                    if i not in matched and compute_iou(pb, gb) > IOU_THRESHOLD:
                        tp += 1
                        matched.add(i)
                        match_found = True
                        break
                if not match_found:
                    fp += 1
            fn += len(gt_boxes) - len(matched)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        aps.append(precision)
        ap_dict[cls] = precision
    return sum(aps) / len(aps), ap_dict