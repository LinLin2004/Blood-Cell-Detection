from utils.iou import compute_iou
from config import IOU_THRESHOLD

def compute_metrics_strict(pred_boxes, pred_types, gt_boxes, gt_types, iou_thresh=IOU_THRESHOLD):
    matched_gt = set()
    matched = 0
    for i, pbox in enumerate(pred_boxes):
        for j, gtbox in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            if compute_iou(pbox, gtbox) >= iou_thresh and pred_types[i] == gt_types[j]:
                matched += 1
                matched_gt.add(j)
                break

    precision = matched / len(pred_boxes) if pred_boxes else 0
    recall = matched / len(gt_boxes) if gt_boxes else 0
    return {'precision': precision, 'recall': recall}

def compute_metrics_soft(pred_boxes, pred_types, gt_boxes, gt_types, iou_thresh=IOU_THRESHOLD):
    matched_gt = set()
    matched_correct = 0
    matched_wrong_cls = 0

    for i, pbox in enumerate(pred_boxes):
        for j, gtbox in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pbox, gtbox)
            if iou >= iou_thresh:
                matched_gt.add(j)
                if pred_types[i] == gt_types[j]:
                    matched_correct += 1
                else:
                    matched_wrong_cls += 1
                break

    total_pred = len(pred_boxes)
    total_gt = len(gt_boxes)

    precision = (matched_correct + 0.5 * matched_wrong_cls) / total_pred if total_pred else 0
    recall = (matched_correct + 0.5 * matched_wrong_cls) / total_gt if total_gt else 0

    return {
        'precision': precision,
        'recall': recall,
        'matched_correct': matched_correct,
        'matched_wrong_cls': matched_wrong_cls
    }

def summarize_metrics(metrics_list_strict, metrics_list_soft):
    ps_s = [m['precision'] for m in metrics_list_strict]
    rs_s = [m['recall'] for m in metrics_list_strict]

    ps_f = [m['precision'] for m in metrics_list_soft]
    rs_f = [m['recall'] for m in metrics_list_soft]
    correct = sum([m['matched_correct'] for m in metrics_list_soft])
    wrong_cls = sum([m['matched_wrong_cls'] for m in metrics_list_soft])

    print("=== Evaluation Summary ===")
    print(f"Strict Precision: {sum(ps_s)/len(ps_s):.3f}, Recall: {sum(rs_s)/len(rs_s):.3f}")
    print(f"Soft Precision: {sum(ps_f)/len(ps_f):.3f}, Recall: {sum(rs_f)/len(rs_f):.3f}")
    print(f"Matched (correct): {correct}, matched (wrong class): {wrong_cls}")
