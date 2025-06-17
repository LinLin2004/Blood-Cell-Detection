import argparse
import os
from detector import preprocess, segmentation, detection, evaluation
from utils import io, visualization
from utils.metrics import compute_confusion, plot_confusion_matrix, compute_map
from config import *
def run_pipeline(image_dir, label_dir, output_dir, metrics_plot_path, cm_plot_path):
    image_paths = io.load_image_paths(image_dir)
    all_metricssoft = []
    all_metricsstrict = []
    all_pred_types, all_gt_types = [], []
    all_pred_boxes_types, all_gt_boxes_types = [], []

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        image = io.load_image(img_path)
        label_path = io.get_label_path(img_path, label_dir)

        pre = preprocess.apply_blur(image)
        enhanced = segmentation.enhance_image(pre)
        masks = segmentation.segment_cells(enhanced)
        boxes, types = detection.detect_cells(masks, image)
        gt_boxes, gt_types = io.load_annotations(label_path)

        metrics_soft = evaluation.compute_metrics_soft(boxes, types, gt_boxes, gt_types)
        metrics_strict = evaluation.compute_metrics_strict(boxes, types, gt_boxes, gt_types)
        all_metricssoft.append(metrics_soft)
        all_metricsstrict.append(metrics_strict)
        all_pred_types.extend(types)
        all_gt_types.extend(gt_types)

        all_pred_boxes_types.append(list(zip(boxes, types)))
        all_gt_boxes_types.append(list(zip(gt_boxes, gt_types)))

        vis = visualization.draw_boxes(image.copy(), boxes, types)
        visualization.save_output(vis, img_path, output_dir)

    evaluation.summarize_metrics(all_metricsstrict, all_metricssoft)
    visualization.plot_metrics(all_metricsstrict, metrics_plot_path)

    cm = compute_confusion(all_pred_types, all_gt_types)
    plot_confusion_matrix(cm, cm_plot_path)
    mAP, ap_dict = compute_map(all_pred_boxes_types, all_gt_boxes_types)
    print(f"Mean Average Precision (mAP): {mAP:.3f}")
    for cls, ap in ap_dict.items():
        print(f"  {cls} AP: {ap:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood Cell Detection using Image Processing")
    parser.add_argument('--image_dir', type=str, default='data/images', help='Directory with input images')
    parser.add_argument('--label_dir', type=str, default='data/annotations', help='Directory with annotation XMLs')
    parser.add_argument('--output_dir', type=str, default='results/predictions', help='Directory to save results')
    parser.add_argument('--metrics_plot', type=str, default='results/metrics_plot.png', help='Path to save precision/recall plot')
    parser.add_argument('--cm_plot', type=str, default='results/confusion_matrix.png', help='Path to save confusion matrix image')
    args = parser.parse_args()

    run_pipeline( args.image_dir,
        args.label_dir,
        args.output_dir,
        args.metrics_plot,
        args.cm_plot,
    )
