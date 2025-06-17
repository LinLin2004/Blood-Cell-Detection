import cv2
import os
import matplotlib.pyplot as plt


def draw_boxes(image, boxes, types):
    color_map = {'RBC': (0, 255, 0), 'WBC': (255, 0, 0), 'Platelet': (0, 0, 255)}
    for (x, y, w, h), cls in zip(boxes, types):
        cv2.rectangle(image, (x, y), (x + w, y + h), color_map.get(cls, (255, 255, 255)), 2)
        cv2.putText(image, cls, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map.get(cls, (255, 255, 255)), 1)
    return image

def save_output(image, img_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(out_dir, name), image)

def plot_metrics(metrics_list, save_path):
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    plt.plot(precisions, label='Precision')
    plt.plot(recalls, label='Recall')
    plt.xlabel('Image Index')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision & Recall per Image')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()