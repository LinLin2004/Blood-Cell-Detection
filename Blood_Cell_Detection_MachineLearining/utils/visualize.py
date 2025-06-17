import cv2
import os

def draw_boxes(img, detections, save_path):
    COLORS = {'RBC': (255, 0, 0), 'WBC': (0, 255, 0), 'Platelets': (0, 0, 255)}
    for x1, y1, x2, y2, label, prob in detections:
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label}:{prob:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
