import cv2
import numpy as np
import config
from sklearn.cluster import KMeans

def circularity(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter * perimeter)

def edge_density(gray_roi):
    edges = cv2.Canny(gray_roi, 50, 150)
    return np.sum(edges > 0) / (gray_roi.shape[0] * gray_roi.shape[1] + 1e-6)

def dominant_color_hue(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    reshaped = hsv[:, :, 0].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init='auto').fit(reshaped)
    centers = kmeans.cluster_centers_.flatten()
    return np.mean(centers)

def detect_cells(fg_mask, original_image):
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, types = [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        circ = circularity(cnt)

        roi = original_image[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hue = hsv[:, :, 0].mean()
        mean_sat = hsv[:, :, 1].mean()

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_val = cv2.mean(gray_roi)[0]
        edge_ratio = edge_density(gray_roi)

        dom_hue = dominant_color_hue(roi)

        # ------------------------------
        # 多特征融合规则分类
        if area > config.RBC_THRESH and circ > 0.7 and 0.8 < aspect_ratio < 1.2 and mean_val > 100:
            cell_type = "RBC"

        elif area > config.WBC_THRESH and 100 < mean_hue < 180 and mean_sat > 60 and edge_ratio > 0.03:
            cell_type = "WBC"

        elif area < config.RBC_THRESH and dom_hue < 100 and edge_ratio < 0.05:
            cell_type = "Platelet"

        else:
            continue
        # ------------------------------

        boxes.append((x, y, w, h))
        types.append(cell_type)

    return boxes, types
