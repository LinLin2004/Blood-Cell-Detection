import cv2
import numpy as np
def enhance_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    eq_v = cv2.equalizeHist(v)
    hsv[:, :, 2] = eq_v
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return gray

def segment_cells(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (5, 5), sigmaX=1)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    return sure_fg

