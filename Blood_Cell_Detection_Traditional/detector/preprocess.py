import cv2

def apply_blur(image):
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1)

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def histogram_equalization(gray):
    return cv2.equalizeHist(gray)
