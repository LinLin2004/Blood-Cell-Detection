import cv2
import os
import numpy as np
from skimage.feature import hog
from xml.etree import ElementTree as ET

def extract_feature_from_patch(patch, feature_type='hog'):
    patch = cv2.resize(patch, (64, 64))
    if feature_type == 'hog':
        features = hog(patch, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),channel_axis=-1,
                     feature_vector=True)
    elif feature_type == 'color_hist':
        features = []
        for i in range(3):
            hist = cv2.calcHist([patch], [i], None, [16], [0, 256])
            features.extend(hist.flatten())
        features = np.array(features)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    return features

def extract_features_from_dataset(data_dir, split='train', feature_type='hog'):
    jpeg_dir = os.path.join(data_dir, 'JPEGImages')
    ann_dir = os.path.join(data_dir, 'Annotations')
    split_file = os.path.join(data_dir, 'ImageSets', f'{split}.txt')

    X, y = [], []
    with open(split_file, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]

    for fname in filenames:
        img_path = os.path.join(jpeg_dir, f'{fname}.jpg')
        ann_path = os.path.join(ann_dir, f'{fname}.xml')
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(ann_path):
            continue

        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = int(float(bbox.find('xmin').text))
            y1 = int(float(bbox.find('ymin').text))
            x2 = int(float(bbox.find('xmax').text))
            y2 = int(float(bbox.find('ymax').text))

            patch = img[y1:y2, x1:x2]
            if patch.shape[0] < 10 or patch.shape[1] < 10:
                continue
            feat = extract_feature_from_patch(patch, feature_type)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)
