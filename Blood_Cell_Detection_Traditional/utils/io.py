import cv2
import os
import json
import xml.etree.ElementTree as ET
def load_image(path):
    return cv2.imread(path)

def load_image_paths(img_dir):
    return [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')]

def get_label_path(img_path, ann_dir):
    name = os.path.basename(img_path).replace('.jpg', '.xml')
    return os.path.join(ann_dir, name)

def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        x = int(bbox.find('xmin').text)
        y = int(bbox.find('ymin').text)
        w = int(bbox.find('xmax').text) - x
        h = int(bbox.find('ymax').text) - y
        boxes.append((x, y, w, h))
        labels.append(label)
    return boxes, labels