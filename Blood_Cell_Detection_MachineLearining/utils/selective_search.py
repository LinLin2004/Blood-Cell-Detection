import selectivesearch
import cv2

def get_candidate_regions(img, scale=500, sigma=0.9, min_size=10):
    _, regions = selectivesearch.selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
    candidates = set()
    h_img, w_img = img.shape[:2]
    for r in regions:
        x, y, w, h = r['rect']
        if w < 20 or h < 20:
            continue
        if w / h > 2.0 or h / w > 2.0:
            continue
        if w * h > w_img * h_img * 0.7:
            continue
        candidates.add((x, y, x + w, y + h))
    return list(candidates)

def get_multi_scale_candidates(img, scales=[150, 300, 500]):
    all_boxes = set()
    for scale in scales:
        boxes = get_candidate_regions(img, scale=scale)
        all_boxes.update(boxes)
    return list(all_boxes)
