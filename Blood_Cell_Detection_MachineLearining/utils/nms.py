import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比 (IoU)
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: IoU 值
    """
    # 计算交集区域的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集区域面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # 计算两个框各自的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - inter_area
    
    # 避免除零错误
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def nms(detections, iou_threshold=0.5):
    """
    非极大值抑制算法 (NMS)
    :param detections: 检测结果列表，每个元素为 (x1, y1, x2, y2, label, score)
    :param iou_threshold: IoU 阈值，用于抑制重叠框 (默认0.5)
    :return: 保留的检测结果列表 [x1, y1, x2, y2, label, score]
    """
    # 如果没有检测结果，直接返回空列表
    if not detections:
        return []
    
    # 将检测结果按分数降序排序
    detections = sorted(detections, key=lambda x: x[5], reverse=True)
    
    # 初始化保留列表
    keep = []
    
    # 按类别分组处理
    class_groups = {}
    for det in detections:
        label = det[4]
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(det)
    
    # 对每个类别独立进行NMS处理
    for label, group in class_groups.items():
        # 当前类别内已排序的检测结果
        sorted_group = sorted(group, key=lambda x: x[5], reverse=True)
        
        while sorted_group:
            # 选择当前最高分的检测框
            current = sorted_group.pop(0)
            keep.append(current)
            
            # 用于存储保留的检测框
            keep_boxes = []
            
            # 遍历剩余的检测框
            for candidate in sorted_group:
                # 计算当前框与候选框的IoU
                current_box = current[:4]
                candidate_box = candidate[:4]
                iou = calculate_iou(current_box, candidate_box)
                
                # 如果IoU小于阈值则保留
                if iou < iou_threshold:
                    keep_boxes.append(candidate)
            
            # 更新当前类别待处理列表
            sorted_group = keep_boxes
            keep_boxes = []
    
    return keep