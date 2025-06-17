# 血液细胞检测项目
## 项目概述
本作业实现了一个基于传统数字图像处理方法的血液细胞检测系统，能够识别并标注以下三种细胞：
- **红细胞 (RBC)** - 红色标注框
- **白细胞 (WBC)** - 蓝色标注框
- **血小板 (Platelets)** - 绿色标注框

项目完全使用传统图像处理技术实现，不使用任何神经网络方法（但包含与神经网络方法的对比实验）。

## 数据集
使用 **BCCD Dataset**：
```bash
git clone https://github.com/Shenggan/BCCD_Dataset
```
数据集结构：
```
BCCD_Dataset/
  ├── Annotations/  # XML标注文件
  ├── JPEGImages/   # 原始血液细胞图像
  └── ImageSets/    # 数据集划分
```
## 📁 项目结构

```bash
Blood_Cell_Detection/
├── Blood_Cell_Detection_Traditional/
├── Blood_Cell_Detection_MachineLearning/
├── Blood_Cell_Detection_YOLOv6/
│   └── [YOLOv6训练相关配置与指令]
└── README.md
````
---

## 🧪 方法说明

### 方法一：传统图像处理

* 使用中值滤波和 CLAHE 进行预处理
* 自适应阈值 + watershed 实现细胞分割
* 多特征（面积、色调、边缘密度等）设计规则分类 RBC/WBC/Platelets
* 使用 IoU + AP + mAP + Precision/Recall 评估性能

### 方法二：Selective Search + HOG + SVM / RF

* 使用 Selective Search 生成候选框
* 提取每个框的 HOG 特征
* 使用支持向量机（SVM）或随机森林（RF）分类
* 采用 NMS 去重，mAP 和每类 AP 评估性能

### 方法三：YOLOv6 深度学习检测

* 使用 VOC 格式构建数据集
* 使用 YOLOv6 框架训练端到端目标检测模型
* 使用 COCO-style AP\@0.50, AP\@0.75, AR 等全面评估

---

## ▶️ 快速开始（YOLOv6）

```bash
# 创建虚拟环境并安装YOLOv6依赖
conda create -n yolov6 python=3.10
conda activate yolov6
pip install -r requirements.txt

# 训练YOLOv6模型
python tools/train.py --data data/bccd.yaml --cfg yolov6s.yaml --epochs 100

# 推理测试
python tools/infer.py --weights runs/train/exp/weights/best_ckpt.pt --source data/test
```
---