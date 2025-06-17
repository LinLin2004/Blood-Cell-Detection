---
# 🔬 血液细胞检测与分类系统（传统方法）

本项目基于传统图像处理和机器学习方法（非深度学习），实现了对血液细胞图像中**红细胞 (RBC)**、**白细胞 (WBC)** 和**血小板 (Platelet)** 的检测、分类与性能评估。

---

## 🧪 实验设计思路

### 1️⃣ 数据准备
- 数据集采用 Pascal VOC 格式的 XML 标注（推荐 BCCD Dataset）。
- 使用 `utils/create_split.py` 可按需划分训练集与测试集。

### 2️⃣ 候选框生成
- 借助 Selective Search (`utils/selective_search.py`) 提取候选区域。
- 利用非极大值抑制（NMS）去除冗余框。

### 3️⃣ 特征提取与训练
- 使用 HOG (`utils/feature_extraction.py`) 提取每个候选框的图像特征。
- 支持两种分类器：`SVM`（支持向量机） 和 `Random Forest (RF)`。
- 模型训练通过 `train_classifier.py` 实现。

### 4️⃣ 检测与评估
- 调用 `detect_and_eval.py`：将测试图像送入检测流程，分类每个候选框，绘图可视化，评估性能。
- 性能指标包括：
  - 每类的 AP（Average Precision）
  - mAP（mean AP）
  - Precision / Recall
  - 混淆矩阵（由 `utils/evaluation.py` 和 `visualize.py` 支持）

---
## 📁 项目目录结构

```
Blood_Cell_Detection_MachineLearning/
├── data/                        # 图像与标注数据（JPEGImages + Annotations）
├── models/                      # 保存的模型（如 svm_model.pkl, rf_model.pkl）
├── svm_results/                 # SVM 检测结果与评估可视化
├── rf_results/                  # RF 检测结果与评估可视化
├── utils/
│   ├── create_split.py          # 数据集划分工具
│   ├── evaluation.py            # IOU、mAP、混淆矩阵等评估工具
│   ├── feature_extraction.py    # 支持 HOG 特征提取
│   ├── nms.py                   # 非极大值抑制
│   ├── selective_search.py      # 候选框提取
│   ├── visualize.py             # 检测结果可视化
├── detect_and_eval.py           # 执行检测与评估的主逻辑
├── train_classifier.py          # 分类器训练
└── main.py                      # 项目入口，命令行控制（train /detect）
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install opencv-python scikit-learn numpy matplotlib lxml
````

### 训练分类器（以 SVM 为例）

```bash
python main.py \
  --mode train \
  --model_path ./models/svm_model.pkl \
  --model_type svm
```

* `--model_type`: `svm` 或 `rf`
* `--feature_type`: 当前支持 `hog`

### 运行检测与评估

```bash
python main.py \
  --mode detect \
  --data_dir ./data \
  --model_path ./models/svm_model.pkl \
  --feature_type hog \
  --iou_thresh 0.5
```

检测结果将自动：

* 绘制图像（保存到 `results/` 或 `rf_results/`）
* 输出 mAP 与每类 AP
* 打印混淆矩阵

---

## 📊 输出示例

```
[INFO] Loaded model from models/svm_model.pkl
[INFO] Detected 17 WBC, 92 RBC, 6 Platelet
[INFO] Evaluation results:
  - RBC AP: 0.823
  - WBC AP: 0.910
  - Platelet AP: 0.695
  - mAP: 0.809
  - Precision: 0.82
  - Recall: 0.78
```

---

## 📌 注意事项

* 标注格式需为 Pascal VOC（XML）。
* 若使用自己数据，请确保图像与 XML 同名、对应在 `data/JPEGImages/` 与 `data/Annotations/` 中。
* 可视化结果保存在 `results/` 或 `rf_results/` 中，便于对比。

---


