
# 🧬 Blood Cell Detection Using Classical Image Processing

本项目基于传统图像处理方法（非深度学习）实现对红细胞（RBC）、白细胞（WBC）、血小板（Platelet）的自动检测、分割与分类评估，使用 **BCCD 血液细胞图像数据集**，支持可视化、精度分析、混淆矩阵与 mAP 等指标。



## 🧪 实验流程设计

### (1) 数据预处理阶段

* 高斯或中值滤波去噪；
* CLAHE + HSV 增强，提升细胞边缘清晰度；
* 转为灰度图，作为分割输入。

### (2) 细胞区域分割

* 使用自适应阈值提取前景；
* 形态学操作清除噪点；
* 距离变换 + watershed 算法分离粘连细胞；
* 保存分割轮廓图和区域面积直方图。

### (3) 基于规则的目标检测与分类

使用以下特征组合判断细胞类型：

| 特征             | 用于作用             |
| -------------- | ---------------- |
| 面积 Area        | 区分大小（血小板更小）      |
| 圆度 Circularity | RBC 更接近圆形        |
| 长宽比            | 区分扁圆 vs 椭圆       |
| 色调 Hue         | 区分紫色核（WBC）       |
| 饱和度 Sat        | Platelet 通常饱和度较低 |
| 灰度亮度           | 排除亮背景或小干扰        |
| 边缘密度           | WBC 边缘更复杂        |
| KMeans 主色调     | 血小板偏灰，主色调值较低     |

## 🧠 特征判断逻辑（分类依据）

细胞类型基于以下特征组合判断：

| 类型       | 面积  | 圆度    | 色调 Hue | 灰度亮度 | 边缘密度 | 饱和度 |
| -------- | --- | ----- | ------ | ---- | ---- | --- |
| RBC      | 大、圆 | >0.75 | 中性     | 明亮   | 低    | 中   |
| WBC      | 中等  | 可变    | 紫色区域   | 偏暗   | 较密   | 高   |
| Platelet | 小、扁 | <0.8  | 灰暗     | 暗淡   | 较稀   | 低   |

---

### (4) 精度评估

* 严格评估：位置 + 分类均正确；
* 宽松评估：位置对但分类错给部分分；
* 输出指标：

  * 平均 Precision / Recall
  * Confusion Matrix
  * mAP（每类平均精度）

### (5) 可视化结果

* 检测框图（带标签）；
* 分割轮廓图（彩色区域 + 红线）；
* 每张图的分割区域面积直方图；
* 混淆矩阵和指标曲线图。

### (6) 参数调试与迭代

* 所有检测参数集中于 `config.py`：

  ```python
  CLASS_NAMES = ['RBC', 'WBC', 'Platelet']
  MIN_AREA = 100
  RBC_THRESH = 1700
  WBC_THRESH = 900
  IOU_THRESHOLD = 0.2
  ```
* 可调整面积阈值、圆度等；
* 利用分类输出 + 预测图分析误判来源并优化规则。

---

## 📁 项目结构

```
Blood_Cell_Detection_Traditional/
├── main.py                         # 主程序入口（支持命令行参数）
├── config.py                       # 各类参数配置（面积阈值、IoU 等）
├── detector/
│   ├── preprocess.py               # 图像预处理模块
│   ├── segmentation.py             # 灰度增强 + 分割 + 可视化
│   ├── detection.py                # 多特征规则分类（面积、圆度、色调等）
│   └── evaluation.py               # 严格+宽松评估方法
├── utils/
│   ├── io.py                       # 图像和标注文件读写
│   ├── metrics.py                  # 混淆矩阵、mAP 等指标计算
│   └── visualization.py           # 可视化绘图工具
├── data/
│   ├── images/                     # 输入图像
│   └── annotations/               # Pascal VOC 格式标注
├── results/
│   ├── predictions/               # 输出检测图像（带框）
│   ├── metrics_plot.png           # 精度/召回率图
│   └── confusion_matrix.png       # 混淆矩阵图
```

---

## ⚙️ 安装依赖

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

---

## 🚀 运行示例

```bash
python main.py \
  --image_dir data/images \
  --label_dir data/annotations \
  --output_dir results/predictions \
  --metrics_plot results/metrics_plot.png \
  --cm_plot results/confusion_matrix.png
```
---

## 📊 输出结果说明

* ✅ **检测图像**：带标签框的预测图保存于 `results/predictions/`；
* ✅ **精度评估**：

  * 严格匹配：IoU + 分类一致；
  * 宽松匹配：IoU 达标，分类错误记为 0.5；
* ✅ **可视化图表**：

  * `metrics_plot.png`：平均精度与召回率；
  * `confusion_matrix.png`：分类混淆热力图；
  * `seg_hist/`：每图像分割区域面积统计；
  * `seg_overlay/`：红线 + 多区域着色分割可视化。

---


## 📈 支持指标

* `Precision / Recall`（每图严格 + 宽松评估）
* `Confusion Matrix`
* `mAP + per-class AP`

---

