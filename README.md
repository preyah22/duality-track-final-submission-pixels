
# 🚀 Duality Track : Offroad Semantic Segmentation using DINOv2

## Submission by:
Team Name: Pixels
---
Team Members: Priya, Parul Singh, Preeti Rani
---
Dataset: https://drive.google.com/drive/folders/1MdR3GYLBf9oR3adYsHpAoul9P-NeWsP6
---
Colab Notebook: https://colab.research.google.com/drive/1VdPt_Ycx555yEsG8Uc33d7bhchfv8w1D?usp=sharing
---
## 📌 Overview

This project focuses on **semantic segmentation of off-road environments** using a deep learning approach.
A **DINOv2 Vision Transformer backbone** is used for feature extraction, combined with a **custom ConvNeXt-style segmentation head** for pixel-wise classification.

---

## 🧠 Model Architecture

* Backbone: **DINOv2 (Pretrained Vision Transformer)**
* Head: **Custom ConvNeXt-style Segmentation Head**
* Task: Multi-class semantic segmentation
* Total Classes: 10

---

## ⚙️ Environment & Dependencies

### 🔹 Requirements

* Python ≥ 3.8
* PyTorch
* torchvision
* numpy
* matplotlib
* opencv-python
* tqdm
* PIL

### 🔹 Install dependencies

```bash
pip install torch torchvision numpy matplotlib opencv-python tqdm pillow
```

---

## 📂 Dataset Structure

```
Offroad_Segmentation_Training_Dataset/
│
├── train/
│   ├── Color_Images/
│   └── Segmentation/
│
├── val/
│   ├── Color_Images/
│   └── Segmentation/
```

---

## ▶️ Step-by-Step Instructions

### 1. Mount Google Drive (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 2. Set Dataset Paths

```python
train_data_dir = "/content/drive/MyDrive/Colab Notebooks/Offroad_Segmentation_Training_Dataset/train"
val_data_dir = "/content/drive/MyDrive/Colab Notebooks/Offroad_Segmentation_Training_Dataset/val"
```

---

### 3. Load Model & Backbone

```python
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
backbone.eval()
```

---

### 4. Train the Model

```python
python segmentation-model.ipynb
```

---

### 5. Save Best Model

* Model automatically saved when validation IoU improves
* Saved file:

```
/content/drive/MyDrive/Seg_model_refined.pth
```

---

## 🔁 How to Reproduce Results

1. Use same dataset structure
2. Set image size to:

```
448 × 448
```

3. Train with:

* Epochs: 20-30
* Batch Size: 4–8
* Learning Rate:

  * Initial: 1e-4
  * Fine-tuning: 3e-5

4. Enable best model checkpointing

---

## 📊 Final Results

* Initial IoU: ~0.34
* Final IoU: ~0.50

Performance improved using:

* Resolution enhancement (224 → 448)
* Learning rate tuning

---

## 📈 Expected Outputs

* Trained model file (`.pth`)
* Loss vs Epoch graph
* IoU vs Epoch graph
* Segmented output masks

---

## 🔍 How to Interpret Results

* **IoU (Intersection over Union):**

  * Measures overlap between prediction and ground truth
  * Higher IoU = better segmentation

* **Loss Curve:**

  * Should decrease over epochs
  * Indicates model learning

* **Output Masks:**

  * Each pixel represents a class label
  * Visual quality shows segmentation performance

---

## ⚠️ Notes

* GPU is recommended (Google Colab)
* Training may stop due to session limits — use checkpointing
* Ensure image size is divisible by 14 (for DINOv2 compatibility)

---

## 🏆 Conclusion

The model successfully segments off-road scenes using a transformer-based approach.
Optimization techniques significantly improved performance, achieving a final IoU of ~0.5.

---

