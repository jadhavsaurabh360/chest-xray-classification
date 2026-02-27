# Multi-Disease Classification from Chest X-Rays Using Deep Learning

A deep learning pipeline for simultaneous multi-label classification of 14 thoracic 
pathologies from chest X-ray images, built on the NIH ChestX-ray14 dataset.

---

## Overview

Radiologists face growing workloads and fatigue-related errors when screening chest X-rays. 
This project explores whether transfer learning with frozen pre-trained backbones can serve 
as a lightweight diagnostic screening aid — flagging likely pathologies to help prioritize 
radiologist attention.

Three architectures were benchmarked (ResNet-101V2, DenseNet-121, EfficientNet-B7) and 
combined into a stacked ensemble. EfficientNet-B7 emerged as the best single model, 
achieving the highest AUC (0.7173) and lowest Hamming Loss (0.2639) among all individual models.

---

## Pipeline

1. **Stratified Sampling** — 10,000 images sampled with balanced disease representation 
   across all 14 classes
2. **Patient-Level Split** — `GroupShuffleSplit` (80/20) ensures no patient appears in 
   both train and test sets, preventing data leakage
3. **Preprocessing** — Images resized to 224×224, normalized to [0, 1]; EfficientNet 
   receives [0, 255] range per its built-in preprocessing expectations
4. **Model Training** — Frozen pre-trained backbones + custom classification heads 
   (GlobalAvgPool → Dense 512 → Dropout → Dense 256 → Dropout → Sigmoid output)
5. **Loss Function** — Weighted binary cross-entropy (pos_weight=2.0) to handle severe 
   class imbalance
6. **Threshold Calibration** — Thresholds swept from 0.15 to 0.50 to find the optimal 
   precision-recall trade-off per model
7. **Stacked Ensemble** — Predicted probabilities from all three models concatenated and 
   fed into a Logistic Regression meta-learner

---

## Models & Results

| Model | Best Threshold | F1 Score | AUC | Hamming Loss |
|---|---|---|---|---|
| ResNet-101V2 | 0.25 | 0.3068 | 0.6946 | 0.3241 |
| DenseNet-121 | 0.25 | 0.3044 | 0.6973 | 0.3418 |
| **EfficientNet-B7** ⭐ | **0.30** | **0.3283** | **0.7173** | **0.2639** |
| Stacked Ensemble | 0.15 | 0.3445 | 0.7161 | 0.2492 |

> EfficientNet-B7 achieves the best individual AUC (0.7173). The stacked 
> ensemble leads on macro F1 (0.3445) & Hamming Loss by combining signals from all three models but at a much lower prediction threshold

**EfficientNet-B7 standout recalls at threshold 0.30:**
- Infiltration: 89.5% | Effusion: 85.2% | Atelectasis: 77.9%

---

## Tech Stack

- **Framework:** TensorFlow / Keras 2.19
- **Models:** ResNet-101V2, DenseNet-121, EfficientNet-B7 (ImageNet weights)
- **Ensemble:** Scikit-learn `MultiOutputClassifier` + `LogisticRegression`
- **Environment:** Google Colab (A100 GPU)
- **Key Libraries:** NumPy, Pandas, Matplotlib, Seaborn, tqdm, Pillow

---

## Key Design Decisions

- **Frozen backbones** — Chosen due to GPU memory constraints on a 10K image subset; 
  fine-tuning top layers would likely improve performance with more resources
- **Sensitivity over specificity** — Lower thresholds (0.15–0.30) are used intentionally. 
  For screening systems, missing a true positive (false negative) is more costly than a 
  false alarm
- **Weighted loss** — Standard BCE treats all classes equally; the pos_weight=2.0 penalty 
  on the positive class helps the model learn rare conditions like Hernia and Emphysema

---

## Limitations & Future Work

- Trained on ~9% of the full NIH dataset (10K of 112K images) due to resource constraints
- No fine-tuning of backbone layers
- Per-class threshold optimization (instead of a single global threshold) would likely 
  improve per-disease recall
- Grad-CAM visualizations for model explainability would be valuable for clinical context

---

## Dataset

See `README_Dataset.md` for dataset details, source, and label structure.

