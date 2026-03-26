# ArtExtract — Task 1: Convolutional-Recurrent Painting Classification

**GSoC 2026 Evaluation Submission — HumanAI Organization**
**Project:** ArtExtract
**Task:** Task 1 — Convolutional-Recurrent Architectures
**Dataset:** https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md

---

## Architecture Decisions — Data-Driven Reasoning

Every architectural choice below is backed by numbers computed directly
from the WikiArt label CSVs. The analysis notebook
`ArtExtract_DataAnalysis.ipynb` contains all code and outputs.

### Dataset Analysis

| CSV              | Images     | Classes |
| ---------------- | ---------- | ------- |
| Style            | 57,025     | 27      |
| Artist           | 13,346     | 23      |
| Genre            | 45,503     | 10      |
| **Total unique** | **57,340** | —       |

Raw dataset sizes (including train and validation sets):
- Style: 81,446 images
- Artist: 19,052 images
- Genre: 64,994 images

Merged dataset (used for training multi-task models):
- Train set: 57,238 images
- Validation set: 24,573 images

Note:
The merged dataset size is smaller because not all images have labels for all tasks 
(style, genre, artist). During preprocessing, datasets are merged on filename, 
and missing labels are handled using masking.

**Label overlap matrix (computed from actual files):**

| Intersection   | Count  | Coverage                                 |
| -------------- | ------ | ---------------------------------------- |
| Style ∩ Artist | 13,226 | 99.1% of artist images have style labels |
| Style ∩ Genre  | 45,290 | 99.5% of genre images have style labels  |
| Artist ∩ Genre | 11,294 | 84.6% of artist images have genre labels |
| All three      | 11,276 | complete coverage                        |

---

## Approach Exploration

### Baseline Considered — Plain CNN (Not Used)
A plain CNN (ResNet50 with global average pooling → linear head) was
considered as a starting point. It was not implemented as a standalone
baseline because the task specifically requires convolutional-recurrent
architectures, and the evaluators are assessing the CNN-RNN design
directly. The LSTM is not an optional enhancement — it is the required
architectural component per the task specification.

### Approach 1 — Full Three-Task Multi-Task (Rejected)

Outer join all three CSVs → one model → three heads.

**Coverage in combined dataset:**

```
Style  : 99.4%  ✅
Genre  : 79.3%  ✅
Artist : 23.3%  ❌
```

**Gradient signal per batch of 32 (computed):**

```
Style  labels per batch : ~32  ✅ dense
Genre  labels per batch : ~25  ✅ dense
Artist labels per batch :  ~7  ❌ sparse — 4.5x less than Style
```

The artist head receives 4.5× less gradient signal per batch than the
style head. The shared backbone becomes style-biased, permanently
handicapping the artist head.

**Solutions considered and rejected:**

| Solution                  | Reason Rejected            |
| ------------------------- | -------------------------- |
| Train 100+ epochs         | Does not fix backbone bias |
| Balanced sampling         | Sparse gradient persists   |
| GradNorm / PCGrad         | Too complex for deadline   |
| Task-specific LR          | Caused instability         |
| Uncertainty-weighted loss | Unstable early training    |

---

### Approach 2 — Intersection Only (Rejected)

Train only on 11,276 images that have all three labels.

* Style classes represented: **16 / 27**
* Missing classes: **11 (unlearnable)**
* Imbalance ratio: **347×**

---

### Approach 3 — Style + Genre Multi-Task + Separate Artist (Chosen ✅)

**Why Style and Genre work together:**

```
Combined dataset : 57,340 images
Style  coverage  : 99.4%
Genre  coverage  : 79.3%

Batch signal:
Style : ~32
Genre : ~25
```

* Dense gradients
* Full class coverage
* Shared visual features (composition, palette, subject)

**Why Artist is separate:**

* Sparse gradient (23.3% coverage)
* Hard fine-grained task
* Requires dedicated optimization
* Avoids multi-task interference

---

## Results Summary

| Task                | Model                        | Val Top-1 | Val Top-5 | Macro F1 |
| ------------------- | ---------------------------- | --------- | --------- | -------- |
| Style (27 classes)  | Model 1 — Multi-Task CNN-RNN | 60.0%     | 94.2%     | 0.562    |
| Genre (10 classes)  | Model 1 — Multi-Task CNN-RNN | 78.5%     | 98.7%     | 0.759    |
| Artist (23 classes) | Model 2 — Artist CNN-RNN     | 78.0%     | 94.8%     | 0.755    |

---

## Repository Structure

```
Task_1/
│
├── model_data_analysis/
│   └── ArtExtract_DataAnalysis_Final.ipynb
│
├── model_notebook/
│   ├── ARTExtract_MODEL_1_FINAL.ipynb
│   └── ArtExtract_MODEL_2_FINAL.ipynb
│
├── notebook_pdf_outputs/
│   ├── ARTExtract_MODEL_1_FINAL.pdf
│   ├── ArtExtract_MODEL_2_FINAL.pdf
│   └── ArtExtract_DataAnalysis_Final.pdf
│
└── README.md
```

---

## Architecture Overview

### Why Two Separate Models?

* Style & Genre share visual cues → joint learning beneficial
* Artist classification is harder → needs isolation
* Prevents gradient imbalance
* Enables independent tuning

---

## Model 1 — Multi-Task CNN-RNN (Style + Genre)

### Architecture

```
Input Image (3 × 224 × 224)
        │
        ▼
┌─────────────────────────────┐
│   ResNet50 Backbone         │
│   Frozen: layers 0–5        │
│   Unfrozen: layer3, layer4  │
│   Output: (B, 2048, 7, 7)   │
└─────────────────────────────┘
        │
        ▼
(B, 49, 2048)   ← spatial tokens
        │
        ▼
┌─────────────────────────────┐
│   Bidirectional LSTM        │
│   hidden_size = 256         │
└─────────────────────────────┘
        │
        ├──────────────┐
        ▼              ▼
 Style Head        Genre Head
```

### Key Design Choices

* Spatial tokenization preserves structure
* LSTM captures spatial relationships
* Partial fine-tuning prevents overfitting
* Masked loss handles missing labels

---

## Model 2 — Artist CNN-RNN

### Architecture

```
Input Image
        │
        ▼
ResNet50 → (B, 2048, 7, 7)
        │
        ▼
(B, 7, 14336) ← row-wise sequence
        │
        ▼
LSTM → last timestep
        │
        ▼
Artist Head
```

### Key Design Choices

* Row-wise tokenization captures composition
* No masking required
* Fully supervised training
* Overfitting controlled via regularization

---

## Evaluation Metrics

* Top-1 Accuracy
* Top-5 Accuracy
* Macro F1
* Per-class Precision / Recall / F1

---

## Outlier Detection

Defined as:

* High-confidence incorrect predictions
* Style/Genre ≥ 80%
* Artist ≥ 90%

Used to identify:

* Mislabelled data
* Ambiguous artworks

---

## Proposed Future Work — Pseudo-Labeling Pipeline

```
Step 1: Train Style+Genre model
Step 2: Train Artist model
Step 3: Predict artist labels on full dataset
Step 4: Keep predictions ≥ 80% confidence
Step 5: Retrain full 3-task model
Step 6: Iterate
```

**Why the 80% confidence threshold is critical:**

Without threshold: at 78% model accuracy, ~22% of pseudo labels are wrong.
With 80% threshold: only the model's most certain predictions are kept —
wrong label rate drops substantially, trading coverage for label quality.

**Why 78% artist accuracy (vs our initial 65% estimate) matters:**
The original pseudo-labeling plan was designed assuming ~65% accuracy,
where noise was a significant concern. At 78%, the model is substantially
more reliable, meaning the 80% confidence threshold retains more images
while producing cleaner labels than initially projected. This makes the
iterative pipeline more viable and faster to converge.

---

## Environment & Reproducibility

* Model 1: Kaggle (T4 GPU)
* Model 2: Colab (T4 GPU)

```
Python      3.10+
PyTorch     2.x
torchvision
scikit-learn
pandas, numpy, matplotlib, seaborn
```

Dataset: Kaggle WikiArt
Labels: ArtGAN CSVs

---

## Contact

Submitted for GSoC 2026 — ArtExtract
HumanAI 

Email: [human-ai@cern.ch](mailto:human-ai@cern.ch)
Subject: Evaluation Test: ArtExtract
