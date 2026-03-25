# ArtExtract — Task 2: Painting Similarity via Deep Metric Learning

**GSoC 2026 Evaluation | HumanAI @ CERN Umbrella Organization**  
**Dataset:** [National Gallery of Art Open Data](https://github.com/NationalGalleryOfArt/opendata)

---

## Task

> *Build a model to find similarities in paintings, e.g. portraits with a similar face or pose.*

---

## Strategy Overview

| Stage | Approach |
|---|---|
| **Similarity proxy** | Same style label → similar; different style → dissimilar |
| **Baseline** | Frozen ResNet50 (ImageNet) + FAISS cosine search |
| **Improved model** | Triplet Loss fine-tuning (projection head on frozen backbone) |
| **Portrait similarity** | Figure/theme-tagged paintings queried separately |
| **Evaluation** | Precision@K per style, mean Precision@10 |

---

## Why This Approach

### Approaches Rejected and Why

| Approach | Reason Rejected |
|---|---|
| **FaceNet** | >60% of paintings have no detectable face; domain gap (trained on photos, not paintings); blind to pose/composition/style |
| **Pose estimation** | Brittle on historical paintings; only works for figure paintings; silent failures on complex scenes |
| **Gram matrices** | Captures texture well but blind to composition and semantic content |
| **Pure style transfer features** | Too abstract; misses subject matter similarity |

### Why ResNet50 + Cosine Similarity

- ResNet50 `layer4` captures mid-level features simultaneously — pose, composition, texture, colour, and semantics
- ImageNet pretrained weights transfer surprisingly well to paintings despite the domain gap
- Cosine similarity on L2-normalised vectors is correct — magnitude is meaningless in embedding space, only direction matters
- Inner product on L2-normalised vectors = cosine similarity (FAISS `IndexFlatIP`)
- FAISS makes retrieval near-instant even over thousands of vectors

### Transfer Learning Reasoning

ImageNet supervision produces a general object recogniser. Fine-tuning with style supervision redirects internal representations to be art-style aware — the same principle as GPT pretraining → chat fine-tuning, or FaceNet ImageNet → face fine-tuning. This is the most principled form of transfer learning: large general dataset → small domain-specific dataset.

---

## Dataset

### Source
National Gallery of Art open dataset — three CSV files:

| File | Contents |
|---|---|
| `objects.csv` | Painting metadata — title, artist, classification, open-access flag |
| `published_images.csv` | IIIF image URLs per object |
| `objects_terms.csv` | Style, theme, and technique labels per object |

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total objects in NGA dataset | 144,192 |
| Paintings (classification filter) | 4,430 (3% of collection) |
| Paintings with images | 4,053 |
| Open-access paintings (downloaded) | 2,888 |
| Paintings with style labels | 2,921 across 12 styles |
| Figure/portrait paintings (theme tags) | ~1,024 |
| Paintings tagged literally as "portrait" | 24 — too sparse for evaluation |

> **Note:** NGA is dominated by prints (64,859), photographs (21,695), and drawings (18,299). Mixing these with paintings would make the embedding space incoherent — similarity scores would reflect *class* differences rather than within-painting visual similarity. We restrict to paintings only.

### Key Engineering Note: Non-Obvious Join Key

Images are joined to objects via `published_images.depictstmsobjectid` → `objects.objectid`. This is not documented prominently in the NGA README and required careful inspection.

### Why Only 12 Style Labels?

The NGA dataset uses broad curatorial style terms. Only 12 styles appear with sufficient coverage for evaluation. Two styles — Neoclassic (n=24) and Symbolist (n=10) — have too few examples for reliable Precision@K measurement.

### Similarity Label Strategy

No explicit pairwise similarity annotations exist in the dataset. We define:
- **Similar** → two paintings share the same style label (e.g., both "Impressionist")
- **Dissimilar** → two paintings have different style labels

This is a standard proxy used in metric learning on art datasets. Its limitations are acknowledged in the evaluation section.

---

## Model Architecture

### Baseline: Frozen ResNet50 Feature Extractor

```
Input Image (3 × 224 × 224)
        │
        ▼
ResNet50 (ImageNet weights, fully frozen)
Remove final FC layer
        │
        ▼
2048-dimensional feature vector
        │
        ▼
L2 normalisation
        │
        ▼
FAISS IndexFlatIP (cosine similarity search)
```

### Fine-Tuned Model: Triplet Loss + Projection Head

```
Input Image
        │
        ▼
ResNet50 backbone (FROZEN — same ImageNet weights)
        │
        ▼
2048-d feature vector
        │
        ▼
Projection Head:
  Linear(2048 → 512) → BatchNorm → ReLU → Dropout(0.3)
  Linear(512  → 256) → L2 normalise
        │
        ▼
256-dimensional embedding
```

**Why freeze the backbone?**  
The NGA dataset (~2,888 open-access paintings) is too small to fully fine-tune ResNet50 without catastrophic forgetting. Freezing the backbone and training only the projection head prevents overfitting while still redirecting the embedding space toward style-discriminative representations.

**Triplet Loss:**

$$\mathcal{L} = \max(0,\ d(a, p) - d(a, n) + \text{margin})$$

- Anchor $a$ — query painting
- Positive $p$ — different painting, same style as anchor
- Negative $n$ — painting from a different style
- Margin = **0.3** (allows soft violations without over-constraining)

---

## Training

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Epochs | 5 |
| Triplet margin | 0.3 |
| Backbone | Frozen |

### Loss Trajectory

| Epoch | Triplet Loss |
|---|---|
| 1 | 0.1615 |
| 2 | 0.0488 |
| 3 | 0.0124 |
| 4 | 0.0102 |
| 5 | 0.0044 |

### Loss Collapse — Key Diagnosis

Loss hit **0.0000 from epoch 3 onwards**. Random triplet sampling produces predominantly *easy* triplets — pairs already well-separated in embedding space that contribute zero gradient. This is why improvement is capped at +9.3% instead of going higher. The fix is **hard negative mining** (see Future Work).

---

## Evaluation

### Metric: Precision@K

This is a retrieval task, not classification. The correct evaluation paradigm is retrieval quality.

For a query painting $q$ with style $s_q$:

$$\text{Precision@K}(q) = \frac{|\{i \in \text{top-K} : \text{style}(i) = s_q\}|}{K}$$

We report **Mean Precision@10** across all style-labelled paintings as the primary metric, and **per-style Precision@10** to understand which styles are most retrievable.

**Why not accuracy?** There is no single "correct" answer for a retrieval query — many paintings can be legitimately similar to a query. Retrieval metrics (Precision@K) are the natural fit.

### Results

| Model | Mean Precision@10 |
|---|---|
| Baseline (frozen ResNet50, 2048-d) | 0.376 |
| Fine-Tuned (Triplet Loss, 256-d) | 0.469 |
| **Improvement** | **+9.3%** |

### Per-Style Baseline Results (Precision@10)

| Style | Precision@10 | n |
|---|---|---|
| Gothic | 0.594 | 63 |
| Renaissance | 0.523 | 410 |
| American Realist | — | — |
| Impressionist | — | — |
| Neoclassic | 0.038 | 24 |
| Symbolist | 0.090 | 10 |

> Gothic and Renaissance score highest — they have the most distinctive visual conventions (dark backgrounds, formal poses, specific iconography). Neoclassic and Symbolist score lowest — too few examples for reliable retrieval.

### Portrait / Figure Similarity

The task specifically mentions *"portraits with a similar face or pose."*  
We demonstrate this using NGA theme tags (`male`, `female`, `family`) to identify ~1,024 figure paintings:

- Portrait queries cluster tightly: cosine similarity scores **0.709–0.732**
- Scene/landscape queries are looser: **0.513–0.546**
- The fine-tuned model retrieves more period-consistent portraits (Renaissance → Baroque, not Naive folk)

### Metric Limitation

Style labels are a proxy. Two paintings can be visually very similar yet carry different style annotations (e.g., Post-Impressionist vs. Impressionist). This introduces label noise into the evaluation — a known limitation of label-based Precision@K.

---

## Qualitative Observations

- **Renaissance portrait query:** Baseline found Naive folk portraits (fooled by pose similarity); fine-tuned model found Renaissance/Baroque portraits (learned period conventions — dark backgrounds, formal poses, rich clothing)
- **Impressionist ballet query:** Both models struggled — complex multi-figure scenes are harder than portraits
- **"Before the Ballet" → "The Dance Lesson":** Near-perfect match (same artist, same subject)

---

## Failure Cases

| Failure Type | Description |
|---|---|
| **Visually similar, different styles** | Post-Impressionist and Impressionist paintings share visual features but are labelled differently |
| **Label noise** | Some NGA style annotations are inconsistent or missing |
| **Low-level bias** | Model may group paintings by dominant colour rather than stylistic convention |
| **Loss collapse** | Random triplet sampling exhausted gradient signal by epoch 3 |
| **Small dataset** | ~2,888 open-access paintings limits generalisation |

---

## Future Work

| Improvement | Expected Impact |
|---|---|
| **Hard negative mining** | Fix easy triplet problem; push Precision@10 toward ~0.55+ |
| **ViT backbone** | Self-attention across all patches from layer 1 captures global composition better; expected ~0.55–0.60 Precision@10 (not implemented — M1 8GB memory constraint) |
| **FaceNet as additional head** | Portrait-specific face similarity score combined with style embedding (hybrid retrieval) |
| **Metadata reranking** | Boost results sharing period, nationality, or theme |
| **Query expansion** | Average query with top-3 results, re-search |
| **SupCon loss** | Uses all positive pairs in batch — more efficient than triplet loss |
| **Cross-class similarity** | Painting ↔ preparatory sketch retrieval |
| **Larger embedding dimension** | 512-d instead of 256-d for finer style distinctions |

---

## Repository Structure

```
Task2/
│
├── task_2_similarity_final.ipynb   ← Main notebook with all outputs
├── task_2_similarity_final.pdf     ← PDF export with rendered outputs
│
└── README.md                       ← This file
```

---

## Environment & Reproducibility

```
Python        3.10+
PyTorch       2.x
torchvision
faiss-cpu
scikit-learn
pandas, numpy, matplotlib, seaborn
Pillow
requests
```

**Hardware used:** Apple M1 (8GB RAM)  
**Key constraint:** `sudo purge` + minimal column loading required throughout to prevent kernel crashes under 8GB RAM.

**Data:**  
Download the NGA open dataset from https://github.com/NationalGalleryOfArt/opendata  
Place CSVs in `opendata/data/`.  
Images are fetched via IIIF API and saved to `nga_images/` — the notebook download cell handles this automatically.

**Saved artefacts:**
- `nga_embeddings.npy` + `nga_ids.npy` — baseline embeddings
- `nga_embeddings_finetuned.npy` + `nga_ids_finetuned.npy` — fine-tuned embeddings
- `finetuned_embedder.pth` — fine-tuned model weights

---

## Contact

Submitted for GSoC 2026 — ArtExtract | HumanAI @ CERN  
**Email:** human-ai@cern.ch  
**Subject:** Evaluation Test: ArtExtract
