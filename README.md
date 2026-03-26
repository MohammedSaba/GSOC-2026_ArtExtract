# ArtExtract — GSoC 2026 Evaluation | Mohammed Saba
```
GSOC-2026_ArtExtract/
│
├── Task_1/
│   ├── model_data_analysis/
│   │   └── ArtExtract_DataAnalysis_Final.ipynb
│   │
│   ├── model_notebook/
│   │   ├── ARTExtract_MODEL_1_FINAL.ipynb
│   │   └── ArtExtract_MODEL_2_FINAL.ipynb
│   │
│   ├── notebook_pdf_outputs/
│   │   ├── ARTExtract_MODEL_1_FINAL.pdf
│   │   ├── ArtExtract_MODEL_2_FINAL.pdf
│   │   └── ArtExtract_DataAnalysis_Final.pdf
│   │
│   └── README.md
│
├── Task_2/
│   ├── notebook/
│   │   └── ArtExtract_Task_2_FINAL.ipynb
│   │
│   ├── notebook_pdf_output/
│   │   ├── ArtExtract_Task_2_FINAL.pdf
│   │   
│   │
│   └── README.md
│
└── README.md  
```
## Results Summary

| Task | Model | Val Top-1 | Val Top-5 | Macro F1 |
|---|---|---|---|---|
| Style (27 classes) | Model 1 — Multi-Task CNN-RNN | 60.0% | 94.2% | 0.562 |
| Genre (10 classes) | Model 1 — Multi-Task CNN-RNN | 78.5% | 98.7% | 0.759 |
| Artist (23 classes) | Model 2 — Artist CNN-RNN | 78.0% | 94.8% | 0.755 |
| Painting Similarity | Triplet Loss Fine-Tuned ResNet50 | Precision@10: 46.9% | — | — |

## Task Details

- **Task 1** — Multi-task CNN-RNN for painting classification (style, genre, artist) using WikiArt. Architecture split into two models to solve artist gradient sparsity. See [`Task_1/README.md`](Task_1/README.md)
- **Task 2** — Painting similarity via deep metric learning on the NGA open dataset. Baseline + triplet loss fine-tuning evaluated with Precision@10. See [`Task_2/README.md`](Task_2/README.md)

Submitted to human-ai@cern.ch | Subject: Evaluation Test: ArtExtract
