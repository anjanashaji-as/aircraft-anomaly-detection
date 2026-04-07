# Anomaly Detection in Aircraft Maintenance Logs

**MSc Computer Science (Artificial Intelligence) — Final Dissertation**

---

## Project Overview

This project presents a machine learning framework for detecting anomalies in unstructured aircraft maintenance logs — without requiring labeled data. It combines classical unsupervised models, hybrid fusion, and deep learning (BERT + Autoencoder) to surface unusual log entries that may indicate latent safety risks.

---

## Models Used

| Model | Features Used | Strength |
|---|---|---|
| One-Class SVM (OC-SVM) | TF-IDF text embeddings | Captures text-based irregularities |
| Isolation Forest (IF) | Structured features (text length, abbreviation hits, etc.) | Efficient on tabular data |
| Hybrid Model (OC-SVM + IF) | Both text & structured | Best practical balance |
| Deep Autoencoder (BERT) | DistilBERT embeddings (768-dim) | Best semantic anomaly detection |

---

## Project Structure

```
aircraft-anomaly-detection/
│
├── Anomaly_Detection_in_Aircraft_Maintenance_Logs.ipynb   # Main notebook
├── Anomaly_Detection_in_Aircraft_Maintenance_Logs.pdf     # Dissertation report
│
├── data/                          # Place your CSV datasets here
│   ├── Aircraft_Annotation_DataFile.csv
│   ├── Aviation_Abbreviation_Dataset.csv
│   ├── Aviation_Grammar_Dataset.csv
│   ├── Aviation_Morphosyntactic_Dataset.csv
│   └── Aviation_TermBanks_Dataset.csv
│
├── dissertation_outputs/          # Auto-generated plots and results
│   ├── pr_text.png
│   ├── pr_if.png
│   ├── pr_hybrid.png
│   ├── pr_deep.png
│   ├── autoencoder_loss.png
│   ├── autoencoder_error_distribution.png
│   ├── umap_autoencoder.png
│   ├── bert_anomaly_heatmap.png
│   ├── results_table.csv
│   ├── topk_deep.csv
│   └── top_terms_lift.csv
│
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/anjanashaji-as/aircraft-anomaly-detection.git
cd aircraft-anomaly-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your datasets
Place the five CSV files into the `data/` folder (see structure above).

### 4. Run the notebook
```bash
jupyter notebook "Anomaly_Detection_in_Aircraft_Maintenance_Logs.ipynb"
```

> **Note:** The notebook works in both **Google Colab** and local Jupyter. When run locally, the Colab file upload step is automatically skipped — just place your CSV files in the `data/` folder as described in Step 3.

---

## Key Results

| Model | AP Score | P@50 | P@100 |
|---|---|---|---|
| Text OC-SVM | 0.331 | 0.36 | 0.40 |
| Isolation Forest | 0.536 | 0.78 | 0.68 |
| Hybrid Model | 0.534 | 0.78 | 0.69 |
| Deep Autoencoder (BERT) | 0.352 | 0.40 | 0.35 |

- **Best semantic detection:** Deep Autoencoder (BERT) — ROC-AUC ~0.72
- **Most interpretable:** Isolation Forest (structured features like log length)
- **Most practical:** Hybrid Model (combines text + structure signals)

---

## Key Features

- Aviation-specific NLP preprocessing (abbreviation expansion, lemmatization, term bank matching)
- Dual feature representation: TF-IDF vectors + structured linguistic indicators
- BERT (DistilBERT) embeddings for deep semantic understanding
- Proxy anomaly labels based on rare vocabulary ratio and corrective action length
- Evaluation via Precision-Recall curves, Precision@K, ROC-AUC, and F1-score
- Visualizations: UMAP projections, reconstruction error histograms, term-lift analysis

---

## Tech Stack

- **Python 3.9**
- **scikit-learn** — OC-SVM, Isolation Forest, Logistic Regression
- **PyTorch** — Deep Autoencoder
- **Hugging Face Transformers** — DistilBERT embeddings
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualizations
- **UMAP** — Dimensionality reduction

---

## Citation

If you use or reference this work, please cite:

```
Anomaly Detection in Aircraft Maintenance Logs.
MSc Dissertation, 2025.
```

---

## License

This project is submitted as an academic dissertation. Please contact the author before reusing any part of this work.
