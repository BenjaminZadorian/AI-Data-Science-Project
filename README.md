# Machine Learning Coursework

This repository contains two machine learning coursework projects covering regression/classification/clustering on weather data, and NLP-based fraud detection on job postings.

---

## Projects

| Notebook | Topic | Type |
|---|---|---|
| `Max-Temp-Prediction.ipynb` | Predicting daily max temperature at LaGuardia Airport | Regression, Classification, Clustering |
| `Fake-Job-Posting-Prediction.ipynb` | Detecting fraudulent job postings | NLP, Deep Learning (Bi-LSTM) |

---

## Project 1: Max Temperature Prediction

**Notebook:** `Max-Temp-Prediction.ipynb`

Predicts the daily maximum temperature (TMAX) at LaGuardia Airport using historical NOAA weather data spanning 1941–2026.

### Dataset
- **Source:** [NOAA Climate Data Online](https://www.ncei.noaa.gov/cdo-web/search) (Station: USW00014732)
- **Size:** 31,077 daily records
- **Features:** `TMIN`, `PRCP`, `AWND`, `PRES`, `month`, `TMAX_lag1`
- **Target:** `TMAX` (daily maximum temperature)

### Workflow

1. **Preprocessing** — Intentionally added 3% missing values and 2% outlier noise to simulate real-world data quality issues, then applied seasonal median imputation to recover them.
2. **Feature Engineering** — Extracted cyclical month encodings (`sin`/`cos`) and a lag-1 TMAX feature to capture day-to-day temperature continuity.
3. **Exploratory Data Analysis** — Correlation heatmaps, seasonality box plots, scatter plots (TMIN vs TMAX, precipitation vs TMAX), histograms, and a full time series plot.
4. **Modelling** — Temporal train/test split at January 2021 to prevent data leakage.

### Models

| Model | Type |
|---|---|
| Linear Regression | Supervised Regression |
| Decision Tree Regressor | Supervised Regression |
| Random Forest Regressor | Supervised Regression |
| Decision Tree Classifier | Supervised Classification (hot/cold day) |
| KMeans | Unsupervised Clustering (4 weather regimes) |

### Key Findings
- **Random Forest** achieved the best regression performance (highest R², lowest RMSE and MAE).
- The **TMAX lag-1 feature** provided a meaningful improvement in R² and RMSE compared to training without it.
- Classification threshold: days above 25°C are labelled as *hot*, below as *cold*.
- KMeans segmented weather into 4 distinct regimes, evaluated with silhouette score.

### Outputs
- `laguardia_noisy_processed.csv` — Preprocessed dataset
- `model_results.csv` — Summary table of all model scores
- `eda_graphs.png` — EDA visualisations
- `feature_importance.png` — Random Forest feature importance chart
- `regression_comparison.png` — Actual vs predicted and residual plots for all three regressors

---

## Project 2: Fake Job Posting Detection

**Notebook:** `Fake-Job-Posting-Prediction.ipynb`

Classifies job postings as real or fraudulent using Natural Language Processing and a Bidirectional LSTM deep learning model.

### Dataset
- **Source:** [Kaggle — Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size:** 17,880 records
- **Features Used:** `title`, `company_profile`, `description`, `requirements`, `benefits`
- **Target:** `fraudulent` (0 = real, 1 = fake)

### Workflow

1. **Text Preprocessing** — Combined five text columns into one, then cleaned the result: lowercased, removed HTML tags, URLs, and non-alphabetic characters.
2. **Stopword Removal** — Used NLTK to remove common English stopwords, reducing noise.
3. **Tokenisation & Padding** — Tokenised to the 20,000 most frequent words, padded/truncated all sequences to 300 tokens.
4. **GloVe Embeddings** — Loaded pre-trained 100-dimensional GloVe vectors to initialise the embedding layer, giving the model prior knowledge of word semantics.
5. **Class Imbalance Handling** — Applied class weights (real: 1.0, fake: 7.0) to penalise missed fraud detections more heavily.
6. **Train / Validation / Test Split** — 70% / 15% / 15% with stratification to preserve the class ratio across all splits.

### Model Architecture

A `Sequential` Bidirectional LSTM:

```
Embedding (GloVe, 100-dim, trainable)
→ Bidirectional LSTM (128 units, returns sequences)
→ Dropout
→ LSTM (64 units)
→ Dropout
→ Dense (32, ReLU)
→ Dense (1, Sigmoid)
```

Compiled with `Adam` and `binary_crossentropy`. Early stopping monitored `val_loss` with patience of 3 epochs.

### Experiments

Four configurations were tested, varying batch size, learning rate, and dropout rates. Results were compared by **F1-score on the fake class**.

| Experiment | Batch Size | Learning Rate | Dropout |
|---|---|---|---|
| Base Model | 64 | 0.001 | 0.4 / 0.3 |
| Larger Batch Size | 128 | 0.001 | 0.4 / 0.3 |
| Lower Learning Rate | 64 | 0.0005 | 0.4 / 0.3 |
| Higher Dropout *(best)* | 64 | 0.001 | 0.5 / 0.4 |

### Outputs
- Learning curve plots (loss & accuracy over epochs) for base and best models
- Confusion matrices for base and best models
- Classification reports with precision, recall, and F1-score
- Example predictions table showing confidence scores

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk
```

Both notebooks were developed in **Google Colab** and reference datasets stored in Google Drive. Update the file paths in the data-loading cells if running locally.

---
