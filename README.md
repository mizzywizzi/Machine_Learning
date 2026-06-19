# Heart Disease Prediction — ML Model Comparison

> **PlutoAcademy AI & ML Internship — Project 02**  
> Binary Classification · Scikit-learn · Python

---

## Dataset

**Heart Disease Dataset** sourced from Kaggle:  
🔗 https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset contains **1,025 rows** and **14 features** describing patient clinical measurements. The target variable (`target`) is binary: `1` = heart disease present, `0` = no heart disease.

---

## Project Structure

```
Machine_learning/
├── heart.csv                  # Raw dataset (from Kaggle)
├── Heart_Disease_ML.ipynb     # Main notebook — fully executed
├── generate_notebook.py       # Script that programmatically builds the notebook
├── charts/                    # All generated charts (PNG, 150 dpi)
│   ├── chart1_correlation_heatmap.png
│   ├── chart2_feature_importance.png
│   ├── chart3_model_comparison.png
│   ├── chart4_roc_curves.png
│   ├── chart5_confusion_matrix.png
│   └── chart6_lr_coefficients.png
└── README.md
```

---

## What's in the Notebook

| Step | Description |
|------|-------------|
| **Step 1** | Load, explore, check for missing values, drop duplicates, one-hot encode categoricals, scale numerics, 80/20 stratified train/test split |
| **Step 2** | Correlation heatmap, exploratory Random Forest importance, feature selection rationale |
| **Step 3** | Train Logistic Regression, Random Forest (200 trees), and KNN (auto-select k=3–19) |
| **Step 4** | Compare all 3 models on Accuracy, Precision, Recall, F1, ROC-AUC; grouped bar chart + ROC curves |
| **Step 5** | Best model analysis — confusion matrix + coefficient plot + 5-line written conclusion |

---

## Model Comparison Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **✅ Logistic Regression** | **0.8525** | **0.8750** | **0.8485** | **0.8615** | **0.8983** |
| KNN (k=3) | 0.8361 | 0.8710 | 0.8182 | 0.8438 | 0.8506 |
| Random Forest | 0.8033 | 0.8182 | 0.8182 | 0.8182 | 0.8874 |

**Winner: Logistic Regression** — best on 4 out of 5 metrics.

---

## Key Findings

- `thalach` (max heart rate), `oldpeak` (ST depression), `cp` (chest pain type), and `ca` (vessel count) are the most predictive features — consistent with clinical literature.
- Logistic Regression outperformed ensemble and instance-based methods on this ~900-row dataset, suggesting a largely linear decision boundary.
- Random Forest showed the highest ROC-AUC potential but underfit the small training set without deeper hyperparameter tuning.

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn jupyter

# Re-generate the notebook from source
python generate_notebook.py

# Execute it
jupyter nbconvert --to notebook --execute Heart_Disease_ML.ipynb --output Heart_Disease_ML.ipynb

# Or open interactively
jupyter notebook Heart_Disease_ML.ipynb
```

---

## Tools Used

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, preprocessing, evaluation |
| `matplotlib` | Custom dark-theme visualisations |
| `seaborn` | Heatmaps and styled plots |
| `jupyter` | Interactive notebook environment |

---

*PlutoAcademy AI & ML Internship Program — © Pluto Academy*
