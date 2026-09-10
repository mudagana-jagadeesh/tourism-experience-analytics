# Tourism Experience Analytics — Full Project

## Contents
- **`Tourism_Experience_Analytics.ipynb`** — full analysis notebook (already executed, all outputs/charts embedded). Data loading, cleaning, EDA (18 charts), hypothesis testing, feature engineering, and three ML objectives: regression, classification, recommendation.
- **`app.py`** — Streamlit deployment app (predicts visit mode + recommends attractions).
- **`data/`** — the 9 source Excel tables.
- **`artifacts/`** — trained models/lookups saved by the notebook.
- **`requirements.txt`** — Python dependencies.

## Production models used (per project brief)
- **Regression (predict Rating): Random Forest Regressor**
- **Classification (predict Visit Mode): Random Forest Classifier**
- **Recommendation: Hybrid** — Truncated SVD (collaborative filtering) + cosine similarity (content-based filtering)

The notebook also trains Linear Regression / Gradient Boosting (regression) and Logistic Regression / XGBoost (classification) purely as comparison baselines in the model-comparison tables, but **Random Forest is the model saved to `artifacts/` and served by `app.py`** for both regression and classification, as specified in the project brief. An XGBoost model is also saved (`xgb_classifier_reference.pkl`) for reference only and is not used by the app.

## How to run

### Notebook
```bash
pip install -r requirements.txt
jupyter notebook Tourism_Experience_Analytics.ipynb
```

### Streamlit app
```bash
pip install -r requirements.txt
streamlit run app.py
```
Make sure `artifacts/` sits next to `app.py`.

## Honest performance notes
- Regression R² is modest (~0.13) — EDA/hypothesis tests show demographic/categorical fields only weakly explain rating.
- Classification (Random Forest) does well on majority classes (Couples/Family), weaker on the rare Business class (1.2% of data).
- The recommendation system is the most production-ready component.

## Before submitting
Add your name / team members and GitHub link in the notebook's title cells.
