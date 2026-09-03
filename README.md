# Predicting House Prices Using Real-Estate Features

Machine learning project for **DSK807: Applied Machine Learning** (University of Southern Denmark, Kolding) that benchmarks multiple regression approaches — from linear models to ensembles and neural networks — on the Ames Housing dataset.

## Overview

The goal is to predict residential sale prices from structural, locational, and quality-related property features, and to compare how different modelling paradigms (linear, tree-based, ensemble, deep learning) perform on structured tabular data.

## Dataset

- **Source:** Ames Housing dataset — 2,930 observations, 82 raw features
- **Target:** `SalePrice` (log-transformed to correct right-skew and reduce the influence of outliers)
- **Feature types:** Numerical (living area, basement size, year built, etc.) and categorical (neighbourhood, quality ratings, etc.)

## Methodology

1. **EDA** — examined price relationships with square footage, room count, and location; found strong location effects and increasing price variance for larger homes.
2. **Preprocessing**
   - Log-transformed `SalePrice`
   - Feature-specific missing value handling (e.g. median imputation for `Lot Frontage`, zero-fill for `Bsmt Full Bath`, `Mas Vnr Area`, `Garage Yr Blt`)
   - One-hot encoding of categoricals → 260 features
3. **Feature selection** — kept features exceeding a correlation threshold of 0.3 **or** a Random Forest importance threshold of 0.003 → 51 final features
4. **Modelling** — trained and tuned:
   - Linear Regression
   - Random Forest
   - Gradient Boosting
   - Stacking Ensemble (RF + Gradient Boosting base learners, Linear Regression meta-learner)
   - Feed-Forward Neural Network (tuned optimizer, learning rate, activation, dropout/L2, layer width)
   - Autoencoder → latent features fed into Random Forest
5. **Evaluation** — RMSE, MAE, R² on a held-out test set, plus MAE broken out by price segment (low/mid-low/mid-high/high)
6. **Interpretability** — SHAP analysis on Random Forest and Gradient Boosting

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.111 | 0.079 | 0.925 |
| Random Forest | 0.124 | 0.084 | 0.907 |
| **Gradient Boosting** | **0.111** | **0.077** | **0.926** |
| Stacking Ensemble | 0.111 | 0.077 | 0.926 |
| FFNN | 0.123 | 0.090 | 0.908 |
| RF + Autoencoder | 0.147 | 0.104 | 0.869 |

**Gradient Boosting** was selected as the final model — it matches the Stacking Ensemble's performance with lower complexity and easier interpretability.

## Key Findings

- Overall quality, living area, basement size, garage capacity, and construction/renovation year are the strongest price drivers (confirmed by correlation, RF importance, and SHAP).
- Neighbourhood shows clear price differences in EDA but has limited *direct* influence post-selection — its effect is largely captured indirectly through correlated structural/quality features.
- Tree-based ensembles outperform neural networks and autoencoders, consistent with known limitations of deep learning on structured tabular data.
- All models show higher error on low-priced homes; performance is best in the mid-price range.

## Future Work

- Use GPS coordinates for geographic refinement instead of neighbourhood labels
- Incorporate external data (e.g. interest rates, economic indicators)
- Build specialized models for the low-price segment to reduce error there

## Repository Contents

- `Ames_Housing_Data_Code.ipynb` — full analysis: EDA, preprocessing, feature selection, model training/tuning, evaluation, and SHAP interpretability
- `Report_housing_data_report.pdf` — written project report

## Requirements

```
pandas
numpy
matplotlib
seaborn
shap
scikit-learn
tensorflow
```

## Authors

- Liva Igaune
- Erika Laizane
- Rabinson Pariyar

*Course: DSK807 — Applied Machine Learning, Department of Mathematics and Computer Science, University of Southern Denmark (SDU), Kolding. Lecturer: Tariq Yousef.*
