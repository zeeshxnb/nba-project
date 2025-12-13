# NBA Playoff Bracket Predictor

**CompSci 178 Final Project**  
**By Atif Usmani and Zeeshan Babul**

A machine learning system that predicts NBA playoff series winners using regular season team statistics.

---

## Results

### 3-Year Test Performance (2020-21, 2021-22, 2022-23)

| Model | CV Accuracy | Test Accuracy | Correct |
|-------|-------------|---------------|---------|
| **Logistic Regression** | 74.0% | **73.2%** | 30/41 |
| Random Forest | 75.0% | 68.3% | 28/41 |
| XGBoost | 72.4% | 65.9% | 27/41 |
| SVM | 75.3% | 65.9% | 27/41 |

### Season Breakdown

| Season | Accuracy | Correct |
|--------|----------|---------|
| 2020-21 | 75% | 9/12 |
| 2021-22 | **87%** | 13/15 |
| 2022-23 | 57% | 8/14 |
| **Overall** | **73%** | **30/41** |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Individual commands
python main.py --evaluate      # Model comparison (3-year test)
python main.py --tune          # Hyperparameter tuning
python main.py --overfit       # Overfitting analysis
python main.py --monte-carlo   # Monte Carlo bracket simulation
python main.py --predict 2022-23  # Predict specific season
```

---

## Project Structure

```
nba-project/
├── data/
│   ├── team/                    # Team statistics CSVs
│   └── playoff_series.csv       # Historical matchup results
├── src/
│   ├── data/
│   │   ├── ingest.py           # Data loading
│   │   └── processing.py       # Feature engineering (differentials)
│   ├── models/
│   │   ├── logistic.py         # Logistic Regression
│   │   ├── xgboost_model.py    # XGBoost
│   │   ├── random_forest.py    # Random Forest
│   │   ├── svm.py              # Support Vector Machine
│   │   └── ensemble.py         # Ensemble (LR + XGB)
│   ├── evaluate/
│   │   ├── compare.py          # Model comparison
│   │   ├── split.py            # Train/test splits (3-year holdout)
│   │   ├── tuning.py           # GridSearchCV tuning
│   │   ├── overfitting.py      # Learning curves
│   │   └── advanced.py         # Feature selection + Monte Carlo
│   ├── predict/
│   │   └── bracket.py          # Bracket simulation
│   └── EDA/
│       └── eda.ipynb           # Exploratory data analysis
├── main.py                      # Main entry point
├── requirements.txt
└── README.md
```

---

## Data

| Dataset | Description |
|---------|-------------|
| Training | 304 playoff series (1996-97 to 2019-20) |
| Test | 41 playoff series (2020-21, 2021-22, 2022-23) |
| Source | NBA Stats API |

### Features (Differentials: Team A - Team B)

| Feature | Description |
|---------|-------------|
| `w_pct_diff` | Win percentage difference |
| `net_rating_diff` | Point differential per 100 possessions |
| `off_rating_diff` | Offensive rating difference |
| `def_rating_diff` | Defensive rating difference |
| `ts_pct_diff` | True shooting % difference |
| `efg_pct_diff` | Effective FG% difference |
| `home_court` | Home court advantage (1/0) |

---

## Models

| Model | Description | Best For |
|-------|-------------|----------|
| Logistic Regression | Linear classifier with L2 regularization | Best generalization |
| XGBoost | Gradient boosted trees | Capturing non-linear patterns |
| Random Forest | Bagged decision trees | Feature importance |
| SVM | Support vector classifier (RBF kernel) | Margin-based classification |
| Ensemble | Average of LR + XGBoost | Combining strengths |

---

## Key Findings

1. **Win percentage is the strongest predictor** - Teams with better regular season records usually win playoff series

2. **Simpler models generalize better** - Logistic Regression (73%) outperformed XGBoost (66%) on the test set

3. **Upsets are inherently unpredictable** - 2022-23 had the most upsets (Miami 8-seed Finals run), causing lower accuracy

4. **~75% is the practical ceiling** - All models converge around 74-76% CV accuracy due to inherent randomness in sports

---

## Something Extra

### 1. Monte Carlo Bracket Simulation
Instead of deterministic predictions, we run 500+ simulated brackets by sampling from win probabilities. This provides championship probability estimates rather than single picks.

```bash
python main.py --monte-carlo
```

### 2. Feature Selection Analysis
Compared 4 feature selection methods (RFE, SelectKBest, Random Forest importance, XGBoost importance) to identify consensus top features.

---

## Course Requirements

- [x] Data exploration and visualization (EDA notebook)
- [x] Multiple models compared (4 models + ensemble)
- [x] Hyperparameter tuning (GridSearchCV)
- [x] Overfitting analysis (learning curves, train/val gap)
- [x] Cross-validation (5-fold)
- [x] Held-out test evaluation (3-year holdout)
- [x] Something extra (Monte Carlo simulation, feature selection)

---

## Authors

**Atif Usmani** and **Zeeshan Babul**  
University of California, Irvine  
CS 178 - Machine Learning
