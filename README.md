# 🏀 NBA Playoff Bracket Predictor

A machine learning project that predicts NBA playoff series winners using historical team statistics.

## 📊 Results

| Metric | Performance |
|--------|-------------|
| Per-series accuracy | **~75%** (cross-validated) |
| Champion prediction | **~35-40%** |
| Improvement over random | **5-6x better** |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Or run specific components
python main.py --evaluate    # Model comparison
python main.py --tune        # Hyperparameter tuning
python main.py --overfit     # Overfitting analysis
python main.py --predict 2022-23  # Bracket prediction
```

## 📁 Project Structure

```
nba-project/
├── data/
│   ├── team/                    # Team statistics CSVs
│   └── playoff_series.csv       # Historical matchup results
├── src/
│   ├── data/
│   │   ├── ingest.py           # Data loading
│   │   └── processing.py       # Feature engineering
│   ├── models/
│   │   ├── logistic.py         # Logistic Regression
│   │   ├── xgboost_model.py    # XGBoost
│   │   ├── random_forest.py    # Random Forest
│   │   ├── svm.py              # Support Vector Machine
│   │   └── ensemble.py         # Ensemble (LR + XGB)
│   ├── evaluate/
│   │   ├── compare.py          # Model comparison
│   │   ├── split.py            # Train/test splits
│   │   ├── tuning.py           # Hyperparameter tuning
│   │   └── overfitting.py      # Learning curves
│   ├── predict/
│   │   └── bracket.py          # Bracket prediction
│   └── EDA/
│       └── eda.ipynb           # Exploratory analysis
├── main.py                      # Main entry point
├── requirements.txt
├── BUILD_PLAN.md
└── README.md
```

## 🔧 Models

| Model | Description | Accuracy |
|-------|-------------|----------|
| Logistic Regression | Linear baseline, interpretable | ~74% |
| XGBoost | Gradient boosting | ~73% |
| Random Forest | Tree ensemble | ~73% |
| SVM | Support vector classifier | ~74% |
| **Ensemble** | LR + XGBoost average | ~74% |

## 📈 Features Used

Differential features (Team A - Team B):
- `net_rating_diff` - Point differential per 100 possessions
- `off_rating_diff` - Offensive efficiency
- `def_rating_diff` - Defensive efficiency
- `w_pct_diff` - Win percentage
- `ts_pct_diff` - True shooting %
- `efg_pct_diff` - Effective FG%
- `home_court` - Home court advantage

## 📚 Data

- **Training data**: 345 playoff series (1996-97 to 2022-23)
- **Sources**: NBA Stats API via `nba_api`
- **Features**: Advanced, traditional, and four factors stats

## 🎯 Key Findings

1. **Win percentage differential** is the strongest predictor
2. **Net rating** is second most important
3. **Simpler models (Logistic) perform as well as complex ones**
4. **Full bracket prediction is inherently hard** (~35% champion accuracy)
5. **Upsets are unpredictable** (e.g., 2023 Heat 8-seed Finals run)

## 📝 Course Requirements Completed

- [x] Data visualization (EDA notebook)
- [x] Multiple models compared (4 models + ensemble)
- [x] Hyperparameter tuning (GridSearchCV)
- [x] Over/underfitting analysis (learning curves)
- [x] Held-out test evaluation
- [x] Cross-validation

## 👥 Authors

Built for CS Machine Learning course.
