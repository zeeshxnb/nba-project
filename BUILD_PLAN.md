# 🏀 NBA Playoff Bracket Predictor - Build Plan

```
╔══════════════════════════════════════════════════════════════════╗
║           PLAYOFF SERIES WINNER PREDICTION MODEL                 ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📊 Data Overview

| File | Purpose |
|------|---------|
| `team_stats_advanced_rs.csv` | NET_RATING, OFF/DEF_RATING, TS%, PACE |
| `team_stats_traditional_rs.csv` | W, L, W_PCT (for seeding) |
| `team_stats_four_factors_rs.csv` | FTA_RATE, opponent stats |
| `team_stats_advanced_po.csv` | Historical playoff performance |

**Training Data:** ~420 playoff series (1997-2024)  
**Expected Accuracy:** 72-78% per series

---

## 🔨 Build Steps

### PHASE 1: Data Preparation
```
┌───────────────────────────────────────────────────┐
│  STEP 1.1  │  Load & merge CSV files              │
├───────────────────────────────────────────────────┤
│  STEP 1.2  │  Create historical series dataset    │
│            │  (who played who, who won)           │
└───────────────────────────────────────────────────┘
```
- [x] 1.1 Complete
- [x] 1.2 Complete

---

### PHASE 2: Feature Engineering
```
┌───────────────────────────────────────────────────┐
│  STEP 2.1  │  Calculate stat differentials        │
│            │  (Team A - Team B for each stat)     │
├───────────────────────────────────────────────────┤
│  STEP 2.2  │  Add seeding & home court            │
└───────────────────────────────────────────────────┘
```
- [x] 2.1 Complete
- [x] 2.2 Complete

---

### PHASE 3: Model Training
```
┌───────────────────────────────────────────────────┐
│  STEP 3.1  │  Logistic Regression (baseline)      │
├───────────────────────────────────────────────────┤
│  STEP 3.2  │  XGBoost + comparison                │
└───────────────────────────────────────────────────┘
```
- [ ] 3.1 Complete
- [ ] 3.2 Complete

---

### PHASE 4: Bracket Predictor
```
┌───────────────────────────────────────────────────┐
│  STEP 4.1  │  predict_series(team_a, team_b)      │
├───────────────────────────────────────────────────┤
│  STEP 4.2  │  predict_bracket(playoff_teams)      │
└───────────────────────────────────────────────────┘
```
- [ ] 4.1 Complete
- [ ] 4.2 Complete

---

### PHASE 5: Evaluation
```
┌───────────────────────────────────────────────────┐
│  STEP 5.1  │  Cross-validation accuracy           │
├───────────────────────────────────────────────────┤
│  STEP 5.2  │  Feature importance analysis         │
└───────────────────────────────────────────────────┘
```
- [ ] 5.1 Complete
- [ ] 5.2 Complete

---

## 📁 Project Structure

```
nba-project/
│
├── data/
│   └── team/
│       ├── team_stats_advanced_rs.csv
│       ├── team_stats_advanced_po.csv
│       ├── team_stats_four_factors_rs.csv
│       └── team_stats_traditional_rs.csv
│
├── src/
│   ├── data_loader.py      ← Phase 1
│   ├── features.py         ← Phase 2
│   ├── model.py            ← Phase 3
│   └── bracket.py          ← Phase 4
│
├── main.py                 ← Ties everything together
├── requirements.txt
└── BUILD_PLAN.md           ← You are here
```

---

## 🎯 Key Features to Engineer

```python
# Differentials (Team A - Team B)
net_rating_diff      # Most predictive
off_rating_diff      
def_rating_diff      
win_pct_diff         
ts_pct_diff          
efg_pct_diff         
tov_pct_diff         
oreb_pct_diff        
fta_rate_diff        

# Categorical
seed_diff            # Higher seed advantage
has_home_court       # 1 if Team A is higher seed
```

---

## 📈 Expected Results

| Metric | Target |
|--------|--------|
| Single series accuracy | 72-78% |
| First round accuracy | 78-85% |
| Full bracket (15 series) | 25-40% |
| Champion prediction | 25-35% |

---

## 🚀 Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run predictions
python main.py

# Run with specific season
python main.py --season 2024
```
