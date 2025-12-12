"""Advanced analysis: feature selection and Monte Carlo simulation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ..data.processing import make_dataset, get_feature_columns


def run_feature_selection():
    """Perform feature selection using multiple methods."""
    df = make_dataset()
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df['team_a_won'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("=" * 70)
    print("FEATURE SELECTION ANALYSIS")
    print("=" * 70)
    print(f"\nOriginal features: {len(feature_cols)}")
    
    # Method 1: RFE
    print("\n--- Method 1: Recursive Feature Elimination (RFE) ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(lr, n_features_to_select=5, step=1)
    rfe.fit(X_scaled, y)
    
    rfe_selected = [f for f, s in zip(feature_cols, rfe.support_) if s]
    print(f"Top 5 features (RFE): {rfe_selected}")
    
    # Method 2: SelectKBest
    print("\n--- Method 2: SelectKBest (ANOVA F-test) ---")
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    
    kbest_scores = list(zip(feature_cols, selector.scores_))
    kbest_sorted = sorted(kbest_scores, key=lambda x: -x[1])[:5]
    print("Top 5 features (F-test):")
    for feat, score in kbest_sorted:
        print(f"  {feat:<22} F-score: {score:.2f}")
    
    # Method 3: Random Forest
    print("\n--- Method 3: Random Forest Feature Importance ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    rf_importance = list(zip(feature_cols, rf.feature_importances_))
    rf_sorted = sorted(rf_importance, key=lambda x: -x[1])[:5]
    print("Top 5 features (RF importance):")
    for feat, imp in rf_sorted:
        print(f"  {feat:<22} Importance: {imp:.3f}")
    
    # Method 4: XGBoost
    print("\n--- Method 4: XGBoost Feature Importance ---")
    xgb = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=42)
    xgb.fit(X, y)
    
    xgb_importance = list(zip(feature_cols, xgb.feature_importances_))
    xgb_sorted = sorted(xgb_importance, key=lambda x: -x[1])[:5]
    print("Top 5 features (XGB importance):")
    for feat, imp in xgb_sorted:
        print(f"  {feat:<22} Importance: {imp:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("CONSENSUS TOP FEATURES")
    print("=" * 70)
    
    all_top = []
    all_top.extend(rfe_selected)
    all_top.extend([f for f, _ in kbest_sorted])
    all_top.extend([f for f, _ in rf_sorted])
    all_top.extend([f for f, _ in xgb_sorted])
    
    from collections import Counter
    feature_counts = Counter(all_top)
    
    print("\nFeatures ranked by selection frequency:")
    for feat, count in feature_counts.most_common():
        bar = "#" * count
        print(f"  {feat:<22} {bar} ({count}/4 methods)")
    
    return feature_counts


def run_monte_carlo_bracket(n_simulations=1000):
    """Monte Carlo simulation of playoff bracket."""
    from ..predict.bracket import PlayoffBracket
    from ..models.ensemble import PlayoffPredictor
    
    print("\n" + "=" * 70)
    print(f"MONTE CARLO BRACKET SIMULATION ({n_simulations:,} runs)")
    print("=" * 70)
    
    west = ['DEN', 'MEM', 'SAC', 'PHX', 'LAC', 'GSW', 'LAL', 'MIN']
    east = ['MIL', 'BOS', 'PHI', 'CLE', 'NYK', 'BKN', 'MIA', 'ATL']
    season = '2022-23'
    
    df = make_dataset()
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df['team_a_won'].values
    
    model = PlayoffPredictor()
    model.train(X, y)
    
    bracket = PlayoffBracket()
    bracket.train()
    
    print(f"\nSimulating {n_simulations:,} playoff brackets...")
    
    champion_counts = {}
    finals_counts = {}
    
    def simulate_series(team_a, team_b, prob_a):
        return team_a if np.random.random() < prob_a else team_b
    
    def simulate_bracket(west_seeds, east_seeds):
        matchup_order = [(0, 7), (3, 4), (2, 5), (1, 6)]
        
        west_r2 = []
        east_r2 = []
        
        for seeds, winners in [(west_seeds, west_r2), (east_seeds, east_r2)]:
            for high, low in matchup_order:
                result = bracket.predict_series(seeds[high], seeds[low], season)
                if 'error' in result:
                    winners.append(seeds[high])
                else:
                    prob_a = result['team_a_prob']
                    winner = simulate_series(seeds[high], seeds[low], prob_a)
                    winners.append(winner)
        
        west_cf = []
        east_cf = []
        
        for r2, cf in [(west_r2, west_cf), (east_r2, east_cf)]:
            for a, b in [(0, 1), (2, 3)]:
                result = bracket.predict_series(r2[a], r2[b], season)
                if 'error' in result:
                    cf.append(r2[a])
                else:
                    prob_a = result['team_a_prob']
                    winner = simulate_series(r2[a], r2[b], prob_a)
                    cf.append(winner)
        
        west_result = bracket.predict_series(west_cf[0], west_cf[1], season)
        if 'error' in west_result:
            west_champ = west_cf[0]
        else:
            west_champ = simulate_series(west_cf[0], west_cf[1], west_result['team_a_prob'])
        
        east_result = bracket.predict_series(east_cf[0], east_cf[1], season)
        if 'error' in east_result:
            east_champ = east_cf[0]
        else:
            east_champ = simulate_series(east_cf[0], east_cf[1], east_result['team_a_prob'])
        
        finals_result = bracket.predict_series(west_champ, east_champ, season)
        if 'error' in finals_result:
            champion = west_champ
        else:
            champion = simulate_series(west_champ, east_champ, finals_result['team_a_prob'])
        
        return champion, (west_champ, east_champ)
    
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    
    for i in range(n_simulations):
        sys.stdout = StringIO()
        champ, finals = simulate_bracket(west, east)
        sys.stdout = old_stdout
        
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
        finals_key = tuple(sorted(finals))
        finals_counts[finals_key] = finals_counts.get(finals_key, 0) + 1
        
        if (i + 1) % 200 == 0:
            print(f"  Completed {i+1:,} simulations...")
    
    print("\n" + "-" * 70)
    print("CHAMPIONSHIP PROBABILITIES (Monte Carlo)")
    print("-" * 70)
    
    sorted_champs = sorted(champion_counts.items(), key=lambda x: -x[1])
    for team, count in sorted_champs[:8]:
        prob = count / n_simulations * 100
        bar = "#" * int(prob / 2)
        print(f"  {team:<6} {prob:5.1f}% {bar}")
    
    print("\n" + "-" * 70)
    print("MOST LIKELY FINALS MATCHUPS")
    print("-" * 70)
    
    sorted_finals = sorted(finals_counts.items(), key=lambda x: -x[1])[:5]
    for teams, count in sorted_finals:
        prob = count / n_simulations * 100
        print(f"  {teams[0]} vs {teams[1]}: {prob:.1f}%")
    
    print("\n" + "-" * 70)
    print("COMPARISON TO ACTUAL RESULT")
    print("-" * 70)
    actual_champ = 'DEN'
    actual_finals = ('DEN', 'MIA')
    
    champ_prob = champion_counts.get(actual_champ, 0) / n_simulations * 100
    finals_key = tuple(sorted(actual_finals))
    finals_prob = finals_counts.get(finals_key, 0) / n_simulations * 100
    
    print(f"  Actual Champion: {actual_champ}")
    print(f"  Model probability for DEN: {champ_prob:.1f}%")
    print(f"  Actual Finals: DEN vs MIA")
    print(f"  Model probability for this matchup: {finals_prob:.1f}%")
    
    return champion_counts, finals_counts


def run_all_advanced():
    """Run all advanced analyses."""
    print("\n" + "=" * 70)
    print("ADVANCED ANALYSIS")
    print("=" * 70)
    
    feature_counts = run_feature_selection()
    champ_counts, finals_counts = run_monte_carlo_bracket(n_simulations=500)
    
    print("\n" + "=" * 70)
    print("ADVANCED ANALYSIS COMPLETE")
    print("=" * 70)
    print("""
Summary:
1. Feature Selection - Compared 4 methods, identified consensus features
2. Monte Carlo Simulation - 500 bracket simulations for probability estimates
""")


if __name__ == "__main__":
    run_all_advanced()
