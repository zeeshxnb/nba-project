"""Model comparison with train/test split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..models.logistic import LogisticModel
from ..models.xgboost_model import XGBoostModel
from ..models.random_forest import RandomForestModel
from ..models.svm import SVMModel
from .split import get_train_test_split, get_data_info, DEFAULT_TEST_SEASONS


def get_all_models():
    """Return dict of all models."""
    return {
        'Logistic Regression': LogisticModel(),
        'XGBoost': XGBoostModel(),
        'Random Forest': RandomForestModel(),
        'SVM': SVMModel()
    }


def evaluate_all_models(test_seasons=None, cv=5):
    """Evaluate all models using cross-validation and held-out test set."""
    if test_seasons is None:
        test_seasons = DEFAULT_TEST_SEASONS
    
    X_train, X_test, y_train, y_test, feature_cols, df, test_df = get_train_test_split(test_seasons)
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"\nMethodology:")
    print(f"  1. Train on: {len(y_train)} series (excluding test seasons)")
    print(f"  2. Cross-validation: {cv}-fold CV on training data")
    print(f"  3. Test on: {len(test_seasons)} seasons ({len(y_test)} series) - NEVER seen during training")
    print()
    
    models = get_all_models()
    results = []
    
    for name, model in models.items():
        cv_result = model.evaluate(X_train, y_train, cv=cv)
        cv_acc = cv_result['mean_accuracy']
        cv_std = cv_result['std']
        
        model.train(X_train, y_train, feature_names=feature_cols)
        
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        
        correct = (y_pred == y_test).sum()
        total = len(y_test)
        
        results.append({
            'Model': name,
            'CV Accuracy': cv_acc,
            'CV Std': cv_std,
            'Test Accuracy': test_acc,
            'Test Correct': f"{correct}/{total}",
            'Precision': test_precision,
            'Recall': test_recall,
            'F1': test_f1
        })
    
    results_df = pd.DataFrame(results)
    
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"\n{'Model':<22} {'CV Acc':<12} {'Test Acc':<12} {'Correct':<10}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<22} {row['CV Accuracy']:.1%} +/-{row['CV Std']:.1%}   "
              f"{row['Test Accuracy']:.1%}         {row['Test Correct']}")
    
    print("-" * 70)
    
    best_cv = results_df.loc[results_df['CV Accuracy'].idxmax()]
    best_test = results_df.loc[results_df['Test Accuracy'].idxmax()]
    
    print(f"\nBest CV Accuracy:   {best_cv['Model']} ({best_cv['CV Accuracy']:.1%})")
    print(f"Best Test Accuracy: {best_test['Model']} ({best_test['Test Accuracy']:.1%})")
    
    return results_df, models


def show_test_predictions(test_seasons=None):
    """Show individual predictions for test set by season."""
    if test_seasons is None:
        test_seasons = DEFAULT_TEST_SEASONS
    
    if isinstance(test_seasons, str):
        test_seasons = [test_seasons]
    
    X_train, X_test, y_train, y_test, feature_cols, df, test_df = get_train_test_split(test_seasons)
    
    model = LogisticModel()
    model.train(X_train, y_train, feature_names=feature_cols)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print("\n" + "=" * 70)
    print(f"INDIVIDUAL PREDICTIONS - {len(test_seasons)} TEST SEASONS")
    print("=" * 70)
    
    total_correct = 0
    total_games = 0
    
    for season in test_seasons:
        season_mask = test_df['season'] == season
        season_df = test_df[season_mask].reset_index(drop=True)
        season_indices = np.where(test_df['season'].values == season)[0]
        
        if len(season_df) == 0:
            continue
        
        print(f"\n--- {season} PLAYOFFS ---")
        print(f"{'Matchup':<20} {'Predicted':<12} {'Actual':<12} {'Prob':<8} {'Result'}")
        print("-" * 60)
        
        season_correct = 0
        for idx, (_, row) in enumerate(season_df.iterrows()):
            pred_idx = season_indices[idx]
            pred = y_pred[pred_idx]
            actual = y_test[pred_idx]
            prob = y_prob[pred_idx]
            
            matchup = f"{row['team_a']} vs {row['team_b']}"
            pred_winner = row['team_a'] if pred == 1 else row['team_b']
            actual_winner = row['team_a'] if actual == 1 else row['team_b']
            conf = prob if pred == 1 else 1 - prob
            correct = "Y" if pred == actual else "N"
            
            if pred == actual:
                season_correct += 1
            
            print(f"{matchup:<20} {pred_winner:<12} {actual_winner:<12} {conf:.0%}      {correct}")
        
        print(f"Season Total: {season_correct}/{len(season_df)} correct ({season_correct/len(season_df):.0%})")
        total_correct += season_correct
        total_games += len(season_df)
    
    print("\n" + "-" * 70)
    print(f"OVERALL: {total_correct}/{total_games} correct ({total_correct/total_games:.0%})")
    print("-" * 70)


def run_full_evaluation(test_seasons=None):
    """Run complete evaluation pipeline."""
    if test_seasons is None:
        test_seasons = DEFAULT_TEST_SEASONS
    
    get_data_info(test_seasons)
    print()
    results_df, models = evaluate_all_models(test_seasons)
    show_test_predictions(test_seasons)
    return results_df


if __name__ == "__main__":
    run_full_evaluation()
