"""Compare all 4 models with proper train/test split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..models.logistic import LogisticModel
from ..models.xgboost_model import XGBoostModel
from ..models.random_forest import RandomForestModel
from ..models.svm import SVMModel
from .split import get_train_test_split, get_data_info


def get_all_models():
    """Return dict of all models."""
    return {
        'Logistic Regression': LogisticModel(),
        'XGBoost': XGBoostModel(),
        'Random Forest': RandomForestModel(),
        'SVM': SVMModel()
    }


def evaluate_all_models(test_season='2022-23', cv=5):
    """
    Evaluate all models using:
    1. Cross-validation on training data (model selection)
    2. Final evaluation on held-out test set (true performance)
    
    Args:
        test_season: Season to hold out for final testing
        cv: Number of cross-validation folds
    
    Returns:
        DataFrame with results for all models
    """
    # Get data split
    X_train, X_test, y_train, y_test, feature_cols, df = get_train_test_split(test_season)
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"\nMethodology:")
    print(f"  1. Train on: 1996-97 to 2022-23 ({len(y_train)} series)")
    print(f"  2. Cross-validation: {cv}-fold CV on training data")
    print(f"  3. Test on: {test_season} ({len(y_test)} series) - NEVER seen during training")
    print()
    
    models = get_all_models()
    results = []
    
    for name, model in models.items():
        # Cross-validation accuracy (on training data only)
        cv_result = model.evaluate(X_train, y_train, cv=cv)
        cv_acc = cv_result['mean_accuracy']
        cv_std = cv_result['std']
        
        # Train on ALL training data
        model.train(X_train, y_train, feature_names=feature_cols)
        
        # Test on held-out test set
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Count correct predictions
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
    
    # Print results table
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"\n{'Model':<22} {'CV Acc':<12} {'Test Acc':<12} {'Correct':<10}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<22} {row['CV Accuracy']:.1%} ±{row['CV Std']:.1%}   "
              f"{row['Test Accuracy']:.1%}         {row['Test Correct']}")
    
    print("-" * 70)
    
    # Find best models
    best_cv = results_df.loc[results_df['CV Accuracy'].idxmax()]
    best_test = results_df.loc[results_df['Test Accuracy'].idxmax()]
    
    print(f"\n🏆 Best CV Accuracy:   {best_cv['Model']} ({best_cv['CV Accuracy']:.1%})")
    print(f"🏆 Best Test Accuracy: {best_test['Model']} ({best_test['Test Accuracy']:.1%})")
    
    return results_df, models


def show_test_predictions(test_season='2022-23'):
    """Show individual predictions for test set."""
    X_train, X_test, y_train, y_test, feature_cols, df = get_train_test_split(test_season)
    test_df = df[df['season'] == test_season].reset_index(drop=True)
    
    # Train best model (Logistic Regression based on earlier results)
    model = LogisticModel()
    model.train(X_train, y_train, feature_names=feature_cols)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print("\n" + "=" * 70)
    print(f"INDIVIDUAL PREDICTIONS - {test_season} PLAYOFFS")
    print("=" * 70)
    print(f"\n{'Matchup':<20} {'Predicted':<12} {'Actual':<12} {'Prob':<8} {'Result'}")
    print("-" * 70)
    
    for i, row in test_df.iterrows():
        matchup = f"{row['team_a']} vs {row['team_b']}"
        pred_winner = row['team_a'] if y_pred[i] == 1 else row['team_b']
        actual_winner = row['team_a'] if y_test[i] == 1 else row['team_b']
        prob = y_prob[i] if y_pred[i] == 1 else 1 - y_prob[i]
        correct = "✓" if y_pred[i] == y_test[i] else "✗"
        
        print(f"{matchup:<20} {pred_winner:<12} {actual_winner:<12} {prob:.0%}      {correct}")
    
    print("-" * 70)
    print(f"Total: {(y_pred == y_test).sum()}/{len(y_test)} correct")


def run_full_evaluation(test_season='2022-23'):
    """Run complete evaluation pipeline."""
    get_data_info(test_season)
    print()
    results_df, models = evaluate_all_models(test_season)
    show_test_predictions(test_season)
    return results_df


if __name__ == "__main__":
    run_full_evaluation()
