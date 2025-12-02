"""Hyperparameter tuning with GridSearchCV."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ..data.processing import make_dataset, get_feature_columns


def tune_all_models(cv=5):
    """
    Run GridSearchCV on all models to find best hyperparameters.
    
    Returns:
        dict with best params and scores for each model
    """
    # Load data
    df = make_dataset()
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df['team_a_won'].values
    
    # Scale for models that need it
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("=" * 70)
    print("HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)
    print(f"\nDataset: {len(df)} samples, {len(feature_cols)} features")
    print(f"Cross-validation: {cv}-fold")
    print()
    
    results = {}
    
    # 1. Logistic Regression
    print("Tuning Logistic Regression...")
    lr_params = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        lr_params, cv=cv, scoring='accuracy', n_jobs=-1
    )
    lr_grid.fit(X_scaled, y)
    results['Logistic Regression'] = {
        'best_params': lr_grid.best_params_,
        'best_score': lr_grid.best_score_,
        'default_score': cross_val_score(
            LogisticRegression(max_iter=1000, random_state=42), 
            X_scaled, y, cv=cv
        ).mean()
    }
    print(f"  Best: {lr_grid.best_score_:.1%} | Params: {lr_grid.best_params_}")
    
    # 2. XGBoost
    print("Tuning XGBoost...")
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        xgb_params, cv=cv, scoring='accuracy', n_jobs=-1
    )
    xgb_grid.fit(X, y)
    results['XGBoost'] = {
        'best_params': xgb_grid.best_params_,
        'best_score': xgb_grid.best_score_,
        'default_score': cross_val_score(
            XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=42),
            X, y, cv=cv
        ).mean()
    }
    print(f"  Best: {xgb_grid.best_score_:.1%} | Params: {xgb_grid.best_params_}")
    
    # 3. Random Forest
    print("Tuning Random Forest...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params, cv=cv, scoring='accuracy', n_jobs=-1
    )
    rf_grid.fit(X, y)
    results['Random Forest'] = {
        'best_params': rf_grid.best_params_,
        'best_score': rf_grid.best_score_,
        'default_score': cross_val_score(
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            X, y, cv=cv
        ).mean()
    }
    print(f"  Best: {rf_grid.best_score_:.1%} | Params: {rf_grid.best_params_}")
    
    # 4. SVM
    print("Tuning SVM...")
    svm_params = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        svm_params, cv=cv, scoring='accuracy', n_jobs=-1
    )
    svm_grid.fit(X_scaled, y)
    results['SVM'] = {
        'best_params': svm_grid.best_params_,
        'best_score': svm_grid.best_score_,
        'default_score': cross_val_score(
            SVC(probability=True, random_state=42),
            X_scaled, y, cv=cv
        ).mean()
    }
    print(f"  Best: {svm_grid.best_score_:.1%} | Params: {svm_grid.best_params_}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<22} {'Default':<12} {'Tuned':<12} {'Improvement':<12}")
    print("-" * 70)
    
    for name, res in results.items():
        default = res['default_score']
        tuned = res['best_score']
        improvement = tuned - default
        sign = '+' if improvement >= 0 else ''
        print(f"{name:<22} {default:.1%}        {tuned:.1%}        {sign}{improvement:.1%}")
    
    print("-" * 70)
    
    # Best overall
    best_model = max(results, key=lambda k: results[k]['best_score'])
    print(f"\n🏆 Best Model (after tuning): {best_model} ({results[best_model]['best_score']:.1%})")
    
    return results


if __name__ == "__main__":
    tune_all_models()

