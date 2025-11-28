"""
Step 3.1: Logistic Regression (baseline)
Step 3.2: XGBoost + comparison
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from .data_loader import load_team_stats, load_series_history
from .features import build_matchup_features, get_feature_columns


class PlayoffPredictor:
    """Predicts playoff series winners using ensemble of models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.logreg = LogisticRegression(max_iter=1000)
        self.xgb = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss')
        self.feature_cols = get_feature_columns()
        self.is_trained = False
    
    def train(self, X, y):
        """Train both models."""
        X_scaled = self.scaler.fit_transform(X)
        self.logreg.fit(X_scaled, y)
        self.xgb.fit(X, y)
        self.is_trained = True
    
    def predict_proba(self, X):
        """Return probability team_a wins (ensemble average)."""
        X_scaled = self.scaler.transform(X)
        logreg_prob = self.logreg.predict_proba(X_scaled)[:, 1]
        xgb_prob = self.xgb.predict_proba(X)[:, 1]
        return (logreg_prob + xgb_prob) / 2
    
    def predict(self, X):
        """Return binary prediction (1 = team_a wins)."""
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def evaluate(self, X, y):
        """Return cross-validated accuracy for both models."""
        X_scaled = self.scaler.fit_transform(X)
        lr_acc = cross_val_score(self.logreg, X_scaled, y, cv=5).mean()
        xgb_acc = cross_val_score(self.xgb, X, y, cv=5).mean()
        return {'logreg': lr_acc, 'xgboost': xgb_acc, 'ensemble': (lr_acc + xgb_acc) / 2}
    
    def feature_importance(self):
        """Return feature importance from XGBoost."""
        if not self.is_trained:
            return {}
        return dict(zip(self.feature_cols, self.xgb.feature_importances_))


def train_model():
    """Load data, build features, train model."""
    team_stats = load_team_stats()
    series = load_series_history()
    df = build_matchup_features(series, team_stats)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df['team_a_won'].values
    
    model = PlayoffPredictor()
    model.train(X, y)
    return model, df


if __name__ == "__main__":
    team_stats = load_team_stats()
    series = load_series_history()
    df = build_matchup_features(series, team_stats)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df['team_a_won'].values
    
    model = PlayoffPredictor()
    scores = model.evaluate(X, y)
    
    print("Cross-validated accuracy:")
    print(f"  Logistic Regression: {scores['logreg']:.1%}")
    print(f"  XGBoost:             {scores['xgboost']:.1%}")
    print(f"  Ensemble:            {scores['ensemble']:.1%}")
    
    # Train and show feature importance
    model.train(X, y)
    print("\nTop 5 features (XGBoost importance):")
    importance = model.feature_importance()
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {imp:.3f}")

