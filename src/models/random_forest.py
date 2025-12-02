"""Random Forest model for playoff series prediction."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


class RandomForestModel:
    """Random Forest classifier for predicting playoff series winners."""
    
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=5):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X, y, feature_names=None):
        """Train the model on features X and labels y."""
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """Return binary predictions (1 = team_a wins)."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Return probability that team_a wins."""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X, y, cv=5):
        """Return cross-validated accuracy."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    def feature_importance(self):
        """Return feature importance scores."""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        return importances
